"""
Train gat_quant on daily market data with sparse report LLM signals.

Data semantics:
- sample dates are daily trading dates from the price files;
- market features enter every sample date when available;
- report LLM features are sparse and keyed by stock_code + aligned_trade_date;
- Guba features are deliberately zeroed out in the Dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from data import DailyReportDataConfig, create_daily_report_dataloaders, NODE_FEATURE_DIM
from models import MarketSentimentGAT
from train import Trainer
from utils import count_parameters, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="训练日频市场 + 稀疏研报 gat_quant")
    parser.add_argument("--panel_csv", type=str, default="./paper_data/paper_event_panel.csv")
    parser.add_argument("--price_dir", type=str, default="/opt/price")
    parser.add_argument("--factor_dir", type=str, default="/opt/tushare_factors")
    parser.add_argument("--universe_csv", type=str, default="")
    parser.add_argument("--num_stocks", type=int, default=0, help="选股数量；0 表示使用全部可用股票")
    parser.add_argument("--target_horizon", type=int, default=60)
    parser.add_argument("--price_lookback", type=int, default=60)
    parser.add_argument("--corr_top_k", type=int, default=4)
    parser.add_argument("--spill_top_k", type=int, default=4)
    parser.add_argument("--corr_threshold", type=float, default=0.3,
                        help="相关性边阈值，低于此值不建边；0.0 = 不过滤")
    parser.add_argument(
        "--feature_mode",
        type=str,
        default="market_report",
        choices=["market_report", "market_only", "report_only"],
        help="日频特征开关：market_report=市场+研报；market_only=纯市场；report_only=只保留稀疏研报",
    )
    parser.add_argument("--graph_mode", type=str, default="full", choices=["full", "no_stock_edges", "sector_style_only"])
    parser.add_argument("--selection_mode", type=str, default="report_count", choices=["report_count", "universe_order"])
    parser.add_argument("--start_date", type=str, default="")
    parser.add_argument("--end_date", type=str, default="")
    parser.add_argument("--require_llm_coverage", action="store_true", default=True,
                        help="过滤掉没有 LLM 打分研报的股票（默认开启）")
    parser.add_argument("--no_require_llm_coverage", dest="require_llm_coverage", action="store_false")
    parser.add_argument("--max_price_start", type=str, default="2022-01-01",
                        help="排除价格数据起始晚于此日期的股票")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--require_full_targets",
        action="store_true",
        help="要求每个样本日所有股票都有未来收益标签；默认关闭，用 target mask 保留更长历史样本。",
    )
    parser.add_argument(
        "--min_valid_targets",
        type=int,
        default=0,
        help="每个样本日至少需要多少只股票有有效标签；0 表示默认使用 80% 股票数。",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "quasi_likelihood", "ic"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="./outputs_daily_report")
    parser.add_argument("--log_dir", type=str, default="./logs_daily_report")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_layers < 2:
        raise ValueError("当前 MarketSentimentGAT 配置下需要 --num_layers >= 2。")

    set_seed(args.seed)

    config = get_config()
    config.data.num_stocks = args.num_stocks

    config.model.hidden_dim = args.hidden_dim
    config.model.num_heads = args.num_heads
    config.model.num_gnn_layers = args.num_layers
    config.model.dropout = args.dropout

    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.loss_type = args.loss_type
    config.training.device = args.device
    config.training.scheduler_type = "none"
    config.training.early_stopping_patience = max(3, args.epochs)
    config.training.seed = args.seed

    config.output_dir = args.output_dir
    config.log_dir = args.log_dir
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    data_config = DailyReportDataConfig(
        panel_csv=args.panel_csv,
        price_dir=args.price_dir,
        factor_dir=args.factor_dir,
        universe_csv=args.universe_csv or None,
        num_stocks=args.num_stocks,
        target_horizon=args.target_horizon,
        max_samples=None if args.max_samples <= 0 else args.max_samples,
        require_full_targets=args.require_full_targets,
        min_valid_targets=args.min_valid_targets,
        price_lookback=args.price_lookback,
        corr_top_k=args.corr_top_k,
        spill_top_k=args.spill_top_k,
        corr_threshold=args.corr_threshold,
        graph_mode=args.graph_mode,
        feature_mode=args.feature_mode,
        selection_mode=args.selection_mode,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        seed=args.seed,
        require_llm_coverage=args.require_llm_coverage,
        max_price_start=args.max_price_start,
    )

    print("=" * 72)
    print("数据口径：日频市场 + 稀疏研报")
    print("- sample date 来自 /opt/price 交易日历，不来自研报事件日期")
    print("- 市场量价特征每日进入模型")
    print("- 研报 LLM 按 stock_code + aligned_trade_date 稀疏填入")
    print("- 无研报时研报特征为 0，has_report_llm = 0")
    print("- 股吧特征全部置 0，不作为有效输入")
    print("- 默认固定股票节点；缺失未来收益标签的位置用 target mask 排除出 loss 和指标")
    print(f"- feature_mode = {args.feature_mode}")
    print(f"- graph_mode = {args.graph_mode}")
    print(f"- require_full_targets = {args.require_full_targets}")
    print(f"- min_valid_targets = {args.min_valid_targets or '80% of num_stocks'}")
    print("=" * 72)

    print("创建 DailyReport DataLoader...")
    train_loader, val_loader, test_loader = create_daily_report_dataloaders(
        data_config,
        batch_size=args.batch_size,
    )
    train_ds = train_loader.dataset
    val_ds   = val_loader.dataset
    test_ds  = test_loader.dataset
    print(
        f"训练日期: {len(train_ds)}, "
        f"验证日期: {len(val_ds)}, "
        f"测试日期: {len(test_ds)}"
    )
    print(f"股票数: {train_ds.num_stocks}")
    print(f"全量日期范围: {train_ds.all_sample_dates[0].date()} to {train_ds.all_sample_dates[-1].date()}")
    print(f"训练切分: {train_ds.sample_dates[0].date()} to {train_ds.sample_dates[-1].date()}")
    print(f"验证切分: {val_ds.sample_dates[0].date()} to {val_ds.sample_dates[-1].date()} (gap={data_config.target_horizon}d)")
    print(f"测试切分: {test_ds.sample_dates[0].date()} to {test_ds.sample_dates[-1].date()} (gap={data_config.target_horizon}d)")

    sample_graph, sample_targets, sample_target_mask = next(iter(train_loader))
    print(f"样本标签形状: {tuple(sample_targets.shape)}")
    print(f"样本有效标签比例: {sample_target_mask.float().mean().item():.4f}")
    print(f"样本边类型: {sample_graph.edge_types}")

    model = MarketSentimentGAT(
        node_feature_dim=NODE_FEATURE_DIM,
        hidden_dim=config.model.hidden_dim,
        num_heads=config.model.num_heads,
        num_gnn_layers=config.model.num_gnn_layers,
        dropout=config.model.dropout,
        output_dim=config.model.output_dim,
    )
    print(f"模型参数量: {count_parameters(model):,}")

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    history = trainer.train()
    print("日频市场 + 稀疏研报训练完成。")
    return history


if __name__ == "__main__":
    main()

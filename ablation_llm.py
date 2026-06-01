"""
消融实验：LLM 情绪维度的贡献

对比三种 feature_mode：
  market_report  —— 完整特征（市场因子 + LLM 情绪）
  market_only    —— 去掉 LLM，纯市场因子
  report_only    —— 去掉市场因子，纯 LLM 情绪

数据只加载一次（共享 shared_state），三组实验在相同的股票集和日期上训练，
结果存入各自的 output_dir，最后打印对比表。

用法:
    python ablation_llm.py \
        --panel_csv ./paper_data/paper_event_panel.csv \
        --price_dir /opt/price \
        --factor_dir /opt/tushare_factors \
        --universe_csv /path/to/csi1000.csv \
        --start_date 2021-01-01 \
        --end_date 2024-12-31 \
        --epochs 10 \
        --output_dir ./ablation_outputs
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config
from data import (
    DailyReportDataConfig,
    DailyReportGraphDataset,
    NODE_FEATURE_DIM,
)
from data.daily_report_dataset import collate_temporal_graphs
from models import MarketSentimentGAT
from train import Trainer
from utils import count_parameters, set_seed


MODES = ["market_report", "market_only", "report_only"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--panel_csv",    default="./paper_data/paper_event_panel.csv")
    p.add_argument("--price_dir",    default="/opt/price")
    p.add_argument("--factor_dir",   default="/opt/tushare_factors")
    p.add_argument("--universe_csv", default="")
    p.add_argument("--start_date",   default="")
    p.add_argument("--end_date",     default="")
    p.add_argument("--num_stocks",   type=int,   default=0)
    p.add_argument("--target_horizon", type=int, default=60)
    p.add_argument("--price_lookback", type=int, default=60)
    p.add_argument("--corr_top_k",   type=int,   default=4)
    p.add_argument("--spill_top_k",  type=int,   default=4)
    p.add_argument("--require_llm_coverage", action="store_true", default=True)
    p.add_argument("--no_require_llm_coverage", dest="require_llm_coverage", action="store_false")
    p.add_argument("--max_price_start", default="2022-01-01")
    p.add_argument("--modes", nargs="+", default=MODES,
                   choices=MODES, help="要跑哪几个 feature_mode")
    p.add_argument("--epochs",      type=int,   default=10)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--hidden_dim",  type=int,   default=64)
    p.add_argument("--num_heads",   type=int,   default=2)
    p.add_argument("--num_layers",  type=int,   default=2)
    p.add_argument("--dropout",     type=float, default=0.3)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--loss_type",   default="mse", choices=["mse", "ic", "quasi_likelihood"])
    p.add_argument("--device",      default="cpu")
    p.add_argument("--output_dir",  default="./ablation_outputs")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def build_base_config(args) -> DailyReportDataConfig:
    return DailyReportDataConfig(
        panel_csv=args.panel_csv,
        price_dir=args.price_dir,
        factor_dir=args.factor_dir,
        universe_csv=args.universe_csv or None,
        num_stocks=args.num_stocks,
        target_horizon=args.target_horizon,
        price_lookback=args.price_lookback,
        corr_top_k=args.corr_top_k,
        spill_top_k=args.spill_top_k,
        require_llm_coverage=args.require_llm_coverage,
        max_price_start=args.max_price_start,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        seed=args.seed,
        feature_mode="market_report",  # 占位，每次运行时覆盖
    )


def make_dataloaders(
    base_config: DailyReportDataConfig,
    feature_mode: str,
    batch_size: int,
    shared_state=None,
    feature_stats=None,
):
    cfg = copy.copy(base_config)
    cfg.feature_mode = feature_mode

    train_ds = DailyReportGraphDataset(
        cfg, split="train",
        shared_state=shared_state,
        feature_stats=feature_stats,
    )
    val_ds = DailyReportGraphDataset(
        cfg, split="val",
        shared_state=train_ds.shared_state,
        feature_stats=train_ds.feature_stats,
    )
    test_ds = DailyReportGraphDataset(
        cfg, split="test",
        shared_state=train_ds.shared_state,
        feature_stats=train_ds.feature_stats,
    )

    kwargs = dict(batch_size=batch_size, num_workers=0, collate_fn=collate_temporal_graphs)
    return (
        DataLoader(train_ds, shuffle=False, **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
        train_ds,
    )


def build_model(args) -> MarketSentimentGAT:
    return MarketSentimentGAT(
        node_feature_dim=NODE_FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_gnn_layers=args.num_layers,
        dropout=args.dropout,
        output_dim=1,
    )


def run_one(args, mode: str, base_config, shared_state, feature_stats) -> dict:
    print(f"\n{'='*60}")
    print(f"  feature_mode = {mode}")
    print(f"{'='*60}")

    out_dir = os.path.join(args.output_dir, mode)
    os.makedirs(out_dir, exist_ok=True)

    set_seed(args.seed)

    train_loader, val_loader, test_loader, train_ds = make_dataloaders(
        base_config, mode, args.batch_size, shared_state, feature_stats
    )

    config = get_config()
    config.model.hidden_dim    = args.hidden_dim
    config.model.num_heads     = args.num_heads
    config.model.num_gnn_layers = args.num_layers
    config.model.dropout       = args.dropout
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr
    config.training.loss_type  = args.loss_type
    config.training.device     = args.device
    config.training.scheduler_type = "none"
    config.training.early_stopping_patience = max(3, args.epochs)
    config.training.seed       = args.seed
    config.output_dir = out_dir
    config.log_dir    = out_dir

    model = build_model(args)
    print(f"参数量: {count_parameters(model):,}")

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    history = trainer.train()

    # 提取最终测试指标
    result = {"mode": mode}
    if history.get("metrics"):
        last = history["metrics"][-1]
        result.update({f"final_{k}": v for k, v in last.items()})
    result["best_val_loss"] = min(history.get("val_loss", [float("nan")]))

    with open(os.path.join(out_dir, "ablation_result.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def print_table(results: list[dict]):
    if not results:
        return
    # 取所有出现过的 metric key
    metric_keys = sorted({k for r in results for k in r if k not in ("mode",)})
    header = f"{'mode':<20}" + "".join(f"{k:>18}" for k in metric_keys)
    print(f"\n{'='*len(header)}")
    print("消融实验结果汇总")
    print("="*len(header))
    print(header)
    print("-"*len(header))
    for r in results:
        row = f"{r['mode']:<20}"
        for k in metric_keys:
            v = r.get(k, float("nan"))
            row += f"{v:>18.4f}" if isinstance(v, float) else f"{str(v):>18}"
        print(row)
    print("="*len(header))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    base_config = build_base_config(args)

    # ── 数据只加载一次 ──────────────────────────────────────────────────────
    print("\n加载数据（只需一次，所有实验共享）…")
    ref_config = copy.copy(base_config)
    ref_config.feature_mode = "market_report"
    ref_ds = DailyReportGraphDataset(ref_config, split="train")
    shared_state  = ref_ds.shared_state
    feature_stats = ref_ds.feature_stats
    print(f"股票数: {ref_ds.num_stocks}，行业数: {ref_ds.num_sectors}")
    print(f"训练样本日: {len(ref_ds)}")

    # ── 逐 mode 训练 ────────────────────────────────────────────────────────
    results = []
    for mode in args.modes:
        r = run_one(args, mode, base_config, shared_state, feature_stats)
        results.append(r)

    # ── 汇总 ────────────────────────────────────────────────────────────────
    print_table(results)
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存至 {args.output_dir}/")


if __name__ == "__main__":
    main()

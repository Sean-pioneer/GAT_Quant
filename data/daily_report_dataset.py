"""
Daily market panel Dataset with sparse report LLM signals.

This Dataset is different from ``paper_event_dataset``:

- sample dates come from the daily price calendar, not report event dates;
- market features are present on normal trading days;
- report LLM features are merged sparsely by ``stock_code + aligned_trade_date``;
- Guba features are always zeroed and are not used as valid inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, HeteroData


# ── 节点特征分组 ────────────────────────────────────────────────────────────
# 5 个 LLM 情绪维度（仅研报日有值，其余日期前向填充）
REPORT_LLM_FEATURES = [
    "fundamental_expectation",
    "rating_momentum",
    "short_term_view",
    "long_term_view",
    "risk_alertness",
]

# 7 个市场因子（每日均有值）
MARKET_FEATURES = [
    "overnight_corr",  # 短期情绪/反转（tushare）
    "roe",             # 长期质量（tushare，季度前填充）
    "mom_20d",         # 短期动量
    "mom_60d",         # 中期动量
    "vol_20d",         # 短期波动率
    "turnover_20d",    # 20日均换手率（tushare）
    "pb",              # 长期估值
]

# 1 个掩码：区分"当日有新研报"与"前向填充值"
MASK_FEATURES = ["has_report_llm"]
# 5 个板块 one-hot（由股票代码推断，非外部数据）
BOARD_BUCKETS = ["SH_Main", "SZ_Main", "ChiNext", "STAR", "Other"]
BOARD_FEATURES = [f"board_{name}" for name in BOARD_BUCKETS]

# 最终节点特征：5+7+1+5 = 18 维，顺序与 NODE_FEATURE_DIM 严格对应
NODE_FEATURE_COLUMNS = (
    REPORT_LLM_FEATURES
    + MARKET_FEATURES
    + MASK_FEATURES
    + BOARD_FEATURES
)

NODE_FEATURE_DIM = 18


def _normalize_stock_code(value) -> str:
    """将各种格式的股票代码（浮点、带后缀、纯数字）统一为6位纯数字字符串。"""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return digits[-6:].zfill(6)
    return text


def _board_from_code(code: str) -> str:
    """根据股票代码前缀推断所属交易板块（SH_Main / SZ_Main / ChiNext / STAR / Other）。"""
    code = _normalize_stock_code(code)
    if code.startswith("688"):
        return "STAR"
    if code.startswith(("300", "301")):
        return "ChiNext"
    if code.startswith(("000", "001", "002", "003")):
        return "SZ_Main"
    if code.startswith(("600", "601", "603", "605")):
        return "SH_Main"
    return "Other"


def _empty_edge_index() -> torch.Tensor:
    """返回形状为 (2, 0) 的空边索引张量，用于无边情况的占位符。"""
    return torch.empty((2, 0), dtype=torch.long)


def collate_temporal_graphs(batch: List[Tuple]) -> Tuple:
    """DataLoader collate 函数：将多个样本的图、目标值和掩码合并为批量张量。"""
    batched_graph = Batch.from_data_list([sample[0] for sample in batch])
    targets = torch.stack([sample[1] for sample in batch], dim=0)
    target_masks = torch.stack([sample[2] for sample in batch], dim=0)
    return batched_graph, targets, target_masks


DAILY_SCALE_FEATURES = MARKET_FEATURES + REPORT_LLM_FEATURES + ["event_count_log"]


@dataclass
class DailyReportDataConfig:
    """Configuration for daily market + sparse report training."""

    panel_csv: str = "./paper_data/paper_event_panel.csv"
    price_dir: str = "/opt/price"
    factor_dir: str = "/opt/tushare_factors"
    universe_csv: Optional[str] = None
    industry_csv: str = "./data/stock_industry.csv"
    num_stocks: int = 128        # 0 = 使用全部可用股票
    num_sectors: int = 8         # 实际由 stock_industry.csv 中的行业数覆盖
    target_horizon: int = 60     # 预测 T+horizon 日收益率
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    max_samples: Optional[int] = None
    require_full_targets: bool = False
    min_valid_targets: int = 0
    price_lookback: int = 60     # 构建相关/溢出边时的历史收益率窗口
    corr_top_k: int = 4          # 每只股票保留 top-k 相关性边
    spill_top_k: int = 4         # 每只股票保留 top-k 波动溢出边
    corr_threshold: float = 0.3  # 低于此绝对相关系数的边不建立，0.0 = 不过滤
    graph_mode: str = "full"     # "full"|"no_stock_edges"|"sector_style_only"
    feature_mode: str = "market_report"  # 消融用："market_report"|"market_only"|"report_only"
    selection_mode: str = "report_count" # 股票排序方式："report_count"|"universe_order"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    seed: int = 42
    require_llm_coverage: bool = True   # 过滤掉没有 LLM 打分的股票
    max_price_start: str = "2022-01-01" # 过滤掉价格数据起始晚于此日期的股票


def _find_price_file(price_dir: Path, code: str) -> Optional[Path]:
    """在 price_dir 下查找指定股票的价格 CSV 文件，自动匹配 sh/sz 前缀，未找到返回 None。"""
    code = _normalize_stock_code(code)
    prefix = "sh" if code.startswith(("6", "9")) else "sz"
    direct = price_dir / f"{prefix}.{code}.csv"
    if direct.exists():
        return direct
    matches = list(price_dir.glob(f"*.{code}.csv"))
    return matches[0] if matches else None


def _first_existing_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """从候选列名列表中返回第一个在 columns 中实际存在的列名，全不存在则返回 None。"""
    column_set = set(columns)
    for name in candidates:
        if name in column_set:
            return name
    return None


def _robust_center_scale(values: pd.Series) -> Tuple[float, float]:
    """计算鲁棒归一化参数 (中位数, IQR-scale)，自动跳过 inf/NaN，常数列退化为 scale=1。"""
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.0, 1.0
    center = float(clean.median())
    q25 = float(clean.quantile(0.25))
    q75 = float(clean.quantile(0.75))
    scale = (q75 - q25) / 1.349  # IQR/1.349 在正态分布下等价于标准差，对异常值鲁棒
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(clean.std(ddof=0))  # 退化到普通标准差
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return center, scale


class DailyReportGraphDataset(Dataset):
    """
    Daily stock graph Dataset.

    Each item is one trading date. The target is future ``target_horizon``
    trading-day return for the selected stocks on that date.
    """

    def __init__(
        self,
        config: DailyReportDataConfig,
        split: str = "train",
        feature_stats: Optional[Mapping[str, Mapping[str, float]]] = None,
        shared_state: Optional[Mapping[str, object]] = None,
    ):
        """
        初始化数据集。

        首次构造（train split）时加载全部数据并拟合归一化统计量；
        val/test 通过 shared_state 和 feature_stats 复用 train 的结果，避免重复 IO。
        """
        self.config = config
        self.split = split
        self.num_stocks = config.num_stocks
        self.num_sectors = config.num_sectors
        self.node_feature_dim = NODE_FEATURE_DIM
        self._edge_cache: Dict[Tuple[pd.Timestamp, Tuple[str, ...], str], Tuple[torch.Tensor, torch.Tensor]] = {}

        self.price_dir = Path(config.price_dir)
        self.report_panel = self._load_report_panel()
        self._industry_map = self._load_industry_map()

        if shared_state is None:
            # 首次加载（通常是 train split）：从头构建股票集和日历
            self.stock_codes = self._select_universe()
            self.num_stocks = len(self.stock_codes)  # 全量时由实际数量决定
            self.daily_raw, self.returns_by_code = self._build_daily_raw_table()
            self.all_dates = list(pd.DatetimeIndex(self.daily_raw["date"].unique()).sort_values())
            self.date_to_idx = {date: idx for idx, date in enumerate(self.all_dates)}
            self.all_sample_dates, self.train_end, self.val_end = self._prepare_sample_date_index()
        else:
            # val/test split 直接复用 train 的股票集、日历和切分点，避免重复 IO
            self.stock_codes = list(shared_state["stock_codes"])
            self.num_stocks = len(self.stock_codes)
            self.daily_raw = shared_state["daily_raw"]  # type: ignore[assignment]
            self.returns_by_code = shared_state["returns_by_code"]  # type: ignore[assignment]
            self.all_dates = list(shared_state["all_dates"])
            self.date_to_idx = dict(shared_state["date_to_idx"])
            self.all_sample_dates = list(shared_state["all_sample_dates"])
            self.train_end = int(shared_state["train_end"])
            self.val_end = int(shared_state["val_end"])

        self._sector_node_map = self._build_sector_map()

        self.shared_state = {
            "stock_codes": self.stock_codes,
            "daily_raw": self.daily_raw,
            "returns_by_code": self.returns_by_code,
            "all_dates": self.all_dates,
            "date_to_idx": self.date_to_idx,
            "all_sample_dates": self.all_sample_dates,
            "train_end": self.train_end,
            "val_end": self.val_end,
        }

        self.feature_stats = (
            dict(feature_stats) if feature_stats is not None else self._fit_feature_stats()
        )
        self._prepare_model_table()
        self._set_split_dates()

    def _load_report_panel(self) -> pd.DataFrame:
        """读取研报事件面板 CSV，校验必要列，清洗股票代码和日期，返回 DataFrame。"""
        required = {
            "stock_code",
            "aligned_trade_date",
            *REPORT_LLM_FEATURES,
            "has_report_llm",
            "event_count",
            "board",
        }
        df = pd.read_csv(self.config.panel_csv, dtype={"stock_code": str})
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"report panel is missing required columns: {missing}")

        df = df.copy()
        df["stock_code"] = df["stock_code"].map(_normalize_stock_code)
        df["aligned_trade_date"] = pd.to_datetime(df["aligned_trade_date"], errors="coerce")
        numeric_cols = [*REPORT_LLM_FEATURES, "has_report_llm", "event_count"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["has_report_llm"] = df["has_report_llm"].fillna(0.0).clip(0.0, 1.0)
        return df.dropna(subset=["stock_code", "aligned_trade_date"])

    def _read_universe_codes(self) -> List[str]:
        """读取候选股票代码列表；未指定 universe_csv 时退化为面板中出现过的所有股票。"""
        if not self.config.universe_csv:
            return sorted(self.report_panel["stock_code"].unique().tolist())

        universe = pd.read_csv(self.config.universe_csv, dtype=str)
        code_col = None
        for candidate in ["stock_code", "code", "证券代码", "成分券代码", "代码"]:
            if candidate in universe.columns:
                code_col = candidate
                break
        if code_col is None:
            code_col = universe.columns[0]

        codes = (
            universe[code_col]
            .astype(str)
            .str.extract(r"(\d{6})")[0]
            .dropna()
            .map(_normalize_stock_code)
            .drop_duplicates()
            .tolist()
        )
        return codes

    def _price_start_date(self, code: str) -> Optional[pd.Timestamp]:
        """返回指定股票价格文件中最早的交易日期，用于过滤数据覆盖不足的股票。"""
        path = _find_price_file(self.price_dir, code)
        if path is None:
            return None
        try:
            df = pd.read_csv(path, usecols=lambda c: c.lstrip("﻿") in
                             {"交易所行情日期", "日期", "date", "Date"})
            df.columns = [str(c).lstrip("﻿") for c in df.columns]
            date_col = _first_existing_column(list(df.columns),
                                              ["交易所行情日期", "日期", "date", "Date"])
            if date_col is None:
                return None
            return pd.to_datetime(df[date_col], errors="coerce").dropna().min()
        except Exception:
            return None

    def _select_universe(self) -> List[str]:
        """从候选股票中依次过滤（LLM覆盖、价格起始日）并按 selection_mode 排序，返回最终股票列表。"""
        candidates = self._read_universe_codes()
        price_available = [code for code in candidates if _find_price_file(self.price_dir, code) is not None]

        # 过滤1：必须有 LLM 打分的研报（排除完全无覆盖 + 有记录但无打分）
        if self.config.require_llm_coverage:
            llm_codes = set(
                self.report_panel[self.report_panel["has_report_llm"] > 0.5]["stock_code"].unique()
            )
            before = len(price_available)
            price_available = [c for c in price_available if c in llm_codes]
            print(f"[universe] LLM 覆盖过滤：{before} → {len(price_available)} 只")

        # 过滤2：价格数据起始日期不得晚于 max_price_start
        if self.config.max_price_start:
            cutoff = pd.Timestamp(self.config.max_price_start)
            before = len(price_available)
            price_available = [
                c for c in price_available
                if (s := self._price_start_date(c)) is not None and s <= cutoff
            ]
            print(f"[universe] 起始日期过滤（<= {self.config.max_price_start}）：{before} → {len(price_available)} 只")

        report_counts = (
            self.report_panel[self.report_panel["has_report_llm"] > 0.5]
            .groupby("stock_code")
            .size()
            .to_dict()
        )
        if self.config.selection_mode == "report_count":
            ordered = sorted(price_available, key=lambda code: (-int(report_counts.get(code, 0)), code))
        elif self.config.selection_mode == "universe_order":
            ordered = price_available
        else:
            raise ValueError("selection_mode must be 'report_count' or 'universe_order'")

        # num_stocks <= 0 表示使用全部可用股票
        if self.config.num_stocks <= 0:
            return ordered
        if len(price_available) < self.config.num_stocks:
            raise ValueError(
                f"only {len(price_available)} universe stocks have price files; "
                f"num_stocks={self.config.num_stocks}"
            )
        return ordered[: self.config.num_stocks]

    def _read_price_features(self, code: str) -> Tuple[pd.DataFrame, pd.Series]:
        """读取单只股票价格文件，计算动量/波动/估值特征，合并 tushare 因子，返回日级特征表和日收益率序列。"""
        path = _find_price_file(self.price_dir, code)
        if path is None:
            raise FileNotFoundError(f"missing price file for {code}")

        price = pd.read_csv(path)
        price.columns = [str(col).lstrip("\ufeff") for col in price.columns]
        date_col = _first_existing_column(price.columns, ["交易所行情日期", "日期", "date", "Date"])
        close_col = _first_existing_column(price.columns, ["收盘价", "close", "Close"])
        pct_col = _first_existing_column(price.columns, ["涨跌幅", "pct_chg", "return"])
        pe_col = _first_existing_column(price.columns, ["滚动市盈率", "pe", "PE"])
        pb_col = _first_existing_column(price.columns, ["市净率", "pb", "PB"])
        turnover_col = _first_existing_column(price.columns, ["换手率", "turnover"])
        if date_col is None or close_col is None:
            raise ValueError(f"price file {path} lacks date/close columns")

        frame = pd.DataFrame()
        frame["date"] = pd.to_datetime(price[date_col], errors="coerce")
        frame["stock_code"] = code
        close = pd.to_numeric(price[close_col], errors="coerce")
        if pct_col is not None:
            returns = pd.to_numeric(price[pct_col], errors="coerce") / 100.0
        else:
            returns = close.pct_change()

        frame["mom_20d"] = close / close.shift(20) - 1.0
        frame["mom_60d"] = close / close.shift(60) - 1.0
        frame["vol_20d"] = returns.rolling(20, min_periods=10).std()
        frame["pb"] = pd.to_numeric(price[pb_col], errors="coerce") if pb_col else np.nan
        frame["target"] = close.shift(-self.config.target_horizon) / close - 1.0
        frame["board"] = _board_from_code(code)

        # 合并 tushare 因子（overnight_corr、roe、turnover_20d）
        factor_path = Path(self.config.factor_dir) / f"{code}.csv"
        if factor_path.exists():
            fac = pd.read_csv(factor_path, index_col="trade_date", parse_dates=True)
            fac = fac[~fac.index.duplicated(keep="last")].sort_index()
            frame = frame.merge(
                fac[["overnight_corr", "roe", "turnover_20d"]],
                left_on="date", right_index=True, how="left",
            )
            # 停牌日因子缺失，前向填充（停牌期间因子值变化极小）
            for col in ["overnight_corr", "roe", "turnover_20d"]:
                frame[col] = frame[col].ffill()
        else:
            frame["overnight_corr"] = np.nan
            frame["roe"] = np.nan
            frame["turnover_20d"] = np.nan

        if self.config.start_date:
            frame = frame[frame["date"] >= pd.Timestamp(self.config.start_date)]
        else:
            start = self.report_panel["aligned_trade_date"].min()
            frame = frame[frame["date"] >= start]
        if self.config.end_date:
            frame = frame[frame["date"] <= pd.Timestamp(self.config.end_date)]
        else:
            end = self.report_panel["aligned_trade_date"].max()
            frame = frame[frame["date"] <= end]

        frame = frame.dropna(subset=["date", "stock_code"]).reset_index(drop=True)
        return_series = pd.Series(
            pd.to_numeric(returns, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
            index=pd.to_datetime(price[date_col], errors="coerce"),
        ).dropna()
        return_series = return_series[~return_series.index.duplicated(keep="last")].sort_index()
        return frame, return_series

    def _build_daily_raw_table(self) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """拼接所有股票的价格特征，按日期左连接研报 LLM 特征，前向填充 LLM 缺失值，生成全量日级宽表。"""
        price_frames: List[pd.DataFrame] = []
        returns_by_code: Dict[str, pd.Series] = {}
        for code in self.stock_codes:
            frame, returns = self._read_price_features(code)
            price_frames.append(frame)
            returns_by_code[code] = returns

        daily = pd.concat(price_frames, ignore_index=True)
        report_cols = [
            "aligned_trade_date",
            "stock_code",
            *REPORT_LLM_FEATURES,
            "has_report_llm",
            "event_count",
        ]
        reports = self.report_panel[self.report_panel["stock_code"].isin(self.stock_codes)][report_cols].copy()
        report_agg = (
            reports.groupby(["aligned_trade_date", "stock_code"], as_index=False)
            .agg(
                {
                    **{col: "mean" for col in REPORT_LLM_FEATURES},
                    "has_report_llm": "max",
                    "event_count": "sum",
                }
            )
            .rename(columns={"aligned_trade_date": "date"})
        )
        daily = daily.merge(report_agg, on=["date", "stock_code"], how="left")
        daily["has_report_llm"] = daily["has_report_llm"].fillna(0.0).clip(0.0, 1.0)
        daily["event_count"] = daily["event_count"].fillna(0.0)
        daily["event_count_log"] = np.log1p(daily["event_count"].clip(lower=0.0))

        # 先清除非研报日的 LLM 值，再前向填充：
        # 无新研报的交易日沿用最近一篇报告的打分，has_report_llm=0 标记非新鲜数据
        for col in REPORT_LLM_FEATURES:
            daily[col] = daily[col].where(daily["has_report_llm"] > 0.5)
        daily = daily.sort_values(["stock_code", "date"])
        for col in REPORT_LLM_FEATURES:
            daily[col] = daily.groupby("stock_code")[col].ffill()

        daily = daily.sort_values(["date", "stock_code"]).reset_index(drop=True)
        return daily, returns_by_code

    def _min_valid_targets(self) -> int:
        """返回样本日被纳入训练所需的最少有效目标股票数量（默认为 max(3, 80% × num_stocks)）。"""
        if self.config.require_full_targets:
            return self.num_stocks
        if self.config.min_valid_targets > 0:
            return min(self.config.min_valid_targets, self.num_stocks)
        return max(3, int(self.num_stocks * 0.8))

    def _prepare_sample_date_index(self) -> Tuple[List[pd.Timestamp], int, int]:
        """过滤出有效样本日期列表，计算 train/val/test 切分边界索引，返回 (candidates, train_end, val_end)。"""
        counts = (
            self.daily_raw.dropna(subset=["target"])
            .groupby("date")["stock_code"]
            .nunique()
        )
        min_valid_targets = self._min_valid_targets()
        candidates = [
            pd.Timestamp(date)
            for date in self.all_dates
            if int(counts.get(pd.Timestamp(date), 0)) >= min_valid_targets
        ]
        if self.config.max_samples is not None:
            candidates = candidates[: self.config.max_samples]
        if len(candidates) < 3:
            raise ValueError("not enough daily dates; lower num_stocks or check price coverage")

        n = len(candidates)
        train_end = max(1, int(n * self.config.train_ratio))
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        val_end = max(train_end + 1, val_end)
        val_end = min(max(val_end, 2), n)
        return candidates, train_end, val_end

    def _set_split_dates(self) -> None:
        """按 split 从全量样本日期中截取当前切分的日期子集，赋值给 self.sample_dates。

        train/val 和 val/test 边界各留 target_horizon 个交易日的 gap，
        避免训练/验证标签（shift(-horizon)）与后续集价格重叠造成泄漏。
        """
        gap = self.config.target_horizon
        if self.split == "train":
            self.sample_dates = self.all_sample_dates[: self.train_end]
        elif self.split == "val":
            start = min(self.train_end + gap, self.val_end)
            self.sample_dates = self.all_sample_dates[start : self.val_end]
        elif self.split == "test":
            start = min(self.val_end + gap, len(self.all_sample_dates))
            self.sample_dates = self.all_sample_dates[start:]
        else:
            raise ValueError(f"unknown split: {self.split}")
        if not self.sample_dates:
            raise ValueError(f"{self.split} split has no samples")

    def _fit_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """在训练集日期上拟合各特征的鲁棒归一化参数 {feature: {mean, std}}，防止数据泄漏。"""
        # 仅用训练集日期拟合归一化统计量，防止 val/test 信息泄漏
        train_dates = set(self.all_sample_dates[: self.train_end])
        train_rows = self.daily_raw[self.daily_raw["date"].isin(train_dates)]
        stats: Dict[str, Dict[str, float]] = {}
        for col in DAILY_SCALE_FEATURES:
            center, scale = _robust_center_scale(train_rows[col])
            stats[col] = {"mean": center, "std": scale}
        return stats

    def _prepare_model_table(self) -> None:
        """对 daily_raw 做归一化、消融归零和板块 one-hot 编码，生成模型直接读取的 model_table 和 by_date 索引。"""
        df = self.daily_raw.copy()
        for col in DAILY_SCALE_FEATURES:
            mean = self.feature_stats[col]["mean"]
            std = self.feature_stats[col]["std"]
            scaled = (df[col].replace([np.inf, -np.inf], np.nan) - mean) / std
            df[col] = scaled.clip(-8.0, 8.0).fillna(0.0)  # ±8σ 截断肥尾；缺失值补 0

        for col in MASK_FEATURES:
            df[col] = df[col].fillna(0.0).clip(0.0, 1.0)

        df = self._apply_feature_mode(df)

        for bucket, feature in zip(BOARD_BUCKETS, BOARD_FEATURES):
            df[feature] = (df["board"] == bucket).astype(float)

        self.model_table = df
        self.by_date = {
            pd.Timestamp(date): frame.set_index("stock_code", drop=False)
            for date, frame in df.groupby("date", sort=True)
        }

    def _apply_feature_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """按 feature_mode 将市场或 LLM 特征列归零，用于消融实验（不改变列结构）。"""
        # 消融实验：将对应特征组归零而非删除列，保持模型结构不变
        mode = self.config.feature_mode
        valid_modes = {"market_report", "market_only", "report_only"}
        if mode not in valid_modes:
            raise ValueError(f"unknown feature_mode: {mode}; available={sorted(valid_modes)}")

        df = df.copy()
        report_cols = set(REPORT_LLM_FEATURES + ["has_report_llm"])
        market_cols = set(MARKET_FEATURES)
        if mode == "market_report":
            return df
        if mode == "market_only":
            for col in report_cols:
                df[col] = 0.0
            return df
        if mode == "report_only":
            for col in market_cols:
                df[col] = 0.0
            return df
        return df

    def __len__(self) -> int:
        """返回当前切分（train/val/test）包含的样本日期数量。"""
        return len(self.sample_dates)

    def __getitem__(self, idx: int) -> Tuple[HeteroData, torch.Tensor, torch.Tensor]:
        """返回第 idx 个交易日的样本：(HeteroData 图, 目标收益率 [N], 有效目标掩码 [N])。"""
        target_date = pd.Timestamp(self.sample_dates[idx])
        target_frame = self.by_date[target_date]
        stock_codes = self._select_target_stocks(target_frame)

        node_features, popularity = self._features_for_date(target_date, stock_codes)
        graph = self._build_graph(
            date=target_date,
            stock_features=node_features,
            popularity=popularity,
            stock_codes=stock_codes,
        )

        target_values = (
            target_frame.reindex(stock_codes)["target"]
            .replace([np.inf, -np.inf], np.nan)
            .to_numpy(dtype=np.float32)
        )
        target_mask = np.isfinite(target_values)
        targets = np.nan_to_num(target_values, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            graph,
            torch.from_numpy(targets),
            torch.from_numpy(target_mask.astype(np.bool_)),
        )

    def _select_target_stocks(self, frame: pd.DataFrame) -> List[str]:
        """返回当前日期要预测的股票列表；require_full_targets=True 时只保留有有效目标值的股票。"""
        if not self.config.require_full_targets:
            return list(self.stock_codes)
        available = set(frame[frame["target"].notna()].index)
        codes = [code for code in self.stock_codes if code in available]
        if len(codes) < self.num_stocks:
            raise ValueError(f"selected fewer than num_stocks on daily date; got {len(codes)}")
        return codes[: self.num_stocks]

    def _features_for_date(
        self,
        date: pd.Timestamp,
        stock_codes: Sequence[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """提取指定日期的节点特征矩阵 [N, 18] 和研报人气值 [N]，缺失股票行填零。"""
        node = np.zeros((self.num_stocks, self.node_feature_dim), dtype=np.float32)
        popularity = np.zeros(self.num_stocks, dtype=np.float32)

        frame = self.by_date.get(pd.Timestamp(date))
        if frame is None:
            return node, popularity

        available = [code for code in stock_codes if code in frame.index]
        if not available:
            return node, popularity

        positions = [stock_codes.index(code) for code in available]
        node[positions] = frame.loc[available, NODE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        popularity[positions] = frame.loc[available, "has_report_llm"].to_numpy(dtype=np.float32)
        return node, popularity

    def _build_graph(
        self,
        date: pd.Timestamp,
        stock_features: np.ndarray,
        popularity: np.ndarray,
        stock_codes: Sequence[str],
    ) -> HeteroData:
        """将节点特征和四类边（belongs_to/contains/correlates_with/spills_volatility_to）组装为 HeteroData。"""
        data = HeteroData()
        stock_x = torch.from_numpy(stock_features.astype(np.float32))
        data["stock"].x = stock_x
        data["stock"].popularity = torch.from_numpy(popularity.astype(np.float32))
        data["stock"].num_nodes = self.num_stocks

        sector_ids = torch.tensor([self._sector_id(code) for code in stock_codes], dtype=torch.long)

        data["sector"].x = self._bucket_means(stock_x, sector_ids, self.num_sectors)
        data["sector"].num_nodes = self.num_sectors

        stock_nodes = torch.arange(self.num_stocks, dtype=torch.long)
        data["stock", "belongs_to", "sector"].edge_index = torch.stack([stock_nodes, sector_ids])
        data["sector", "contains", "stock"].edge_index = torch.stack([sector_ids, stock_nodes])

        if self.config.graph_mode == "sector_style_only":
            corr_edges = _empty_edge_index()
            spill_edges = _empty_edge_index()
        else:
            corr_edges, spill_edges = self._edges_for_date(stock_codes, date)
        data["stock", "correlates_with", "stock"].edge_index = corr_edges
        data["stock", "spills_volatility_to", "stock"].edge_index = spill_edges
        data["stock"].y = torch.zeros(self.num_stocks, dtype=torch.float32)
        return data

    def _edges_for_date(
        self,
        stock_codes: Sequence[str],
        date: pd.Timestamp,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算指定日期的相关性边和溢出边，结果按 (date, codes, graph_mode) 缓存。"""
        key = (pd.Timestamp(date), tuple(stock_codes), self.config.graph_mode)
        cached = self._edge_cache.get(key)
        if cached is not None:
            return cached
        returns = self._return_matrix(stock_codes, date)
        corr_edges = self._correlation_edges(returns, self.config.corr_top_k)
        spill_edges = self._spillover_edges(returns, self.config.spill_top_k)
        self._edge_cache[key] = (corr_edges, spill_edges)
        return corr_edges, spill_edges

    def _load_industry_map(self) -> Dict[str, str]:
        """读取 stock_industry.csv，返回股票代码 → 行业名称的映射字典；文件不存在时返回空字典。"""
        path = Path(self.config.industry_csv)
        if not path.exists():
            return {}
        df = pd.read_csv(path, dtype=str)
        code_col = "symbol" if "symbol" in df.columns else df.columns[0]
        result: Dict[str, str] = {}
        for _, row in df.iterrows():
            code = _normalize_stock_code(row.get(code_col, ""))
            industry = str(row.get("industry", "")).strip()
            if code and industry and industry != "nan":
                result[code] = industry
        return result

    def _build_sector_map(self) -> Dict[str, int]:
        """将当前股票集涉及的所有行业映射到连续整数 ID，同时更新 self.num_sectors。"""
        industries = sorted(set(
            self._industry_map.get(_normalize_stock_code(code), "Other")
            for code in self.stock_codes
        ))
        # 1-to-1：每个唯一行业对应一个 sector 节点，num_sectors 由此决定而非 config
        self.num_sectors = len(industries)
        return {ind: i for i, ind in enumerate(industries)}

    def _sector_id(self, code: str) -> int:
        """返回指定股票所属行业的 sector 节点 ID，未知行业默认归入 ID 0。"""
        industry = self._industry_map.get(_normalize_stock_code(code), "Other")
        return self._sector_node_map.get(industry, 0)

    @staticmethod
    def _bucket_means(stock_x: torch.Tensor, ids: torch.Tensor, num_buckets: int) -> torch.Tensor:
        """按 sector ID 对股票特征做均值池化，生成 sector 节点特征矩阵 [num_buckets, feat_dim]。"""
        # sector 节点特征 = 该板块内所有股票特征的均值（无独立数据源）
        out = torch.zeros(num_buckets, stock_x.size(1), dtype=stock_x.dtype)
        counts = torch.zeros(num_buckets, 1, dtype=stock_x.dtype)
        out.index_add_(0, ids, stock_x)
        counts.index_add_(0, ids, torch.ones(stock_x.size(0), 1, dtype=stock_x.dtype))
        return out / counts.clamp_min(1.0)

    def _return_matrix(self, stock_codes: Sequence[str], date: pd.Timestamp) -> np.ndarray:
        """提取截至 date 前 price_lookback 天的日收益率矩阵 [num_stocks, lookback]，不足部分左填零。"""
        mat = np.zeros((self.num_stocks, self.config.price_lookback), dtype=np.float32)
        cutoff = pd.Timestamp(date)
        for i, code in enumerate(stock_codes):
            series = self.returns_by_code.get(code, pd.Series(dtype=np.float32))
            if series.empty:
                continue
            history = series.loc[series.index < cutoff].tail(self.config.price_lookback)
            values = history.to_numpy(dtype=np.float32)
            if values.size:
                mat[i, -values.size :] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return mat

    def _correlation_edges(self, returns: np.ndarray, top_k: int) -> torch.Tensor:
        """基于历史收益率的 |Pearson 相关系数|，为每只股票构造 top-k 相关性有向边。"""
        # 用 |Pearson 相关| 度量同向共振，取绝对值是因为负相关同样意味着强联动
        if self.config.graph_mode == "no_stock_edges":
            return _empty_edge_index()
        if returns.shape[0] <= 1 or top_k <= 0:
            return _empty_edge_index()
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(returns)
        score = np.abs(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
        np.fill_diagonal(score, -np.inf)
        if self.config.corr_threshold > 0:
            score[score < self.config.corr_threshold] = -np.inf
        return self._topk_edges(score, top_k)

    def _spillover_edges(self, returns: np.ndarray, top_k: int) -> torch.Tensor:
        """基于滞后1期交叉相关系数，为每只股票构造 top-k 波动溢出有向边（i → j 表示 i 影响 j）。"""
        # 滞后1期交叉相关：股票 i 的昨日收益率与股票 j 的今日收益率的相关性
        # 捕捉波动溢出方向（i→j），有别于同期相关性边
        if self.config.graph_mode == "no_stock_edges":
            return _empty_edge_index()
        if returns.shape[0] <= 1 or returns.shape[1] < 3 or top_k <= 0:
            return _empty_edge_index()
        src_lag = returns[:, :-1].astype(np.float64, copy=False)
        dst_now = returns[:, 1:].astype(np.float64, copy=False)
        src_centered = src_lag - src_lag.mean(axis=1, keepdims=True)
        dst_centered = dst_now - dst_now.mean(axis=1, keepdims=True)
        src_norm = np.linalg.norm(src_centered, axis=1)
        dst_norm = np.linalg.norm(dst_centered, axis=1)
        denom = src_norm[:, None] * dst_norm[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            score = (src_centered @ dst_centered.T) / denom
        score = np.abs(np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0))
        np.fill_diagonal(score, -np.inf)
        if self.config.corr_threshold > 0:
            score[score < self.config.corr_threshold] = -np.inf
        return self._topk_edges(score, top_k)

    def _topk_edges(self, score: np.ndarray, top_k: int) -> torch.Tensor:
        """将得分矩阵转换为每行保留 top-k 邻居的有向边张量 [2, N×k]。"""
        n = score.shape[0]
        k = min(top_k, max(n - 1, 0))
        if n <= 1 or k == 0:
            return _empty_edge_index()
        src: List[int] = []
        dst: List[int] = []
        for i in range(n):
            row = score[i]
            finite = np.isfinite(row)
            if finite.sum() == 0:
                # 全 NaN（如该股票历史数据全缺）时退化为环形邻居，保持图连通
                neighbors = [(i + step + 1) % n for step in range(k)]
            else:
                order = np.argsort(-np.where(finite, row, -np.inf), kind="mergesort")
                neighbors = [int(j) for j in order[:k] if j != i]
                step = 1
                while len(neighbors) < k:
                    candidate = (i + step) % n
                    if candidate != i and candidate not in neighbors:
                        neighbors.append(candidate)
                    step += 1
            src.extend([i] * k)
            dst.extend(neighbors[:k])
        return torch.tensor([src, dst], dtype=torch.long)


def create_daily_report_dataloaders(
    config: DailyReportDataConfig,
    batch_size: int = 2,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """一次性创建 train/val/test 三个 DataLoader，val/test 自动复用 train 的归一化统计量和股票集。"""
    train_dataset = DailyReportGraphDataset(config, split="train")
    val_dataset = DailyReportGraphDataset(
        config,
        split="val",
        feature_stats=train_dataset.feature_stats,
        shared_state=train_dataset.shared_state,
    )
    test_dataset = DailyReportGraphDataset(
        config,
        split="test",
        feature_stats=train_dataset.feature_stats,
        shared_state=train_dataset.shared_state,
    )

    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_temporal_graphs,
    }
    train_loader = DataLoader(train_dataset, shuffle=False, **kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader

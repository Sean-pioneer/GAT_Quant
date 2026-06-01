"""
Build a paper-grade event panel for gat_quant.

This script consolidates the already processed report, Guba, market and label
features into one reproducible stock-date panel plus a dataset card. It does not
scan the 85GB raw Guba directory unless explicitly asked; the default input is
the event table produced from LLM-scored reports and LLM-scored Guba posts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


REPORT_LLM_FEATURES = [
    "fundamental_expectation",
    "rating_momentum",
    "short_term_view",
    "long_term_view",
    "risk_alertness",
]

GUBA_LLM_FEATURES = [
    "guba_sentiment",
    "guba_intensity",
    "guba_confidence",
]

GUBA_ATTENTION_FEATURES = ["guba_n_posts"]

MARKET_FEATURES = [
    "mom_20d",
    "mom_60d",
    "mom_120d",
    "vol_20d",
    "pe",
    "pb",
    "turnover",
]

DERIVED_FEATURES = [
    "report_direction",
    "guba_direction",
    "divergence",
    "-fe_scaled",
    "combo_signal",
]


def normalize_stock_code(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return text.zfill(6)
    return text


def board_from_code(code: str) -> str:
    code = normalize_stock_code(code)
    if code.startswith("688"):
        return "STAR"
    if code.startswith(("300", "301")):
        return "ChiNext"
    if code.startswith(("000", "001", "002", "003")):
        return "SZ_Main"
    if code.startswith(("600", "601", "603", "605")):
        return "SH_Main"
    if code.startswith(("8", "9")):
        return "BSE_or_B"
    return "Other"


def safe_read_header(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            return next(csv.reader(f))
    except Exception:
        return []


def estimate_guba_score_coverage(guba_dir: str, max_files: int = 512) -> Dict[str, float]:
    if not guba_dir or not os.path.isdir(guba_dir):
        return {"sampled_files": 0, "files_with_score_col": 0, "score_col_rate": 0.0}

    files = [name for name in os.listdir(guba_dir) if name.endswith(".csv")]
    sample = files[:max_files]
    with_score = 0
    for name in sample:
        header = safe_read_header(os.path.join(guba_dir, name))
        if "score" in header:
            with_score += 1

    sampled = len(sample)
    return {
        "total_csv_files": len(files),
        "sampled_files": sampled,
        "files_with_score_col": with_score,
        "score_col_rate": float(with_score / sampled) if sampled else 0.0,
    }


def aggregate_event_panel(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    required = {
        "stock_code",
        "aligned_trade_date",
        target_col,
        *REPORT_LLM_FEATURES,
        *GUBA_LLM_FEATURES,
        *GUBA_ATTENTION_FEATURES,
        *MARKET_FEATURES,
        *DERIVED_FEATURES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")

    df = df.copy()
    df["stock_code"] = df["stock_code"].map(normalize_stock_code)
    df["aligned_trade_date"] = pd.to_datetime(df["aligned_trade_date"], errors="coerce")

    numeric_cols = sorted(required - {"stock_code", "aligned_trade_date"})
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["stock_code", "aligned_trade_date", target_col])
    df["has_report_llm"] = df[REPORT_LLM_FEATURES].notna().any(axis=1).astype(int)
    df["has_guba_llm"] = df[GUBA_LLM_FEATURES].notna().any(axis=1).astype(int)
    df["has_guba_posts"] = (df["guba_n_posts"].fillna(0) > 0).astype(int)

    agg_map = {col: "mean" for col in numeric_cols}
    agg_map.update(
        {
            "has_report_llm": "max",
            "has_guba_llm": "max",
            "has_guba_posts": "max",
        }
    )

    panel = (
        df.groupby(["aligned_trade_date", "stock_code"], as_index=False)
        .agg(agg_map)
        .sort_values(["aligned_trade_date", "stock_code"])
    )
    panel["board"] = panel["stock_code"].map(board_from_code)
    panel["event_count"] = (
        df.groupby(["aligned_trade_date", "stock_code"]).size().reindex(
            pd.MultiIndex.from_frame(panel[["aligned_trade_date", "stock_code"]])
        )
        .to_numpy()
    )
    panel["date"] = panel["aligned_trade_date"].dt.strftime("%Y-%m-%d")
    return panel


def write_dataset_card(output_dir: str, metadata: Dict) -> None:
    lines = [
        "# gat_quant paper dataset",
        "",
        "## Inputs",
        f"- Event table: `{metadata['inputs']['event_csv']}`",
        f"- Price directory: `{metadata['inputs']['price_dir']}`",
        f"- Guba directory: `{metadata['inputs']['guba_dir']}`",
        "",
        "## LLM semantic factors",
        "- Report LLM fields: " + ", ".join(REPORT_LLM_FEATURES),
        "- Guba LLM fields: " + ", ".join(GUBA_LLM_FEATURES + GUBA_ATTENTION_FEATURES),
        "",
        "## Leakage controls",
        "- Report rows are aligned to `aligned_trade_date`.",
        "- Guba scores are aggregated from posts strictly before the event date in the upstream builder.",
        "- Labels use forward returns and are not used in graph construction.",
        "- Train/validation/test splits in the Dataset are chronological.",
        "- Feature normalization in the Dataset uses train-period statistics only.",
        "",
        "## Output profile",
        f"- Rows: {metadata['panel']['rows']}",
        f"- Stocks: {metadata['panel']['stocks']}",
        f"- Dates: {metadata['panel']['dates']}",
        f"- Date range: {metadata['panel']['date_min']} to {metadata['panel']['date_max']}",
        f"- Report LLM coverage: {metadata['coverage']['report_llm_rate']:.2%}",
        f"- Guba LLM coverage: {metadata['coverage']['guba_llm_rate']:.2%}",
        f"- Guba post coverage: {metadata['coverage']['guba_posts_rate']:.2%}",
        "",
        "## Graph construction plan",
        "- Stock-sector edges: board/proxy sector buckets from stock code; replaceable with a formal industry table.",
        "- Stock-stock correlation edges: rolling historical return correlation from `/opt/price`.",
        "- Stock-stock volatility spillover edges: lagged return-to-return correlation from `/opt/price`.",
        "- Institution/style edges: style-bucket proxy until a real institutional holdings table is provided.",
        "",
    ]
    with open(os.path.join(output_dir, "DATASET_CARD.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gat_quant paper event panel")
    parser.add_argument(
        "--event_csv",
        default="/opt/report_factor_test_60d_v2/enriched_events_with_guba.csv",
    )
    parser.add_argument("--price_dir", default="/opt/price")
    parser.add_argument("--guba_dir", default="/opt/guba_data")
    parser.add_argument("--output_dir", default="./paper_data")
    parser.add_argument("--target_col", default="forward_60d")
    parser.add_argument("--guba_score_sample_files", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    events = pd.read_csv(args.event_csv)
    panel = aggregate_event_panel(events, args.target_col)

    output_csv = os.path.join(args.output_dir, "paper_event_panel.csv")
    panel.to_csv(output_csv, index=False, encoding="utf-8-sig")

    guba_coverage = estimate_guba_score_coverage(args.guba_dir, args.guba_score_sample_files)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "event_csv": args.event_csv,
            "price_dir": args.price_dir,
            "guba_dir": args.guba_dir,
        },
        "outputs": {
            "panel_csv": output_csv,
        },
        "features": {
            "report_llm": REPORT_LLM_FEATURES,
            "guba_llm": GUBA_LLM_FEATURES,
            "guba_attention": GUBA_ATTENTION_FEATURES,
            "market": MARKET_FEATURES,
            "derived": DERIVED_FEATURES,
            "masks": ["has_report_llm", "has_guba_llm", "has_guba_posts"],
        },
        "panel": {
            "rows": int(len(panel)),
            "stocks": int(panel["stock_code"].nunique()),
            "dates": int(panel["aligned_trade_date"].nunique()),
            "date_min": str(panel["aligned_trade_date"].min().date()),
            "date_max": str(panel["aligned_trade_date"].max().date()),
        },
        "coverage": {
            "report_llm_rate": float(panel["has_report_llm"].mean()),
            "guba_llm_rate": float(panel["has_guba_llm"].mean()),
            "guba_posts_rate": float(panel["has_guba_posts"].mean()),
            "raw_guba_score_sample": guba_coverage,
        },
        "leakage_controls": [
            "chronological split",
            "train-period normalization only",
            "Guba posts strictly before event date in upstream builder",
            "rolling price edges use historical prices only",
        ],
        "institution_node_note": "Current paper Dataset uses style-bucket proxy nodes until real institutional holdings are available.",
    }

    with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    write_dataset_card(args.output_dir, metadata)

    print(json.dumps(metadata["panel"], ensure_ascii=False, indent=2))
    print(json.dumps(metadata["coverage"], ensure_ascii=False, indent=2))
    print(f"Saved panel to {output_csv}")


if __name__ == "__main__":
    main()

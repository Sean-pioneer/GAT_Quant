"""检查研报面板对中证 1000 成分股的覆盖情况.

用法:
    python check_report_coverage.py \
        --panel_csv ../paper_data/paper_event_panel.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def load_codes() -> list[str]:
    try:
        import constituent_codes as cc
    except ImportError:
        print("ERROR: 找不到 constituent_codes 模块", file=sys.stderr)
        sys.exit(1)

    for attr in ("CODES", "codes", "CONSTITUENT_CODES", "constituent_codes", "CSI1000"):
        if hasattr(cc, attr):
            raw = getattr(cc, attr)
            break
    else:
        raw = next((v for v in vars(cc).values() if isinstance(v, list)), None)
        if raw is None:
            print("ERROR: constituent_codes 中没有列表变量", file=sys.stderr)
            sys.exit(1)

    codes = []
    for c in raw:
        digits = "".join(ch for ch in str(c) if ch.isdigit())[-6:]
        if digits:
            codes.append(digits.zfill(6))
    return sorted(set(codes))


def normalize(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[-6:].zfill(6) if digits else text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel_csv", default="../paper_data/paper_event_panel.csv")
    parser.add_argument("--start", default="", help="只统计该日期之后的研报（如 2022-01-01）")
    parser.add_argument("--end",   default="", help="只统计该日期之前的研报（如 2026-01-01）")
    args = parser.parse_args()

    csi1000 = set(load_codes())
    print(f"中证 1000 成分股：{len(csi1000)} 只\n")

    panel = pd.read_csv(args.panel_csv, dtype={"stock_code": str})
    panel["stock_code"] = panel["stock_code"].map(normalize)
    panel["aligned_trade_date"] = pd.to_datetime(panel["aligned_trade_date"], errors="coerce")

    if args.start:
        panel = panel[panel["aligned_trade_date"] >= pd.Timestamp(args.start)]
    if args.end:
        panel = panel[panel["aligned_trade_date"] <= pd.Timestamp(args.end)]

    if args.start or args.end:
        print(f"时间过滤：{args.start or '最早'} ~ {args.end or '最晚'}\n")

    # 所有出现在面板里的股票
    panel_codes = set(panel["stock_code"].dropna().unique())

    # 有 LLM 打分的股票
    if "has_report_llm" in panel.columns:
        llm_codes = set(
            panel[panel["has_report_llm"].fillna(0) > 0.5]["stock_code"].dropna().unique()
        )
    else:
        llm_codes = panel_codes

    in_panel     = csi1000 & panel_codes
    has_llm      = csi1000 & llm_codes
    no_coverage  = csi1000 - panel_codes

    print(f"有研报记录（任意类型）：{len(in_panel)} 只 ({len(in_panel)/len(csi1000)*100:.1f}%)")
    print(f"有 LLM 打分：          {len(has_llm)} 只 ({len(has_llm)/len(csi1000)*100:.1f}%)")
    print(f"完全没有覆盖：         {len(no_coverage)} 只 ({len(no_coverage)/len(csi1000)*100:.1f}%)")

    if no_coverage:
        print("\n=== 完全没有研报覆盖的股票 ===")
        for code in sorted(no_coverage):
            print(f"  {code}")

    # 有记录但 LLM 打分为 0 的
    partial = in_panel - has_llm
    if partial:
        print(f"\n=== 有记录但无 LLM 打分的股票（{len(partial)} 只）===")
        for code in sorted(partial):
            print(f"  {code}")

    # 每只股票的研报数量分布
    print("\n=== LLM 研报数量分布（CSI1000 内） ===")
    counts = (
        panel[panel["stock_code"].isin(csi1000)]
        .groupby("stock_code")["has_report_llm"]
        .apply(lambda s: (s.fillna(0) > 0.5).sum())
        .rename("llm_count")
    )
    bins = [0, 1, 5, 20, 50, 100, float("inf")]
    labels = ["0篇", "1-4篇", "5-19篇", "20-49篇", "50-99篇", "100篇+"]
    cut = pd.cut(counts, bins=bins, labels=labels, right=False)
    print(cut.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()

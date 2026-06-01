"""检查 /opt/price 目录中中证 1000 成分股的价格文件完整性.

用法:
    # 只检查文件是否存在、行数、起止日期
    python check_price_data.py --price_dir /opt/price --start 2021-01-01 --end 2024-12-31

    # 同时检查每只股票在交易日历内是否有缺口
    python check_price_data.py --price_dir /opt/price --start 2021-01-01 --end 2024-12-31 --check_gaps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from tushare_client import get_pro


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


def load_trading_calendar(start: str, end: str) -> pd.DatetimeIndex:
    """从 tushare 拉取上交所交易日历."""
    pro = get_pro()
    df = pro.trade_cal(
        exchange="SSE",
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        is_open="1",
        fields="cal_date",
    )
    if df is None or df.empty:
        print("ERROR: 无法从 tushare 拉取交易日历", file=sys.stderr)
        sys.exit(1)
    return pd.DatetimeIndex(pd.to_datetime(df["cal_date"]))


def find_price_file(price_dir: Path, code: str) -> Path | None:
    prefix = "sh" if code.startswith(("6", "9")) else "sz"
    direct = price_dir / f"{prefix}.{code}.csv"
    if direct.exists():
        return direct
    matches = list(price_dir.glob(f"*.{code}.csv"))
    return matches[0] if matches else None


def read_dates(path: Path, date_col_candidates: list[str] | None = None) -> pd.DatetimeIndex | None:
    """读取文件的日期列，返回 DatetimeIndex；失败返回 None."""
    if date_col_candidates is None:
        date_col_candidates = ["交易所行情日期", "日期", "date", "Date", "trade_date"]
    try:
        df = pd.read_csv(path, nrows=0)
        cols = [str(c).lstrip("﻿") for c in df.columns]
        date_col = next((c for c in date_col_candidates if c in cols), None)
        # tushare 因子文件以 trade_date 为索引，读出来第一列就是日期
        if date_col is None:
            date_col = cols[0]
        df = pd.read_csv(path, usecols=[date_col], parse_dates=[date_col])
        df.columns = ["date"]
        return pd.DatetimeIndex(df["date"].dropna().unique())
    except Exception:
        return None


def check_file(path: Path, start: str, end: str, min_rows: int) -> dict:
    result = {"ok": True, "issues": [], "rows": 0, "date_min": "?", "date_max": "?"}
    dates = read_dates(path)
    if dates is None or dates.empty:
        result["ok"] = False
        result["issues"].append("文件为空或找不到日期列")
        return result

    result["rows"] = len(dates)
    result["date_min"] = str(dates.min().date())
    result["date_max"] = str(dates.max().date())

    if len(dates) < min_rows:
        result["ok"] = False
        result["issues"].append(f"行数不足 ({len(dates)} < {min_rows})")

    if start and dates.min() > pd.Timestamp(start) + pd.Timedelta(days=30):
        result["ok"] = False
        result["issues"].append(f"起始日期偏晚 ({dates.min().date()} > {start})")

    if end and dates.max() < pd.Timestamp(end) - pd.Timedelta(days=30):
        result["ok"] = False
        result["issues"].append(f"结束日期偏早 ({dates.max().date()} < {end})")

    return result


def check_gaps(path: Path, calendar: pd.DatetimeIndex) -> list[str]:
    """返回文件在交易日历中缺失的日期（仅在文件自身起止范围内查找缺口）."""
    dates = read_dates(path)
    if dates is None or dates.empty:
        return []
    expected = calendar[(calendar >= dates.min()) & (calendar <= dates.max())]
    missing = expected.difference(dates)
    return [str(d.date()) for d in missing]


def check_dir(
    label: str,
    codes: list[str],
    find_fn,
    start: str,
    end: str,
    min_rows: int,
    calendar: pd.DatetimeIndex | None,
) -> int:
    """通用检查逻辑，返回有问题的股票数量."""
    missing_files: list[str] = []
    bad: list[dict] = []
    gap_report: list[dict] = []
    ok_count = 0

    for code in codes:
        path = find_fn(code)
        if path is None:
            missing_files.append(code)
            continue

        info = check_file(path, start, end, min_rows)

        if calendar is not None:
            gaps = check_gaps(path, calendar)
            if gaps:
                gap_report.append({"code": code, "count": len(gaps), "dates": gaps})

        if info["ok"]:
            ok_count += 1
        else:
            bad.append({"code": code, **info})

    print(f"── {label} ──")
    print(f"  [OK]            {ok_count} 只")
    print(f"  [文件缺失]      {len(missing_files)} 只")
    print(f"  [文件有问题]    {len(bad)} 只")
    if calendar is not None:
        print(f"  [有交易日缺口]  {len(gap_report)} 只")
    print()

    if missing_files:
        print("  === 缺失文件 ===")
        for code in missing_files:
            print(f"    {code}")
        print()

    if bad:
        print("  === 文件有问题 ===")
        for item in bad:
            date_range = f"{item['date_min']} ~ {item['date_max']}"
            issues = "; ".join(item["issues"])
            print(f"    {item['code']}  rows={item['rows']}  {date_range}  [{issues}]")
        print()

    if gap_report:
        print("  === 有交易日缺口 ===")
        for item in sorted(gap_report, key=lambda x: -x["count"]):
            sample = ", ".join(item["dates"][:5])
            tail = f"… 共 {item['count']} 天" if item["count"] > 5 else f"共 {item['count']} 天"
            print(f"    {item['code']}  {sample}  {tail}")
        print()

    return len(missing_files) + len(bad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--price_dir", default="", help="价格文件目录（不传则跳过）")
    parser.add_argument("--factor_dir", default="", help="tushare 因子目录（不传则跳过）")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--min_rows", type=int, default=200)
    parser.add_argument("--check_gaps", action="store_true", help="检查交易日缺口（从 tushare 拉日历）")
    args = parser.parse_args()

    codes = load_codes()
    print(f"共 {len(codes)} 只成分股\n")

    calendar: pd.DatetimeIndex | None = None
    if args.check_gaps:
        print("从 tushare 拉取交易日历…")
        calendar = load_trading_calendar(args.start, args.end)
        print(f"交易日历: {calendar.min().date()} ~ {calendar.max().date()}，共 {len(calendar)} 个交易日\n")

    if not args.price_dir and not args.factor_dir:
        print("ERROR: 请至少传 --price_dir 或 --factor_dir 之一", file=sys.stderr)
        sys.exit(1)

    total_bad = 0

    if args.price_dir:
        price_dir = Path(args.price_dir)
        total_bad += check_dir(
            label=f"price 目录: {price_dir}",
            codes=codes,
            find_fn=lambda code: find_price_file(price_dir, code),
            start=args.start,
            end=args.end,
            min_rows=args.min_rows,
            calendar=calendar,
        )

    if args.factor_dir:
        factor_dir = Path(args.factor_dir)
        total_bad += check_dir(
            label=f"factor 目录: {factor_dir}",
            codes=codes,
            find_fn=lambda code: (factor_dir / f"{code}.csv") if (factor_dir / f"{code}.csv").exists() else None,
            start=args.start,
            end=args.end,
            min_rows=args.min_rows,
            calendar=calendar,
        )

    if total_bad == 0:
        print("全部检查通过。")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

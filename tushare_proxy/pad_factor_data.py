"""对 /opt/tushare_factors 目录中的因子文件做前向填充，补全停牌日缺口.

用法:
    python pad_factor_data.py \
        --factor_dir /opt/tushare_factors \
        --start 2021-01-01 \
        --end 2024-12-31
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from tushare_client import get_pro


def load_trading_calendar(start: str, end: str) -> pd.DatetimeIndex:
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


def pad_file(path: Path, calendar: pd.DatetimeIndex) -> int:
    """前向填充单个因子文件，返回补全的行数."""
    df = pd.read_csv(path, index_col="trade_date", parse_dates=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    file_start = df.index.min()
    file_end   = df.index.max()
    target_cal = calendar[(calendar >= file_start) & (calendar <= file_end)]

    missing = target_cal.difference(df.index)
    if missing.empty:
        return 0

    df = df.reindex(target_cal).ffill()
    df.index.name = "trade_date"
    df.to_csv(path)
    return len(missing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor_dir", default="/opt/tushare_factors")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-01-01")
    args = parser.parse_args()

    factor_dir = Path(args.factor_dir)
    csv_files = sorted(factor_dir.glob("*.csv"))
    print(f"找到 {len(csv_files)} 个因子文件\n")
    print("从 tushare 拉取交易日历…")
    calendar = load_trading_calendar(args.start, args.end)
    print(f"交易日历: {calendar.min().date()} ~ {calendar.max().date()}，共 {len(calendar)} 个交易日\n")

    total_padded = 0
    errors: list[str] = []

    for path in csv_files:
        try:
            n = pad_file(path, calendar)
            if n > 0:
                total_padded += n
                print(f"  [pad] {path.name}: +{n} 行")
        except Exception as e:
            errors.append(f"{path.name}: {e}")
            print(f"  [ERR] {path.name}: {e}")

    print(f"\n完成：共补全 {total_padded} 行，{len(errors)} 个文件出错")
    if errors:
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()

"""批量拉取中证 1000 成分股的 tushare 因子并保存到本地.

用法:
    python batch_fetch.py \
        --out_dir ../tushare_factors \
        --start 2021-01-01 \
        --end 2024-12-31
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from factor_fetch import fetch_all_factors

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_codes() -> list[str]:
    """从 constituent_codes 模块加载股票代码列表."""
    try:
        import constituent_codes as cc
    except ImportError:
        logger.error("找不到 constituent_codes 模块，请确认该文件在同目录或 PYTHONPATH 中")
        sys.exit(1)

    # 自动查找模块内的列表变量
    for attr in ("CODES", "codes", "CONSTITUENT_CODES", "constituent_codes", "CSI1000"):
        if hasattr(cc, attr):
            raw = getattr(cc, attr)
            break
    else:
        # 找不到已知名称时，取第一个 list 类型的属性
        raw = next(
            (v for v in vars(cc).values() if isinstance(v, list)),
            None,
        )
        if raw is None:
            logger.error("constituent_codes 模块中未找到列表变量")
            sys.exit(1)

    # 统一为 6 位纯数字格式
    codes = []
    for c in raw:
        digits = "".join(ch for ch in str(c) if ch.isdigit())[-6:]
        if digits:
            codes.append(digits.zfill(6))
    return sorted(set(codes))


def to_ts_code(code: str) -> str:
    """6 位代码 → tushare ts_code."""
    if code.startswith(("60", "68", "90", "11")):
        return f"{code}.SH"
    if code.startswith(("00", "30", "20")):
        return f"{code}.SZ"
    if code.startswith(("43", "83", "87", "92")):
        return f"{code}.BJ"
    return code


def fetch_and_save(code: str, start: str, end: str, out_dir: Path, overwrite: bool,
                   retries: int = 3, retry_sleep: float = 5.0) -> bool:
    out_path = out_dir / f"{code}.csv"
    if out_path.exists() and not overwrite:
        logger.info(f"[skip] {code} 已存在")
        return True
    for attempt in range(1, retries + 1):
        try:
            df = fetch_all_factors(to_ts_code(code), start, end)
            if df.empty:
                logger.warning(f"[warn] {code} 无数据")
                return False
            df.index.name = "trade_date"
            df.to_csv(out_path)
            logger.info(f"[ok]   {code}  {len(df)} 行")
            return True
        except Exception as e:
            if attempt < retries:
                logger.warning(f"[retry {attempt}/{retries}] {code}: {e}")
                time.sleep(retry_sleep)
            else:
                logger.error(f"[err]  {code}: {e}")
                return False
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="../tushare_factors",
                        help="因子文件输出目录")
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--sleep", type=float, default=0.5,
                        help="每只票请求后等待秒数（共享代理限频）")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的文件")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    codes = load_codes()
    logger.info(f"共 {len(codes)} 只股票，输出到 {out_dir.resolve()}")

    ok, fail = 0, 0
    for i, code in enumerate(codes, 1):
        success = fetch_and_save(code, args.start, args.end, out_dir, args.overwrite)
        if success:
            ok += 1
        else:
            fail += 1
        if i % 20 == 0:
            logger.info(f"进度 {i}/{len(codes)}  成功={ok} 失败={fail}")
        time.sleep(args.sleep)

    logger.info(f"完成：成功 {ok} / 失败 {fail} / 共 {len(codes)}")


if __name__ == "__main__":
    main()

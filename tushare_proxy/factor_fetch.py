"""四个经典因子的数据爬取 + 计算(走 tushare 代理,自包含版).

因子清单:
    1. overnight_corr  隔夜相关性   —— pro.daily          (open/close/pre_close)
    2. roe             ROE 质量      —— pro.fina_indicator  (roe + ann_date)
    3. turnover_20d    换手率因子     —— pro.daily_basic     (turnover_rate)
    4. value_bp        价值因子(BP)  —— pro.daily_basic     (pb)

每个 fetch_* 函数都返回一个 index=trade_date 的 Series,方便后续对齐到
forward 收益做 IC / 分层回测.fetch_all_factors 把四个拼成一张日频表.

用法:
    from factor_fetch import fetch_all_factors
    df = fetch_all_factors('000001.SZ', '2023-01-01', '2024-12-31')
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from tushare_client import get_pro

logger = logging.getLogger(__name__)

# 共享代理 token 有限频,批量拉多只票时每次请求后歇一下
_RATE_LIMIT_SLEEP = 0.2


def _norm_code(code: str) -> str:
    """'000001' / 'sz000001' / '000001.SZ' → '000001.SZ'(tushare ts_code 格式)."""
    code = code.strip().upper()
    if code.endswith(('.SH', '.SZ', '.BJ')):
        return code
    if code.startswith(('SH', 'SZ', 'BJ')):
        return f'{code[2:]}.{code[:2]}'
    bare = code
    if bare.startswith(('60', '68', '90', '11')):
        return f'{bare}.SH'
    if bare.startswith(('00', '30', '20')):
        return f'{bare}.SZ'
    if bare.startswith(('43', '83', '87', '92')):
        return f'{bare}.BJ'
    return bare


def _d(date: str | None) -> str | None:
    """'2024-01-31' / '20240131' → '20240131';None 透传."""
    if date is None:
        return None
    return date.replace('-', '')


def _shift_date(date: str, days: int) -> str:
    """将日期字符串平移 days 个自然日，返回 'YYYYMMDD' 格式."""
    return (pd.Timestamp(date) + pd.Timedelta(days=days)).strftime('%Y%m%d')


# --------------------------------------------------------------------------- #
# 1. 隔夜相关性
# --------------------------------------------------------------------------- #
def fetch_overnight_corr(ts_code: str, start: str, end: str,
                         window: int = 20) -> pd.Series:
    """隔夜相关性因子:过去 window 日「隔夜收益」与「日内收益」的滚动相关系数.

    隔夜收益 = open_t / pre_close_t - 1   (跳空)
    日内收益 = close_t / open_t   - 1
    经典逻辑:二者相关性越负,越倾向于反转;常作反转/情绪因子.

    Returns:
        Series index=trade_date(Timestamp), name='overnight_corr'
    """
    # 多拉 window*2 个自然日保证滚动窗口前 window-1 行有足够历史
    fetch_start = _shift_date(start, -(window * 2))
    pro = get_pro()
    df = pro.daily(ts_code=_norm_code(ts_code), start_date=_d(fetch_start),
                   end_date=_d(end), fields='trade_date,open,close,pre_close')
    time.sleep(_RATE_LIMIT_SLEEP)
    if df is None or df.empty:
        return pd.Series(dtype=float, name='overnight_corr')

    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').set_index('trade_date')

    overnight = df['open'] / df['pre_close'] - 1
    intraday  = df['close'] / df['open'] - 1
    corr = overnight.rolling(window).corr(intraday)
    # 裁剪到请求的起始日期，丢弃预热期
    return corr.loc[corr.index >= pd.Timestamp(start)].rename('overnight_corr')


# --------------------------------------------------------------------------- #
# 2. ROE(质量)
# --------------------------------------------------------------------------- #
def fetch_roe(ts_code: str, start: str, end: str) -> pd.Series:
    """ROE 因子(净资产收益率,单位 %).

    用 ann_date(公告日)作为因子可用日期,而非 end_date(报告期),
    避免「报告期已到但财报还没公布」造成的前瞻性误差.

    Returns:
        Series index=ann_date(Timestamp), name='roe'.财报频率(季度),
        后续可 reindex 到交易日再 ffill.
    """
    pro = get_pro()
    df = pro.fina_indicator(ts_code=_norm_code(ts_code), start_date=_d(start),
                            end_date=_d(end), fields='ann_date,end_date,roe')
    time.sleep(_RATE_LIMIT_SLEEP)
    if df is None or df.empty:
        return pd.Series(dtype=float, name='roe')

    df = df.dropna(subset=['ann_date'])
    df['ann_date'] = pd.to_datetime(df['ann_date'])
    # 同一报告期可能被多次公告(更正),按公告日排序后每个报告期保留最早一次
    df = (df.sort_values(['ann_date', 'end_date'])
            .drop_duplicates(subset=['end_date'], keep='first')
            .set_index('ann_date')
            .sort_index())
    roe = df['roe'].rename('roe')
    # 同一公告日可能有多个报告期，保留最新报告期的 ROE
    roe = roe[~roe.index.duplicated(keep='last')]
    return roe


# --------------------------------------------------------------------------- #
# 3. 换手率因子
# --------------------------------------------------------------------------- #
def fetch_turnover(ts_code: str, start: str, end: str,
                   window: int = 20) -> pd.Series:
    """换手率因子:过去 window 日平均换手率(%).

    经典「低换手溢价」:平均换手率越低,未来收益往往越高(流动性/关注度反向).

    Returns:
        Series index=trade_date(Timestamp), name='turnover_20d'
    """
    fetch_start = _shift_date(start, -(window * 2))
    pro = get_pro()
    df = pro.daily_basic(ts_code=_norm_code(ts_code), start_date=_d(fetch_start),
                         end_date=_d(end), fields='trade_date,turnover_rate')
    time.sleep(_RATE_LIMIT_SLEEP)
    if df is None or df.empty:
        return pd.Series(dtype=float, name=f'turnover_{window}d')

    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').set_index('trade_date')
    mean_to = df['turnover_rate'].rolling(window).mean()
    return mean_to.loc[mean_to.index >= pd.Timestamp(start)].rename(f'turnover_{window}d')


# --------------------------------------------------------------------------- #
# 4. 价值因子(BP = 1 / PB)
# --------------------------------------------------------------------------- #
def fetch_value_bp(ts_code: str, start: str, end: str) -> pd.Series:
    """价值因子 BP(账面市值比 = 1 / PB).

    Fama-French HML 用的经典价值代理.BP 越高越「便宜」.
    PB<=0(负净资产)置为 NaN,避免符号反转污染因子.

    Returns:
        Series index=trade_date(Timestamp), name='value_bp'
    """
    pro = get_pro()
    df = pro.daily_basic(ts_code=_norm_code(ts_code), start_date=_d(start),
                         end_date=_d(end), fields='trade_date,pb')
    time.sleep(_RATE_LIMIT_SLEEP)
    if df is None or df.empty:
        return pd.Series(dtype=float, name='value_bp')

    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.sort_values('trade_date').set_index('trade_date')
    pb = df['pb'].where(df['pb'] > 0)            # 负/零净资产剔除
    return (1.0 / pb).rename('value_bp')


# --------------------------------------------------------------------------- #
# 汇总:四因子拼成一张日频表
# --------------------------------------------------------------------------- #
def fetch_all_factors(ts_code: str, start: str, end: str) -> pd.DataFrame:
    """拉四个因子并按交易日对齐成一张表.

    ROE 是季度频率,这里 reindex 到日线交易日并 ffill(用最近一期已公布的 ROE).

    Returns:
        DataFrame index=trade_date,
        columns=[overnight_corr, roe, turnover_20d, value_bp]
    """
    oc  = fetch_overnight_corr(ts_code, start, end)
    # 往前多拉 180 天，确保 start 日期前最近一期财报能被 ffill 覆盖
    roe = fetch_roe(ts_code, _shift_date(start, -180), end)
    to  = fetch_turnover(ts_code, start, end)
    bp  = fetch_value_bp(ts_code, start, end)

    # 以日频因子的交易日为基准索引
    daily_idx = oc.index.union(to.index).union(bp.index).sort_values()
    out = pd.DataFrame(index=daily_idx)
    out['overnight_corr'] = oc
    out['turnover_20d']   = to
    out['value_bp']       = bp
    # ROE 按公告日 ffill 到交易日(每个交易日取「当时最新已公布」的 ROE)
    out['roe'] = roe.reindex(daily_idx, method='ffill')
    return out[['overnight_corr', 'roe', 'turnover_20d', 'value_bp']]


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    df = fetch_all_factors('000001.SZ', '2023-01-01', '2024-12-31')
    print(df.dropna().tail(10))
    print('\n因子覆盖行数:')
    print(df.describe().loc['count'])

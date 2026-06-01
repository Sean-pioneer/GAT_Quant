import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

PRICE_DIR  = '/home/zhangke/original_sever_data/price'
EVENTS_CSV = '/home/zhangke/report_factor_test_60d_v2/enriched_events.csv'
TRAIN_END  = pd.Timestamp('2023-12-31')
TEST_START = pd.Timestamp('2024-01-01')
HORIZONS   = [10, 20, 60, 120, 252]
OUT_DIR    = '/home/zhangke/paper_output'

# ── Load events ───────────────────────────────────────────────────────────────
ev = pd.read_csv(EVENTS_CSV)
ev['aligned_trade_date'] = pd.to_datetime(ev['aligned_trade_date'])
ev['stock_code'] = ev['stock_code'].astype(str).str.zfill(6)
ev['signal'] = -ev['fundamental_expectation']
ev = ev[['stock_code', 'aligned_trade_date', 'signal']].dropna()
print("Events loaded:", len(ev))

# ── Load prices ───────────────────────────────────────────────────────────────
print("Loading price data...")
price_dict = {}
needed = set(ev['stock_code'])
for fname in os.listdir(PRICE_DIR):
    if not fname.endswith('.csv'):
        continue
    code = fname.replace('.csv', '').split('.')[-1].zfill(6)
    if code not in needed:
        continue
    try:
        df = pd.read_csv(os.path.join(PRICE_DIR, fname),
                         usecols=[0, 5], header=0,
                         names=['date', 'close'], skiprows=1)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna().sort_values('date')
        df = df[df['date'] >= '2021-01-01']
        price_dict[code] = df.set_index('date')['close']
    except Exception:
        continue
print("Stocks loaded:", len(price_dict))

# ── Forward returns ───────────────────────────────────────────────────────────
def get_fwd_return(code, date, n):
    if code not in price_dict:
        return np.nan
    s = price_dict[code]
    idx = s.index.searchsorted(date)
    if idx >= len(s) or s.index[idx] != date:
        return np.nan
    end_idx = idx + n
    if end_idx >= len(s):
        return np.nan
    return s.iloc[end_idx] / s.iloc[idx] - 1

print("Computing forward returns...")
for h in HORIZONS:
    col = f'fwd_{h}d'
    ev[col] = [get_fwd_return(r.stock_code, r.aligned_trade_date, h)
               for r in ev.itertuples()]
    print(f"  fwd_{h}d: {ev[col].notna().sum()} valid")

# ── Monthly IC per horizon ────────────────────────────────────────────────────
ev['ym'] = ev['aligned_trade_date'].dt.to_period('M')

ic_series = {}
for h in HORIZONS:
    col = f'fwd_{h}d'
    rows = []
    for ym, g in ev.groupby('ym'):
        g2 = g[g['signal'].notna() & g[col].notna()]
        if len(g2) < 10:
            continue
        ic, _ = spearmanr(g2['signal'], g2[col])
        rows.append({'date': g2['aligned_trade_date'].max(), 'ic': ic})
    ic_series[h] = pd.DataFrame(rows).sort_values('date').set_index('date')['ic']

# ── Plot 1: Monthly IC time series per horizon (subplots) ─────────────────────
fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(12, 3 * len(HORIZONS)),
                         sharex=True)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for ax, h, color in zip(axes, HORIZONS, colors):
    s = ic_series[h]
    ax.bar(s.index, s.values, width=20, color=color, alpha=0.6, label=f'{h}d IC')
    # 3-month rolling mean
    roll = s.rolling(3, min_periods=2).mean()
    ax.plot(roll.index, roll.values, color=color, linewidth=1.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(TEST_START, color='red', linewidth=1.2, linestyle='--', label='Train/Test split')
    ax.set_ylabel(f'{h}d IC', fontsize=9)
    mean_tr = s[s.index <= TRAIN_END].mean()
    mean_te = s[s.index >= TEST_START].mean()
    ax.set_title(f'Horizon {h}d  |  Train IC={mean_tr:.3f}  Test IC={mean_te:.3f}', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

axes[-1].set_xlabel('Date')
plt.suptitle('Monthly Spearman IC by Holding Horizon (-fundamental_expectation)',
             fontsize=12, y=1.01)
plt.tight_layout()
path1 = os.path.join(OUT_DIR, 'horizon_ic_timeseries.png')
plt.savefig(path1, dpi=150, bbox_inches='tight')
plt.close()
print("Saved:", path1)

# ── Plot 2: Cumulative IC comparison ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for h, color in zip(HORIZONS, colors):
    s = ic_series[h]
    cumsum = s.cumsum()
    ax.plot(cumsum.index, cumsum.values, color=color, linewidth=1.5, label=f'{h}d')

ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(TEST_START, color='red', linewidth=1.2, linestyle='--', label='Train/Test split')
ax.set_title('Cumulative IC by Holding Horizon (-fundamental_expectation)', fontsize=12)
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative IC')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.tight_layout()
path2 = os.path.join(OUT_DIR, 'horizon_ic_cumulative.png')
plt.savefig(path2, dpi=150, bbox_inches='tight')
plt.close()
print("Saved:", path2)

print("Done.")

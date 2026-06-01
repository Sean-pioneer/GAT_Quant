import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os
import warnings
warnings.filterwarnings('ignore')

PRICE_DIR  = '/home/zhangke/original_sever_data/price'
EVENTS_CSV = '/home/zhangke/report_factor_test_60d_v2/enriched_events.csv'
TRAIN_END  = '2023-12-31'
TEST_START = '2024-01-01'
HORIZONS   = [5, 10, 20, 30, 45, 60, 90, 120, 180, 252]

# ── Load events ───────────────────────────────────────────────────────────────
ev = pd.read_csv(EVENTS_CSV)
ev['aligned_trade_date'] = pd.to_datetime(ev['aligned_trade_date'])
ev['stock_code'] = ev['stock_code'].astype(str).str.zfill(6)
ev['signal'] = -ev['fundamental_expectation']
ev = ev[['stock_code', 'aligned_trade_date', 'signal']].dropna()
print("Events loaded:", len(ev))

# ── Load close prices for all stocks (2021+) ─────────────────────────────────
print("Loading price data...")
price_dict = {}
needed = set(ev['stock_code'])

for fname in os.listdir(PRICE_DIR):
    if not fname.endswith('.csv'):
        continue
    raw_code = fname.replace('.csv', '')        # e.g. sz.000001
    code = raw_code.split('.')[-1].zfill(6)    # e.g. 000001
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

print("Stocks with price data:", len(price_dict))

# ── Compute forward returns at each horizon ───────────────────────────────────
print("Computing forward returns...")

def get_fwd_return(code, date, n_days):
    if code not in price_dict:
        return np.nan
    s = price_dict[code]
    idx = s.index.searchsorted(date)
    if idx >= len(s):
        return np.nan
    actual_start = s.index[idx]
    if actual_start != date:    # date not a trading day
        return np.nan
    end_idx = idx + n_days
    if end_idx >= len(s):
        return np.nan
    return s.iloc[end_idx] / s.iloc[idx] - 1

for h in HORIZONS:
    col = f'fwd_{h}d'
    ev[col] = [get_fwd_return(r.stock_code, r.aligned_trade_date, h)
               for r in ev.itertuples()]
    n_valid = ev[col].notna().sum()
    print(f"  fwd_{h}d: {n_valid} valid ({100*n_valid/len(ev):.1f}%)")

# ── IC analysis ───────────────────────────────────────────────────────────────
ev['ym'] = ev['aligned_trade_date'].dt.to_period('M')

def nw_tstat(x, lags=None):
    T = len(x)
    if T < 5: return np.nan
    if lags is None: lags = int(np.floor(4*(T/100)**(2/9)))
    mu = x.mean(); g0 = np.var(x, ddof=1); nw = g0
    for k in range(1, lags+1):
        gk = np.cov(x[k:], x[:-k])[0,1]; nw += 2*(1-k/(lags+1))*gk
    se = np.sqrt(nw/T)
    return mu/se if se > 0 else np.nan

print()
print("%-10s %8s %8s %8s %8s %6s" % ('Horizon','Full_IC','Train_IC','Train_t','Test_IC','N_mo'))
print("-" * 56)

results = []
for h in HORIZONS:
    col = f'fwd_{h}d'
    ics = []
    mask = ev['signal'].notna() & ev[col].notna()
    sub = ev[mask]
    for ym, g in sub.groupby('ym'):
        if len(g) < 10: continue
        ic, _ = spearmanr(g['signal'], g[col])
        ics.append({'ym': ym, 'ic': ic, 'date': g['aligned_trade_date'].iloc[0]})
    d = pd.DataFrame(ics).sort_values('date')
    tr = d[d['date'] <= TRAIN_END]['ic']
    te = d[d['date'] >= TEST_START]['ic']
    full_ic  = d['ic'].mean()
    train_ic = tr.mean()
    train_t  = nw_tstat(tr.values)
    test_ic  = te.mean()
    n_mo     = len(d)
    results.append(dict(horizon=h, full_ic=full_ic, train_ic=train_ic,
                        train_t=train_t, test_ic=test_ic, n_months=n_mo))
    print("%-10s %8.4f %8.4f %8.3f %8.4f %6d" % (
        f'{h}d', full_ic, train_ic, train_t, test_ic, n_mo))

pd.DataFrame(results).to_csv('/home/zhangke/horizon_ic_results.csv', index=False)
print("\nSaved to /home/zhangke/horizon_ic_results.csv")

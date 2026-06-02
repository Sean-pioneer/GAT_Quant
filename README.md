# MarketSentimentGAT — 日频异构图量化交易系统

本项目面向 A 股横截面收益预测，将市场量价因子与研报 LLM 语义信号统一到异构图注意力网络（`MarketSentimentGAT`）中，用于训练、消融实验和指标评估。

当前主线：

```
日频价格文件 + tushare 因子 + 稀疏研报 LLM 前向填充 + CSI1000 全量股票 + target mask
```

---

## 目录

1. [系统架构](#1-系统架构)
2. [特征定义](#2-特征定义)
3. [数据流程](#3-数据流程)
4. [图结构](#4-图结构)
5. [模型结构](#5-模型结构)
6. [训练](#6-训练)
7. [消融实验](#7-消融实验)
8. [数据检查与准备工具](#8-数据检查与准备工具)
9. [安装与环境](#9-安装与环境)
10. [文件说明](#10-文件说明)
11. [外部数据依赖](#11-外部数据依赖)
12. [已知局限与待改进](#12-已知局限与待改进)

---

## 1. 系统架构

```
/opt/price/          /opt/tushare_factors/    paper_event_panel.csv
(日频价格文件)         (tushare 量化因子)         (研报 LLM 打分面板)
      │                      │                        │
      └──────────────────────┴────────────────────────┘
                             │
                  DailyReportGraphDataset
                  ┌──────────────────────┐
                  │ 股票选取 & 过滤        │
                  │ 特征工程 & 标准化      │
                  │ LLM 前向填充 + 衰减    │
                  │ 异构图构建             │
                  └──────────────────────┘
                             │
                    MarketSentimentGAT
                  ┌──────────────────────┐
                  │ HeteroGATConv × 2    │
                  │   节点级注意力        │
                  │   语义级注意力        │
                  │ MLP Predictor        │
                  └──────────────────────┘
                             │
                  predictions [batch, num_stocks]
                  (未来 N 交易日收益率预测，默认 10d)
```

---

## 2. 特征定义

### 节点特征（`NODE_FEATURE_DIM = 18`）

| 类别 | 特征名 | 维度 | 说明 |
|---|---|---|---|
| 研报 LLM | `fundamental_expectation` | 1 | 基本面预期 |
| 研报 LLM | `rating_momentum` | 1 | 评级动量 |
| 研报 LLM | `short_term_view` | 1 | 短期观点 |
| 研报 LLM | `long_term_view` | 1 | 长期观点 |
| 研报 LLM | `risk_alertness` | 1 | 风险提示 |
| 市场因子 | `overnight_corr` | 1 | 隔夜收益与日内收益 20 日滚动相关（情绪/反转因子） |
| 市场因子 | `roe` | 1 | 净资产收益率（季度公告日前向填充） |
| 市场因子 | `mom_20d` | 1 | 20 日价格动量 |
| 市场因子 | `mom_60d` | 1 | 60 日价格动量 |
| 市场因子 | `vol_20d` | 1 | 20 日收益率波动率 |
| 市场因子 | `turnover_20d` | 1 | 20 日平均换手率 |
| 市场因子 | `pb` | 1 | 市净率 |
| Mask | `has_report_llm` | 1 | 该日期是否有最新研报（1=有，0=前向填充） |
| 板块独热 | `board_SH_Main / SZ_Main / ChiNext / STAR / Other` | 5 | 上市板块 |

> **LLM 前向填充 + 时间衰减**：无新研报的交易日沿用最近一篇研报的 LLM 打分，按 `exp(-days_since_last_report / llm_decay_days)` 衰减，`has_report_llm` 置 0 以标记非最新。默认半衰期 30 交易日。

> **股吧特征全部置 0**：股吧 LLM 特征不作为有效输入，保留字段仅为数据格式兼容。

---

## 3. 数据流程

### 3.1 数据来源

| 数据 | 路径 | 说明 |
|---|---|---|
| 日频价格 | `/opt/price/{sh\|sz}.{code}.csv` | 收盘价、涨跌幅、PB 等 |
| tushare 因子 | `/opt/tushare_factors/{code}.csv` | overnight_corr、roe、turnover_20d（从 2022 年起） |
| 研报 LLM 面板 | `/opt/paper_event_panel.csv` | 每行一条研报事件，含 5 维 LLM 打分 |
| 股票池 | `./data/csi1000.csv` | CSI1000 成分股列表 |
| 行业映射 | `data/stock_industry.csv` | 股票代码 → 申万行业，用于构造 sector 节点 |

### 3.2 股票过滤（`_select_universe`）

按以下顺序过滤候选股票：

1. 必须有价格文件（`/opt/price/`）
2. `require_llm_coverage=True`：必须在研报面板中有 `has_report_llm > 0.5` 的记录
3. `max_price_start="2022-01-01"`：价格文件起始日期不得晚于此日期（排除新股）

### 3.3 市场特征计算

```python
mom_20d = close / close.shift(20) - 1        # 20 日动量
mom_60d = close / close.shift(60) - 1        # 60 日动量
vol_20d = returns.rolling(20, min_periods=10).std()
pb      = 从价格文件读取
# tushare 因子：从 /opt/tushare_factors/{code}.csv 按日期 left join
overnight_corr, roe, turnover_20d
```

### 3.4 标签构造

```python
target = close.shift(-target_horizon) / close - 1   # 默认 target_horizon=10
```

`shift(-N)` 按**行**移位，行 = 交易日，自动跳过节假日。末尾 N 行无标签，由 `target_mask` 排除出 loss。

### 3.5 特征标准化

仅用训练集拟合：

```python
center = median(train_rows[col])
scale  = IQR(train_rows[col]) / 1.349        # robust scaler
scaled = clip((x - center) / scale, -8, 8)   # NaN → 0
```

val/test 使用训练集统计量，不泄漏未来信息。

### 3.6 LLM 前向填充与时间衰减

无新研报的交易日沿用最近一篇研报的打分：

```python
# 1. 清除非研报日的 LLM 值
for col in REPORT_LLM_FEATURES:
    daily[col] = daily[col].where(daily["has_report_llm"] > 0.5)

# 2. 前向填充
for col in REPORT_LLM_FEATURES:
    daily[col] = daily.groupby("stock_code")[col].ffill()

# 3. 时间衰减（可选，默认 llm_decay_days=30）
days_since = 距上次研报的交易日数
decay = exp(-days_since / llm_decay_days)
LLM_score *= decay
```

`llm_decay_days=0` 可关闭衰减，回退到纯前向填充。

### 3.7 时间切分

```
全部交易日 → 按时间排序 → train 70% / val 15% / test 15%
每个样本日要求至少 80% 股票有有效标签（可调 --min_valid_targets）
```

---

## 4. 图结构

### 节点类型

| 节点 | 数量 | 特征维度 | 说明 |
|---|---|---|---|
| `stock` | num_stocks | 18 | 每只股票一个节点 |
| `sector` | 实际行业数 | 18 | 由 `data/stock_industry.csv` 映射，当前为所含股票特征均值池化 |

> ⚠️ **已知局限**：sector 节点特征为 `_bucket_means`（行业内股票均值），无独立数据源。引入申万行业指数收益率等独立数据可显著提升信息量。

### 边类型

| 边 | 方向 | 构造方式 (默认 edge_mode=fundamentals) |
|---|---|---|
| `belongs_to` | stock → sector | 股票归属行业（静态） |
| `contains` | sector → stock | 反向（静态） |
| `correlates_with` | stock ↔ stock | 见下方"边构造模式" |
| `spills_volatility_to` | stock ↔ stock | fundamentals/static 模式下禁用 |

### 边构造模式 (`--edge_mode`)

| 模式 | 方法 | 边稳定性 | 说明 |
|---|---|---|---|
| `fundamentals`（默认） | 余弦相似度(pb, roe, turnover, board) | 高 | 基于正交基本面特征，不来自收益，避免与节点特征重叠 |
| `static` | 同行业全连接 | 最高 | 完全不用历史数据，最干净可解释 |
| `returns` | Pearson(returns), lag-1 cross-corr | 低 | 原始方法：边来自历史收益，与动量特征同源 |

> **设计理由**：`returns` 模式中边和节点特征（mom_20d/mom_60d/vol_20d）都来自同一份历史收益，图结构只是动量信号的复读机。`fundamentals` 和 `static` 用与 target 正交的数据建边，消除了信息冗余。

### 图模式 (`--graph_mode`)

| 模式 | 含义 |
|---|---|
| `full` | 全部可用边 |
| `no_stock_edges` | 去掉 stock↔stock 边，仅保留 sector 边 |
| `sector_style_only` | 仅 sector 边，stock-stock 边全空 |
| `random_stock_edges` | stock-stock 边随机打乱（证伪对照：验证模型是否真的从边结构学习） |

---

## 5. 模型结构

```
输入: 单帧 HeteroData（stock + sector 节点，边类型取决于配置）

Step 1: 异构图卷积（HeteroGATConv × num_gnn_layers）
  Level 1 — 节点级注意力（每类边独立 GATv2Conv）
    h_sector = GATv2Conv(stock → sector)
    h1       = GATv2Conv(sector → stock)
    h2       = GATv2Conv(stock -[correlates]→ stock)
    h3       = GATv2Conv(stock -[spills]→ stock)

  Level 2 — 语义级注意力（学习各入路径的重要性权重）
    scores  = MLP([h1, h2, h3])           [N, 3, 1]
    weights = softmax(scores, dim=1)
    h_stock = Σ weights_i × h_i           [N, hidden_dim]

  → stock_emb [N, hidden_dim]

Step 2: MLP 预测
  Linear(hidden_dim → hidden_dim/2)
  BatchNorm1d + ReLU + Dropout
  Linear(hidden_dim/2 → 1)
  → predictions [B, N]  (收益率预测信号)
```

### 主要超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `hidden_dim` | 256 | 隐藏层维度 |
| `num_heads` | 4 | GAT 注意力头数 |
| `num_layers` | 2 | GNN 层数（必须 ≥ 2） |
| `dropout` | 0.3 | Dropout 率 |
| `target_horizon` | 10 | 预测目标（交易日） |
| `loss_type` | mse | mse / ic / quasi_likelihood |

---

## 6. 训练

### 6.1 默认参数训练

```bash
python train_daily_report.py \
    --panel_csv /opt/paper_event_panel.csv \
    --price_dir /opt/price \
    --universe_csv ./data/csi1000.csv \
    --num_stocks 0 \
    --target_horizon 10 \
    --edge_mode fundamentals \
    --llm_decay_days 30 \
    --loss_type ic \
    --device cuda
```

### 6.2 完整训练命令（全部参数显式指定）

```bash
python train_daily_report.py \
    --panel_csv /opt/paper_event_panel.csv \
    --price_dir /opt/price \
    --universe_csv ./data/csi1000.csv \
    --num_stocks 0 \
    --target_horizon 10 \
    --price_lookback 60 \
    --corr_top_k 4 \
    --corr_threshold 0.3 \
    --feature_mode market_report \
    --graph_mode full \
    --edge_mode fundamentals \
    --llm_decay_days 30 \
    --selection_mode report_count \
    --start_date 2022-01-01 \
    --epochs 20 \
    --batch_size 8 \
    --hidden_dim 256 \
    --num_heads 4 \
    --num_layers 2 \
    --dropout 0.3 \
    --lr 1e-4 \
    --loss_type ic \
    --device cuda \
    --output_dir ./outputs_daily_report \
    --log_dir ./logs_daily_report \
    --seed 42
```

### 6.3 快速烟雾测试

```bash
python train_daily_report.py \
    --panel_csv /opt/paper_event_panel.csv \
    --price_dir /opt/price \
    --universe_csv ./data/csi1000.csv \
    --num_stocks 128 \
    --target_horizon 10 \
    --edge_mode static \
    --llm_decay_days 0 \
    --epochs 10 \
    --batch_size 4 \
    --hidden_dim 64 \
    --num_heads 2 \
    --num_layers 2 \
    --loss_type ic \
    --device cuda \
    --output_dir ./smoke_test_out
```

### 6.4 边构造消融对比实验

```bash
# 实验 1: 基本面相似度建边（默认）
python train_daily_report.py --edge_mode fundamentals --target_horizon 10 --loss_type ic

# 实验 2: 纯行业边（最干净）
python train_daily_report.py --edge_mode static --target_horizon 10 --loss_type ic

# 实验 3: 收益相关建边（原始方法）
python train_daily_report.py --edge_mode returns --target_horizon 10 --loss_type ic

# 实验 4: 随机打乱边（证伪对照）
python train_daily_report.py --graph_mode random_stock_edges --target_horizon 10 --loss_type ic
```

### 6.5 关键参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--num_stocks 0` | 0 | 使用全部过滤后的可用股票 |
| `--target_horizon` | 10 | 预测 N 交易日后的收益率 |
| `--edge_mode` | fundamentals | 边构造方式：fundamentals / static / returns |
| `--llm_decay_days` | 30 | LLM 前向填充半衰期（交易日），0=不衰减 |
| `--graph_mode` | full | full / no_stock_edges / sector_style_only / random_stock_edges |
| `--require_llm_coverage` | True | 过滤无 LLM 研报覆盖的股票 |
| `--max_price_start` | 2022-01-01 | 过滤此日期后上市的新股 |
| `--corr_threshold` | 0.3 | 相关性弱于此值的边不建立 |
| `--feature_mode` | market_report | market_report / market_only / report_only |
| `--batch_size` | 8 | 每批图数量 |
| `--epochs` | 20 | 训练轮数 |
| `--loss_type` | mse | mse / ic / quasi_likelihood |

### 6.6 训练输出

每个 epoch 记录：

| 指标 | 说明 |
|---|---|
| `daily_ic` | 日截面预测与真实收益的 Pearson 均值（**主要指标**） |
| `daily_rank_ic` | 日截面 Spearman 秩相关均值 |
| `daily_ic_ir` | IC / IC_std，衡量 IC 稳定性 |
| `direction_accuracy` | 涨跌方向预测准确率 |

> IC 参考：> 0.03 勉强可用，> 0.05 有效，> 0.08 较好。

---

## 7. 消融实验

比较不同特征配置和边构造模式的预测能力，数据只加载一次、复用于所有实验：

```bash
python ablation_llm.py \
    --panel_csv /opt/paper_event_panel.csv \
    --price_dir /opt/price \
    --universe_csv ./data/csi1000.csv \
    --start_date 2022-01-01 \
    --epochs 10 \
    --modes market_report market_only report_only \
    --output_dir ./ablation_outputs
```

| `feature_mode` | 含义 |
|---|---|
| `market_report` | 完整特征（市场因子 + LLM 情绪） |
| `market_only` | 去掉 LLM，纯市场因子 |
| `report_only` | 去掉市场因子，纯 LLM 情绪 |

结果保存至 `ablation_outputs/summary.json`，并打印对比表。

---

## 8. 数据检查与准备工具

所有工具位于 `tushare_proxy/` 目录，在服务器上运行。

### 8.1 批量拉取 tushare 因子

```bash
cd tushare_proxy
python batch_fetch.py \
    --out_dir /opt/tushare_factors \
    --start 2022-01-01 \
    --end 2026-01-01
```

因子包含：`overnight_corr`（隔夜相关性）、`roe`（净资产收益率）、`turnover_20d`（20 日均换手率）、`value_bp`（账面市值比）。

### 8.2 因子文件停牌日前向填充

```bash
python pad_factor_data.py \
    --factor_dir /opt/tushare_factors \
    --start 2022-01-01 \
    --end 2026-01-01
```

### 8.3 数据完整性检查

```bash
python check_price_data.py \
    --price_dir /opt/price \
    --start 2022-01-01 \
    --end 2026-01-01 \
    --check_gaps

python check_price_data.py \
    --factor_dir /opt/tushare_factors \
    --start 2022-01-01 \
    --end 2026-01-01 \
    --check_gaps
```

### 8.4 研报覆盖率检查

```bash
python check_report_coverage.py \
    --panel_csv ../paper_data/paper_event_panel.csv \
    --start 2022-01-01 \
    --end 2026-01-01
```

---

## 9. 安装与环境

**GPU 要求：CUDA Compute Capability ≥ 7.0（V100 及以上）**

```bash
# 创建 conda 环境
conda create -n gat python=3.11 -y
conda activate gat

# PyTorch 2.6（支持 SM 7.0 的最新版本，兼容 V100）
pip install --index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"

# 其他依赖
pip install torch-geometric
pip install pandas numpy scipy scikit-learn tqdm
```

> **注意**：PyTorch 2.7+ 最低要求 SM 7.5，V100（SM 7.0）需使用 PyTorch 2.6。

核心依赖版本：

| 包 | 版本 |
|---|---|
| Python | 3.11 |
| PyTorch | 2.6.0+cu124 |
| torch-geometric | ≥ 2.4.0 |
| pandas | ≥ 2.0.0 |
| numpy | ≥ 1.24.0 |

---

## 10. 文件说明

### 根目录

| 文件 | 说明 |
|---|---|
| `train_daily_report.py` | 主训练入口 |
| `ablation_llm.py` | 消融实验脚本 |
| `train.py` | 通用 `Trainer`（含 target mask 支持） |
| `config.py` | 模型与训练默认配置 |
| `utils.py` | 损失函数、评估指标、日志、checkpoint |
| `setup_gpu_env.sh` | GPU 环境安装脚本（PyTorch 2.6+cu124） |

### `data/`

| 文件 | 说明 |
|---|---|
| `daily_report_dataset.py` | 核心 Dataset：日频市场 + 稀疏研报 LLM，含三种边构造模式和 LLM 时间衰减 |
| `stock_industry.csv` | 股票代码 → 申万行业映射（构造 sector 节点） |

### `models/`

| 文件 | 说明 |
|---|---|
| `market_sentiment_gat.py` | 主模型 `MarketSentimentGAT` |
| `layers/hetero_conv.py` | `HeteroGATConv`（两级注意力）、`StackedHeteroGATConv` |

### `tushare_proxy/`

| 文件 | 说明 |
|---|---|
| `tushare_client.py` | tushare 代理客户端 |
| `factor_fetch.py` | 单股因子拉取 |
| `batch_fetch.py` | CSI1000 批量因子拉取 |
| `pad_factor_data.py` | 停牌日因子前向填充 |
| `check_price_data.py` | 价格/因子文件完整性检查 |
| `check_report_coverage.py` | 研报 LLM 对 CSI1000 的覆盖情况检查 |

---

## 11. 外部数据依赖

| 路径 | 说明 |
|---|---|
| `/opt/price/{sh\|sz}.{code}.csv` | 日频价格文件（每股一个 CSV） |
| `/opt/tushare_factors/{code}.csv` | tushare 量化因子（由 `batch_fetch.py` 生成，从 2022 年起） |
| `/opt/paper_event_panel.csv` | 研报 LLM 打分面板（原始版本，不使用 denoised） |
| `data/stock_industry.csv` | 股票行业映射表 |
| `data/csi1000.csv` | CSI1000 成分股列表（传入 `--universe_csv`） |

新机器复现训练的准备顺序：

1. 准备 `/opt/price/`
2. 运行 `batch_fetch.py` 生成 `/opt/tushare_factors/`（从 2022-01-01 起）
3. 运行 `pad_factor_data.py` 补全停牌日
4. 准备 `/opt/paper_event_panel.csv`
5. 准备 `data/stock_industry.csv` 和 `data/csi1000.csv`
6. 运行 `train_daily_report.py`（加 `--start_date 2022-01-01`）

---

## 12. 已知局限与待改进

### 已修复

| 问题 | 状态 | 改动 |
|---|---|---|
| 边与节点特征同源（returns → mom/vol/corr都来自同一份数据） | ✅ 已修 | 新增 `edge_mode=fundamentals/static`，用正交基本面特征建边 |
| LLM 前向填充无时间衰减（3月前研报权重=昨天研报） | ✅ 已修 | 新增 `llm_decay_days`，默认半衰期 30 交易日 |
| 默认预测 60d 太激进（10d IC 更显著） | ✅ 已修 | `target_horizon` 默认改为 10 |
| 默认 batch_size/epochs 太小 | ✅ 已修 | batch_size 2→8, epochs 3→20 |

### 待改进

| 优先级 | 问题 | 说明 |
|---|---|---|
| 🔴 P0 | Sector 节点无独立数据 | `_bucket_means` 纯行业均值池化，无增量信息。需引入申万行业指数收益率或行业 ETF 数据 |
| 🔴 P0 | 缺少 multi-horizon 消融 | 需跑 `for h in 10 20 60; do ... --target_horizon $h` 对比 IC |
| 🔴 P1 | 缺少随机边证伪实验 | 跑 `--graph_mode random_stock_edges` 确认模型真正从图结构学习 |
| 🟡 P2 | 单帧图无时序建模 | 当前每天独立构图，无法学"IC持续高/低"的动态。可加 GRU 滑动窗口 |
| 🟡 P2 | 无实际 IC 数字 | README 只写了 IC>0.03 标准，需报告模型在各 horizon/edge_mode 下的实际 IC |

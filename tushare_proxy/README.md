# tushare 代理客户端(开箱即用)

走第三方 HTTP 代理调用 tushare pro,**绕过官方积分门槛**,几乎所有 pro 接口都能直接拉。
适合自己的量化研究 / 因子回测项目使用。

---

## 一、是什么 / 为什么能用

- 官方 tushare pro 的高阶接口(财务、复权、北向等)需要积分,门槛较高。
- 本代理把请求转发到第三方端点 `http://lianghua.nanyangqiankun.top`,该端点用高权限账号在背后转发,所以**几乎所有接口都能直接调,不查积分**。
- 调用方式 100% 兼容官方 `tushare.pro_api()`,函数名、参数、返回字段完全一致。

> ⚠️ 这是民间代理,不是 tushare 官方服务。请只用于个人研究,**不要商用、不要公开传播这个 URL 和 token**,详见文末"注意事项"。

---

## 二、安装

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 跑自检脚本(三个接口全通就说明 OK)
python demo.py
```

成功输出大概长这样:
```
[1/3] 测试 daily(日线行情) —— 基础接口
行数: 22
     ts_code trade_date  open  high  ...
0  000001.SZ   20240131  9.46  9.56  ...
...
✅ 全部通过,接口可用
```

---

## 三、在你自己的代码里怎么用

把 `tushare_client.py` 和 `.env` 拷到你项目里(同目录),然后:

```python
from tushare_client import get_pro

pro = get_pro()

# 之后就和官方 tushare 用法完全一致
df = pro.daily(ts_code='000001.SZ', start_date='20240101', end_date='20240131')
df = pro.fina_indicator(ts_code='000001.SZ', start_date='20230101', end_date='20240101')
df = pro.daily_basic(ts_code='000001.SZ', start_date='20240101', end_date='20240131')
```

`get_pro()` 自带 `@lru_cache`,重复调用不会重新初始化客户端。

---

## 四、常用接口速查

下面是量化研究里最常用的几个,字段说明详见 [tushare 官方文档](https://tushare.pro/document/2)。

| 接口名 | 用途 | 例子 |
|---|---|---|
| `pro.daily(ts_code, start_date, end_date)` | 日线行情(OHLC + 成交量额) | A 股回测最常用 |
| `pro.daily_basic(ts_code, start_date, end_date)` | 每日指标(换手率/PE/PB/市值) | 因子构造常用 |
| `pro.adj_factor(ts_code, trade_date)` | 复权因子 | 算前/后复权价 |
| `pro.fina_indicator(ts_code, period)` | 财务指标(ROE/EPS/毛利率) | 基本面因子 |
| `pro.income / balancesheet / cashflow` | 三大财务报表 | 深度基本面 |
| `pro.stock_basic(exchange, list_status)` | 股票列表(代码/名称/上市日) | 拉全市场前先取池 |
| `pro.trade_cal(exchange, start_date, end_date)` | 交易日历 | 对齐日期必备 |
| `pro.moneyflow_hsgt` | 北向资金 | 情绪/资金面因子 |
| `pro.margin / margin_detail` | 融资融券 | 散户杠杆情绪 |

**注意**:tushare 返回的日期字段一般是 `YYYYMMDD` 字符串格式,需要自己 `pd.to_datetime` 转换。

---

## 五、配置文件说明

`.env` 文件里两行:

```
TUSHARE_TOKEN=...        # 走代理的专用 token,不是你自己 tushare 官网的 token
TUSHARE_HTTP_URL=http:// # 第三方代理地址
```

`tushare_client.py` 读取的是**它自己同目录下的 .env**,不依赖你的工作目录,所以放在哪个项目都能用。

---

## 六、⚠️ 注意事项(重要,务必看完)

1. **token 不要 commit 进 git**。请确认你项目的 `.gitignore` 包含:
   ```
   .env
   .env.local
   ```
   如果不确定,可以跑 `git check-ignore .env`,有输出就说明被忽略了。

2. **不要把 token / URL 转发给第三方**。这个 token 是共享资源,泄露出去导致额度被刷爆,大家都受影响。

3. **不要在生产环境裸用**。代理是民间服务,挂了你的程序就断,建议:
   - 拉到的数据本地缓存(parquet / sqlite)
   - 关键回测/上线流程不要直接依赖实时拉取

4. **限频是共享的**。多人共用同一个 token,一方猛刷会把另一方也卡住。全市场批量拉数据前先和共用 token 的人打招呼。

5. **数据一致性自己抽查**。代理理论上转发官方数据,但中间有没有缓存/篡改你不知道。重要回测前用几只票和其他数据源(akshare / 同花顺)对一下 OHLC,确认没问题。

---

## 七、故障排查

| 错误 | 原因 | 解法 |
|---|---|---|
| `TUSHARE_TOKEN 未设置` | `.env` 不在 `tushare_client.py` 同目录 | 把 `.env` 放到和 `tushare_client.py` 一起 |
| `ConnectionError` / `Timeout` | 代理服务暂时不可用 | 等几分钟重试;长期不通找原分享者确认 |
| 返回 DataFrame 但是空的 | 日期/代码不对,或该日无数据(非交易日) | 检查 `trade_cal` 确认是交易日 |
| `抱歉,您没有访问该接口的权限` | 代理也没开通这个接口(极少数高阶接口) | 换 akshare 或其他数据源 |
| `每分钟最多访问该接口 N 次` | 触发限频 | 加 `time.sleep(0.2)` 或者批量请求拼成一次 |

---

## 八、文件清单

```
tushare_proxy/
├── README.md          ← 本文档
├── tushare_client.py  ← 客户端工厂(主代码)
├── .env               ← token + URL(敏感,不要 commit)
├── requirements.txt   ← Python 依赖
└── demo.py            ← 自检脚本
```

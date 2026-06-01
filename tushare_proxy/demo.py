"""一键自检脚本:验证 token + 代理是否能正常拉数据.

跑法:
    python demo.py

如果三个接口都打印出 DataFrame 就 OK.
"""
from tushare_client import get_pro


def main():
    pro = get_pro()

    print('=' * 60)
    print('[1/3] 测试 daily(日线行情) —— 基础接口')
    print('=' * 60)
    df = pro.daily(ts_code='000001.SZ', start_date='20240101', end_date='20240131')
    print(f'行数: {len(df)}')
    print(df.head())

    print('\n' + '=' * 60)
    print('[2/3] 测试 fina_indicator(财务指标) —— 官方需 2000 积分')
    print('=' * 60)
    df = pro.fina_indicator(ts_code='000001.SZ', start_date='20230101', end_date='20240101')
    print(f'行数: {len(df)}')
    print(df[['ts_code', 'end_date', 'roe', 'roe_yoy', 'eps']].head())

    print('\n' + '=' * 60)
    print('[3/3] 测试 daily_basic(每日指标:换手率/PE/PB等)')
    print('=' * 60)
    df = pro.daily_basic(
        ts_code='000001.SZ', start_date='20240101', end_date='20240131',
        fields='ts_code,trade_date,turnover_rate,pe_ttm,pb,total_mv'
    )
    print(f'行数: {len(df)}')
    print(df.head())

    print('\n✅ 全部通过,接口可用')


if __name__ == '__main__':
    main()

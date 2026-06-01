"""tushare 客户端工厂(走第三方 HTTP 代理,绕过官方积分限制).

用法:
    from tushare_client import get_pro
    pro = get_pro()
    df = pro.daily(ts_code='000001.SZ', start_date='20240101', end_date='20240131')

token 与 HTTP URL 从同目录 .env 读取(TUSHARE_TOKEN / TUSHARE_HTTP_URL).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import tushare as ts
from dotenv import load_dotenv

# 加载本文件同目录下的 .env(不依赖调用方的工作目录)
load_dotenv(Path(__file__).parent / '.env')


@lru_cache(maxsize=1)
def get_pro(timeout: int = 30):
    """返回配置好代理 URL 的 tushare pro 客户端.

    Args:
        timeout: HTTP 请求超时(秒).代理偶尔会卡,设短一点便于失败重试.
    """
    token = os.environ.get('TUSHARE_TOKEN')
    if not token:
        raise RuntimeError(
            'TUSHARE_TOKEN 未设置.请在 tushare_client.py 同目录建 .env,'
            '加入 TUSHARE_TOKEN=xxx 和 TUSHARE_HTTP_URL=http://...'
        )

    pro = ts.pro_api(token, timeout=timeout)

    http_url = os.environ.get('TUSHARE_HTTP_URL')
    if http_url:
        # 关键一行:把官方端点替换为第三方代理,这样不查积分
        pro._DataApi__http_url = http_url
    return pro


if __name__ == '__main__':
    pro = get_pro()
    df = pro.daily(ts_code='000001.SZ', start_date='20240101', end_date='20240131')
    print(df)

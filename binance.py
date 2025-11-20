import requests
import hmac
import hashlib
import time
from urllib.parse import urlencode

# ===== 幣安 API 設定 =====
API_KEY = "your_api_key_here"  # 替換成你的 API Key
API_SECRET = "your_api_secret_here"  # 替換成你的 API Secret

BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"


def get_server_time():
    """取得幣安伺服器時間。

    Returns:
        int: 伺服器時間戳記 (毫秒)。
    """
    url = f"{BINANCE_FUTURES_BASE_URL}/fapi/v1/time"
    response = requests.get(url)
    return response.json()["serverTime"]


def generate_signature(query_string, secret):
    """產生 HMAC SHA256 簽名。

    Args:
        query_string: 查詢字串。
        secret: API Secret。

    Returns:
        str: 簽名字串。
    """
    return hmac.new(
        secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def get_account_balance():
    """取得幣安合約帳戶餘額。

    Returns:
        dict: 包含帳戶資訊的字典,包括總餘額、可用餘額等。

    潛在問題:
        1. API Key 和 Secret 需要有讀取權限
        2. 需要處理 API 限流問題
        3. 時間戳記需要與伺服器同步,否則會出現簽名錯誤
    """
    endpoint = "/fapi/v2/account"

    # 取得伺服器時間以避免時間戳記不同步問題
    timestamp = get_server_time()

    # 建立查詢參數
    params = {
        "timestamp": timestamp,
        "recvWindow": 5000,  # 接收視窗時間 (毫秒)
    }

    # 產生查詢字串
    query_string = urlencode(params)

    # 產生簽名
    signature = generate_signature(query_string, API_SECRET)

    # 加入簽名到參數
    params["signature"] = signature

    # 設定請求標頭
    headers = {"X-MBX-APIKEY": API_KEY}

    # 發送請求
    url = f"{BINANCE_FUTURES_BASE_URL}{endpoint}"
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()

        # 解析餘額資訊
        total_wallet_balance = float(data.get("totalWalletBalance", 0))
        available_balance = float(data.get("availableBalance", 0))
        total_unrealized_profit = float(data.get("totalUnrealizedProfit", 0))

        # 取得各資產餘額
        assets = data.get("assets", [])
        usdt_asset = next((asset for asset in assets if asset["asset"] == "USDT"), None)

        balance_info = {
            "total_wallet_balance": total_wallet_balance,
            "available_balance": available_balance,
            "total_unrealized_profit": total_unrealized_profit,
            "usdt_wallet_balance": float(usdt_asset["walletBalance"])
            if usdt_asset
            else 0,
            "usdt_available_balance": float(usdt_asset["availableBalance"])
            if usdt_asset
            else 0,
        }

        return balance_info
    else:
        print(f"❌ API 請求失敗: {response.status_code}")
        print(f"錯誤訊息: {response.text}")
        return None


def get_simple_balance():
    """取得簡化版的 USDT 餘額資訊。

    Returns:
        float: USDT 可用餘額。
    """
    balance_info = get_account_balance()

    if balance_info:
        print(f"📊 帳戶餘額資訊:")
        print(f"   總錢包餘額: {balance_info['total_wallet_balance']:.2f} USDT")
        print(f"   可用餘額: {balance_info['available_balance']:.2f} USDT")
        print(f"   未實現盈虧: {balance_info['total_unrealized_profit']:.2f} USDT")
        print(f"   USDT 錢包餘額: {balance_info['usdt_wallet_balance']:.2f} USDT")
        print(f"   USDT 可用餘額: {balance_info['usdt_available_balance']:.2f} USDT")

        return balance_info["usdt_available_balance"]
    else:
        return 0.0


# ===== 使用範例 =====
if __name__ == "__main__":
    # 方法 1: 取得完整餘額資訊
    balance_info = get_account_balance()
    if balance_info:
        print(f"\n✅ 成功取得帳戶資訊")
        print(f"可用餘額: {balance_info['available_balance']:.2f} USDT")

    # 方法 2: 取得簡化版 USDT 餘額
    print("\n" + "=" * 50)
    usdt_balance = get_simple_balance()

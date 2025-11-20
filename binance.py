import requests
import hmac
import hashlib
import time
import os
import logging
from urllib.parse import urlencode
from functools import wraps

# ===== 日誌設定 =====
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("binance_api.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# ===== 幣安 API 設定 =====
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"

# 檢查 API 金鑰是否已設定
if not API_KEY or not API_SECRET:
    logger.warning("⚠️ BINANCE_API_KEY 或 BINANCE_API_SECRET 未設定,請設定環境變數")
    # 如果是測試環境,可以使用預設值
    if os.getenv("ENV") != "production":
        API_KEY = "your_api_key_here"
        API_SECRET = "your_api_secret_here"
        logger.warning("⚠️ 使用預設 API 金鑰 (僅供測試)")


# ===== 重試裝飾器 =====
def retry_on_failure(max_retries=3, delay=1):
    """重試裝飾器。

    Args:
        max_retries: 最大重試次數。
        delay: 重試間隔 (秒)。
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(
                            f"❌ {func.__name__} 失敗 (已重試 {max_retries} 次): {e}"
                        )
                        raise
                    logger.warning(
                        f"⚠️ {func.__name__} 第 {attempt + 1} 次嘗試失敗: {e}"
                    )
                    logger.info(f"   {delay} 秒後重試...")
                    time.sleep(delay)
            return None

        return wrapper

    return decorator


# ===== 統一 API 請求處理 =====
def _make_request(method, endpoint, params=None, headers=None):
    """統一的 API 請求處理函數。

    Args:
        method: HTTP 方法 ("GET", "POST", "DELETE")。
        endpoint: API 端點。
        params: 請求參數。
        headers: 請求標頭。

    Returns:
        dict: API 回應資料。

    Raises:
        RuntimeError: 當 API 請求失敗時。
    """
    url = f"{BINANCE_FUTURES_BASE_URL}{endpoint}"

    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=params, timeout=10)
        elif method == "POST":
            response = requests.post(url, headers=headers, params=params, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, params=params, timeout=10)
        else:
            raise ValueError(f"不支援的 HTTP 方法: {method}")

        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        raise RuntimeError(f"❌ API 請求逾時: {endpoint}")
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"❌ 無法連接到伺服器: {endpoint}")
    except requests.exceptions.HTTPError as e:
        error_msg = response.text if response else str(e)
        raise RuntimeError(f"❌ API 請求失敗 ({response.status_code}): {error_msg}")
    except Exception as e:
        raise RuntimeError(f"❌ 未預期的錯誤: {e}")


@retry_on_failure(max_retries=3, delay=2)
def get_server_time():
    """取得幣安伺服器時間。

    Returns:
        int: 伺服器時間戳記 (毫秒)。

    Raises:
        RuntimeError: 當無法連接到伺服器時。
    """
    endpoint = "/fapi/v1/time"
    data = _make_request("GET", endpoint)
    return data["serverTime"]


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

    Raises:
        RuntimeError: 當 API 請求失敗時。
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
    data = _make_request("GET", endpoint, params=params, headers=headers)

    # 解析餘額資訊
    total_wallet_balance = float(data.get("totalWalletBalance", 0))
    available_balance = float(data.get("availableBalance", 0))
    total_unrealized_profit = float(data.get("totalUnrealizedProfit", 0))

    # 取得各資產餘額
    assets = data.get("assets", [])
    usdt_asset = next((asset for asset in assets if asset.get("asset") == "USDT"), None)

    balance_info = {
        "total_wallet_balance": total_wallet_balance,
        "available_balance": available_balance,
        "total_unrealized_profit": total_unrealized_profit,
        "usdt_wallet_balance": float(usdt_asset.get("walletBalance", 0))
        if usdt_asset
        else 0,
        "usdt_available_balance": float(usdt_asset.get("availableBalance", 0))
        if usdt_asset
        else 0,
    }

    logger.info(f"✅ 成功取得帳戶餘額: {available_balance:.2f} USDT")
    return balance_info


def get_simple_balance():
    """取得簡化版的 USDT 餘額資訊。

    Returns:
        float: USDT 可用餘額。

    Raises:
        RuntimeError: 當 API 請求失敗時。
    """
    balance_info = get_account_balance()

    print(f"📊 帳戶餘額資訊:")
    print(f"   總錢包餘額: {balance_info['total_wallet_balance']:.2f} USDT")
    print(f"   可用餘額: {balance_info['available_balance']:.2f} USDT")
    print(f"   未實現盈虧: {balance_info['total_unrealized_profit']:.2f} USDT")
    print(f"   USDT 錢包餘額: {balance_info['usdt_wallet_balance']:.2f} USDT")
    print(f"   USDT 可用餘額: {balance_info['usdt_available_balance']:.2f} USDT")

    return balance_info["usdt_available_balance"]


def set_leverage(symbol, leverage):
    """設定合約槓桿倍數。

    Args:
        symbol: 交易對符號 (例如: "BTCUSDT")。
        leverage: 槓桿倍數 (1-125)。

    Returns:
        dict: API 回應資料。

    潛在問題:
        1. 不同交易對的最大槓桿倍數不同
        2. 需要有交易權限的 API Key

    Raises:
        ValueError: 當參數不符合規則時。
        RuntimeError: 當 API 請求失敗時。
    """
    # 驗證參數
    if not isinstance(leverage, int) or leverage < 1 or leverage > 125:
        raise ValueError(f"❌ 槓桿倍數必須在 1-125 之間,目前為 {leverage}")

    endpoint = "/fapi/v1/leverage"

    timestamp = get_server_time()

    params = {
        "symbol": symbol,
        "leverage": leverage,
        "timestamp": timestamp,
        "recvWindow": 5000,
    }

    query_string = urlencode(params)
    signature = generate_signature(query_string, API_SECRET)
    params["signature"] = signature

    headers = {"X-MBX-APIKEY": API_KEY}

    data = _make_request("POST", endpoint, params=params, headers=headers)

    logger.info(f"✅ 槓桿設定成功: {symbol} = {leverage}x")
    print(f"✅ 槓桿設定成功: {symbol} = {leverage}x")
    return data


def place_market_order(symbol, side, quantity, leverage=None, reduce_only=False):
    """下市價單 (合約)。

    Args:
        symbol: 交易對符號 (例如: "BTCUSDT")。
        side: 訂單方向 ("BUY" 或 "SELL")。
        quantity: 數量 (BTC 數量,需符合最小單位)。
        leverage: 槓桿倍數 (1-125),若提供則會先設定槓桿。
        reduce_only: 是否為只減倉訂單 (平倉用)。

    Returns:
        dict: 訂單資訊,包含訂單 ID、成交價格等。

    潛在問題:
        1. 數量必須符合交易對的最小單位 (MIN_QTY)
        2. 名義價值必須大於最小值 (MIN_NOTIONAL)
        3. 餘額不足會導致下單失敗
        4. 市價單會立即成交,可能有滑價

    Raises:
        ValueError: 當參數不符合規則時。
        RuntimeError: 當 API 請求失敗時。
    """
    # 驗證參數
    if quantity <= 0:
        raise ValueError(f"❌ 數量必須大於 0,目前為 {quantity}")

    if side not in ["BUY", "SELL"]:
        raise ValueError(f"❌ 訂單方向必須為 BUY 或 SELL,目前為 {side}")

    # 如果提供槓桿參數,先設定槓桿
    if leverage is not None:
        set_leverage(symbol, leverage)

    endpoint = "/fapi/v1/order"

    timestamp = get_server_time()

    params = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": quantity,
        "timestamp": timestamp,
        "recvWindow": 5000,
    }

    if reduce_only:
        params["reduceOnly"] = "true"

    query_string = urlencode(params)
    signature = generate_signature(query_string, API_SECRET)
    params["signature"] = signature

    headers = {"X-MBX-APIKEY": API_KEY}

    data = _make_request("POST", endpoint, params=params, headers=headers)

    logger.info(
        f"✅ 市價單成功: {side} {quantity} {symbol} (訂單 ID: {data['orderId']})"
    )
    print(f"✅ 市價單成功: {side} {quantity} {symbol}")
    print(f"   訂單 ID: {data['orderId']}")
    print(f"   成交價格: {data.get('avgPrice', 'N/A')}")
    return data


def place_limit_order(
    symbol, side, quantity, price, leverage=None, reduce_only=False, time_in_force="GTC"
):
    """下限價單 (合約)。

    Args:
        symbol: 交易對符號 (例如: "BTCUSDT")。
        side: 訂單方向 ("BUY" 或 "SELL")。
        quantity: 數量 (BTC 數量)。
        price: 限價價格。
        leverage: 槓桿倍數 (1-125),若提供則會先設定槓桿。
        reduce_only: 是否為只減倉訂單。
        time_in_force: 訂單有效期 ("GTC", "IOC", "FOK")。

    Returns:
        dict: 訂單資訊。

    潛在問題:
        1. 限價單不一定會成交
        2. 價格必須符合 tick size 規則
        3. GTC 訂單會一直掛單直到成交或取消

    Raises:
        ValueError: 當參數不符合規則時。
        RuntimeError: 當 API 請求失敗時。
    """
    # 驗證參數
    if quantity <= 0:
        raise ValueError(f"❌ 數量必須大於 0,目前為 {quantity}")

    if price <= 0:
        raise ValueError(f"❌ 價格必須大於 0,目前為 {price}")

    if side not in ["BUY", "SELL"]:
        raise ValueError(f"❌ 訂單方向必須為 BUY 或 SELL,目前為 {side}")

    if time_in_force not in ["GTC", "IOC", "FOK"]:
        raise ValueError(f"❌ 訂單有效期必須為 GTC、IOC 或 FOK,目前為 {time_in_force}")

    # 如果提供槓桿參數,先設定槓桿
    if leverage is not None:
        set_leverage(symbol, leverage)

    endpoint = "/fapi/v1/order"

    timestamp = get_server_time()

    params = {
        "symbol": symbol,
        "side": side,
        "type": "LIMIT",
        "quantity": quantity,
        "price": price,
        "timeInForce": time_in_force,
        "timestamp": timestamp,
        "recvWindow": 5000,
    }

    if reduce_only:
        params["reduceOnly"] = "true"

    query_string = urlencode(params)
    signature = generate_signature(query_string, API_SECRET)
    params["signature"] = signature

    headers = {"X-MBX-APIKEY": API_KEY}

    data = _make_request("POST", endpoint, params=params, headers=headers)

    logger.info(
        f"✅ 限價單成功: {side} {quantity} {symbol} @ {price} (訂單 ID: {data['orderId']})"
    )
    print(f"✅ 限價單成功: {side} {quantity} {symbol} @ {price}")
    print(f"   訂單 ID: {data['orderId']}")
    return data


def place_stop_market_order(
    symbol, side, quantity, stop_price, leverage=None, reduce_only=False
):
    """下止損市價單 (Stop-Loss Market Order)。

    Args:
        symbol: 交易對符號。
        side: 訂單方向 ("BUY" 或 "SELL")。
        quantity: 數量。
        stop_price: 觸發價格。
        leverage: 槓桿倍數 (1-125),若提供則會先設定槓桿。
        reduce_only: 是否為只減倉訂單。

    Returns:
        dict: 訂單資訊。

    潛在問題:
        1. 止損單只有在價格觸及 stop_price 時才會觸發
        2. 觸發後會以市價成交,可能有滑價

    Raises:
        ValueError: 當參數不符合規則時。
        RuntimeError: 當 API 請求失敗時。
    """
    # 驗證參數
    if quantity <= 0:
        raise ValueError(f"❌ 數量必須大於 0,目前為 {quantity}")

    if stop_price <= 0:
        raise ValueError(f"❌ 觸發價格必須大於 0,目前為 {stop_price}")

    if side not in ["BUY", "SELL"]:
        raise ValueError(f"❌ 訂單方向必須為 BUY 或 SELL,目前為 {side}")

    # 如果提供槓桿參數,先設定槓桿
    if leverage is not None:
        set_leverage(symbol, leverage)

    endpoint = "/fapi/v1/order"

    timestamp = get_server_time()

    params = {
        "symbol": symbol,
        "side": side,
        "type": "STOP_MARKET",
        "quantity": quantity,
        "stopPrice": stop_price,
        "timestamp": timestamp,
        "recvWindow": 5000,
    }

    if reduce_only:
        params["reduceOnly"] = "true"

    query_string = urlencode(params)
    signature = generate_signature(query_string, API_SECRET)
    params["signature"] = signature

    headers = {"X-MBX-APIKEY": API_KEY}

    data = _make_request("POST", endpoint, params=params, headers=headers)

    logger.info(
        f"✅ 止損單成功: {side} {quantity} {symbol} @ Stop {stop_price} (訂單 ID: {data['orderId']})"
    )
    print(f"✅ 止損單成功: {side} {quantity} {symbol} @ Stop {stop_price}")
    print(f"   訂單 ID: {data['orderId']}")
    return data


def place_take_profit_market_order(
    symbol, side, quantity, stop_price, leverage=None, reduce_only=False
):
    """下止盈市價單 (Take-Profit Market Order)。

    Args:
        symbol: 交易對符號。
        side: 訂單方向 ("BUY" 或 "SELL")。
        quantity: 數量。
        stop_price: 觸發價格。
        leverage: 槓桿倍數 (1-125),若提供則會先設定槓桿。
        reduce_only: 是否為只減倉訂單。

    Returns:
        dict: 訂單資訊。

    Raises:
        ValueError: 當參數不符合規則時。
        RuntimeError: 當 API 請求失敗時。
    """
    # 驗證參數
    if quantity <= 0:
        raise ValueError(f"❌ 數量必須大於 0,目前為 {quantity}")

    if stop_price <= 0:
        raise ValueError(f"❌ 觸發價格必須大於 0,目前為 {stop_price}")

    if side not in ["BUY", "SELL"]:
        raise ValueError(f"❌ 訂單方向必須為 BUY 或 SELL,目前為 {side}")

    # 如果提供槓桿參數,先設定槓桿
    if leverage is not None:
        set_leverage(symbol, leverage)

    endpoint = "/fapi/v1/order"

    timestamp = get_server_time()

    params = {
        "symbol": symbol,
        "side": side,
        "type": "TAKE_PROFIT_MARKET",
        "quantity": quantity,
        "stopPrice": stop_price,
        "timestamp": timestamp,
        "recvWindow": 5000,
    }

    if reduce_only:
        params["reduceOnly"] = "true"

    query_string = urlencode(params)
    signature = generate_signature(query_string, API_SECRET)
    params["signature"] = signature

    headers = {"X-MBX-APIKEY": API_KEY}

    data = _make_request("POST", endpoint, params=params, headers=headers)

    logger.info(
        f"✅ 止盈單成功: {side} {quantity} {symbol} @ TP {stop_price} (訂單 ID: {data['orderId']})"
    )
    print(f"✅ 止盈單成功: {side} {quantity} {symbol} @ TP {stop_price}")
    print(f"   訂單 ID: {data['orderId']}")
    return data


def get_order_status(symbol, order_id):
    """查詢訂單狀態。

    Args:
        symbol: 交易對符號。
        order_id: 訂單 ID。

    Returns:
        dict: 訂單詳細資訊,包含狀態、成交價格、成交數量等。

    訂單狀態說明:
        - NEW: 新訂單,尚未成交
        - PARTIALLY_FILLED: 部分成交
        - FILLED: 完全成交
        - CANCELED: 已取消
        - REJECTED: 被拒絕
        - EXPIRED: 已過期

    Raises:
        RuntimeError: 當 API 請求失敗時。
    """
    endpoint = "/fapi/v1/order"

    timestamp = get_server_time()

    params = {
        "symbol": symbol,
        "orderId": order_id,
        "timestamp": timestamp,
        "recvWindow": 5000,
    }

    query_string = urlencode(params)
    signature = generate_signature(query_string, API_SECRET)
    params["signature"] = signature

    headers = {"X-MBX-APIKEY": API_KEY}

    data = _make_request("GET", endpoint, params=params, headers=headers)

    logger.info(f"📋 訂單狀態查詢: {symbol} 訂單 ID {order_id} - {data['status']}")
    print(f"📋 訂單狀態: {data['status']}")
    print(f"   訂單 ID: {data['orderId']}")
    print(f"   類型: {data['type']}")
    print(f"   方向: {data['side']}")
    print(f"   數量: {data['origQty']}")
    print(f"   已成交: {data['executedQty']}")
    print(f"   平均成交價: {data.get('avgPrice', 'N/A')}")
    return data


def get_all_open_orders(symbol=None):
    """查詢所有未成交訂單。

    Args:
        symbol: 交易對符號 (可選,不填則查詢所有交易對)。

    Returns:
        list: 未成交訂單列表。

    Raises:
        RuntimeError: 當 API 請求失敗時。
    """
    endpoint = "/fapi/v1/openOrders"

    timestamp = get_server_time()

    params = {"timestamp": timestamp, "recvWindow": 5000}

    if symbol:
        params["symbol"] = symbol

    query_string = urlencode(params)
    signature = generate_signature(query_string, API_SECRET)
    params["signature"] = signature

    headers = {"X-MBX-APIKEY": API_KEY}

    orders = _make_request("GET", endpoint, params=params, headers=headers)

    logger.info(f"📋 未成交訂單數量: {len(orders)}")
    print(f"📋 未成交訂單數量: {len(orders)}")
    for order in orders:
        print(
            f"   - {order['symbol']}: {order['side']} {order['origQty']} @ {order.get('price', 'MARKET')}"
        )
    return orders


def cancel_order(symbol, order_id):
    """取消訂單。

    Args:
        symbol: 交易對符號。
        order_id: 訂單 ID。

    Returns:
        dict: 取消結果。

    Raises:
        RuntimeError: 當 API 請求失敗時。
    """
    endpoint = "/fapi/v1/order"

    timestamp = get_server_time()

    params = {
        "symbol": symbol,
        "orderId": order_id,
        "timestamp": timestamp,
        "recvWindow": 5000,
    }

    query_string = urlencode(params)
    signature = generate_signature(query_string, API_SECRET)
    params["signature"] = signature

    headers = {"X-MBX-APIKEY": API_KEY}

    data = _make_request("DELETE", endpoint, params=params, headers=headers)

    logger.info(f"✅ 訂單已取消: {symbol} 訂單 ID {order_id}")
    print(f"✅ 訂單已取消: {order_id}")
    return data


def cancel_all_open_orders(symbol):
    """取消指定交易對的所有未成交訂單。

    Args:
        symbol: 交易對符號。

    Returns:
        dict: 取消結果。

    Raises:
        RuntimeError: 當 API 請求失敗時。
    """
    endpoint = "/fapi/v1/allOpenOrders"

    timestamp = get_server_time()

    params = {"symbol": symbol, "timestamp": timestamp, "recvWindow": 5000}

    query_string = urlencode(params)
    signature = generate_signature(query_string, API_SECRET)
    params["signature"] = signature

    headers = {"X-MBX-APIKEY": API_KEY}

    data = _make_request("DELETE", endpoint, params=params, headers=headers)

    logger.info(f"✅ 所有訂單已取消: {symbol}")
    print(f"✅ 所有訂單已取消: {symbol}")
    return data


def get_position_info(symbol=None):
    """查詢當前持倉資訊。

    Args:
        symbol: 交易對符號 (可選)。

    Returns:
        list: 持倉資訊列表。

    Raises:
        RuntimeError: 當 API 請求失敗時。
    """
    endpoint = "/fapi/v2/positionRisk"

    timestamp = get_server_time()

    params = {"timestamp": timestamp, "recvWindow": 5000}

    if symbol:
        params["symbol"] = symbol

    query_string = urlencode(params)
    signature = generate_signature(query_string, API_SECRET)
    params["signature"] = signature

    headers = {"X-MBX-APIKEY": API_KEY}

    positions = _make_request("GET", endpoint, params=params, headers=headers)

    # 只顯示有持倉的交易對
    active_positions = [p for p in positions if float(p.get("positionAmt", 0)) != 0]

    logger.info(f"📊 當前持倉數量: {len(active_positions)}")
    print(f"📊 當前持倉數量: {len(active_positions)}")
    for pos in active_positions:
        print(f"   - {pos['symbol']}: {pos['positionAmt']} @ {pos['entryPrice']}")
        print(f"     未實現盈虧: {pos['unRealizedProfit']} USDT")
        print(f"     槓桿: {pos['leverage']}x")

    return positions


# ===== 使用範例 =====
if __name__ == "__main__":
    try:
        # 方法 1: 取得完整餘額資訊
        balance_info = get_account_balance()
        if balance_info:
            print(f"\n✅ 成功取得帳戶資訊")
            print(f"可用餘額: {balance_info['available_balance']:.2f} USDT")

        # 1. 下市價單 (開多單) - 自動設定槓桿為 15x
        # order = place_market_order("BTCUSDT", "BUY", 0.001, leverage=15)

        # 2. 下限價單 - 自動設定槓桿為 20x
        # order = place_limit_order("BTCUSDT", "BUY", 0.001, 95000, leverage=20)

        # 3. 下止損單 - 自動設定槓桿為 15x
        # stop_order = place_stop_market_order("BTCUSDT", "SELL", 0.001, 94000, leverage=15, reduce_only=True)

        # 4. 下止盈單 - 自動設定槓桿為 15x
        # tp_order = place_take_profit_market_order("BTCUSDT", "SELL", 0.001, 96000, leverage=15, reduce_only=True)

        # 5. 查詢訂單狀態
        # if order:
        #     get_order_status("BTCUSDT", order['orderId'])

        # 6. 查詢所有未成交訂單
        # get_all_open_orders("BTCUSDT")

        # 7. 查詢持倉
        # get_position_info("BTCUSDT")

        # 8. 取消訂單
        # if order:
        #     cancel_order("BTCUSDT", order['orderId'])

        # 9. 取消所有訂單
        # cancel_all_open_orders("BTCUSDT")

        # 方法 2: 取得簡化版 USDT 餘額
        print("\n" + "=" * 50)
        usdt_balance = get_simple_balance()

    except ValueError as e:
        logger.error(f"參數錯誤: {e}")
        print(f"參數錯誤: {e}")
    except RuntimeError as e:
        logger.error(f"執行錯誤: {e}")
        print(f"執行錯誤: {e}")
    except Exception as e:
        logger.error(f"未預期的錯誤: {e}")
        print(f"未預期的錯誤: {e}")

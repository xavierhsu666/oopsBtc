import requests
import pandas as pd
import numpy as np
import time

# ===== 可調參數 =====
BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "BTCUSDT"
INTERVAL_MAIN = "5m"
INTERVAL_REF = "15m"
LIMIT = 200

INITIAL_BALANCE = 1000  # 初始資金
LEVERAGE = 15  # 槓桿倍數
RR_THRESHOLD = 1.2  # RR 條件

USE_RSI_FILTER = False  # 是否啟用 RSI 過濾
USE_TREND_FILTER = True  # 是否啟用順勢條件
USE_RR_FILTER = True  # 是否啟用 RR 條件
USE_STOP_TAKE_M15 = True  # 是否使用 M15 高低作為止盈止損


# ===== 取得歷史K線資料 =====
def get_klines(symbol, interval, limit):
    url = f"{BASE_URL}?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms") + pd.Timedelta(hours=8)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(
        float
    )
    return df


# ===== 計算技術指標 =====
def calculate_indicators(df):
    df["EMA9"] = df["close"].ewm(span=9).mean()
    df["EMA21"] = df["close"].ewm(span=21).mean()
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


# ===== 找最近一根 M15 的高低 =====
def get_last_m15_levels(m15_df, current_time):
    ref = m15_df[m15_df["timestamp"] <= current_time].iloc[-1]
    return ref["high"], ref["low"]


# ===== 產生交易訊號 =====
def generate_signal(latest, df_ref):
    m15_high, m15_low = get_last_m15_levels(df_ref, latest["timestamp"])
    m15_trend_up = (
        df_ref[df_ref["timestamp"] <= latest["timestamp"]].iloc[-1]["EMA9"]
        > df_ref[df_ref["timestamp"] <= latest["timestamp"]].iloc[-1]["EMA21"]
    )
    m15_trend_down = not m15_trend_up

    # 多單
    if latest["EMA9"] > latest["EMA21"]:
        if USE_RSI_FILTER and latest["RSI"] >= 70:
            return None
        if USE_TREND_FILTER and not m15_trend_up:
            return None
        entry_price = latest["close"]
        if USE_STOP_TAKE_M15 and (m15_high <= entry_price or m15_low >= entry_price):
            return None
        stop_loss = m15_low if USE_STOP_TAKE_M15 else latest["low"]
        take_profit = m15_high if USE_STOP_TAKE_M15 else latest["high"]
        rr = (take_profit - entry_price) / (entry_price - stop_loss)
        if USE_RR_FILTER and rr <= RR_THRESHOLD:
            return None
        return {
            "signal": "LONG",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "rr": rr,
        }

    # 空單
    elif latest["EMA9"] < latest["EMA21"]:
        if USE_RSI_FILTER and latest["RSI"] <= 30:
            return None
        if USE_TREND_FILTER and not m15_trend_down:
            return None
        entry_price = latest["close"]
        if USE_STOP_TAKE_M15 and (m15_low >= entry_price or m15_high <= entry_price):
            return None
        stop_loss = m15_high if USE_STOP_TAKE_M15 else latest["high"]
        take_profit = m15_low if USE_STOP_TAKE_M15 else latest["low"]
        rr = (entry_price - take_profit) / (stop_loss - entry_price)
        if USE_RR_FILTER and rr <= RR_THRESHOLD:
            return None
        return {
            "signal": "SHORT",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "rr": rr,
        }

    return None


# ===== 回測策略 =====
def backtest(df_main, df_ref):
    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0
    trades = []
    trade_details = []

    df_ref = calculate_indicators(df_ref)

    for i in range(21, len(df_main)):
        latest = df_main.iloc[i]

        # 平倉檢查
        if position != 0:
            if position > 0:  # 多單
                if latest["low"] <= stop_loss or latest["high"] >= take_profit:
                    pnl = (latest["close"] - entry_price) * position
                    balance += pnl
                    trades.append(pnl)
                    margin = (abs(position) * entry_price) / LEVERAGE
                    trade_details.append(
                        {
                            "方向": "多單",
                            "槓桿": LEVERAGE,
                            "開倉大小": round(abs(position), 4),
                            "保證金": round(margin, 2),
                            "進場時間": entry_time,
                            "進場價格": entry_price,
                            "出場時間": latest["timestamp"],
                            "出場價格": latest["close"],
                            "盈虧": round(pnl, 2),
                            "RR": rr,
                        }
                    )
                    position = 0
            else:  # 空單
                if latest["high"] >= stop_loss or latest["low"] <= take_profit:
                    pnl = (entry_price - latest["close"]) * abs(position)
                    balance += pnl
                    trades.append(pnl)
                    margin = (abs(position) * entry_price) / LEVERAGE
                    trade_details.append(
                        {
                            "方向": "空單",
                            "槓桿": LEVERAGE,
                            "開倉大小": round(abs(position), 4),
                            "保證金": round(margin, 2),
                            "進場時間": entry_time,
                            "進場價格": entry_price,
                            "出場時間": latest["timestamp"],
                            "出場價格": latest["close"],
                            "盈虧": round(pnl, 2),
                            "RR": rr,
                        }
                    )
                    position = 0

        # 開倉條件
        if position == 0:
            signal_info = generate_signal(latest, df_ref)
            if signal_info:
                entry_price = signal_info["entry_price"]
                stop_loss = signal_info["stop_loss"]
                take_profit = signal_info["take_profit"]
                rr = signal_info["rr"]
                entry_time = latest["timestamp"]
                position = (
                    (balance * LEVERAGE) / entry_price
                    if signal_info["signal"] == "LONG"
                    else -(balance * LEVERAGE) / entry_price
                )

    # 統計結果
    win_trades = [t for t in trades if t > 0]
    lose_trades = [t for t in trades if t <= 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0
    total_pnl = sum(trades)
    max_drawdown = min(trades) if trades else 0

    print("回測結果:")
    print(f"初始資金: {INITIAL_BALANCE} USDT")
    print(f"槓桿倍數: {LEVERAGE}x")
    print(f"最終資金: {balance:.2f} USDT")
    print(f"總盈虧: {total_pnl:.2f} USDT")
    print(f"交易次數: {len(trades)}")
    print(f"勝率: {win_rate:.2f}%")
    print(f"最大單筆虧損: {max_drawdown:.2f} USDT")

    trade_df = pd.DataFrame(trade_details)
    trade_df.to_excel("trade_df.xlsx")
    print("交易明細已匯出至 trade_df.xlsx")


# ===== 實盤掃描 =====
def pratical_scanner():
    print(f"\n[Scanner Detail]")
    position = 0

    while True:
        try:
            df_main = get_klines(SYMBOL, INTERVAL_MAIN, LIMIT)
            df_ref = get_klines(SYMBOL, INTERVAL_REF, LIMIT)
            df_main = calculate_indicators(df_main)
            df_ref = calculate_indicators(df_ref)

            latest = df_main.iloc[-1]
            signal_info = generate_signal(latest, df_ref)

            trend = "UP" if latest["EMA9"] > latest["EMA21"] else "DOWN"
            print(
                f"{latest['timestamp']}: Close: {round(latest['close'], 2)}, EMA Trend: {trend}, RSI: {round(latest['RSI'], 2)}"
            )

            if position == 0 and signal_info:
                print(
                    f"{latest['timestamp']}: {signal_info['signal']} 訊號, 進場 {signal_info['entry_price']}, TP {signal_info['take_profit']}, SL {signal_info['stop_loss']}, RR {round(signal_info['rr'], 2)}"
                )

            time.sleep(0.3)
        except Exception as e:
            print("錯誤:", e)
            time.sleep(0.3)


# ===== 執行回測 =====
# df_main = get_klines(SYMBOL, INTERVAL_MAIN, LIMIT)
# df_ref = get_klines(SYMBOL, INTERVAL_REF, LIMIT)
# df_main = calculate_indicators(df_main)
# backtest(df_main, df_ref)

# ===== 執行實盤 =====
pratical_scanner()

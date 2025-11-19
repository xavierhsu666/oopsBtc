import requests
import pandas as pd
import numpy as np
import time

# ===== 可調參數 =====
BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "BTCUSDT"
INTERVAL_MAIN = "5m"
INTERVAL_REF = "15m"
LIMIT = 100

INITIAL_BALANCE = 36.28  # 初始資金
LEVERAGE = 15  # 槓桿倍數
RR_THRESHOLD = 1.2  # RR 條件
FEE_RATE = 0.0004  # 幣安合約市價單 taker 費率

USE_RSI_FILTER = False
USE_TREND_FILTER = True
USE_RR_FILTER = True
USE_STOP_TAKE_M15 = True


# ===== 取得歷史K線資料 =====
def get_klines(symbol, interval, limit):
    url = f"{BASE_URL}?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms") + pd.Timedelta(hours=8)
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
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


def get_last_m15_levels(m15_df, current_time):
    ref = m15_df[m15_df["timestamp"] <= current_time].iloc[-1]
    return ref["high"], ref["low"]


# ===== 強平價計算 =====
def calc_liquidation_price(entry_price, leverage, side):
    if side == "LONG":
        return entry_price * (1 - 1 / leverage)
    else:  # SHORT
        return entry_price * (1 + 1 / leverage)


# ===== 實盤掃描 =====
def pratical_scanner():
    print(f"\n[Scanner Detail]")
    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0

    while True:
        try:
            df_main = get_klines(SYMBOL, INTERVAL_MAIN, LIMIT)
            df_ref = get_klines(SYMBOL, INTERVAL_REF, LIMIT)
            df_main = calculate_indicators(df_main)
            df_ref = calculate_indicators(df_ref)

            latest = df_main.iloc[-1]
            if position == 0:
                m15_high, m15_low = get_last_m15_levels(df_ref, latest["timestamp"])
                m15_trend_up = (
                    df_ref[df_ref["timestamp"] <= latest["timestamp"]].iloc[-1]["EMA9"]
                    > df_ref[df_ref["timestamp"] <= latest["timestamp"]].iloc[-1]["EMA21"]
                )
                m15_trend_down = not m15_trend_up
                trend = "UP" if m15_trend_up else "DOWN"

                EMA_trend = (
                    "=" if latest["EMA9"] == latest["EMA21"]
                    else ">" if latest["EMA9"] > latest["EMA21"]
                    else "<"
                )
                print(
                    f"{latest['timestamp']}: Close: {round(latest['close'], 2)}, EMA9 {EMA_trend} EMA21, RSI: {round(latest['RSI'], 2)}, Trend: {trend}"
                )

                # 多單
                if latest["EMA9"] > latest["EMA21"]:
                    if USE_RSI_FILTER and latest["RSI"] >= 70:
                        continue
                    if USE_TREND_FILTER and not m15_trend_up:
                        continue
                    entry_price = latest["close"]
                    if USE_STOP_TAKE_M15 and (m15_high <= entry_price or m15_low >= entry_price):
                        continue
                    stop_loss = m15_low if USE_STOP_TAKE_M15 else latest["low"]
                    take_profit = m15_high if USE_STOP_TAKE_M15 else latest["high"]
                    rr = (take_profit - entry_price) / (entry_price - stop_loss)
                    if USE_RR_FILTER and rr <= RR_THRESHOLD:
                        continue
                    position = (balance * LEVERAGE) / entry_price

                    # 強平價檢查
                    liquidation_price = calc_liquidation_price(entry_price, LEVERAGE, "LONG")
                    if stop_loss <= liquidation_price:
                        print(f"跳過: 止損({stop_loss}) <= 強平價({liquidation_price})")
                        continue

                    # 手續費檢查
                    sch_profit = abs(entry_price - take_profit) * position
                    open_fee = entry_price * abs(position) * FEE_RATE
                    close_fee = take_profit * abs(position) * FEE_RATE
                    total_fee = open_fee + close_fee
                    if sch_profit - total_fee <= 0:
                        continue

                    print(f"{latest['timestamp']}: 做多, price={entry_price}, TP={take_profit}, SL={stop_loss}, 預估盈虧={sch_profit:.2f}, 手續費={total_fee:.2f}, 強平價={liquidation_price:.2f}")

                # 空單
                elif latest["EMA9"] < latest["EMA21"]:
                    if USE_RSI_FILTER and latest["RSI"] <= 30:
                        continue
                    if USE_TREND_FILTER and not m15_trend_down:
                        continue
                    entry_price = latest["close"]
                    if USE_STOP_TAKE_M15 and (m15_low >= entry_price or m15_high <= entry_price):
                        continue
                    stop_loss = m15_high if USE_STOP_TAKE_M15 else latest["high"]
                    take_profit = m15_low if USE_STOP_TAKE_M15 else latest["low"]
                    rr = (entry_price - take_profit) / (stop_loss - entry_price)
                    if USE_RR_FILTER and rr <= RR_THRESHOLD:
                        continue
                    position = -(balance * LEVERAGE) / entry_price

                    # 強平價檢查
                    liquidation_price = calc_liquidation_price(entry_price, LEVERAGE, "SHORT")
                    if stop_loss >= liquidation_price:
                        print(f"跳過: 止損({stop_loss}) >= 強平價({liquidation_price})")
                        continue

                    # 手續費檢查
                    sch_profit = abs(entry_price - take_profit) * abs(position)
                    open_fee = entry_price * abs(position) * FEE_RATE
                    close_fee = take_profit * abs(position) * FEE_RATE
                    total_fee = open_fee + close_fee
                    if sch_profit - total_fee <= 0:
                        continue

                    print(f"{latest['timestamp']}: 做空, price={entry_price}, TP={take_profit}, SL={stop_loss}, 預估盈虧={sch_profit:.2f}, 手續費={total_fee:.2f}, 強平價={liquidation_price:.2f}")

            time.sleep(15)
        except Exception as e:
            print("錯誤:", e)
            time.sleep(15)


# ===== 執行實盤掃描 =====
pratical_scanner()

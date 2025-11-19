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
FEE_RATE = 0.0005  # 手續費率

USE_RSI_FILTER = False
USE_TREND_FILTER = True
USE_RR_FILTER = True
USE_STOP_TAKE_M15 = True

MIN_QTY = 0.001  # BTCUSDT Futures 最小單位
MIN_NOTIONAL = 5  # 最小名義價值 (USDT)

TELEGRAM_TOKEN = "你的TelegramBotToken"
CHAT_ID = "你的ChatID"


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


# ===== Telegram通知 =====
def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram通知錯誤: {e}")


def notify_close(
    signal_info, exit_price, position, entry_time, net_pnl, total_fee, reason
):
    direction = signal_info["signal"]
    entry_price = signal_info["entry_price"]
    qty = abs(position)
    message = (
        f"{direction} 訂單完成({reason})\n"
        f"進場時間: {entry_time}\n"
        f"進場價: {entry_price}\n"
        f"出場價: {exit_price}\n"
        f"倉位: {qty:.4f} BTC\n"
        f"淨盈虧: {net_pnl:.2f} USDT\n"
        f"總手續費: {total_fee:.2f} USDT"
    )
    send_telegram_message(message)

def notify_open(
    signal_info, exit_price, position, entry_time, net_pnl, total_fee, reason
):
    direction = signal_info["signal"]
    entry_price = signal_info["entry_price"]
    qty = abs(position)
    sch_profit = abs(signal_info["take_profit"] - signal_info["entry_price"])*qty
    sch_loss = abs(signal_info["stop_loss"] - signal_info["entry_price"])*qty*
        
    message = (
        f"🟢 {direction} 開倉 @ {entry_price}\n"
        f"進場時間: {entry_time}\n"
        f"倉位: {qty:.4f} BTC\n"
        f"進場價: {entry_price}\n"
        f"止盈價 (TP): {signal_info["take_profit"]:.2f}\n"
        f"止損價 (SL): {signal_info["stop_loss"]:.2f}\n"
        f"總手續費預估: {total_fee:.2f} USDT"
        f"預估損益: +{sch_profit} / -{sch_loss} USDT\n"
    )
    send_telegram_message(message)

# ===== 倉位調整 =====
def adjust_position_size(position, entry_price):
    adjusted = round(position // MIN_QTY * MIN_QTY, 3)
    if adjusted * entry_price < MIN_NOTIONAL:
        return 0
    return adjusted


# ===== 實盤掃描 =====
def pratical_scanner():
    print(f"\n[Scanner Detail]")
    position = 0
    entry_time = None
    signal_info = None

    while True:
        try:
            df_main = get_klines(SYMBOL, INTERVAL_MAIN, LIMIT)
            df_ref = get_klines(SYMBOL, INTERVAL_REF, LIMIT)
            df_main = calculate_indicators(df_main)
            df_ref = calculate_indicators(df_ref)

            latest = df_main.iloc[-1]
            if position == 0:
                signal_info = generate_signal(latest, df_ref)
                if signal_info:
                    entry_time = latest["timestamp"]
                    position = (INITIAL_BALANCE * LEVERAGE) / signal_info["entry_price"]
                    position = adjust_position_size(
                        position, signal_info["entry_price"]
                    )
                    print(
                        f"{entry_time}: {signal_info['signal']} 訊號, 進場 {signal_info['entry_price']}, TP {signal_info['take_profit']}, SL {signal_info['stop_loss']}, RR {round(signal_info['rr'], 2)}"
                    )

            # 平倉檢查
            if position != 0 and signal_info:
                df_check = get_klines(SYMBOL, INTERVAL_MAIN, 1)
                current_price = df_check.iloc[-1]["close"]

                if (
                    position > 0
                    and (
                        current_price >= signal_info["take_profit"]
                        or current_price <= signal_info["stop_loss"]
                    )
                ) or (
                    position < 0
                    and (
                        current_price <= signal_info["take_profit"]
                        or current_price >= signal_info["stop_loss"]
                    )
                ):
                    pnl = (
                        (current_price - signal_info["entry_price"]) * position
                        if position > 0
                        else (signal_info["entry_price"] - current_price)
                        * abs(position)
                    )
                    open_fee = signal_info["entry_price"] * abs(position) * FEE_RATE
                    close_fee = current_price * abs(position) * FEE_RATE
                    total_fee = open_fee + close_fee
                    net_pnl = pnl - total_fee

                    reason = (
                        "止盈"
                        if (
                            (
                                position > 0
                                and current_price >= signal_info["take_profit"]
                            )
                            or (
                                position < 0
                                and current_price <= signal_info["take_profit"]
                            )
                        )
                        else "止損"
                    )

                    print(
                        f"交易完成: 出場價={current_price}, 盈虧={net_pnl:.2f} USDT (手續費={total_fee:.2f})"
                    )
                    notify_close(
                        signal_info,
                        current_price,
                        position,
                        entry_time,
                        net_pnl,
                        total_fee,
                        reason,
                    )

                    position = 0
                    signal_info = None

            time.sleep(15)
        except Exception as e:
            print("錯誤:", e)
            time.sleep(15)


# ===== 執行實盤 =====
pratical_scanner()

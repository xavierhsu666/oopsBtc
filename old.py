import requests
import pandas as pd
import numpy as np
import time
from typing import Tuple, Optional, Literal, Dict, Any, List
import atexit

# ===== 模式與回測參數控制 (與之前相同) =====
MODE = "BACKTEST"  # 調整為 BACKTEST 或 LIVE
K_BAR_COUNT = 800  # 回測專用參數

# ===== 交易與連接參數 (與之前相同) =====
BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "BTCUSDT"
INTERVAL_MAIN = "5m"
INTERVAL_REF = "15m"
LIMIT = 500

INITIAL_BALANCE = 100
LEVERAGE = 15
RR_THRESHOLD = 1.2
FEE_RATE = 0.0005

# ===== 策略濾網參數 =====
USE_RSI_FILTER = True
USE_TREND_FILTER = True
USE_RR_FILTER = True
# 新增: 控制是否使用 M15 反轉 K 線極值作為 TP/SL
USE_REVERSAL_LEVELS = True
# 舊參數: 僅在 USE_REVERSAL_LEVELS = False 時，使用 M15 K 線的高低點
USE_STOP_TAKE_M15 = False


MIN_QTY = 0.001
MIN_NOTIONAL = 5

TELEGRAM_TOKEN = "8311467265:AAHRI8fd7xHgx4HZH4FEBQ78vCx9wwsc6w0"
CHAT_ID = "1188811502"


# [以下輔助函數 get_klines, calculate_indicators, calc_liquidation_price, adjust_position_size, send_telegram_message 保持不變]
# 為了精簡，這裡省略重複的輔助函數程式碼，但它們在您的實際檔案中必須存在。
# --- START OF OMITTED HELPER FUNCTIONS ---


def get_klines(
    symbol: str, interval: str, limit: int, start_time: Optional[int] = None
) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    url = f"{BASE_URL}"
    data = requests.get(url, params=params).json()
    if (
        not isinstance(data, list)
        or len(data) == 0
        or (isinstance(data[0], dict) and "code" in data[0])
    ):
        if MODE == "LIVE":
            print(f"Error fetching klines for {symbol}@{interval}: {data}")
        return pd.DataFrame()
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
    return df.drop(
        columns=[
            "close_time",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
        errors="ignore",
    )


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
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


def get_last_m15_levels(m15_df: pd.DataFrame, current_time) -> Tuple[float, float]:
    ref = m15_df[m15_df["timestamp"] <= current_time]
    if ref.empty:
        return m15_df.iloc[0]["high"], m15_df.iloc[0]["low"]
    return ref.iloc[-1]["high"], ref.iloc[-1]["low"]


def calc_liquidation_price(
    entry_price: float, leverage: int, side: Literal["LONG", "SHORT"]
) -> float:
    if side == "LONG":
        return entry_price * (1 - 1 / leverage)
    else:
        return entry_price * (1 + 1 / leverage)


def adjust_position_size(position: float, entry_price: float) -> float:
    abs_position = abs(position)
    adjusted_abs = round(abs_position // MIN_QTY * MIN_QTY, 3)
    if adjusted_abs * entry_price < MIN_NOTIONAL:
        return 0.0
    return adjusted_abs if position >= 0 else -adjusted_abs


def send_telegram_message(message: str):
    if MODE != "LIVE":
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram通知錯誤: {e}")


# [monitor_and_close_order, pratical_scanner, calculate_metrics, backtester 需在 check_signals 之後定義]

# --- END OF OMITTED HELPER FUNCTIONS ---


# ===== 核心優化：尋找 M15 反轉 K 線的極值 =====


def get_m15_reversal_levels(
    m15_df: pd.DataFrame, current_time, side: Literal["LONG", "SHORT"]
) -> Tuple[Optional[float], Optional[float]]:
    """
    尋找 M15 級別上，最近一次 EMA 交叉的反轉 K 線的極值 (High/Low)。

    對於 LONG (多單): 尋找最近一次「空頭趨勢結束，轉為多頭」的反轉 K 棒。
    對於 SHORT (空單): 尋找最近一次「多頭趨勢結束，轉為空頭」的反轉 K 棒。

    返回: (Stop_Loss_Level, Take_Profit_Level)
    """

    # 確保 df 是按時間排序的，並且包含指標
    df = m15_df[m15_df["timestamp"] < current_time].copy()  # 僅使用當前時間之前的數據

    if df.shape[0] < 2:
        return None, None  # 數據不足

    # 1. 判斷趨勢方向
    # 趨勢向上: EMA9 > EMA21
    df["Trend_Up"] = df["EMA9"] > df["EMA21"]

    target_reversal_k = None

    if side == "LONG":
        # 尋找「空轉多」的反轉 K 線 (前一根 Trend_Up=False, 當前 Trend_Up=True)
        # 這裡從倒數第二根開始迭代，因為最新一根(i-1)可能正在形成或已經用在趨勢判斷
        for i in range(len(df) - 1, 0, -1):
            current_k = df.iloc[i]
            prev_k = df.iloc[i - 1]

            # 判斷是否為「空轉多」的交叉 K 線
            if current_k["Trend_Up"] and not prev_k["Trend_Up"]:
                # 找到反轉K線: 這是反轉K線本身或其後續K線
                # 這裡定義反轉點為趨勢正式確認的 K 線
                target_reversal_k = current_k
                break

        if target_reversal_k is not None:
            # 多單: SL 設在反轉 K 線的 Low，TP 設在反轉 K 線的 High
            return target_reversal_k["low"], target_reversal_k["high"]

    elif side == "SHORT":
        # 尋找「多轉空」的反轉 K 線 (前一根 Trend_Up=True, 當前 Trend_Up=False)
        for i in range(len(df) - 1, 0, -1):
            current_k = df.iloc[i]
            prev_k = df.iloc[i - 1]

            # 判斷是否為「多轉空」的交叉 K 線
            if not current_k["Trend_Up"] and prev_k["Trend_Up"]:
                # 找到反轉K線: 這是反轉K線本身或其後續K線
                target_reversal_k = current_k
                break

        if target_reversal_k is not None:
            # 空單: SL 設在反轉 K 線的 High，TP 設在反轉 K 線的 Low
            return target_reversal_k["high"], target_reversal_k["low"]

    # 如果沒找到反轉 K 線，則返回 None
    return None, None


# ===== 核心交易邏輯 (check_signals) - 納入反轉 K 線邏輯 =====


def check_signals(
    latest: pd.Series, df_ref: pd.DataFrame, current_balance: float
) -> Optional[Dict[str, Any]]:
    """檢查指標，判斷是否有開倉信號。"""
    entry_price = latest["close"]

    # 判斷 M15 趨勢 (用於趨勢濾網)
    m15_trend_up = (
        df_ref[df_ref["timestamp"] <= latest["timestamp"]].iloc[-1]["EMA9"]
        > df_ref[df_ref["timestamp"] <= latest["timestamp"]].iloc[-1][
            "EMA21"
        ]
    )
    m15_trend_down = not m15_trend_up

    side: Optional[Literal["LONG", "SHORT"]] = None
    if latest["EMA9"] > latest["EMA21"]:
        side = "LONG"
    elif latest["EMA9"] < latest["EMA21"]:
        side = "SHORT"
    if side is None:
        return None

    # --- 篩選與止盈止損價設定 ---

    stop_loss, take_profit = 0.0, 0.0

    # 邏輯 1: 使用 M15 反轉 K 線極值 (新邏輯)
    if USE_REVERSAL_LEVELS:
        # 獲取 M15 反轉 K 線的極值
        reversal_sl, reversal_tp = get_m15_reversal_levels(
            df_ref, latest["timestamp"], side
        )

        if reversal_sl is None or reversal_tp is None:
            if MODE == "LIVE":
                print("跳過: M15 反轉 K 線數據不足或未找到。")
            return None  # 找不到反轉 K 線則跳過

        stop_loss = reversal_sl
        take_profit = reversal_tp

        # 額外檢查: 反轉 K 線的 SL/TP 必須與進場價有合理的距離
        if side == "LONG" and (entry_price <= stop_loss or entry_price >= take_profit):
            if MODE == "LIVE":
                print("跳過: LONG 進場價不在反轉 K 線極值之間。")
            return None
        if side == "SHORT" and (entry_price >= stop_loss or entry_price <= take_profit):
            if MODE == "LIVE":
                print("跳過: SHORT 進場價不在反轉 K 線極值之間。")
            return None

    # 邏輯 2: 使用 M15 K 線的高低點 (舊邏輯)
    elif USE_STOP_TAKE_M15:
        m15_high, m15_low = get_last_m15_levels(df_ref, latest["timestamp"])
        if side == "LONG":
            stop_loss, take_profit = m15_low, m15_high
            if m15_high <= entry_price or m15_low >= entry_price:
                return None
        elif side == "SHORT":
            stop_loss, take_profit = m15_high, m15_low
            if m15_low >= entry_price or m15_high <= entry_price:
                return None

    # 邏輯 3: 使用 M5 K 線的高低點 (最原始邏輯)
    else:
        if side == "LONG":
            stop_loss, take_profit = latest["low"], latest["high"]
        elif side == "SHORT":
            stop_loss, take_profit = latest["high"], latest["low"]

    # --- 共同濾網 (RSI, 趨勢, RR, 倉位, 強平, 手續費) ---

    if side == "LONG":
        if USE_RSI_FILTER and latest["RSI"] >= 70:
            return None
        if USE_TREND_FILTER and not m15_trend_up:
            return None

        if entry_price - stop_loss <= 0:
            return None
        rr = (take_profit - entry_price) / (entry_price - stop_loss)
        if USE_RR_FILTER and rr <= RR_THRESHOLD:
            return None

        position = adjust_position_size(
            (current_balance * LEVERAGE) / entry_price, entry_price
        )
        if position == 0:
            return None

        liquidation_price = calc_liquidation_price(entry_price, LEVERAGE, "LONG")
        if stop_loss <= liquidation_price:
            if MODE == "LIVE":
                print(
                    f"跳過 (LONG): 止損({stop_loss:.2f}) <= 強平價({liquidation_price:.2f})"
                )
            return None

    elif side == "SHORT":
        if USE_RSI_FILTER and latest["RSI"] <= 30:
            return None
        if USE_TREND_FILTER and not m15_trend_down:
            return None

        if stop_loss - entry_price <= 0:
            return None
        rr = (entry_price - take_profit) / (stop_loss - entry_price)
        if USE_RR_FILTER and rr <= RR_THRESHOLD:
            return None

        position = adjust_position_size(
            -(current_balance * LEVERAGE) / entry_price, entry_price
        )
        if position == 0:
            return None

        liquidation_price = calc_liquidation_price(entry_price, LEVERAGE, "SHORT")
        if stop_loss >= liquidation_price:
            if MODE == "LIVE":
                print(
                    f"跳過 (SHORT): 止損({stop_loss:.2f}) >= 強平價({liquidation_price:.2f})"
                )
            return None

    # 計算預估手續費
    sch_profit = abs(entry_price - take_profit) * abs(position)
    open_fee = entry_price * abs(position) * FEE_RATE
    close_fee = take_profit * abs(position) * FEE_RATE
    total_fee = open_fee + close_fee

    if sch_profit - total_fee <= 0:
        return None

    return {
        "timestamp": latest["timestamp"],
        "side": side,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "position": position,
        "total_fee": total_fee,
        "liquidation_price": liquidation_price,
        "sch_profit": sch_profit,
    }


# [以下為 monitor_and_close_order, pratical_scanner, calculate_metrics, backtester 程式碼，請確保這些函數與您上一個程式碼版本保持一致]
# --- START OF OMITTED MAIN FUNCTIONS ---


def monitor_and_close_order(
    order_info: Dict[str, Any], interval_main: str, symbol: str
) -> float:
    entry_price, position, take_profit, stop_loss, side = (
        order_info["entry_price"],
        order_info["position"],
        order_info["take_profit"],
        order_info["stop_loss"],
        order_info["side"],
    )
    print(f"\n--- 開始監控 {side} 訂單 ---")
    while True:
        df_check = get_klines(symbol, interval_main, 1)
        if df_check.empty:
            time.sleep(15)
            continue
        current_price = df_check.iloc[-1]["close"]
        is_close_signal = (
            side == "LONG"
            and (current_price >= take_profit or current_price <= stop_loss)
        ) or (
            side == "SHORT"
            and (current_price <= take_profit or current_price >= stop_loss)
        )
        if is_close_signal:
            pnl = (
                (current_price - entry_price) * position
                if position > 0
                else (entry_price - current_price) * abs(position)
            )
            open_fee = entry_price * abs(position) * FEE_RATE
            close_fee = current_price * abs(position) * FEE_RATE
            total_fee = open_fee + close_fee
            net_pnl = pnl - total_fee
            close_type = "止盈" if pnl > 0 else ("止損" if pnl < 0 else "平手")
            print(
                f"交易完成 ({close_type}): 出場價={current_price:.2f}, 盈虧={net_pnl:.2f} USDT (手續費={total_fee:.2f})"
            )
            message = f"🚨 **{side} 訂單完成 ({close_type})** 🚨\n進場時間: {order_info['timestamp']}\n進場價: {entry_price:.2f}\n出場價: {current_price:.2f}\n倉位: {abs(position):.3f} BTC\n🚀 **淨盈虧: {net_pnl:.2f} USDT**\n總手續費: {total_fee:.2f} USDT"
            send_telegram_message(message)
            return net_pnl
        time.sleep(15)


def pratical_scanner():
    print(f"💰 初始資金: {INITIAL_BALANCE} USDT")
    print("-" * 30)
    balance, position = INITIAL_BALANCE, 0.0
    while True:
        try:
            df_main = calculate_indicators(get_klines(SYMBOL, INTERVAL_MAIN, LIMIT))
            df_ref = calculate_indicators(get_klines(SYMBOL, INTERVAL_REF, LIMIT))
            if df_main.empty or df_ref.empty:
                time.sleep(15)
                continue
            latest = df_main.iloc[-1]

            if position == 0:
                signal = check_signals(latest, df_ref, balance)
                trend_str = (
                    "UP"
                    if latest["EMA9"] > latest["EMA21"]
                    else ("DOWN" if latest["EMA9"] < latest["EMA21"] else "SIDE")
                )
                EMA_trend_str = (
                    "="
                    if latest["EMA9"] == latest["EMA21"]
                    else ">" if latest["EMA9"] > latest["EMA21"] else "<"
                )
                print(
                    f"{latest['timestamp']}: Close: {round(latest['close'], 2)}, EMA9 {EMA_trend_str} EMA21, RSI: {round(latest['RSI'], 2)}, Trend: {trend_str}"
                )

                if signal:
                    position, side = signal["position"], signal["side"]
                    print(
                        f"\n🎉 **發現 {side} 信號!** - Price: {signal['entry_price']:.2f}, Qty: {abs(position):.3f} BTC"
                    )
                    print(
                        f"TP: {signal['take_profit']:.2f}, SL: {signal['stop_loss']:.2f}"
                    )
                    print(
                        f"預估淨利: {signal['sch_profit'] - signal['total_fee']:.2f} USDT, 強平價: {signal['liquidation_price']:.2f}"
                    )
                    tele_message = f"🟢 **{side} 開倉** @ {signal['entry_price']:.2f}\n倉位: {abs(position):.3f} BTC\n目標價 (TP): {signal['take_profit']:.2f}\n止損價 (SL): {signal['stop_loss']:.2f}\n槓桿: {LEVERAGE}x"
                    send_telegram_message(tele_message)
                    net_pnl = monitor_and_close_order(signal, INTERVAL_MAIN, SYMBOL)
                    balance += net_pnl
                    position = 0.0
                    print(f"\n--- 資金更新: {balance:.2f} USDT ---\n")
                else:
                    time.sleep(0.3)
                    continue
            else:
                time.sleep(15)
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            time.sleep(15)


def calculate_metrics(
    trades: List[Dict[str, Any]], initial_balance: float
) -> Dict[str, Any]:
    if not trades:
        return {
            "總交易次數": 0,
            "淨盈虧 (USDT)": 0.0,
            "最終餘額 (USDT)": initial_balance,
            "總體報酬率 (%)": 0.0,
            "勝率 (%)": 0.0,
            "盈虧比 (R)": 0.0,
            "最大回撤 (%)": 0.0,
        }
    df_trades = pd.DataFrame(trades)
    df_trades["net_pnl_acc"] = df_trades["net_pnl"].cumsum()
    df_trades["equity"] = initial_balance + df_trades["net_pnl_acc"]
    total_trades, winning_trades, losing_trades = (
        len(df_trades),
        len(df_trades[df_trades["net_pnl"] > 0]),
        len(df_trades[df_trades["net_pnl"] < 0]),
    )
    total_pnl = df_trades["net_pnl"].sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    avg_win, avg_loss = (
        df_trades[df_trades["net_pnl"] > 0]["net_pnl"].mean(),
        df_trades[df_trades["net_pnl"] < 0]["net_pnl"].mean(),
    )
    risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss else 0.0
    peak = df_trades["equity"].cummax()
    drawdown = (peak - df_trades["equity"]) / peak
    mdd = drawdown.max()
    return {
        "總交易次數": total_trades,
        "獲利次數": winning_trades,
        "虧損次數": losing_trades,
        "淨盈虧 (USDT)": round(total_pnl, 2),
        "最終餘額 (USDT)": round(initial_balance + total_pnl, 2),
        "總體報酬率 (%)": round(total_pnl / initial_balance * 100, 2),
        "勝率 (%)": round(win_rate * 100, 2),
        "平均獲利 (USDT)": round(avg_win, 2) if not pd.isna(avg_win) else 0.0,
        "平均虧損 (USDT)": round(avg_loss, 2) if not pd.isna(avg_loss) else 0.0,
        "盈虧比 (R)": round(risk_reward_ratio, 2),
        "最大回撤 (%)": round(mdd * 100, 2),
    }


def backtester(k_limit: int):
    print(f"📈 **開始回測**：使用最近 {k_limit} 根 {INTERVAL_MAIN} K 線數據")
    print("-" * 40)
    buffer = 21
    df_main = calculate_indicators(get_klines(SYMBOL, INTERVAL_MAIN, k_limit + buffer))
    df_ref = calculate_indicators(get_klines(SYMBOL, INTERVAL_REF, k_limit + buffer))
    if df_main.empty or df_ref.empty:
        print("❌ 數據獲取失敗或不足，無法進行回測。")
        return
    df_main_slice = df_main.iloc[buffer:].reset_index(drop=True)
    current_balance, current_position, current_order, trades_log = (
        INITIAL_BALANCE,
        0.0,
        None,
        [],
    )
    print(
        f"🌐 回測區間: {df_main_slice.iloc[0]['timestamp']} ~ {df_main_slice.iloc[-1]['timestamp']}"
    )
    print(f"💰 初始資金: {current_balance:.2f} USDT")
    print("-" * 40)
    for i in range(len(df_main_slice)):
        latest_k = df_main_slice.iloc[i]
        if current_position != 0.0:
            k_high, k_low = latest_k["high"], latest_k["low"]
            entry_price, stop_loss, take_profit, position, side = (
                current_order["entry_price"],
                current_order["stop_loss"],
                current_order["take_profit"],
                current_order["position"],
                current_order["side"],
            )
            close_price, triggered = latest_k["close"], False
            if side == "LONG":
                if k_low <= stop_loss:
                    close_price, triggered = stop_loss, True
                elif k_high >= take_profit:
                    close_price, triggered = take_profit, True
            elif side == "SHORT":
                if k_high >= stop_loss:
                    close_price, triggered = stop_loss, True
                elif k_low <= take_profit:
                    close_price, triggered = take_profit, True
            liquidation_price = current_order["liquidation_price"]
            if (side == "LONG" and k_low <= liquidation_price) or (
                side == "SHORT" and k_high >= liquidation_price
            ):
                close_price, triggered = liquidation_price, True
                print(f"‼️ 強平發生! {latest_k['timestamp']}")
            if triggered:
                pnl = (
                    (close_price - entry_price) * position
                    if position > 0
                    else (entry_price - close_price) * abs(position)
                )
                open_fee = entry_price * abs(position) * FEE_RATE
                close_fee = close_price * abs(position) * FEE_RATE
                net_pnl = pnl - (open_fee + close_fee)
                current_balance += net_pnl
                trades_log.append(
                    {
                        "timestamp_in": current_order["timestamp"],
                        "timestamp_out": latest_k["timestamp"],
                        "side": side,
                        "entry_price": entry_price,
                        "close_price": close_price,
                        "net_pnl": net_pnl,
                        "total_fee": open_fee + close_fee,
                        "balance": current_balance,
                    }
                )
                print(
                    f"[平倉] {latest_k['timestamp']} | {side} @ {entry_price:.2f} -> {close_price:.2f} | 盈虧: {net_pnl:+.2f} | 餘額: {current_balance:.2f}"
                )
                current_position, current_order = 0.0, None
        if current_position == 0.0:
            signal = check_signals(latest_k, df_ref, current_balance)
            if signal:
                current_order, current_position = signal, signal["position"]
                print(
                    f"[開倉] {latest_k['timestamp']} | {signal['side']} @ {signal['entry_price']:.2f} | TP: {signal['take_profit']:.2f}, SL: {signal['stop_loss']:.2f}"
                )
    metrics = calculate_metrics(trades_log, INITIAL_BALANCE)
    print("\n" + "=" * 40)
    print("🏆 **回測績效報告**")
    print("=" * 40)
    results_table = {
        "指標": [
            "總交易次數",
            "最終餘額 (USDT)",
            "總體報酬率 (%)",
            "淨盈虧 (USDT)",
            "勝率 (%)",
            "盈虧比 (R)",
            "最大回撤 (%)",
        ],
        "數值": [
            metrics["總交易次數"],
            metrics["最終餘額 (USDT)"],
            metrics["總體報酬率 (%)"],
            metrics["淨盈虧 (USDT)"],
            metrics["勝率 (%)"],
            metrics["盈虧比 (R)"],
            metrics["最大回撤 (%)"],
        ],
    }
    print(pd.DataFrame(results_table).to_markdown(index=False))
    print("-" * 40)
    print(f"✅ 回測結束。總交易次數: {metrics['總交易次數']}")


# --- END OF OMITTED MAIN FUNCTIONS ---


# ===== 執行區塊 (切換模式) =====
# 註冊退出函數 (不需傳遞參數)
def exit_handler():
    # 當程式退出時，執行您要的清理和通知動作
    send_telegram_message("✅ 程式已通過 atexit 順利結束運行。")
    print("atexit handler 執行完畢。")


if __name__ == "__main__":
    if MODE == "LIVE":
        try:
            pratical_scanner()
        except Exception as e:
            print(f"回測執行發生錯誤: {e}")
        finally:
            send_telegram_message("✅ 程式已通過 atexit 順利結束運行。")
            # atexit 會在 KeyboardInterrupt 後執行
            print("\n收到 Ctrl+C，等待 atexit 執行...")

    elif MODE == "BACKTEST":
        try:
            backtester(K_BAR_COUNT)
        except Exception as e:
            print(f"回測執行發生錯誤: {e}")

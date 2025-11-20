import requests
import pandas as pd
import numpy as np
import time
import datetime as dt

# ===== 可調參數 =====
BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "BTCUSDT"
INTERVAL_MAIN = "5m"
INTERVAL_REF = "15m"
LIMIT = 100
BACKTEST_LIMIT = 1000

INITIAL_BALANCE = 100  # 初始資金
LEVERAGE = 15  # 槓桿倍數
RR_THRESHOLD = 1.5  # RR 條件

USE_RSI_FILTER = True  # 是否啟用 RSI 過濾
USE_TREND_FILTER = True  # 是否啟用順勢條件
USE_RR_FILTER = True  # 是否啟用 RR 條件
USE_STOP_TAKE_M15 = True  # 是否使用 M15 高低作為止盈止損
USE_SL_OPTIMIZER = True  # 是否根據ATR使用SL優化器
SL_OPTIMIZER_THRESHOLD = 0.3  # ATR - SL優化器 TH
TAKER_FEE_RATE = 0.0005  # 吃單方(市價單)
MAKER_FEE_RATE = 0.0002  # 掛單方(限價單)
MIN_PROFIT = 0.5
MAX_LOSS = INITIAL_BALANCE * 0.02

MIN_QTY = 0.001  # BTCUSDT Futures 最小單位
MIN_NOTIONAL = 5  # 最小名義價值 (USDT)

TELEGRAM_TOKEN = "8311467265:AAHRI8fd7xHgx4HZH4FEBQ78vCx9wwsc6w0"
CHAT_ID = "1188811502"


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
    """計算技術指標。

    Args:
        df: 包含 OHLC 資料的 DataFrame。

    Returns:
        DataFrame: 新增指標欄位後的 DataFrame。
    """
    df["EMA9"] = df["close"].ewm(span=9).mean()
    df["EMA21"] = df["close"].ewm(span=21).mean()

    # RSI 計算
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # ATR (Average True Range) 計算
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=14).mean()

    return df


# ===== 找最近一根 M15 的高低 =====
def get_last_m15_levels(m15_df, current_time):
    ref = m15_df[m15_df["timestamp"] <= current_time].iloc[-1]
    return ref["high"], ref["low"]


# ===== 產生交易訊號 =====
def generate_signal(df_main, df_ref, balance):
    latest = df_main.iloc[-1]
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    m15_high, m15_low = get_last_m15_levels(df_ref, latest["timestamp"])
    m15_trend_up = (
        df_ref[df_ref["timestamp"] <= latest["timestamp"]].iloc[-1]["EMA9"]
        > df_ref[df_ref["timestamp"] <= latest["timestamp"]].iloc[-1]["EMA21"]
    )
    m15_trend_down = not m15_trend_up

    m5_trend_up = (
        df_main[df_main["timestamp"] <= latest["timestamp"]].iloc[-1]["EMA9"]
        > df_main[df_main["timestamp"] <= latest["timestamp"]].iloc[-1]["EMA21"]
    )
    m5_trend_down = not m5_trend_up

    entry_price = latest["close"]
    position = (balance * LEVERAGE) / entry_price
    position = adjust_position_size(position, entry_price)

    # 多單
    if latest["EMA9"] > latest["EMA21"]:
        if USE_RSI_FILTER and latest["RSI"] >= 70:
            print(f"LONG RSI 過濾>=70: {latest['RSI']} 不做交易")
            return None
        if USE_TREND_FILTER and (m15_trend_down or m5_trend_down):
            print(
                f"LONG Trend Filter 順勢單: M5Down={m5_trend_down}, M15Down={m15_trend_down} 不做交易"
            )
            return None
        # print(
        #     USE_STOP_TAKE_M15 and (m15_high <= entry_price or m15_low >= entry_price),
        #     m15_high,
        #     entry_price,
        #     m15_low,
        # )
        if USE_STOP_TAKE_M15 and (m15_high <= entry_price or m15_low >= entry_price):
            print(
                f"LONG USE_STOP_TAKE_M15 價格不再區間內 low-entry-high: {m15_low}-{entry_price}-{m15_high} 不做交易"
            )
            return None
        stop_loss = m15_low if USE_STOP_TAKE_M15 else latest["low"]
        take_profit = m15_high if USE_STOP_TAKE_M15 else latest["high"]
        rr = (take_profit - entry_price) / (entry_price - stop_loss)

        # 根據ATR 優化stop_loss, take_profit設置ex 先判斷開倉價gap
        if USE_SL_OPTIMIZER:
            stop_loss = (
                stop_loss
                if abs(stop_loss - entry_price)
                <= latest["ATR"] * SL_OPTIMIZER_THRESHOLD
                else stop_loss - latest["ATR"] * SL_OPTIMIZER_THRESHOLD
            )
            rr = (take_profit - entry_price) / (entry_price - stop_loss)

        if USE_RR_FILTER and rr <= RR_THRESHOLD:
            print(f"LONG RR 過濾<={RR_THRESHOLD}: {rr} 不做交易")
            return None

        open_fee = (entry_price * abs(position)) * TAKER_FEE_RATE
        profit_close_fee = (take_profit * abs(position)) * MAKER_FEE_RATE
        loss_close_fee = (stop_loss * abs(position)) * MAKER_FEE_RATE
        sch_loss = (abs(stop_loss - entry_price) * position) * -1
        sch_profit = abs(take_profit - entry_price) * position
        if (sch_profit - profit_close_fee - open_fee) <= MIN_PROFIT:
            print(
                f"LONG MIN_PROFIT<={MIN_PROFIT}: {sch_profit - profit_close_fee - open_fee} 不做交易"
            )
            return None
        if abs(sch_loss - loss_close_fee - open_fee) <= MAX_LOSS:
            print(
                f"LONG MAX_LOSS<={MAX_LOSS}: {sch_loss - loss_close_fee - open_fee} 不做交易"
            )
            return None
        # print(1)
        return {
            "signal": "LONG",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "sch_loss": sch_loss,
            "sch_profit": sch_profit,
            "rr": rr,
            "open_fee": open_fee,
            "profit_close_fee": profit_close_fee,
            "loss_close_fee": loss_close_fee,
            "entry_time": now,
            "position": position,
        }

    # 空單
    elif latest["EMA9"] < latest["EMA21"]:
        if USE_RSI_FILTER and latest["RSI"] <= 30:
            print(f"SHORT RSI 過濾<=30: {latest['RSI']} 不做交易")
            return None
        if USE_TREND_FILTER and (m15_trend_up or m5_trend_up):
            print(
                f"SHORT Trend Filter 順勢單: M5UP={m15_trend_up}, M15UP={m5_trend_up} 不做交易"
            )
            return None
        entry_price = latest["close"]
        if USE_STOP_TAKE_M15 and (m15_low >= entry_price or m15_high <= entry_price):
            print(
                f"SHORT USE_STOP_TAKE_M15 價格不再區間內 low-entry-high: {m15_low}-{entry_price}-{m15_high} 不做交易"
            )
            return None
        stop_loss = m15_high if USE_STOP_TAKE_M15 else latest["high"]
        take_profit = m15_low if USE_STOP_TAKE_M15 else latest["low"]
        rr = (entry_price - take_profit) / (stop_loss - entry_price)
        # 根據ATR 優化stop_loss, take_profit設置ex 先判斷開倉價gap
        if USE_SL_OPTIMIZER:
            stop_loss = (
                stop_loss
                if abs(stop_loss - entry_price)
                <= latest["ATR"] * SL_OPTIMIZER_THRESHOLD
                else stop_loss + latest["ATR"] * SL_OPTIMIZER_THRESHOLD
            )
            rr = (entry_price - take_profit) / (stop_loss - entry_price)

        open_fee = (entry_price * abs(position)) * TAKER_FEE_RATE
        profit_close_fee = (take_profit * abs(position)) * MAKER_FEE_RATE
        loss_close_fee = (stop_loss * abs(position)) * MAKER_FEE_RATE
        sch_loss = (abs(stop_loss - entry_price) * position) * -1
        sch_profit = abs(take_profit - entry_price) * position

        if USE_RR_FILTER and rr <= RR_THRESHOLD:
            print(f"SHORT RR 過濾<={RR_THRESHOLD}: {rr} 不做交易")
            return None
        if (sch_profit - profit_close_fee - open_fee) <= MIN_PROFIT:
            print(
                f"SHORT MIN_PROFIT<={MIN_PROFIT}: {sch_profit - profit_close_fee - open_fee} 不做交易"
            )
            return None
        if abs(sch_loss - loss_close_fee - open_fee) <= MAX_LOSS:
            print(
                f"SHORT MAX_LOSS<={MAX_LOSS}: {sch_loss - loss_close_fee - open_fee} 不做交易"
            )
            return None
        return {
            "signal": "SHORT",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "sch_loss": sch_loss,
            "sch_profit": sch_profit,
            "rr": rr,
            "open_fee": open_fee,
            "profit_close_fee": profit_close_fee,
            "loss_close_fee": loss_close_fee,
            "entry_time": now,
            "position": position,
        }

    return None


def send_telegram_message(message: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram通知錯誤: {e}")


def notify_close(signal_info, balance):
    ret = "止盈" if signal_info["pnl"] > 0 else "止損"
    ret_ico = "💰" if signal_info["pnl"] > 0 else "💸"
    message = (
        f"{ret_ico} {signal_info['signal']} 平倉({ret}) 損益:{signal_info['net_pnl']:.2f} USDT \n"
        f"平倉時間: {signal_info['close_time']}\n"
        f"倉位: {signal_info['position']:.4f} BTC\n"
        f"開倉價: {signal_info['entry_price']}\n"
        f"平倉價: {signal_info['close_price']}\n"
        f"損益: {signal_info['pnl']:.2f} USDT\n"
        f"總手續費預估: {signal_info['total_fee']:.2f} USDT\n"
        f"淨損益: {signal_info['net_pnl']:.2f} USDT\n"
        f"\n"
        f"帳戶餘額: {balance:.2f}  USDT\n"
    )
    send_telegram_message(message)


def notify_open(signal_info):
    ret = "🚀" if signal_info["signal"] == "LONG" else "🛝"
    message = (
        f"{ret} {signal_info['signal']} 開倉 @ {signal_info['entry_price']}\n"
        f"進場時間: {signal_info['entry_time']}\n"
        f"進場價: {signal_info['entry_price']}\n"
        f"保證金金額: {(signal_info['position'] * signal_info['entry_price']) / LEVERAGE:.2f}\n"
        f"槓桿: {LEVERAGE:.2f}\n"
        f"倉位: {signal_info['position']:.4f} BTC\n"
        f"\n"
        f"TP/SL: {signal_info['take_profit']:.2f} / {signal_info['stop_loss']:.2f}\n"
        f"PNL: {signal_info['sch_profit']:.2f} / {signal_info['sch_loss']:.2f}\n"
        f"TP手續費預估: {signal_info['open_fee'] + signal_info['profit_close_fee']:.2f} USDT\n"  # 注意：原代碼這裡沒有 \n，我建議加上以確保下一行資訊完整
        f"SL手續費預估: {signal_info['open_fee'] + signal_info['loss_close_fee']:.2f} USDT\n"  # 注意：原代碼這裡沒有 \n，我建議加上以確保下一行資訊完整
    )
    send_telegram_message(message)


def notify_startup():
    # 假設所有變數已定義
    rsi_status = "✅ 啟用" if USE_RSI_FILTER else "❌ 關閉"
    trend_status = "✅ 啟用" if USE_TREND_FILTER else "❌ 關閉"
    rr_threshold_display = f"RR > {RR_THRESHOLD}"
    rr_status = f"✅ 啟用 ({rr_threshold_display})" if USE_RR_FILTER else "❌ 關閉"
    stop_take_logic = "M15 高低點" if USE_STOP_TAKE_M15 else "M5 K線高低點"
    current_time_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ⚠️ 修正: 使用 f-string 並採用 HTML 標籤
    message = (
        f"🚀 <b>Opps Trader Bot 啟動通知</b> 🤖\n\n"
        f"--- <b>核心配置</b> ---\n"
        f"* <b>交易對:</b> <code>{SYMBOL}</code>\n"
        f"* <b>主週期:</b> <code>{INTERVAL_MAIN}</code>\n"
        f"* <b>參考週期:</b> <code>{INTERVAL_REF}</code>\n"
        f"* <b>初始資金:</b> <code>{INITIAL_BALANCE:.2f} USDT</code>\n"
        f"* <b>槓桿倍數:</b> <code>{LEVERAGE}x</code>\n\n"
        f"--- <b>策略濾網</b> ---\n"
        f"* <b>RSI 過濾:</b> {rsi_status}\n"
        f"* <b>順勢過濾 (M15):</b> {trend_status}\n"
        f"* <b>風報比:</b> {rr_status}\n"
        f"* <b>止盈/損依據:</b> <code>{stop_take_logic}</code>\n\n"
        f"--- <b>狀態</b> ---\n"
        f"<b>掃描開始時間:</b> <code>{current_time_str}</code>\n"
    )
    send_telegram_message(message)


def notify_close_Bot(trade):
    # ⚠️ 修正: 使用 f-string 並採用 HTML 標籤
    message = summarize_trade_performance(trade)
    send_telegram_message(message)


def summarize_trade_performance(trade_list):
    """
    總結交易列表的投資表現，並以精簡的 Telegram HTML 格式輸出。
    """
    # 修正點：確保輸入是 DataFrame
    if isinstance(trade_list, pd.DataFrame):
        df = trade_list
    elif isinstance(trade_list, list):
        df = pd.DataFrame(trade_list)
    else:
        return "❌ 交易紀錄類型錯誤，無法進行總結。"

    # 修正點：使用 .empty 檢查 DataFrame 是否為空
    if df.empty:
        return "❌ 交易紀錄為空，無法進行總結。"

    # 2. 計算核心指標
    total_trades = len(df)
    winning_trades = len(df[df["net_pnl"] > 0])
    losing_trades = len(df[df["net_pnl"] < 0])
    total_net_pnl = df["net_pnl"].sum()
    total_fee = df["total_fee"].sum()

    # 3. 衍伸指標
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    avg_pnl = df["net_pnl"].mean()

    # 修正點 A: 確保在沒有獲勝/虧損交易時，平均值為 0
    avg_winning_pnl = (
        df[df["net_pnl"] > 0]["net_pnl"].mean() if winning_trades > 0 else 0.00
    )
    avg_losing_pnl = (
        df[df["net_pnl"] < 0]["net_pnl"].mean() if losing_trades > 0 else 0.00
    )

    # 4. 風險報酬比 (Profit Factor)
    total_gross_profit = df[df["net_pnl"] > 0]["net_pnl"].sum()
    total_gross_loss = df[df["net_pnl"] < 0]["net_pnl"].sum()

    # 修正點 B: 使用 np.isclose 檢查總虧損是否接近 0 (更穩定的浮點數比較)
    if np.isclose(total_gross_loss, 0) or total_gross_loss >= 0:
        profit_factor = np.inf if total_gross_profit > 0 else 0.00
    else:
        profit_factor = total_gross_profit / abs(total_gross_loss)

    # 5. 格式化輸出
    summary_text = (
        f"🚀 <b>交易總結報告</b> (USDⓈ-M)\n\n"
        f"--- <b>核心績效</b> ---\n"
        f"🏆 **總淨盈虧:** <b><code>${total_net_pnl:,.2f}</code></b>\n"
        f"📈 **總交易次數:** <code>{total_trades}</code> 次\n"
        f"💰 **最終餘額:** <code>${df['balance'].iloc[-1]:,.2f}</code>\n\n"
        f"--- <b>風險指標</b> ---\n"
        f"✅ **勝率 (Win Rate):** <code>{win_rate:,.2f}%</code>\n"
        f"⚖️ **獲利因子 (PF):** <code>{profit_factor:,.2f}</code>\n"
        f"💡 **平均單次淨利:** <code>${avg_pnl:,.2f}</code>\n\n"
        f"--- <b>交易明細</b> ---\n"
        f"🟢 獲勝次數: <code>{winning_trades}</code> / 平均: <code>${avg_winning_pnl:,.2f}</code>\n"
        f"🔴 虧損次數: <code>{losing_trades}</code> / 平均: <code>${abs(avg_losing_pnl):,.2f}</code>\n"
        f"💸 **總手續費:** <code>${total_fee:,.2f}</code>\n\n"
        f"<i>(PF > 1.0 策略總體盈利)</i>"
    )

    return summary_text


# 在 pratical_scanner 函數的最開始調用
# def pratical_scanner():
#     notify_startup() # 👈 新增這行
#     print(f"\n[Scanner Detail]")
#     # ... 繼續原本的邏輯


# ===== 回測策略 =====


def backtest():
    """執行回測策略。

    Returns:
        None: 結果會印出並匯出至 Excel。
    """
    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    rr = 0
    entry_time = None
    trades = []
    trade_details = []
    signal_info = None

    # 取得完整資料
    df_main_b = get_klines(SYMBOL, INTERVAL_MAIN, BACKTEST_LIMIT)
    df_ref_b = get_klines(SYMBOL, INTERVAL_REF, BACKTEST_LIMIT)

    df_main_b = calculate_indicators(df_main_b)
    df_ref_b = calculate_indicators(df_ref_b)

    # 從第 21 根 K 線開始回測
    for i in range(21, len(df_main_b)):
        current_time = df_main_b.iloc[i]["timestamp"]

        # ✅ 根據當前 M5 時間,取對應的 M5 和 M15 資料
        df_main = df_main_b[df_main_b["timestamp"] <= current_time].iloc[-22:]
        df_ref = df_ref_b[df_ref_b["timestamp"] <= current_time].iloc[-22:]

        # 檢查資料是否足夠
        if len(df_main) < 22 or len(df_ref) < 22:
            continue

        latest = df_main.iloc[-1]

        # ===== 平倉檢查 =====
        if position != 0 and signal_info is not None:
            exit_price = None

            if position > 0:  # 多單
                if latest["low"] <= stop_loss:
                    exit_price = stop_loss
                elif latest["high"] >= take_profit:
                    exit_price = take_profit

                if exit_price:
                    pnl = (exit_price - entry_price) * position

            else:  # 空單
                if latest["high"] >= stop_loss:
                    exit_price = stop_loss
                elif latest["low"] <= take_profit:
                    exit_price = take_profit

                if exit_price:
                    pnl = (entry_price - exit_price) * abs(position)

            # 處理平倉
            if exit_price:
                signal_info["close_price"] = exit_price

                # 手續費計算
                open_fee = signal_info["open_fee"]
                close_fee = (exit_price * abs(position)) * TAKER_FEE_RATE
                signal_info["close_fee"] = close_fee
                total_fee = open_fee + close_fee
                signal_info["total_fee"] = total_fee

                # 淨損益
                net_pnl = pnl - total_fee
                signal_info["net_pnl"] = net_pnl
                signal_info["pnl"] = pnl
                signal_info["close_time"] = latest["timestamp"]

                # 更新餘額
                margin = (abs(position) * entry_price) / LEVERAGE
                balance = balance + margin + net_pnl
                signal_info["balance"] = balance

                # 記錄交易
                trades.append(net_pnl)
                trade_details.append(
                    {
                        "方向": "多單" if position > 0 else "空單",
                        "槓桿": LEVERAGE,
                        "開倉大小": round(abs(position), 4),
                        "保證金": round(margin, 2),
                        "進場時間": entry_time,
                        "進場價格": entry_price,
                        "出場時間": latest["timestamp"],
                        "出場價格": exit_price,
                        "盈虧": round(pnl, 2),
                        "淨盈虧": round(net_pnl, 2),
                        "手續費": round(total_fee, 2),
                        "RR": rr,
                        "balance": round(balance, 2),  # ✅ 新增
                    }
                )
                position = 0
                signal_info = None

                if balance <= 0:
                    print(f"餘額不足,無法繼續開倉")
                    break

        # ===== 開倉條件 =====
        if position == 0:
            signal_info = generate_signal(df_main, df_ref, balance)
            if signal_info:
                entry_price = signal_info["entry_price"]
                stop_loss = signal_info["stop_loss"]
                take_profit = signal_info["take_profit"]
                rr = signal_info["rr"]
                entry_time = latest["timestamp"]
                position = (
                    signal_info["position"]
                    if signal_info["signal"] == "LONG"
                    else -signal_info["position"]
                )

                # 扣除保證金
                margin = (abs(position) * entry_price) / LEVERAGE

                if balance < margin:
                    print(
                        f"⚠️ 餘額不足,無法開倉 (需要 {margin:.2f} USDT,剩餘 {balance:.2f} USDT)"
                    )
                    signal_info = None
                    continue

                balance -= margin

    # ===== 處理最後未平倉的持倉 =====
    if position != 0 and signal_info is not None:
        exit_price = df_main_b.iloc[-1]["close"]

        if position > 0:
            pnl = (exit_price - entry_price) * position
        else:
            pnl = (entry_price - exit_price) * abs(position)

        open_fee = signal_info["open_fee"]
        close_fee = (exit_price * abs(position)) * TAKER_FEE_RATE
        total_fee = open_fee + close_fee
        net_pnl = pnl - total_fee

        margin = (abs(position) * entry_price) / LEVERAGE
        balance = balance + margin + net_pnl

        trades.append(net_pnl)
        trade_details.append(
            {
                "方向": "多單" if position > 0 else "空單",
                "槓桿": LEVERAGE,
                "開倉大小": round(abs(position), 4),
                "保證金": round(margin, 2),
                "進場時間": entry_time,
                "進場價格": entry_price,
                "出場時間": df_main_b.iloc[-1]["timestamp"],
                "出場價格": exit_price,
                "盈虧": round(pnl, 2),
                "淨盈虧": round(net_pnl, 2),
                "手續費": round(total_fee, 2),
                "RR": rr,
                "balance": round(balance, 2),  # ✅ 新增
            }
        )
        print(f"⚠️ 回測結束時強制平倉: 出場價={exit_price}, 淨盈虧={net_pnl:.2f} USDT")

    # ===== 統計結果 =====
    win_trades = [t for t in trades if t > 0]
    lose_trades = [t for t in trades if t <= 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0
    total_pnl = sum(trades)
    max_drawdown = min(trades) if trades else 0

    print("回測結果:")
    print(f"初始資金: {INITIAL_BALANCE} USDT")
    print(f"槓桿倍數: {LEVERAGE}x")
    print(f"最終資金: {balance:.2f} USDT")
    print(f"總淨盈虧: {total_pnl:.2f} USDT")
    print(f"交易次數: {len(trades)}")
    print(f"勝率: {win_rate:.2f}%")
    print(f"最大單筆虧損: {max_drawdown:.2f} USDT")

    trade_df = pd.DataFrame(trade_details)
    trade_df.to_excel("trade_df.xlsx", index=False)
    print("交易明細已匯出至 trade_df.xlsx")


def adjust_position_size(position, entry_price):
    # 四捨五入到最小單位
    adjusted = round(position // MIN_QTY * MIN_QTY, 3)
    # 檢查名義價值
    if adjusted * entry_price < MIN_NOTIONAL:
        return 0  # 不符合要求，返回 0 表示不開倉
    return adjusted


def is_after_six_am() -> bool:
    """
    判斷當前系統時間是否大於早上 6:00。

    Returns:
        bool: True 如果現在時間 > 06:00:00，否則為 False。
    """
    # 獲取當前時間
    now = dt.datetime.now()

    # 創建一個當天的早上 6:00 時間點
    six_am_today = dt.datetime(
        year=now.year, month=now.month, day=now.day, hour=6, minute=0, second=0
    )

    # 比較當前時間是否晚於 6:00
    if now > six_am_today:
        return True
    else:
        return False


# ===== 實盤掃描 =====
def pratical_scanner():
    trade = []
    notify_startup()
    print(f"\n[Scanner Detail]")
    position = 0
    balance = INITIAL_BALANCE
    while True:
        try:
            df_main = get_klines(SYMBOL, INTERVAL_MAIN, LIMIT)
            df_ref = get_klines(SYMBOL, INTERVAL_REF, LIMIT)
            df_main = calculate_indicators(df_main)
            df_ref = calculate_indicators(df_ref)

            latest = df_main.iloc[-1]
            signal_info = generate_signal(df_main, df_ref, balance)

            trend = "UP" if latest["EMA9"] > latest["EMA21"] else "DOWN"
            print(
                f"{latest['timestamp']}: Close: {round(latest['close'], 2)}, EMA Trend: {trend}, RSI: {round(latest['RSI'], 2)}"
            )

            if position == 0 and signal_info:
                position = (
                    signal_info["position"]
                    if signal_info["signal"] == "LONG"
                    else signal_info["position"] * -1
                )
                print(
                    f"{signal_info['entry_time']}: {signal_info['signal']} 訊號, 進場 {signal_info['entry_price']}, TP {signal_info['take_profit']:.2f}({signal_info['sch_profit']:.2f} USDT), SL {signal_info['stop_loss']}({signal_info['sch_loss']:.2f} USDT), RR {round(signal_info['rr'], 2)}"
                )
                notify_open(signal_info)

                if signal_info["signal"] == "LONG":
                    while True:
                        df_check = get_klines(SYMBOL, INTERVAL_MAIN, 1)
                        current_price = df_check.iloc[-1]["close"]

                        if position > 0 and (
                            current_price >= signal_info["take_profit"]
                            or current_price <= signal_info["stop_loss"]
                        ):
                            # 計算盈虧
                            if position > 0:
                                pnl = (
                                    current_price - signal_info["entry_price"]
                                ) * position
                            else:
                                pnl = (
                                    signal_info["entry_price"] - current_price
                                ) * abs(position)

                            signal_info["close_price"] = current_price
                            # 手續費
                            open_fee = signal_info["open_fee"]
                            close_fee = (
                                signal_info["profit_close_fee"]
                                if pnl
                                else signal_info["loss_close_fee"]
                            )
                            signal_info["close_fee"] = close_fee

                            total_fee = open_fee + close_fee
                            signal_info["total_fee"] = total_fee

                            net_pnl = pnl - total_fee
                            signal_info["net_pnl"] = net_pnl

                            signal_info["pnl"] = pnl
                            signal_info["close_time"] = dt.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )

                            balance = balance + signal_info["net_pnl"]
                            signal_info["balance"] = balance
                            trade.append(signal_info)
                            position = 0
                            print(
                                f"交易完成: 出場價={current_price}, 盈虧={net_pnl:.2f} USDT (手續費={total_fee:.2f})"
                            )
                            notify_close(signal_info, balance)
                            break
                elif signal_info["signal"] == "SHORT":
                    # 監控迴圈
                    while True:
                        df_check = get_klines(SYMBOL, INTERVAL_MAIN, 1)
                        current_price = df_check.iloc[-1]["close"]

                        if position < 0 and (
                            current_price <= signal_info["take_profit"]
                            or current_price >= signal_info["stop_loss"]
                        ):
                            # 計算盈虧
                            if position > 0:
                                pnl = (
                                    current_price - signal_info["entry_price"]
                                ) * position
                            else:
                                pnl = (
                                    signal_info["entry_price"] - current_price
                                ) * abs(position)

                            signal_info["close_price"] = current_price
                            # 手續費
                            open_fee = signal_info["open_fee"]
                            close_fee = (
                                signal_info["profit_close_fee"]
                                if pnl
                                else signal_info["loss_close_fee"]
                            )
                            signal_info["close_fee"] = close_fee

                            total_fee = open_fee + close_fee
                            signal_info["total_fee"] = total_fee

                            net_pnl = pnl - total_fee
                            signal_info["net_pnl"] = net_pnl

                            signal_info["pnl"] = pnl
                            signal_info["close_time"] = dt.datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                            balance = balance + signal_info["net_pnl"]
                            signal_info["balance"] = balance
                            position = 0
                            print(
                                f"交易完成: 出場價={current_price}, 盈虧={net_pnl:.2f} USDT (手續費={total_fee:.2f})"
                            )
                            notify_close(signal_info, balance)
                            trade.append(signal_info)
                            break
                        time.sleep(0.3)

            if len(trade) >= 10 or balance <= 0:
                trade = pd.DataFrame(trade)
                trade.to_excel("trade_df.xlsx", index=False)
                notify_close_Bot(trade)
                break
            time.sleep(0.3)

        except Exception as e:
            print("錯誤:", e)
            time.sleep(15)


def optimize_parameters():
    """優化策略參數以找出最佳回測結果。

    使用網格搜索測試不同參數組合,並返回最佳配置。

    Returns:
        dict: 包含最佳參數組合和對應績效的字典。
    """
    # 定義參數搜索空間
    param_grid = {
        "LEVERAGE": [15],
        "RR_THRESHOLD": [1.2, 1.5],
        "USE_RSI_FILTER": [True, False],
        "USE_TREND_FILTER": [True, False],
        "USE_RR_FILTER": [True, False],
        "USE_STOP_TAKE_M15": [True, False],
        "USE_SL_OPTIMIZER": [True, False],
        "SL_OPTIMIZER_THRESHOLD": [0.3, 0.5],
    }

    best_result = {
        "total_pnl": -float("inf"),
        "params": {},
        "win_rate": 0,
        "total_trades": 0,
        "final_balance": 0,
        "max_drawdown": 0,
        "profit_factor": 0,
    }

    results_list = []

    # 計算總組合數
    import itertools

    param_combinations = list(
        itertools.product(
            param_grid["LEVERAGE"],
            param_grid["RR_THRESHOLD"],
            param_grid["USE_RSI_FILTER"],
            param_grid["USE_TREND_FILTER"],
            param_grid["USE_RR_FILTER"],
            param_grid["USE_STOP_TAKE_M15"],
            param_grid["USE_SL_OPTIMIZER"],
            param_grid["SL_OPTIMIZER_THRESHOLD"],
        )
    )

    total_combinations = len(param_combinations)
    print(f"開始參數優化,總共 {total_combinations} 種組合...")

    for idx, params in enumerate(param_combinations, 1):
        # 解包參數
        (
            leverage,
            rr_threshold,
            use_rsi,
            use_trend,
            use_rr,
            use_stop_m15,
            use_sl_opt,
            sl_opt_th,
        ) = params

        # 暫存全局變數
        global LEVERAGE, RR_THRESHOLD, USE_RSI_FILTER, USE_TREND_FILTER
        global \
            USE_RR_FILTER, \
            USE_STOP_TAKE_M15, \
            USE_SL_OPTIMIZER, \
            SL_OPTIMIZER_THRESHOLD

        original_params = {
            "LEVERAGE": LEVERAGE,
            "RR_THRESHOLD": RR_THRESHOLD,
            "USE_RSI_FILTER": USE_RSI_FILTER,
            "USE_TREND_FILTER": USE_TREND_FILTER,
            "USE_RR_FILTER": USE_RR_FILTER,
            "USE_STOP_TAKE_M15": USE_STOP_TAKE_M15,
            "USE_SL_OPTIMIZER": USE_SL_OPTIMIZER,
            "SL_OPTIMIZER_THRESHOLD": SL_OPTIMIZER_THRESHOLD,
        }

        # 設置新參數
        LEVERAGE = leverage
        RR_THRESHOLD = rr_threshold
        USE_RSI_FILTER = use_rsi
        USE_TREND_FILTER = use_trend
        USE_RR_FILTER = use_rr
        USE_STOP_TAKE_M15 = use_stop_m15
        USE_SL_OPTIMIZER = use_sl_opt
        SL_OPTIMIZER_THRESHOLD = sl_opt_th

        try:
            # 執行回測
            result = backtest_with_return()

            # 記錄結果
            result["params"] = {
                "LEVERAGE": leverage,
                "RR_THRESHOLD": rr_threshold,
                "USE_RSI_FILTER": use_rsi,
                "USE_TREND_FILTER": use_trend,
                "USE_RR_FILTER": use_rr,
                "USE_STOP_TAKE_M15": use_stop_m15,
                "USE_SL_OPTIMIZER": use_sl_opt,
                "SL_OPTIMIZER_THRESHOLD": sl_opt_th,
            }
            results_list.append(result)

            # 更新最佳結果(以總盈虧為主要指標)
            if result["total_pnl"] > best_result["total_pnl"]:
                best_result = result.copy()

            # 進度顯示
            if idx % 10 == 0 or idx == total_combinations:
                print(
                    f"進度: {idx}/{total_combinations} ({idx / total_combinations * 100:.1f}%)"
                )

        except Exception as e:
            print(f"參數組合 {idx} 執行失敗: {e}")
            continue
        finally:
            # 恢復原始參數
            LEVERAGE = original_params["LEVERAGE"]
            RR_THRESHOLD = original_params["RR_THRESHOLD"]
            USE_RSI_FILTER = original_params["USE_RSI_FILTER"]
            USE_TREND_FILTER = original_params["USE_TREND_FILTER"]
            USE_RR_FILTER = original_params["USE_RR_FILTER"]
            USE_STOP_TAKE_M15 = original_params["USE_STOP_TAKE_M15"]
            USE_SL_OPTIMIZER = original_params["USE_SL_OPTIMIZER"]
            SL_OPTIMIZER_THRESHOLD = original_params["SL_OPTIMIZER_THRESHOLD"]

    # 輸出結果
    print("\n" + "=" * 60)
    print("參數優化完成!")
    print("=" * 60)
    print(f"\n最佳參數組合:")
    for key, value in best_result["params"].items():
        print(f"  {key}: {value}")

    print(f"\n最佳績效:")
    print(f"  總淨盈虧: {best_result['total_pnl']:.2f} USDT")
    print(f"  最終資金: {best_result['final_balance']:.2f} USDT")
    print(
        f"  報酬率: {(best_result['final_balance'] / INITIAL_BALANCE - 1) * 100:.2f}%"
    )
    print(f"  交易次數: {best_result['total_trades']}")
    print(f"  勝率: {best_result['win_rate']:.2f}%")
    print(f"  獲利因子: {best_result['profit_factor']:.2f}")
    print(f"  最大單筆虧損: {best_result['max_drawdown']:.2f} USDT")

    # 匯出所有結果
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values("total_pnl", ascending=False)
    results_df.to_excel("optimization_results.xlsx", index=False)
    print(f"\n所有優化結果已匯出至 optimization_results.xlsx")

    return best_result


def backtest_with_return():
    """執行回測並返回結果(不印出詳細資訊)。

    Returns:
        dict: 包含回測績效指標的字典。
    """
    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    rr = 0
    entry_time = None
    trades = []
    signal_info = None

    # 取得完整資料
    df_main_b = get_klines(SYMBOL, INTERVAL_MAIN, BACKTEST_LIMIT)
    df_ref_b = get_klines(SYMBOL, INTERVAL_REF, BACKTEST_LIMIT)

    df_main_b = calculate_indicators(df_main_b)
    df_ref_b = calculate_indicators(df_ref_b)

    # 從第 21 根 K 線開始回測
    for i in range(21, len(df_main_b)):
        current_time = df_main_b.iloc[i]["timestamp"]

        df_main = df_main_b[df_main_b["timestamp"] <= current_time].iloc[-22:]
        df_ref = df_ref_b[df_ref_b["timestamp"] <= current_time].iloc[-22:]

        if len(df_main) < 22 or len(df_ref) < 22:
            continue

        latest = df_main.iloc[-1]

        # 平倉檢查
        if position != 0 and signal_info is not None:
            exit_price = None

            if position > 0:
                if latest["low"] <= stop_loss:
                    exit_price = stop_loss
                elif latest["high"] >= take_profit:
                    exit_price = take_profit

                if exit_price:
                    pnl = (exit_price - entry_price) * position

            else:
                if latest["high"] >= stop_loss:
                    exit_price = stop_loss
                elif latest["low"] <= take_profit:
                    exit_price = take_profit

                if exit_price:
                    pnl = (entry_price - exit_price) * abs(position)

            if exit_price:
                open_fee = signal_info["open_fee"]
                close_fee = (exit_price * abs(position)) * TAKER_FEE_RATE
                total_fee = open_fee + close_fee
                net_pnl = pnl - total_fee

                margin = (abs(position) * entry_price) / LEVERAGE
                balance = balance + margin + net_pnl

                trades.append(net_pnl)
                position = 0
                signal_info = None

                if balance <= 0:
                    break

        # 開倉條件
        if position == 0:
            signal_info = generate_signal(df_main, df_ref, balance)
            if signal_info:
                entry_price = signal_info["entry_price"]
                stop_loss = signal_info["stop_loss"]
                take_profit = signal_info["take_profit"]
                rr = signal_info["rr"]
                entry_time = latest["timestamp"]
                position = (
                    signal_info["position"]
                    if signal_info["signal"] == "LONG"
                    else -signal_info["position"]
                )

                margin = (abs(position) * entry_price) / LEVERAGE

                if balance < margin:
                    signal_info = None
                    continue

                balance -= margin

    # 處理最後未平倉的持倉
    if position != 0 and signal_info is not None:
        exit_price = df_main_b.iloc[-1]["close"]

        if position > 0:
            pnl = (exit_price - entry_price) * position
        else:
            pnl = (entry_price - exit_price) * abs(position)

        open_fee = signal_info["open_fee"]
        close_fee = (exit_price * abs(position)) * TAKER_FEE_RATE
        total_fee = open_fee + close_fee
        net_pnl = pnl - total_fee

        margin = (abs(position) * entry_price) / LEVERAGE
        balance = balance + margin + net_pnl

        trades.append(net_pnl)

    # 統計結果
    win_trades = [t for t in trades if t > 0]
    lose_trades = [t for t in trades if t <= 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0
    total_pnl = sum(trades)
    max_drawdown = min(trades) if trades else 0

    # 計算獲利因子
    total_profit = sum(win_trades) if win_trades else 0
    total_loss = abs(sum(lose_trades)) if lose_trades else 0
    profit_factor = (
        total_profit / total_loss
        if total_loss > 0
        else (float("inf") if total_profit > 0 else 0)
    )

    return {
        "total_pnl": total_pnl,
        "final_balance": balance,
        "total_trades": len(trades),
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
        "win_trades": len(win_trades),
        "lose_trades": len(lose_trades),
    }


# ===== 執行回測參數優化 =====
# best_params = optimize_parameters()
# ===== 執行回測 =====
backtest()

# ===== 執行實盤 =====
# pratical_scanner()


# send_telegram_message("hi")

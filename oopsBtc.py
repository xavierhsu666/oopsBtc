# from binance.client import Client
# from binance.enums import *
import requests
import pandas as pd
import numpy as np


# ===== Binance API 管理類 =====
class BinanceAPIManager:
    def __init__(self, api_key, api_secret, testnet=True):
        self.client = Client(api_key, api_secret, testnet=testnet)

    def set_leverage(self, symbol, leverage):
        self.client.futures_change_leverage(symbol=symbol, leverage=leverage)

    def place_order(self, symbol, side, quantity, take_profit=None, stop_loss=None):
        # 市價單
        order = self.client.futures_create_order(
            symbol=symbol, side=side, type=ORDER_TYPE_MARKET, quantity=quantity
        )
        print("市價單已送出:", order)

        # 止盈單
        if take_profit:
            tp_order = self.client.futures_create_order(
                symbol=symbol,
                side=SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
                type=ORDER_TYPE_STOP_MARKET,
                stopPrice=take_profit,
                closePosition=True,
            )
            print("止盈單已送出:", tp_order)

        # 止損單
        if stop_loss:
            sl_order = self.client.futures_create_order(
                symbol=symbol,
                side=SIDE_SELL if side == SIDE_BUY else SIDE_BUY,
                type=ORDER_TYPE_STOP_MARKET,
                stopPrice=stop_loss,
                closePosition=True,
            )
            print("止損單已送出:", sl_order)

        return order

    def get_balance(self):
        return self.client.futures_account_balance()

    def get_position(self, symbol):
        positions = self.client.futures_position_information(symbol=symbol)
        return positions


# ===== 策略類 =====
class TradingStrategy:
    BASE_URL = "https://fapi.binance.com/fapi/v1/klines"

    def __init__(
        self,
        initial_balance=1000,
        leverage=15,
        rr_threshold=1.2,
        use_rsi=False,
        use_trend=True,
        use_rr=True,
        use_m15=True,
    ):
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.rr_threshold = rr_threshold
        self.use_rsi = use_rsi
        self.use_trend = use_trend
        self.use_rr = use_rr
        self.use_m15 = use_m15

    def get_klines(self, symbol, interval, limit):
        url = f"{self.BASE_URL}?symbol={symbol}&interval={interval}&limit={limit}"
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
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms") + pd.Timedelta(
            hours=8
        )
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        return df

    def calculate_indicators(self, df):
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

    def get_last_m15_levels(self, m15_df, current_time):
        ref = m15_df[m15_df["timestamp"] <= current_time].iloc[-1]
        return ref["high"], ref["low"]

    def backtest(self, df_main, df_ref):
        balance = self.initial_balance
        position = 0
        entry_price = 0
        trades = []
        trade_details = []

        df_ref = self.calculate_indicators(df_ref)

        for i in range(21, len(df_main)):
            latest = df_main.iloc[i]

            # 平倉檢查
            if position != 0:
                if position > 0:  # 多單
                    if latest["low"] <= stop_loss or latest["high"] >= take_profit:
                        pnl = (latest["close"] - entry_price) * position
                        balance += pnl
                        trades.append(pnl)
                        position = 0
                else:  # 空單
                    if latest["high"] >= stop_loss or latest["low"] <= take_profit:
                        pnl = (entry_price - latest["close"]) * abs(position)
                        balance += pnl
                        trades.append(pnl)
                        position = 0

            # 開倉條件
            if position == 0:
                m15_high, m15_low = self.get_last_m15_levels(
                    df_ref, latest["timestamp"]
                )
                m15_trend_up = (
                    df_ref[df_ref["timestamp"] <= latest["timestamp"]].iloc[-1]["EMA9"]
                    > df_ref[df_ref["timestamp"] <= latest["timestamp"]].iloc[-1][
                        "EMA21"
                    ]
                )
                m15_trend_down = not m15_trend_up

                # 多單
                if latest["EMA9"] > latest["EMA21"]:
                    if self.use_rsi and latest["RSI"] >= 70:
                        continue
                    if self.use_trend and not m15_trend_up:
                        continue
                    entry_price = latest["close"]
                    if self.use_m15 and (
                        m15_high <= entry_price or m15_low >= entry_price
                    ):
                        continue
                    stop_loss = m15_low if self.use_m15 else latest["low"]
                    take_profit = m15_high if self.use_m15 else latest["high"]
                    rr = (take_profit - entry_price) / (entry_price - stop_loss)
                    if self.use_rr and rr <= self.rr_threshold:
                        continue
                    position = (balance * self.leverage) / entry_price
                    entry_time = latest["timestamp"]

                # 空單
                elif latest["EMA9"] < latest["EMA21"]:
                    if self.use_rsi and latest["RSI"] <= 30:
                        continue
                    if self.use_trend and not m15_trend_down:
                        continue
                    entry_price = latest["close"]
                    if self.use_m15 and (
                        m15_low >= entry_price or m15_high <= entry_price
                    ):
                        continue
                    stop_loss = m15_high if self.use_m15 else latest["high"]
                    take_profit = m15_low if self.use_m15 else latest["low"]
                    rr = (entry_price - take_profit) / (stop_loss - entry_price)
                    if self.use_rr and rr <= self.rr_threshold:
                        continue
                    position = -(balance * self.leverage) / entry_price
                    entry_time = latest["timestamp"]

        print(f"回測完成，最終資金: {balance:.2f} USDT，交易次數: {len(trades)}")


# 初始化策略
strategy = TradingStrategy(
    initial_balance=1000,
    leverage=15,
    rr_threshold=1.2,
    use_rsi=False,
    use_trend=True,
    use_rr=True,
    use_m15=True,
)

# 抓資料
df_main = strategy.get_klines("BTCUSDT", "5m", 1000)
df_ref = strategy.get_klines("BTCUSDT", "15m", 1000)
df_main = strategy.calculate_indicators(df_main)

# 回測
strategy.backtest(df_main, df_ref)

# 初始化幣安 API
# api_manager = BinanceAPIManager(
#     api_key="你的API_KEY", api_secret="你的API_SECRET", testnet=True
# )
# api_manager.set_leverage("BTCUSDT", 15)

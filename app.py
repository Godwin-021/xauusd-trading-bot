import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
import time
import os
import joblib
from collections import Counter
import logging
import streamlit as st
import threading

# Configure logging
logging.basicConfig(
    filename='xauusd_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# User settings
LOGIN = int(os.getenv("MT5_LOGIN", 104486))  # MT5 account number (from env or default)
PASSWORD = os.getenv("MT5_PASSWORD", "6Rj@7v!EZ!")  # MT5 password
SERVER = os.getenv("MT5_SERVER", "TradewillGlobal-Server")  # MT5 server
SYMBOL = "XAUUSD"  # Trading symbol: Gold vs. US Dollar
TIMEFRAME = mt5.TIMEFRAME_M5  # 5-minute chart
LOT_SIZE = 0.1  # Lot size for XAU/USD
PROFIT_TARGET = 0.05  # 5% profit target
SL_ATR_MULTI = 1.5  # Stop-loss multiplier for ATR
RSI_PERIOD = 14
EMA_FAST = 12
EMA_SLOW = 26
MOMENTUM_PERIOD = 10  # For Magic Histogram approximation
POWERBALL_PERIOD = 20  # Period for Powerball breakout
POWERBALL_VOLUME_MULTIPLIER = 1.5  # Volume confirmation threshold
MODEL_FILE = "xauusd_model.joblib"
FEATURE_FILE = "xauusd_features.csv"
RETRAIN_EVERY = 50  # Retrain model every 50 trades
WARMUP_TRADES = 30  # Initial trades for data collection

# Global variable for bot running state
running = False

# Initialize MT5 connection
def initialize_mt5():
    if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
        logging.error("MT5 initialization failed: %s", mt5.last_error())
        return False
    logging.info("MT5 initialized successfully")
    return True

# Fetch market data
def get_market_data(symbol, timeframe, bars=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        logging.error("Error fetching rates: %s", mt5.last_error())
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.columns = [col.upper() for col in df.columns]
    return df

# Calculate technical indicators
def calculate_indicators(df):
    df['EMA_FAST'] = df['CLOSE'].ewm(span=EMA_FAST, adjust=False).mean()
    df['EMA_SLOW'] = df['CLOSE'].ewm(span=EMA_SLOW, adjust=False).mean()
    
    delta = df['CLOSE'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['MOMENTUM'] = df['CLOSE'].pct_change(periods=MOMENTUM_PERIOD) * 100
    
    df['TR'] = np.maximum(df['HIGH'] - df['LOW'], 
                         np.maximum(abs(df['HIGH'] - df['CLOSE'].shift(1)), 
                                    abs(df['LOW'] - df['CLOSE'].shift(1))))
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    df['POWERBALL_UPPER'] = df['HIGH'].rolling(window=POWERBALL_PERIOD).max()
    df['POWERBALL_LOWER'] = df['LOW'].rolling(window=POWERBALL_PERIOD).min()
    df['VOLUME_MA'] = df['REAL_VOLUME'].rolling(window=POWERBALL_PERIOD).mean()
    df['POWERBALL_BUY'] = (df['CLOSE'] > df['POWERBALL_UPPER'].shift(1)) & \
                          (df['REAL_VOLUME'] > df['VOLUME_MA'] * POWERBALL_VOLUME_MULTIPLIER) & \
                          (df['MOMENTUM'] > 0)
    df['POWERBALL_SELL'] = (df['CLOSE'] < df['POWERBALL_LOWER'].shift(1)) & \
                           (df['REAL_VOLUME'] > df['VOLUME_MA'] * POWERBALL_VOLUME_MULTIPLIER) & \
                           (df['MOMENTUM'] < 0)
    
    return df

# Detect candlestick patterns
def detect_patterns(df: pd.DataFrame):
    pats = ["None"] * len(df)
    for i, _ in enumerate(df.index):
        o, h, l, c = df.iloc[i][["OPEN", "HIGH", "LOW", "CLOSE"]]
        body = abs(c - o)
        lower = min(o, c) - l
        upper = h - max(o, c)

        if lower >= 2 * body and upper <= body:
            pats[i] = "hammer"
            continue
        if upper >= 2 * body and lower <= body:
            pats[i] = "shootingstar"
            continue
        if abs(c - o) <= (h - l) * 0.1:
            pats[i] = "doji"
            continue
        if i >= 1:
            prev_o, prev_c = df.iloc[i-1][["OPEN", "CLOSE"]]
            curr_o, curr_c = o, c
            if (prev_c < prev_o and curr_c > curr_o and
                curr_o < prev_c and curr_c > prev_o):
                pats[i] = "bullishengulfing"
                continue
            if (prev_c > prev_o and curr_c < curr_o and
                curr_o > prev_c and curr_c < prev_o):
                pats[i] = "bearishengulfing"
                continue
            prev_body = abs(prev_c - prev_o)
            curr_body = abs(curr_c - curr_o)
            if prev_body > 0 and curr_body > 0:
                if prev_o > prev_c and curr_o < curr_c:
                    if curr_o > prev_c and curr_c < prev_o:
                        pats[i] = "bullishharami"
                        continue
                if prev_o < prev_c and curr_o > curr_c:
                    if curr_o < prev_c and curr_c > prev_o:
                        pats[i] = "bearishharami"
                        continue
        if i >= 2:
            o1, c1 = df.iloc[i-2][["OPEN", "CLOSE"]]
            o2, c2 = df.iloc[i-1][["OPEN", "CLOSE"]]
            o3, c3 = o, c
            is_bearish1 = c1 < o1
            is_small2 = abs(c2 - o2) < abs(c1 - o1) * 0.5
            is_bullish3 = c3 > o3
            if is_bearish1 and is_small2 and c3 > ((o1 + c1) / 2) and is_bullish3:
                if min(o2, c2) < c1 and max(o2, c2) > c1:
                    pats[i] = "morningstar"
                    continue
            is_bullish1 = c1 > o1
            is_bearish3 = c3 < o3
            if is_bullish1 and is_small2 and c3 < ((o1 + c1) / 2) and is_bearish3:
                if max(o2, c2) > c1 and min(o2, c2) < c1:
                    pats[i] = "eveningstar"
                    continue
    return pats

# Prepare features for ML model
def prepare_features(df, patterns):
    df['TREND'] = (df['EMA_FAST'] > df['EMA_SLOW']).astype(int)
    df['RSI_SIGNAL'] = ((df['RSI'] < 30) | (df['RSI'] > 70)).astype(int)
    df['MOMENTUM_SIGNAL'] = (df['MOMENTUM'] > 0).astype(int)
    df['POWERBALL_BUY_SIGNAL'] = df['POWERBALL_BUY'].astype(int)
    df['POWERBALL_SELL_SIGNAL'] = df['POWERBALL_SELL'].astype(int)
    
    bullish_patterns = ['hammer', 'bullishengulfing', 'bullishharami', 'morningstar']
    bearish_patterns = ['shootingstar', 'bearishengulfing', 'bearishharami', 'eveningstar']
    df['BULLISH_PATTERN'] = [1 if p in bullish_patterns else 0 for p in patterns]
    df['BEARISH_PATTERN'] = [1 if p in bearish_patterns else 0 for p in patterns]
    df['DOJI'] = [1 if p == 'doji' else 0 for p in patterns]
    
    features = df[['TREND', 'RSI_SIGNAL', 'MOMENTUM_SIGNAL', 'POWERBALL_BUY_SIGNAL', 
                   'POWERBALL_SELL_SIGNAL', 'BULLISH_PATTERN', 'BEARISH_PATTERN', 'DOJI']].dropna()
    return features

# Train or load ML model
def train_model(df, patterns):
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    else:
        model = LogisticRegression()
    
    features = prepare_features(df, patterns)
    if len(features) < WARMUP_TRADES:
        return model, False
    
    df['TARGET'] = (df['CLOSE'].shift(-1) > df['CLOSE']).astype(int)
    X = features.iloc[:-1]
    y = df['TARGET'].iloc[:-1]
    
    if len(X) >= WARMUP_TRADES:
        model.fit(X, y)
        joblib.dump(model, MODEL_FILE)
        return model, True
    return model, False

# Get trade signal
def get_trade_signal(df, patterns, model):
    latest = df.iloc[-1]
    latest_pattern = patterns[-1]
    features = prepare_features(df.iloc[-1:], [latest_pattern])
    prob = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else 0.5
    
    logging.info(f"Trend: {'Up' if latest['EMA_FAST'] > latest['EMA_SLOW'] else 'Down'}")
    logging.info(f"RSI: {latest['RSI']:.2f}, Momentum: {latest['MOMENTUM']:.2f}")
    logging.info(f"Pattern: {latest_pattern}, Powerball: {'Buy' if latest['POWERBALL_BUY'] else 'Sell' if latest['POWERBALL_SELL'] else 'None'}")
    logging.info(f"Win Probability: {prob:.2%}")
    
    bullish_patterns = ['hammer', 'bullishengulfing', 'bullishharami', 'morningstar']
    bearish_patterns = ['shootingstar', 'bearishengulfing', 'bearishharami', 'eveningstar']
    
    if (prob >= 0.6 and latest['EMA_FAST'] > latest['EMA_SLOW'] and
        latest['RSI'] < 30 and latest['MOMENTUM'] > 0 and 
        latest_pattern in bullish_patterns and latest['POWERBALL_BUY']):
        return "BUY", latest['ATR'] * SL_ATR_MULTI
    elif (prob >= 0.6 and latest['EMA_FAST'] < latest['EMA_SLOW'] and
          latest['RSI'] > 70 and latest['MOMENTUM'] < 0 and 
          latest_pattern in bearish_patterns and latest['POWERBALL_SELL']):
        return "SELL", latest['ATR'] * SL_ATR_MULTI
    return None, None

# Place trade
def place_trade(symbol, action, lot_size, price, sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": price - sl if action == "BUY" else price + sl,
        "tp": price * (1 + PROFIT_TARGET) if action == "BUY" else price * (1 - PROFIT_TARGET),
        "deviation": 10,
        "magic": 123456,
        "comment": "AI Trade",
        "type_time": mt5.ORDER_TIME_GTC
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error("Order failed: %s", result.comment)
        return False
    logging.info(f"Order placed: {action} at {price}, SL: {request['sl']}, TP: {request['tp']}")
    return True

# Monitor and manage trades
def monitor_trades(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions:
        for pos in positions:
            if pos.profit / pos.volume >= PROFIT_TARGET * pos.price_open:
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": mt5.symbol_info_tick(symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask,
                    "deviation": 10,
                    "magic": 123456,
                    "comment": "Close AI Trade"
                }
                mt5.order_send(close_request)
                logging.info(f"Closed trade {pos.ticket} with {pos.profit:.2f} profit")

# Bot loop function
def bot_loop():
    global running
    if not initialize_mt5():
        st.error("Failed to initialize MT5")
        return
    
    trade_count = 0
    while running:
        try:
            df = get_market_data(SYMBOL, TIMEFRAME)
            if df is None:
                time.sleep(60)
                continue
            
            df = calculate_indicators(df)
            patterns = detect_patterns(df)
            
            model, trained = train_model(df, patterns)
            
            action, sl = get_trade_signal(df, patterns, model)
            
            if action and not mt5.positions_get(symbol=SYMBOL):
                tick = mt5.symbol_info_tick(SYMBOL)
                price = tick.ask if action == "BUY" else tick.bid
                if place_trade(SYMBOL, action, LOT_SIZE, price, sl, price * (1 + PROFIT_TARGET)):
                    trade_count += 1
            
            monitor_trades(SYMBOL)
            
            if trade_count >= RETRAIN_EVERY:
                model, trained = train_model(df, patterns)
                trade_count = 0
            
            df.to_csv(FEATURE_FILE, index=False)
            
            counts = Counter(patterns)
            log_lines = [f"{k}={v}" for k, v in counts.items() if k != "None"]
            logging.info(f"Pattern counts: {', '.join(log_lines)}")
            
            time.sleep(300)
            
        except Exception as e:
            logging.error(f"Error: {e}")
            time.sleep(60)
    
    mt5.shutdown()
    logging.info("Bot stopped")

# Streamlit app
st.title("XAU/USD Trading Bot")

st.write("This app controls a trading bot for XAU/USD using MT5.")

# Input for MT5 credentials (optional, if not set in env)
mt5_login = st.text_input("MT5 Login", value=os.getenv("MT5_LOGIN", "123456"))
mt5_password = st.text_input("MT5 Password", type="password", value=os.getenv("MT5_PASSWORD", "your_password"))
mt5_server = st.text_input("MT5 Server", value=os.getenv("MT5_SERVER", "MetaQuotes-Demo"))

os.environ["MT5_LOGIN"] = mt5_login
os.environ["MT5_PASSWORD"] = mt5_password
os.environ["MT5_SERVER"] = mt5_server

# Start/Stop buttons
if st.button("Start Bot"):
    if not running:
        running = True
        bot_thread = threading.Thread(target=bot_loop)
        bot_thread.start()
        st.success("Bot started!")
    else:
        st.warning("Bot is already running.")

if st.button("Stop Bot"):
    if running:
        running = False
        st.success("Bot stopped.")
    else:
        st.warning("Bot is not running.")

# Display logs
st.subheader("Live Logs")
logs_placeholder = st.empty()
while True:
    with open('xauusd_bot.log', 'r') as f:
        logs = f.read()
    logs_placeholder.text_area("Logs", logs, height=300)
    time.sleep(5)

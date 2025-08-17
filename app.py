import streamlit as st
import time
import os

# Streamlit app
st.title("XAU/USD Trading Bot Monitor")

st.write("This app monitors the XAU/USD trading bot running in MetaTrader 5.")

# Input for log file path (optional, defaults to repository)
log_file_path = st.text_input("Log File Path", value="xauusd_bot.log")

# Display trade status
st.subheader("Trade Status")
status_placeholder = st.empty()

# Display logs
st.subheader("Live Logs")
logs_placeholder = st.empty()

# Monitor loop
while True:
    try:
        # Check if log file exists
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r') as f:
                logs = f.read()
            logs_placeholder.text_area("Logs", logs, height=300)
            
            # Parse logs for trade status (simplified)
            open_trades = 0
            last_trade = "No trades"
            for line in logs.split('\n'):
                if "Order placed" in line:
                    last_trade = line
                    open_trades += 1
                elif "Closed trade" in line:
                    open_trades = max(0, open_trades - 1)
            
            status_placeholder.write(f"Open Trades: {open_trades}\nLast Trade: {last_trade}")
        else:
            logs_placeholder.write("Log file not found. Ensure the EA is running and logging to the specified path.")
        
        time.sleep(5)  # Update every 5 seconds
    except Exception as e:
        logs_placeholder.write(f"Error reading logs: {e}")
        time.sleep(5)
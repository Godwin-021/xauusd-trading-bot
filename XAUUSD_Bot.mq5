//+------------------------------------------------------------------+
//| XAUUSD Trading Bot EA                                            |
//| Trades XAU/USD on M5 timeframe with 0.1 lot size, 5% profit target|
//| Uses Powerball Signals, candlestick patterns, EMA, RSI, momentum  |
//+------------------------------------------------------------------+
#property copyright "Your Name"
#property link      "https://github.com/your-username/xauusd-trading-bot"
#property version   "1.00"

#include <Trade/Trade.mqh>

// Input parameters
input int    Login = 106486;              // MT5 Account Number
input string Password = "6Rj@7v!EZ!";  // MT5 Password
input string Server = "TradewillGlobal-Server";  // MT5 Server
input string SymbolName = "XAUUSD";       // Trading Symbol
input double LotSize = 0.1;               // Lot Size
input double ProfitTarget = 0.05;         // Profit Target (5%)
input double SL_ATR_Multi = 1.5;          // Stop-Loss ATR Multiplier
input int    RSI_Period = 14;             // RSI Period
input int    EMA_Fast = 12;               // Fast EMA Period
input int    EMA_Slow = 26;               // Slow EMA Period
input int    Momentum_Period = 10;        // Momentum Period
input int    Powerball_Period = 20;       // Powerball Breakout Period
input double Volume_Multiplier = 1.5;     // Volume Confirmation Threshold
input int    Retrain_Every = 50;          // Retrain Model Every X Trades
input int    Warmup_Trades = 30;          // Initial Trades for Data

// Global variables
CTrade trade;
int trade_count = 0;
double atr[];
double ema_fast[];
double ema_slow[];
double rsi[];
double momentum[];
double powerball_upper[];
double powerball_lower[];
double volume_ma[];
bool powerball_buy[];
bool powerball_sell[];

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize MT5
   if(!TerminalInfoInteger(TERMINAL_CONNECTED))
   {
      Print("MT5 initialization failed. Check credentials.");
      return(INIT_FAILED);
   }
   
   // Set up arrays
   ArraySetAsSeries(atr, true);
   ArraySetAsSeries(ema_fast, true);
   ArraySetAsSeries(ema_slow, true);
   ArraySetAsSeries(rsi, true);
   ArraySetAsSeries(momentum, true);
   ArraySetAsSeries(powerball_upper, true);
   ArraySetAsSeries(powerball_lower, true);
   ArraySetAsSeries(volume_ma, true);
   ArraySetAsSeries(powerball_buy, true);
   ArraySetAsSeries(powerball_sell, true);
   
   Print("EA initialized for XAUUSD");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("EA deinitialized: ", reason);
}

//+------------------------------------------------------------------+
//| Calculate indicators                                              |
//+------------------------------------------------------------------+
bool CalculateIndicators(double &close[], double &open[], double &high[], double &low[], long &volume[], int bars)
{
   // Copy price data
   if(CopyClose(SymbolName, PERIOD_M5, 0, bars, close) <= 0 ||
      CopyOpen(SymbolName, PERIOD_M5, 0, bars, open) <= 0 ||
      CopyHigh(SymbolName, PERIOD_M5, 0, bars, high) <= 0 ||
      CopyLow(SymbolName, PERIOD_M5, 0, bars, low) <= 0 ||
      CopyTickVolume(SymbolName, PERIOD_M5, 0, bars, volume) <= 0)
   {
      Print("Error copying price data: ", GetLastError());
      return false;
   }
   
   // Resize arrays
   ArrayResize(atr, bars);
   ArrayResize(ema_fast, bars);
   ArrayResize(ema_slow, bars);
   ArrayResize(rsi, bars);
   ArrayResize(momentum, bars);
   ArrayResize(powerball_upper, bars);
   ArrayResize(powerball_lower, bars);
   ArrayResize(volume_ma, bars);
   ArrayResize(powerball_buy, bars);
   ArrayResize(powerball_sell, bars);
   
   // Calculate EMA
   int ema_fast_handle = iMA(SymbolName, PERIOD_M5, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   int ema_slow_handle = iMA(SymbolName, PERIOD_M5, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   if(CopyBuffer(ema_fast_handle, 0, 0, bars, ema_fast) <= 0 ||
      CopyBuffer(ema_slow_handle, 0, 0, bars, ema_slow) <= 0)
   {
      Print("Error calculating EMA: ", GetLastError());
      return false;
   }
   
   // Calculate RSI
   int rsi_handle = iRSI(SymbolName, PERIOD_M5, RSI_Period, PRICE_CLOSE);
   if(CopyBuffer(rsi_handle, 0, 0, bars, rsi) <= 0)
   {
      Print("Error calculating RSI: ", GetLastError());
      return false;
   }
   
   // Calculate Momentum
   for(int i = 0; i < bars - Momentum_Period; i++)
   {
      momentum[i] = ((close[i] / close[i + Momentum_Period]) - 1) * 100;
   }
   
   // Calculate ATR
   int atr_handle = iATR(SymbolName, PERIOD_M5, 14);
   if(CopyBuffer(atr_handle, 0, 0, bars, atr) <= 0)
   {
      Print("Error calculating ATR: ", GetLastError());
      return false;
   }
   
   // Calculate Powerball Signals (Donchian Channel + Volume + Momentum)
   for(int i = 0; i < bars - Powerball_Period; i++)
   {
      powerball_upper[i] = high[iHighest(SymbolName, PERIOD_M5, MODE_HIGH, Powerball_Period, i)];
      powerball_lower[i] = low[iLowest(SymbolName, PERIOD_M5, MODE_LOW, Powerball_Period, i)];
      double sum_volume = 0;
      for(int j = i; j < i + Powerball_Period; j++)
         sum_volume += volume[j];
      volume_ma[i] = sum_volume / Powerball_Period;
      
      powerball_buy[i] = close[i] > powerball_upper[i + 1] &&
                         volume[i] > volume_ma[i] * Volume_Multiplier &&
                         momentum[i] > 0;
      powerball_sell[i] = close[i] < powerball_lower[i + 1] &&
                          volume[i] > volume_ma[i] * Volume_Multiplier &&
                          momentum[i] < 0;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Detect candlestick patterns                                       |
//+------------------------------------------------------------------+
string DetectPattern(double &open[], double &high[], double &low[], double &close[], int index)
{
   double o = open[index];
   double h = high[index];
   double l = low[index];
   double c = close[index];
   double body = MathAbs(c - o);
   double lower = MathMin(o, c) - l;
   double upper = h - MathMax(o, c);
   
   // Hammer
   if(lower >= 2 * body && upper <= body)
      return "hammer";
   
   // Shooting Star
   if(upper >= 2 * body && lower <= body)
      return "shootingstar";
   
   // Doji
   if(MathAbs(c - o) <= (h - l) * 0.1)
      return "doji";
   
   if(index >= 1)
   {
      double prev_o = open[index + 1];
      double prev_c = close[index + 1];
      double curr_o = o;
      double curr_c = c;
      
      // Bullish Engulfing
      if(prev_c < prev_o && curr_c > curr_o &&
         curr_o < prev_c && curr_c > prev_o)
         return "bullishengulfing";
      
      // Bearish Engulfing
      if(prev_c > prev_o && curr_c < curr_o &&
         curr_o > prev_c && curr_c < prev_o)
         return "bearishengulfing";
      
      double prev_body = MathAbs(prev_c - prev_o);
      double curr_body = MathAbs(curr_c - curr_o);
      if(prev_body > 0 && curr_body > 0)
      {
         // Bullish Harami
         if(prev_o > prev_c && curr_o < curr_c &&
            curr_o > prev_c && curr_c < prev_o)
            return "bullishharami";
         
         // Bearish Harami
         if(prev_o < prev_c && curr_o > curr_c &&
            curr_o < prev_c && curr_c > prev_o)
            return "bearishharami";
      }
   }
   
   if(index >= 2)
   {
      double o1 = open[index + 2];
      double c1 = close[index + 2];
      double o2 = open[index + 1];
      double c2 = close[index + 1];
      double o3 = o;
      double c3 = c;
      
      // Morning Star
      bool is_bearish1 = c1 < o1;
      bool is_small2 = MathAbs(c2 - o2) < MathAbs(c1 - o1) * 0.5;
      bool is_bullish3 = c3 > o3;
      if(is_bearish1 && is_small2 && c3 > ((o1 + c1) / 2) && is_bullish3 &&
         MathMin(o2, c2) < c1 && MathMax(o2, c2) > c1)
         return "morningstar";
      
      // Evening Star
      bool is_bullish1 = c1 > o1;
      bool is_bearish3 = c3 < o3;
      if(is_bullish1 && is_small2 && c3 < ((o1 + c1) / 2) && is_bearish3 &&
         MathMax(o2, c2) > c1 && MathMin(o2, c2) < c1)
         return "eveningstar";
   }
   
   return "None";
}

//+------------------------------------------------------------------+
//| Get trade signal                                                  |
//+------------------------------------------------------------------+
bool GetTradeSignal(double &close[], double &open[], double &high[], double &low[], long &volume[], string &action, double &sl)
{
   if(!CalculateIndicators(close, open, high, low, volume, 100))
      return false;
   
   string pattern = DetectPattern(open, high, low, close, 0);
   bool bullish_patterns[] = {"hammer", "bullishengulfing", "bullishharami", "morningstar"};
   bool bearish_patterns[] = {"shootingstar", "bearishengulfing", "bearishharami", "eveningstar"};
   bool is_bullish = false;
   bool is_bearish = false;
   
   for(int i = 0; i < ArraySize(bullish_patterns); i++)
      if(pattern == bullish_patterns[i])
         is_bullish = true;
   for(int i = 0; i < ArraySize(bearish_patterns); i++)
      if(pattern == bearish_patterns[i])
         is_bearish = true;
   
   // Log to file for Streamlit monitoring
   string log_file = "xauusd_bot.log";
   FileWrite(FileOpen(log_file, FILE_WRITE|FILE_TXT|FILE_COMMON),
             StringFormat("Trend: %s, RSI: %.2f, Momentum: %.2f, Pattern: %s, Powerball: %s",
                          ema_fast[0] > ema_slow[0] ? "Up" : "Down",
                          rsi[0], momentum[0], pattern,
                          powerball_buy[0] ? "Buy" : powerball_sell[0] ? "Sell" : "None"));
   
   // Simplified signal without ML (ML not implemented in MQL5 for simplicity)
   if(ema_fast[0] > ema_slow[0] && rsi[0] < 30 && momentum[0] > 0 && is_bullish && powerball_buy[0])
   {
      action = "BUY";
      sl = atr[0] * SL_ATR_Multi;
      return true;
   }
   else if(ema_fast[0] < ema_slow[0] && rsi[0] > 70 && momentum[0] < 0 && is_bearish && powerball_sell[0])
   {
      action = "SELL";
      sl = atr[0] * SL_ATR_Multi;
      return true;
   }
   
   action = "";
   sl = 0;
   return false;
}

//+------------------------------------------------------------------+
//| Place trade                                                       |
//+------------------------------------------------------------------+
bool PlaceTrade(string action, double price, double sl, double tp)
{
   trade.SetExpertMagicNumber(123456);
   trade.SetDeviationInPoints(10);
   
   if(action == "BUY")
   {
      if(!trade.Buy(LotSize, SymbolName, price, price - sl, price * (1 + ProfitTarget)))
      {
         Print("Buy order failed: ", trade.ResultComment());
         // Log to file
         FileWrite(FileOpen("xauusd_bot.log", FILE_WRITE|FILE_TXT|FILE_COMMON),
                   StringFormat("Buy order failed: %s", trade.ResultComment()));
         return false;
      }
   }
   else if(action == "SELL")
   {
      if(!trade.Sell(LotSize, SymbolName, price, price + sl, price * (1 - ProfitTarget)))
      {
         Print("Sell order failed: ", trade.ResultComment());
         FileWrite(FileOpen("xauusd_bot.log", FILE_WRITE|FILE_TXT|FILE_COMMON),
                   StringFormat("Sell order failed: %s", trade.ResultComment()));
         return false;
      }
   }
   
   // Log to file
   FileWrite(FileOpen("xauusd_bot.log", FILE_WRITE|FILE_TXT|FILE_COMMON),
             StringFormat("Order placed: %s at %.2f, SL: %.2f, TP: %.2f",
                          action, price, price - sl, price * (1 + ProfitTarget)));
   return true;
}

//+------------------------------------------------------------------+
//| Monitor and close trades                                          |
//+------------------------------------------------------------------+
void MonitorTrades()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket) && PositionGetString(POSITION_SYMBOL) == SymbolName)
      {
         double profit = PositionGetDouble(POSITION_PROFIT);
         double volume = PositionGetDouble(POSITION_VOLUME);
         double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
         if(profit / volume >= ProfitTarget * open_price)
         {
            trade.PositionClose(ticket);
            // Log to file
            FileWrite(FileOpen("xauusd_bot.log", FILE_WRITE|FILE_TXT|FILE_COMMON),
                      StringFormat("Closed trade %d with profit: %.2f", ticket, profit));
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check for new M5 bar
   static datetime last_bar;
   datetime current_bar = iTime(SymbolName, PERIOD_M5, 0);
   if(last_bar != current_bar)
   {
      last_bar = current_bar;
      
      double close[];
      double open[];
      double high[];
      double low[];
      long volume[];
      
      string action;
      double sl;
      
      if(GetTradeSignal(close, open, high, low, volume, action, sl))
      {
         if(PositionsTotal() == 0) // No open positions
         {
            MqlTick tick;
            SymbolInfoTick(SymbolName, tick);
            double price = action == "BUY" ? tick.ask : tick.bid;
            if(PlaceTrade(action, price, sl, price * (1 + (action == "BUY" ? ProfitTarget : -ProfitTarget))))
               trade_count++;
         }
      }
      
      MonitorTrades();
      
      if(trade_count >= Retrain_Every)
      {
         // ML retraining not implemented; reset counter for simplicity
         trade_count = 0;
      }
   }
}
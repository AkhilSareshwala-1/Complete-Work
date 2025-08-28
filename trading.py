import os
import math
import time
import pandas as pd
from datetime import datetime, time as dt_time
from kiteconnect import KiteConnect
import ta

try:
    import streamlit as st
except ImportError:
    st = None

import matplotlib.pyplot as plt

# --- Zerodha API Setup --- #
api_key = "fispy0ikjo710gqt"
kite = KiteConnect(api_key=api_key)
with open("access_token.txt", "r") as f:
    kite.set_access_token(f.read().strip())

# --- User-Defined Zerodha Intraday Leverage --- #
INTRADAY_LEVERAGE = 5  # Edit as per Zerodha's current MIS leverage

# --- Stock Universe (NSE Tradingsymbols) --- #
COMPANY_SYMBOL_MAP = {
    "HDFC Bank": "HDFCBANK",
    "ICICI Bank": "ICICIBANK",
    "State Bank of India": "SBIN",
    "Kotak Mahindra Bank": "KOTAKBANK",
    "Axis Bank": "AXISBANK",
    "Infosys": "INFY",
    "TCS": "TCS",
    "Wipro": "WIPRO",
    "Tech Mahindra": "TECHM",
    "HCL Technologies": "HCLTECH",
    "Reliance Industries": "RELIANCE",
    "ONGC": "ONGC",
    "BPCL": "BPCL",
    "IOC": "IOC",
    "Bharti Airtel": "BHARTIARTL",
    "Sun Pharma": "SUNPHARMA",
    "Cipla": "CIPLA",
    "Hindustan Unilever": "HINDUNILVR",
    "ITC": "ITC",
    "Nestle India": "NESTLEIND",
    "Britannia": "BRITANNIA",
    "Dabur": "DABUR",
    "Tata Steel": "TATASTEEL",
    "JSW Steel": "JSWSTEEL",
    "Hindalco": "HINDALCO",
    "Coal India": "COALINDIA",
    "Tata Motors": "TATAMOTORS",
    "Mahindra & Mahindra": "M&M",
    "Hero MotoCorp": "HEROMOTOCO",
    "Bajaj Finance": "BAJAJFINSERV",
    "SBI Life": "SBILIFE",
    "HDFC Life": "HDFCLIFE",
    "ICICI Prudential": "ICICIPRULI",
    "NTPC": "NTPC",
    "Power Grid Corp": "POWERGRID",
    "Adani Ports": "ADANIPORTS",
    "Adani Enterprises": "ADANIENT",
    "Larsen & Toubro": "LT",
    "Grasim": "GRASIM",
    "GAIL": "GAIL",
    "Eicher Motors": "EICHERMOT",
    "Shriram Finance": "SHRIRAMFIN",
    "Trent": "TRENT",
    "Apollo Hospitals": "APOLLOHOSP",
    "Bajaj Finserv": "BAJAJFINSV",
    "Titan Company": "TITAN",
    "Asian Paints": "ASIANPAINT",
    "IndusInd Bank": "INDUSINDBK"
}

def get_instrument_token(symbol):
    for inst in kite.instruments("NSE"):
        if inst["tradingsymbol"] == symbol:
            return inst["instrument_token"]
    return None

def compute_confidence(atr, day_high, day_low, vwap, range_pct):
    confidence = 0
    if atr > 0.5:
        confidence += 1
    if 0.2 * (day_high - day_low) < atr < 1.5 * (day_high - day_low):
        confidence += 1
    if 1.0 < range_pct < 4.5:
        confidence += 1
    if day_low < vwap < day_high:
        confidence += 1
    return confidence

def fetch_intraday_data(date):
    filename = f"intraday_{date.strftime('%Y%m%d')}.csv"
    if os.path.exists(filename):
        return filename

    data = []
    for company, symbol in COMPANY_SYMBOL_MAP.items():
        try:
            token = get_instrument_token(symbol)
            if not token:
                continue
            candles = kite.historical_data(
                token,
                datetime.combine(date, dt_time(9, 15)),
                datetime.combine(date, dt_time(11, 30)),
                "minute"
            )
            df = pd.DataFrame(candles)
            if df.empty: continue
            df['VWAP'] = (df['high'] + df['low'] + df['close']) / 3
            df['ATR3'] = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'], 3
            ).average_true_range()
            day_high = df['high'].max()
            day_low = df['low'].min()
            vwap = df['VWAP'].iloc[-1]
            atr = df['ATR3'].iloc[-1]
            range_pct = (day_high - day_low) / day_low * 100 if day_low > 0 else 0
            confidence = compute_confidence(atr, day_high, day_low, vwap, range_pct)
            buy_level = round(vwap - 0.3 * atr - 1, 2)
            sell_level = round(vwap + 0.4 * atr + 1, 2)
            if confidence >= 3:
                data.append({
                    'Symbol': symbol,
                    'Company': company,
                    'Buy_Level': buy_level,
                    'Sell_Level': sell_level,
                    'VWAP': round(vwap, 2),
                    'ATR': round(atr, 2),
                    'Day_High': round(day_high, 2),
                    'Day_Low': round(day_low, 2),
                    'Range_Pct': round(range_pct, 2),
                    'Confidence_Score': confidence,
                    'Avg_Close': round(df["close"].mean(), 2),
                    'Avg_High': round(df["high"].mean(), 2),
                    'Avg_Low': round(df["low"].mean(), 2),
                    'Avg_VWAP': round(df["VWAP"].mean(), 2),
                    'Avg_ATR3': round(df["ATR3"].mean(), 2),
                    'Last_Close': round(df["close"].iloc[-1], 2)
                })
            time.sleep(0.1)
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
    pd.DataFrame(data).to_csv(filename, index=False)
    return filename

def check_level_hits(signal_file, date):
    df = pd.read_csv(signal_file)
    results = []
    for _, row in df.iterrows():
        try:
            token = get_instrument_token(row['Symbol'])
            candles = kite.historical_data(
                token,
                datetime.combine(date, dt_time(11, 30)),
                datetime.combine(date, dt_time(15, 30)),
                "minute"
            )
            highs = [c['high'] for c in candles]
            lows = [c['low'] for c in candles]
            buy_hit = any(l <= row['Buy_Level'] <= h for l, h in zip(lows, highs))
            sell_hit = any(l <= row['Sell_Level'] <= h for l, h in zip(lows, highs))
            results.append({
                **row,
                'Buy_Hit': '✅' if buy_hit else '❌',
                'Sell_Hit': '✅' if sell_hit else '❌'
            })
            time.sleep(0.1)
        except:
            continue
    result_file = f"results_{date.strftime('%Y%m%d')}.csv"
    pd.DataFrame(results).to_csv(result_file, index=False)
    return result_file

def portfolio_optimizer(df, capital, min_stocks=3, max_stocks=7, leverage=INTRADAY_LEVERAGE):
    df = df.copy()
    df['Potential_Profit'] = df['Sell_Level'] - df['Buy_Level']
    # Filter: Only consider stocks below ₹3500
    df = df[df['Buy_Level'] < 3500]
    df = df.sort_values(['Confidence_Score', 'Potential_Profit'], ascending=[False, False]).reset_index(drop=True)
    best = df.head(max_stocks)
    chosen = []
    spent = 0.0
    for idx, row in best.iterrows():
        if len(chosen) >= max_stocks:
            break
        price = row['Buy_Level']
        if price <= 0:
            continue
        margin_price = price / leverage
        max_possible = math.floor((capital - spent) / margin_price) if (capital - spent) > 0 else 0
        if max_possible > 0:
            chosen.append({
                **row,
                'Qty': max_possible,
                'Total_Buy': price * max_possible,
                'Margin_Used': margin_price * max_possible  # For dashboard display
            })
            spent += margin_price * max_possible
    if len(chosen) < min_stocks and (len(df) >= min_stocks):
        remaining_df = df.loc[~df.index.isin([c['Symbol'] for c in chosen])]
        for idx, row in remaining_df.iterrows():
            if len(chosen) >= min_stocks:
                break
            price = row['Buy_Level']
            if price <= 0:
                continue
            margin_price = price / leverage
            if spent + margin_price <= capital:
                chosen.append({
                    **row,
                    'Qty': 1,
                    'Total_Buy': price,
                    'Margin_Used': margin_price
                })
                spent += margin_price
    return pd.DataFrame(chosen), spent

def calculate_weighted_profit(portfolio_df, actual_df):
    profit = 0.0
    for _, row in portfolio_df.iterrows():
        hit_row = actual_df[actual_df['Symbol'] == row['Symbol']].iloc[0]
        qty = row['Qty']
        if hit_row['Buy_Hit'] == '✅':
            if hit_row['Sell_Hit'] == '✅':
                profit += (hit_row['Sell_Level'] - hit_row['Buy_Level']) * qty
            else:
                profit -= qty  # Penalty when buy triggers, but sell doesn't
    return round(profit, 2)

def run_full_analysis(date=None, invest_amt=50000):
    MIN_STOCKS = 3
    MAX_STOCKS = 7
    date = pd.to_datetime(date).date() if date else datetime.now().date()
    intraday_file = fetch_intraday_data(date)
    if os.path.getsize(intraday_file) == 0:
        return pd.DataFrame(), pd.DataFrame(), 0, date, ""
    signal_file = intraday_file
    result_file = check_level_hits(signal_file, date)
    all_trades = pd.read_csv(result_file)
    portfolio, spent = portfolio_optimizer(all_trades, invest_amt, MIN_STOCKS, MAX_STOCKS)
    if portfolio.empty:
        return all_trades, portfolio, 0, spent, date, result_file
    profit = calculate_weighted_profit(portfolio, all_trades)
    return all_trades, portfolio, profit, spent, date, result_file

def launch_dashboard():
    st.title("VWAP+ATR Intraday Optimized Portfolio Backtest (with Intraday Leverage)")
    st.markdown(
        "Select a date, investment amount, and click 'Submit' to process. "
        "Only stocks with buy level under ₹3500 and allocation based on Zerodha intraday leverage will be considered."
    )
    with st.form(key='date_form'):
        d = st.date_input("Select Backtest Date", value=datetime.now().date())
        capital = st.number_input("Capital to Invest (₹)", value=50000, step=1000, min_value=1000)
        submit_clicked = st.form_submit_button("Submit")

    if submit_clicked:
        with st.spinner("Running optimized portfolio backtest..."):
            all_trades, portfolio, profit, spent, date_used, results_file = run_full_analysis(str(d), capital)
        st.subheader(f"Date: {date_used}")
        if portfolio.empty:
            st.error("No stocks could be bought under current logic—try lowering minimum stocks or increasing capital.")
            return
        st.metric("Total Weighted Portfolio Profit/Loss", f"₹{profit}")
        st.metric("Total Margin Utilized", f"₹{round(spent,2)}")
        st.metric("Unused Capital", f"₹{round(capital-spent,2)}")
        st.markdown("### Selected Portfolio (Top Confidence & Potential, Leverage Applied)")
        st.dataframe(
            portfolio[['Symbol','Company','Qty','Buy_Level','Sell_Level',
                       'Confidence_Score','Potential_Profit','Buy_Hit',
                       'Sell_Hit','Total_Buy','Margin_Used']],
            use_container_width=True
        )

        st.markdown("### All Qualifying Stocks (Confidence Score ≥ 3)")
        st.dataframe(all_trades, use_container_width=True)

        st.markdown("### Profit Distribution for Portfolio")
        portfolio['Stock_Profit'] = portfolio.apply(
            lambda row: (row['Sell_Level']-row['Buy_Level'])*row['Qty'] if row['Buy_Hit']=='✅' and row['Sell_Hit']=='✅'
            else -row['Qty'] if row['Buy_Hit']=='✅' and row['Sell_Hit']!='✅'
            else 0, axis=1
        )
        fig, ax = plt.subplots()
        portfolio['Stock_Profit'].plot(kind='bar', color='green', ax=ax)
        ax.set_title('Per-Stock Profit from Allocated Portfolio')
        ax.set_ylabel('Profit (₹)')
        ax.set_xlabel('Stock')
        st.pyplot(fig)

        st.download_button(
            label="Download Portfolio as CSV",
            data=portfolio.to_csv(index=False),
            file_name=f"Optimized_Portfolio_{date_used.strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
        st.write("---")
        st.caption(
            "Backtest, portfolio optimization and dashboard by Your Name, powered by Streamlit."
        )

if __name__ == "__main__":
    launch_dashboard()

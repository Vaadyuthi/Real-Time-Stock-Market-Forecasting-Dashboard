from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly
import streamlit as st
import yfinance as yf

st.set_page_config(layout="wide", page_title="Real Time Stock Market & Prophet Forecasting Dashboard")

def fetch_stock_data(ticker, period=None, interval='1m', start=None, end=None):
    """Fetches market data from yfinance and flattens MultiIndex columns."""
    try:
        if start and end:
            data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        else:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data
    except Exception:
        return pd.DataFrame()

def process_data(data):
    """Converts timezones and normalizes column headers."""
    if data.empty:
        return data
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    
    if 'Date' in data.columns:
        data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data

@st.cache_data
def load_historical_data(ticker, start_date, end_date):
    """Cached function to fetch daily historical data for forecasting."""
    data = fetch_stock_data(ticker, start=start_date, end=end_date, interval='1d')
    if not data.empty:
        data.reset_index(inplace=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
    return data

@st.cache_data
def get_company_info(ticker):
    """Fetches company profile details and business summary from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info
        return info
    except Exception:
        return {}

st.title('Real Time Stock Market & Forecasting Dashboard')

st.sidebar.header('Chart & Model Parameters')

all_stocks = (
    "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX", "AMD", "INTC", "ORCL", "CRM", "AVGO", "CSCO",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA",
    "WMT", "COST", "PG", "JNJ", "PFE", "UNH", "KO", "PEP", "DIS", "BA", "CAT",
    "SPY", "QQQ", "DIA", "IWM", "VOO", "^GSPC", "^IXIC", "^DJI",
    "BTC-USD", "ETH-USD", "SOL-USD", "EURUSD=X", "GBPUSD=X"
)

selected_stock = st.sidebar.selectbox("Search / Select stock or asset:", all_stocks)
custom_stock = st.sidebar.text_input("Or enter any custom ticker (e.g., RELIANCE.NS, BABA, SHOP):", "").upper()

if custom_stock.strip():
    selected_stock = custom_stock.strip()

n_years = st.sidebar.slider("Years of forecast prediction:", 1, 5, 2)
period = n_years * 365

st.header('Real-Time Market Snapshot')

stock_symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'NVDA', 'TSLA']
cols = st.columns(len(stock_symbols))

for idx, symbol in enumerate(stock_symbols):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    real_time_data = fetch_stock_data(symbol, start=start_date, end=end_date, interval='1m')
    
    if not real_time_data.empty:
        real_time_data = process_data(real_time_data)
        
        last_price = float(real_time_data['Close'].iloc[-1])
        open_price = float(real_time_data['Open'].iloc[0])
        change = last_price - open_price
        pct_change = (change / open_price) * 100
        
        cols[idx].metric(
            label=symbol, 
            value=f"${last_price:.2f}", 
            delta=f"{change:.2f} ({pct_change:.2f}%)"
        )

st.markdown("---")

st.header(f"Analysis for: {selected_stock}")

comp_info = get_company_info(selected_stock)

if comp_info:
    company_name = comp_info.get('longName', selected_stock)

START_DATE = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

data_load_state = st.text("Loading historical data...")
hist_data = load_historical_data(selected_stock, START_DATE, TODAY)

if hist_data.empty:
    st.error(f"Could not fetch chart data for '{selected_stock}'. Please verify the ticker symbol.")
else:
    data_load_state.text("Loading historical data... Done!")

    hist_data['SMA_20'] = hist_data['Close'].rolling(window=20).mean()
    hist_data['SMA_50'] = hist_data['Close'].rolling(window=50).mean()

    m1, m2, m3, m4 = st.columns(4)
    latest_close = float(hist_data['Close'].iloc[-1])
    high_52 = float(hist_data['High'].tail(252).max())
    low_52 = float(hist_data['Low'].tail(252).min())
    total_volume = int(hist_data['Volume'].iloc[-1])

    m1.metric("Current Price", f"${latest_close:.2f}")
    m2.metric("52-Week High", f"${high_52:.2f}")
    m3.metric("52-Week Low", f"${low_52:.2f}")
    m4.metric("Latest Volume", f"{total_volume:,}")

    st.subheader('Raw Data (Latest Records)')
    st.write(hist_data.tail())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_data['Date'], y=hist_data['Open'], name="Stock Open", line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=hist_data['Date'], y=hist_data['Close'], name="Stock Close", line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=hist_data['Date'], y=hist_data['SMA_20'], name="20-Day SMA", line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=hist_data['Date'], y=hist_data['SMA_50'], name="50-Day SMA", line=dict(color='magenta')))
    
    fig.update_layout(
        title_text=f'{selected_stock} Time Series Data with Range Slider & Technical Indicators',
        xaxis_rangeslider_visible=True,
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header(f'Prophet Price Forecast ({n_years} Year{"s" if n_years > 1 else ""})')

    df_train = hist_data[['Date', 'Close']].copy()
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
    df_train.dropna(inplace=True)

    with st.spinner("Training time-series forecasting model..."):
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

    st.subheader('Forecast Tail Data')
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    st.subheader('Interactive Forecast Plot')
    fig_forecast = plot_plotly(m, forecast)
    fig_forecast.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.subheader('Forecast Trends & Seasonal Components')
    fig_components = m.plot_components(forecast)
    st.pyplot(fig_components)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os

# --- Configuration ---
FACTOR_TICKERS = {
    "Value": "VLUE",
    "Momentum": "MTUM",
    "Quality": "QUAL",
    "Size": "SIZE",
    "Low Volatility": "USMV",
    "Growth": "IVW"
}
BENCHMARK = "SPY"
ALL_TICKERS = list(FACTOR_TICKERS.values()) + [BENCHMARK]
COLORS = px.colors.qualitative.Plotly

def setup_page():
    st.set_page_config(
        page_title="US Factor Momentum Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Custom CSS for dark mode aesthetics
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Data Fetching ---
# --- Data Fetching ---
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

@st.cache_data(ttl=3600*12)  # Cache for 12 hours
def fetch_data(tickers, lookback_years=2):
    """
    Fetch Adjusted Close data for the given tickers.
    Uses local CSV caching:
    - Checks data/{ticker}.csv
    - If exists, fetches only new data since last date in CSV.
    - If not exists, fetches full history.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    end_date = datetime.today()
    # Ensure end_date includes today by setting time to end of day? 
    # yfinance usually handles "today" implies up to now or last close.
    # Let's keep strict date objects for comparison to ease logic.
    today_date = end_date.date()
    
    global_start_date = end_date - timedelta(days=lookback_years * 365 + 30) # Buffer
    
    # 1. Determine fetch requirements per ticker
    fetch_plan = {} # start_date -> [tickers]
    existing_data = {} # ticker -> dataframe
    
    for ticker in tickers:
        file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
        start_date = global_start_date
        
        if os.path.exists(file_path):
            try:
                # Read CSV
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # Standardize column name internally to ticker for compatibility
                if 'adj_close' in df.columns:
                    df = df.rename(columns={'adj_close': ticker})
                elif len(df.columns) == 1:
                     # Fallback if migration hasn't happened yet for some reason
                     df.columns = [ticker]
                
                existing_data[ticker] = df
                
                if not df.empty:
                    last_date = df.index[-1].date()
                    if last_date < today_date:
                         # Need new data from last_date + 1 day
                         # BUT if last_date + 1 == today, and market is open/closed, we try to fetch today.
                         # If today is weekend or no data out yet, yf might fail or return empty.
                         start_date_candidate = datetime.combine(last_date + timedelta(days=1), datetime.min.time())
                         # Only fetch if start_date <= today. (Equality means we try to fetch today's data)
                         if start_date_candidate.date() <= today_date:
                            start_date = start_date_candidate
                         else:
                            continue
                    else:
                        # Up to date
                        continue 
            except Exception as e:
                st.error(f"Error reading {file_path}: {e}")
                # Fallback to full download
                existing_data.pop(ticker, None)
        
        # Add to fetch plan
        # We use strict timestamp for Grouping
        # If start_date is datetime, use it. if it's based on lookback (which is datetime), ok.
        
        # Fix: ensure start_date is datetime object for grouping
        if isinstance(start_date, datetime):
            s_key = start_date
        else:
             # Should be datetime based on logic above
             s_key = start_date
             
        if s_key not in fetch_plan:
            fetch_plan[s_key] = []
        fetch_plan[s_key].append(ticker)

    # 2. Fetch Data in Groups
    # Refactored batch fetcher that takes specific list
    
    for start_dt, group_tickers in fetch_plan.items():
        # Rate limit per batch of 2 within the group
        batch_size = 2
        delay = 1.0
        
        for i in range(0, len(group_tickers), batch_size):
            batch = group_tickers[i:i+batch_size]
            try:
                # Check if start_dt is in future or today?
                if start_dt.date() > today_date:
                    continue
                    
                # yf.download
                # Note: start is inclusive.
                b_data = yf.download(batch, start=start_dt, end=end_date, progress=False, threads=True)
                
                if not b_data.empty:
                    # Clean data structure
                    if 'Adj Close' in b_data.columns.get_level_values(0):
                         process_df = b_data['Adj Close']
                    elif 'Close' in b_data.columns.get_level_values(0):
                         process_df = b_data['Close']
                    else:
                        # If single ticker result, columns are not MultiIndex in same way usually?
                        # yfinance usually returns MultiIndex if list passed, even size 1?
                        # If size 1, it might just be columns: Open, High... 
                        # Let's standardize.
                        if len(batch) == 1:
                            # It returns DataFrame with columns Open, Close etc.
                             if 'Adj Close' in b_data.columns:
                                 process_df = pd.DataFrame({batch[0]: b_data['Adj Close']})
                             elif 'Close' in b_data.columns:
                                 process_df = pd.DataFrame({batch[0]: b_data['Close']})
                             else:
                                 process_df = pd.DataFrame()
                        else:
                             # MultiIndex case (handled above mostly, but double check)
                             process_df = pd.DataFrame() # Should have been caught by if 'Adj Close' check?
                             # Re-eval:
                             if isinstance(b_data.columns, pd.MultiIndex):
                                  if 'Adj Close' in b_data.columns.get_level_values(0):
                                       process_df = b_data['Adj Close']
                                  elif 'Close' in b_data.columns.get_level_values(0):
                                       process_df = b_data['Close']
                             else:
                                  # Should not happen with list input unless flattened?
                                  st.write(f"Unexpected columns for {batch}: {b_data.columns}")
                                  continue

                    # Update CSVs
                    for t in batch:
                        if t in process_df.columns:
                            new_series = process_df[t].dropna()
                            if not new_series.empty:
                                new_df = pd.DataFrame(new_series)
                                new_df.columns = [t]
                                
                                # Combine with existing
                                if t in existing_data:
                                    # Append
                                    combined = pd.concat([existing_data[t], new_df])
                                    combined = combined[~combined.index.duplicated(keep='last')]
                                    existing_data[t] = combined
                                else:
                                    existing_data[t] = new_df
                                    
                                # Save with generic format
                                save_df = existing_data[t].copy()
                                save_df.columns = ['adj_close']
                                save_df.index.name = 'date'
                                
                                file_path = os.path.join(DATA_DIR, f"{t}.csv")
                                save_df.to_csv(file_path)
                                
            except Exception as e:
                st.error(f"Error fetching batch {batch}: {e}")
            
            time.sleep(delay)

    # 3. Assemble Final DataFrame
    final_df_list = []
    for t in tickers:
        if t in existing_data:
            # Rename column to t if not already (it should be)
            d = existing_data[t]
            # Ensure column is named t
            if d.shape[1] == 1:
                d.columns = [t]
            final_df_list.append(d)
            
    if not final_df_list:
        return pd.DataFrame()
        
    return pd.concat(final_df_list, axis=1).ffill().dropna()

# --- Calculations ---
def calculate_momentum(df, window_12m_exclude=21):
    """
    Calculate momentum scores.
    12M-1M: 12-month return excluding last 1 month (approx 21 days).
    Standard Momentum (12-1): P(t-21) / P(t-252) - 1
    Snapshots: 1M (21d), 3M (63d), 6M (126d)
    """
    mom_data = pd.DataFrame(index=df.index)
    
    # Standard 12M-1M Momentum
    # Return from t-252 to t-21
    # Mom_12_1 = Price(t-21) / Price(t-252) - 1
    # We can shift the series to vectorize
    
    lag_1m = 21
    lag_12m = 252
    
    # Ensure sufficient data
    if len(df) < lag_12m:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # 12M-1M Momentum
    # P_t-21
    p_lag_1m = df.shift(lag_1m)
    # P_t-252
    p_lag_12m = df.shift(lag_12m)
    
    momentum_12_1 = (p_lag_1m / p_lag_12m) - 1
    
    # Snapshot momentums (Simple returns)
    mom_1m = df.pct_change(21)
    mom_3m = df.pct_change(63)
    mom_6m = df.pct_change(126)
    mom_12m = df.pct_change(252)
    
    return momentum_12_1, mom_1m, mom_3m, mom_6m, mom_12m

def calculate_rs(df, benchmark_col, normalization_window=None):
    """
    Calculate Relative Strength vs Benchmark.
    RS = Price_Factor / Price_Benchmark
    Normalized to start at 1.0.
    """
    rs_df = pd.DataFrame(index=df.index)
    benchmark_price = df[benchmark_col]
    
    for col in df.columns:
        if col != benchmark_col:
            ratio = df[col] / benchmark_price
            # Normalize
            if normalization_window:
                # Normalize based on the first valid index in the window
                start_val = ratio.iloc[-normalization_window]
                rs_df[col] = ratio / start_val
            else:
                # Normalize based on the very first value available
                rs_df[col] = ratio / ratio.iloc[0]
                
    return rs_df

def calculate_zscore(momentum_series, window=252):
    """
    Calculate rolling Z-Score of the momentum series.
    Z = (Value - Mean) / StdDev
    """
    rolling_mean = momentum_series.rolling(window=window).mean()
    rolling_std = momentum_series.rolling(window=window).std()
    z_score = (momentum_series - rolling_mean) / rolling_std
    return z_score

# --- Visualizations ---
def plot_rs_chart(rs_df, window):
    """
    Plot Relative Strength Line Chart.
    """
    # Slice last 'window' days
    if window:
        plot_data = rs_df.iloc[-window:]
        # Renormalize to 1.0 at start of plot
        plot_data = plot_data.div(plot_data.iloc[0])
    else:
        plot_data = rs_df

    fig = px.line(plot_data, title="Relative Strength vs SPY (Normalized to 1.0)",
                  labels={"value": "Relative Strength", "index": "Date", "variable": "Factor"},
                  color_discrete_sequence=COLORS)
    fig.update_layout(template="plotly_dark", hovermode="x unified")
    return fig

def plot_momentum_heatmap(current_ranks):
    """
    Plot Ranking Heatmap.
    current_ranks: DataFrame with factors as index and Timeframes as columns. 
    Values are Ranks (1=Best).
    """
    # Inverse ranks for color intensity (High rank #1 should be bright/green, Low rank #6 dark/red)
    # Actually, let's just color by rank.
    
    fig = go.Figure(data=go.Heatmap(
        z=current_ranks.values,
        x=current_ranks.columns,
        y=current_ranks.index,
        colorscale='Viridis_r', # Reverse Viridis so 1 (High Rank) is brightest
        text=current_ranks.values,
        texttemplate="%{text}",
        showscale=False
    ))
    fig.update_layout(title="Factor Momentum Rank (1 = Best)", template="plotly_dark")
    return fig

def plot_zscore_bar(z_scores):
    """
    Plot Bar Chart of Current Z-Scores.
    """
    # z_scores is a Series: index=Ticker, value=Z-Score
    
    # Color condition
    colors = ['#ef5350' if x < -1.5 else '#66bb6a' if x > 1.5 else '#42a5f5' for x in z_scores.values]
    
    fig = go.Figure(data=[go.Bar(
        x=z_scores.index,
        y=z_scores.values,
        marker_color=colors,
        text=z_scores.values.round(2),
        textposition='auto'
    )])
    fig.update_layout(title="Current Momentum Z-Score (Rolling 1Y)", 
                      yaxis_title="Z-Score",
                      template="plotly_dark")
    fig.add_hline(y=1.5, line_dash="dash", line_color="green", annotation_text="Overbought (+1.5)")
    fig.add_hline(y=-1.5, line_dash="dash", line_color="red", annotation_text="Oversold (-1.5)")
    return fig

# --- Main App ---
def main():
    setup_page()
    
    # Sidebar
    st.sidebar.title("Configuration")
    lookback_days = st.sidebar.slider("Chart Lookback (Days)", 30, 756, 378, help="Number of trading days to show on RS chart.")
    
    # Fetch Data
    with st.spinner("Fetching market data..."):
        prices = fetch_data(ALL_TICKERS)
        
    if prices.empty:
        st.stop()
        
    # Process Data
    # Calculate Moments
    mom_12_1, mom_1m, mom_3m, mom_6m, mom_12m = calculate_momentum(prices)
    
    # Map Tickers to Factor Names for Display
    # Create mapping: {TICKER: Name}
    ticker_to_name = {v: k for k, v in FACTOR_TICKERS.items()}
    
    # Helper to clean column names if needed (yfinance MultiIndex often)
    # If MultiIndex (Price, Ticker), yf.download(..., group_by='column') gives simple columns if 1 price type
    # Current call returns 'Adj Close' only, so columns are Tickers.
    
    # Calculate RS
    rs_df = calculate_rs(prices, BENCHMARK, normalization_window=lookback_days)
    rs_df_disp = rs_df.rename(columns=ticker_to_name)
    
    # Current Momentum Values (Last available row)
    # Ensure we use valid data (dropna)
    last_idx = mom_12_1.dropna().index[-1]
    
    cur_12_1 = mom_12_1.loc[last_idx]
    cur_1m = mom_1m.loc[last_idx]
    cur_3m = mom_3m.loc[last_idx]
    cur_6m = mom_6m.loc[last_idx]
    
    # Create Metrics DataFrame
    metrics = pd.DataFrame({
        "1M": cur_1m,
        "3M": cur_3m,
        "6M": cur_6m,
        "12M-1M": cur_12_1
    })
    # Filter only Factor Tickers (exclude SPY if it's there)
    factor_tickers_list = list(FACTOR_TICKERS.values())
    metrics = metrics.loc[[t for t in factor_tickers_list if t in metrics.index]]
    metrics.index = metrics.index.map(ticker_to_name)
    
    # Rank (Higher return = Rank 1)
    ranks = metrics.rank(ascending=False, method='min')
    
    # Z-Scores Calculation (on 12M-1M series)
    z_scores_df = calculate_zscore(mom_12_1)[factor_tickers_list]
    current_z_score = z_scores_df.iloc[-1]
    current_z_score.index = current_z_score.index.map(ticker_to_name)
    
    # --- UI Layout ---
    st.title("Factor Momentum Dashboard")
    st.markdown(f"**Data Date:** {last_idx.strftime('%Y-%m-%d')}")
    
    # Top Level KPI: Best Factor (12M-1M)
    best_factor = ranks['12M-1M'].idxmin()
    best_val_12_1 = metrics.loc[best_factor, '12M-1M']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Top Momentum Factor (12M-1M)", best_factor, f"{best_val_12_1:.1%}")
    col2.metric("Benchmark (SPY) Price", f"${prices[BENCHMARK].iloc[-1]:.2f}")
    
    st.divider()
    
    # Charts
    tab1, tab2 = st.tabs(["Relative Strength", "Momentum Analysis"])
    
    with tab1:
        st.plotly_chart(plot_rs_chart(rs_df_disp, lookback_days), use_container_width=True)
        st.markdown("*Relative Strength is calculated as Factor Price / SPY Price, rebased to 1.0 at start of period.*")

    with tab2:
        col_hm, col_bar = st.columns([1, 1])
        
        with col_hm:
            st.subheader("Momentum Ranks")
            st.plotly_chart(plot_momentum_heatmap(ranks), use_container_width=True)
            st.dataframe(metrics.style.format("{:.1%}"), use_container_width=True)

        with col_bar:
            st.subheader("Momentum Z-Scores (12M-1M)")
            st.plotly_chart(plot_zscore_bar(current_z_score), use_container_width=True)
            st.caption("Z-Score measures how many standard deviations the current momentum is from its 1-year average. Extremes (>1.5 or <-1.5) may indicate mean reversion risks.")

if __name__ == "__main__":
    main()


import sys
import os
import pandas as pd
# Add current directory to path so we can import app
sys.path.append(os.getcwd())

try:
    from app import fetch_data, calculate_momentum, calculate_rs, calculate_zscore, ALL_TICKERS, FACTOR_TICKERS
    print("Import successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_logic():
    print("Testing Data Fetching...")
    # Fetch small amount of data to test
    # We mock streamlit cache if needed or just let it run (it might complain about streamlit not running, so we might need to mock st.cache_data)
    # Since we imported from app, and app uses decorators, it might fail if not in streamlit context?
    # Actually st.cache_data usually works fine in script but warns or just runs.
    # Exception: st commands like st.error might fail if called.
    
    # Prerequisite: Check if we can bypass the streamlit specific parts or just try.
    # To be safe, let's just try-catch the fetch.
    try:
        df = fetch_data(ALL_TICKERS, lookback_years=2) 
        if df.empty:
            print("Data fetching returned empty. Using dummy data.")
            raise ValueError("Empty Data")
    except Exception as e:
        print(f"Fetch data failed (likely due to streamlit context or network): {e}")
        # Use dummy data for calculation verification
        print("Using dummy data for logic verification.")
        dates = pd.date_range("2023-01-01", periods=300)
        data = {t: pd.Series(range(300), index=dates) + (i*10) for i, t in enumerate(ALL_TICKERS)}
        df = pd.DataFrame(data)

    print(f"Data Shape: {df.shape}")
    
    print("Testing Momentum Calculation...")
    vals = calculate_momentum(df)
    if isinstance(vals, tuple):
        print("Momentum calculation returned tuple (Expected).")
        # Check shapes
        if not vals[0].empty:
             print("12M-1M Momentum has data.")
    else:
        print("Momentum calculation failed result format.")

    print("Testing RS Calculation...")
    rs = calculate_rs(df, "SPY")
    if not rs.empty:
        print("RS Calculation successful.")

    print("Testing Z-Score...")
    z = calculate_zscore(vals[0])
    if not z.empty:
        print("Z-Score Calculation successful.")
    
    print("Verification Complete.")

if __name__ == "__main__":
    test_logic()

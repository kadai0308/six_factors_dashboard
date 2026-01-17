# US Factor Momentum Dashboard

A Streamlit application that monitors and visualizes the momentum of 6 key US investment factors using ETF proxies.

## Factors Tickers
- **Value**: `VLUE`
- **Momentum**: `MTUM`
- **Quality**: `QUAL`
- **Size**: `SIZE`
- **Low Volatility**: `USMV`
- **Growth**: `IVW`
- **Benchmark**: `SPY`

## Features
- **Relative Strength (RS)**: Compares Factor ETF prices vs. SPY (normalized to 1.0).
- **Momentum ranking**: Ranks factors based on 1M, 3M, 6M, and 12M-1M returns.
- **Z-Score**: Identifies extreme momentum readings (Overbought/Oversold) using a 1-year rolling window.

## Installation

1.  **Prerequisites**: Python 3.10+
2.  **Install Dependencies**:

```bash
pip install streamlit pandas numpy plotly yfinance scipy
```

## Usage

Run the application:

```bash
streamlit run app.py
```

## Verification

You can run the included test script to verify data fetching and calculation logic:

```bash
python test_app.py
```

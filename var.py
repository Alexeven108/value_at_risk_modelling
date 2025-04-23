import streamlit as st
import numpy as np
from scipy.stats import norm
from var_garch_egarch import download_returns, garch_var, egarch_var

# Page setup
st.set_page_config(page_title="Value at Risk Dashboard", layout="centered")
st.title("📊 Value at Risk Dashboard (GARCH/EGARCH Enhanced)")

# Sidebar Inputs
st.sidebar.header("User Input")
popular_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "JPM", "BAC", "NFLX"]
ticker = st.sidebar.selectbox("Choose a stock", popular_stocks)
confidence = st.sidebar.slider("Confidence Level", 0.90, 0.99, 0.95)
investment = st.sidebar.number_input("Investment Amount ($)", min_value=1000, value=1_000_000, step=1000)

# Main Button and Output
if st.button("💡 Calculate VaR"):
    try:
        returns = download_returns(ticker)

        # Historical VaR
        hist_var = -np.percentile(returns, (1 - confidence) * 100)

        # Parametric VaR (Normal distribution)
        mu, sigma = returns.mean(), returns.std()
        param_var = -norm.ppf(1 - confidence, mu, sigma)

        # GARCH & EGARCH VaR
        garch = garch_var(returns, confidence)
        egarch = egarch_var(returns, confidence)

        st.subheader(f"📉 1-Day Value at Risk for {ticker.upper()}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Historical VaR", f"${hist_var * investment / 100:,.2f}")
            st.metric("GARCH VaR", f"${garch * investment / 100:,.2f}")
        with col2:
            st.metric("Parametric VaR", f"${param_var * investment / 100:,.2f}")
            st.metric("EGARCH VaR", f"${egarch * investment / 100:,.2f}")

    except ValueError as e:
        st.error(f"⚠️ {str(e)}")
    except Exception as e:
        st.exception(e)

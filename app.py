import streamlit as st
from var_garch_egarch import download_returns, garch_var, egarch_var
import numpy as np
from scipy.stats import norm

st.set_page_config(page_title="Value at Risk Dashboard", layout="centered")
st.title("📊 Value at Risk Dashboard (GARCH/EGARCH Enhanced)")

popular_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "JPM", "BAC", "NFLX"]
ticker = st.selectbox(
    "Choose a stock",
    popular_stocks,
    help="Choose from the top traded stocks on NASDAQ/NYSE"
)
confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95)
investment = st.number_input("Investment Amount", min_value=1000, value=1_000_000)

if st.button("Calculate VaR"):
  try:
    returns = download_returns(ticker)

    # Historical
    hist_var = -np.percentile(returns, (1 - confidence) * 100)

    # Parametric
    mu, sigma = returns.mean(), returns.std()
    param_var = -norm.ppf(1 - confidence, mu, sigma)

    # GARCH & EGARCH
    garch = garch_var(returns, confidence)
    egarch = egarch_var(returns, confidence)

    st.subheader(f"1-Day VaR Results for {ticker.upper()}")
    st.metric("Historical VaR", f"${hist_var * investment / 100:,.2f}")
    st.metric("Parametric VaR", f"${param_var * investment / 100:,.2f}")
    st.metric("GARCH VaR", f"${garch * investment / 100:,.2f}")
    st.metric("EGARCH VaR", f"${egarch * investment / 100:,.2f}")

  except ValueError as e:
      st.error(str(e))
  except Exception as e:
      st.exception(e)
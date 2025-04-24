# 📊 Value at Risk Dashboard (GARCH/EGARCH Enhanced)

This Streamlit app provides an interactive dashboard to calculate the **1-Day Value at Risk (VaR)** for selected stocks using multiple models:

- 📈 **Historical VaR**
- 📊 **Parametric VaR** (Normal distribution assumption)
- 🔁 **GARCH(1,1) VaR**
- 🔀 **EGARCH(1,1) VaR**

> Designed for finance students, risk managers, and quants who want quick yet advanced insight into stock risk.

---

## 🚀 Features

- ✅ Choose from top tickers: `AAPL`, `MSFT`, `TSLA`, `GOOGL`, etc.
- 🎚 Set confidence levels from **90% to 99%**
- 💰 Input custom investment amounts
- 📉 Calculate 1-day VaR using:
  - Empirical distribution
  - Gaussian assumptions
  - GARCH/EGARCH volatility models
- 📊 Side-by-side metric comparison
- 🛡️ Error handling for invalid inputs and model failures

---

## 🛠 How It Works

```python
# 1. Get historical returns
returns = download_returns(ticker)

# 2. Calculate VaR
hist_var = -np.percentile(returns, (1 - confidence) * 100)
param_var = -norm.ppf(1 - confidence, returns.mean(), returns.std())
garch = garch_var(returns, confidence)
egarch = egarch_var(returns, confidence)
```
## Requirements

Install the necessary packages with:

- pip install streamlit numpy scipy arch pandas yfinance



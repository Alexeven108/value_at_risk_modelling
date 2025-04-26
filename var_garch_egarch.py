import numpy as np
import yfinance as yf
from arch import arch_model
from scipy.stats import norm


def download_returns(ticker="AAPL", period="1y"):
    try:
        # Avoid 'group_by' and directly download data
        data = yf.download(ticker, period=period, auto_adjust=True)

        # Check if 'Adj Close' exists, otherwise fall back to 'Close'
        if 'Adj Close' in data.columns:
            close = data['Adj Close']
        elif 'Close' in data.columns:
            close = data['Close']
        else:
            raise ValueError(f"[DEBUG] No 'Adj Close' or 'Close' found for {ticker}")

        # Calculate log returns
        returns = 100 * np.log(close / close.shift(1)).dropna()
        return returns
    except Exception as e:
        raise ValueError(f"⚠️ No valid data returned for '{ticker}'. Error: {str(e)}")


def garch_var(returns, level=0.99):
    model = arch_model(returns, vol='GARCH', p=1, q=1)
    res = model.fit(disp="off")
    forecast = res.forecast(horizon=1)
    sigma = np.sqrt(forecast.variance.values[-1][0])
    var = norm.ppf(1 - level) * sigma
    return -var

def egarch_var(returns, level=0.99):
    model = arch_model(returns, vol='EGARCH', p=1, q=1)
    res = model.fit(disp="off")
    forecast = res.forecast(horizon=1)
    sigma = np.sqrt(forecast.variance.values[-1][0])
    var = norm.ppf(1 - level) * sigma
    return -var

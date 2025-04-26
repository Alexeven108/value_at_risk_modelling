import numpy as np
import yfinance as yf
from arch import arch_model
from scipy.stats import norm


def download_returns(ticker: str = "AAPL", period: str = "1y") -> np.ndarray:
    try:
        data = yf.download(ticker, period=period, auto_adjust=True)
        close = data['Close']
        returns = 100 * np.log(close / close.shift(1)).dropna()
        return returns
    except Exception as e:
        raise ValueError(f"⚠️ Failed to download or process data for '{ticker}': {e}")


def model_var(returns: np.ndarray, level: float = 0.99, model_type: str = "GARCH") -> float:
    model = arch_model(returns, vol=model_type, p=1, q=1)
    res = model.fit(disp="off")
    forecast = res.forecast(horizon=1)
    sigma = np.sqrt(forecast.variance.values[-1][0])
    var = norm.ppf(1 - level) * sigma
    return -var

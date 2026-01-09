import yfinance as yf
import pandas as pd

def get_asset_returns(tickers, start_date='2020-01-01', end_date='2023-12-31'):
    """
    Fetch real asset returns for portfolio optimization
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    
    # Calculate statistics for optimization
    expected_returns = returns.mean() * 252  # Annualize
    covariance = returns.cov() * 252  # Annualize
    
    return {
        'returns': returns,
        'expected_returns': expected_returns,
        'covariance': covariance
    }
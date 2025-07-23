import yfinance as yf
import numpy as np
from datetime import datetime

def get_market_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    # Current price
    spot_price = hist['Close'].iloc[-1]
    
    # Historical volatility (annualized)
    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    volatility = returns.std() * np.sqrt(252)
    
    # Dividend yield
    div_yield = stock.info.get('dividendYield', 0.0)
    if div_yield is None:
        div_yield = 0.0
    
    return spot_price, volatility, div_yield

def time_to_expiry(expiry_date):
    return (datetime.strptime(expiry_date, "%Y-%m-%d") - datetime.today()).days / 365

def payoff(price, strike, option_type, position_type):
    """Basic payoff for call/put, buy/sell"""
    if option_type == "call":
        value = max(price - strike, 0)
    else:
        value = max(strike - price, 0)
    
    return value if position_type == "buy" else -value

def get_option_chain_data(ticker):
    """
    Returns available expiries and option chain for the given ticker.
    """
    stock = yf.Ticker(ticker)
    expiries = stock.options  # list of expiry dates

    # Fetch first expiry by default
    if not expiries:
        return [], None

    expiry = expiries[0]
    chain = stock.option_chain(expiry)
    return expiries, chain

import numpy as np 
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd 
import scipy.optimize as optimization

#stock tickers
stocks = ["AAPL","WMT", "TSLA", "DB", "GE"]

#historical data
start_date = "2010-01-01"
end_date = "2020-01-01"

#download data
def download_data():
    #name of stock is key, stock values (2010-2020) is value
    stock_data = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)["Close"]
    return pd.DataFrame(stock_data)

def show_data(data):
    data.plot(figsize=(10,6))
    plt.title("Stock Prices (2010-2020)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend(stocks)
    plt.grid()
    plt.show()

def calculate_returns(data):
    log_returns = np.log(data / data.shift(1)) #formula ln(S(t+1)/S(t))
    return log_returns[1:] #why ln? because of normalization 

def show_statistics(returns):
    NUMBER_OF_TRADING_DAYS = 252
    print("Mean Returns:")
    print(returns.mean() * NUMBER_OF_TRADING_DAYS) #instead of daily metrics, we are after annualized metrics.
    print("\nCovariance Matrix:")
    print(returns.cov())


if __name__ == "__main__":
    data = download_data()
    print(data.head())
    show_data(data)
    returns = calculate_returns(data)
    print(returns.head())

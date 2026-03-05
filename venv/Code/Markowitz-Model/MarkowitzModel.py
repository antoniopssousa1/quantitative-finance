import numpy as np 
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd 
import scipy.optimize as optimization


NUMBER_OF_TRADING_DAYS = 252
NUMBER_OF_PORTFOLIOS = 10000


#stock tickers
stocks = ["AAPL","WMT", "TSLA", "DB", "GE"]

#historical data
start_date = "2010-01-01"
end_date = "2026-01-01"

#download data
def download_data():
    #name of stock is key, stock values (2010-2020) is value
    stock_data = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)["Close"]
    return pd.DataFrame(stock_data)

def show_data(data):
    data.plot(figsize=(12,8))
    plt.title("Stock Prices (2010-2026)")
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
    print(returns.cov()*NUMBER_OF_TRADING_DAYS) #instead of daily metrics, we are after annualized metrics.

def show_mean_variance(returns, weights):
    portfolio_return = np.sum(returns.mean() * weights) * NUMBER_OF_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUMBER_OF_TRADING_DAYS, weights)))
    print(f"Expected Portfolio Return: {portfolio_return:.2%}")
    print(f"Portfolio Volatility: {portfolio_volatility:.4f}")

def show_portfolio(returns, volatilities):
    plt.figure(figsize=(12,8))
    plt.scatter(volatilities, returns, c=returns/volatilities, marker='o')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.title('Portfolio Optimization - Risk vs Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.grid(True)
    plt.show()


def generate_portfolios(returns):
    portfolios_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(NUMBER_OF_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w) #normalize weights to sum to 1
        portfolio_weights.append(w)

        portfolio_return = np.sum(returns.mean() * w) * NUMBER_OF_TRADING_DAYS
        portfolio_risk = np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUMBER_OF_TRADING_DAYS, w)))

        portfolios_means.append(portfolio_return)
        portfolio_risks.append(portfolio_risk)
    return np.array(portfolios_means), np.array(portfolio_risks), np.array(portfolio_weights)

def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUMBER_OF_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUMBER_OF_TRADING_DAYS, weights)))
    return np.array([portfolio_return, portfolio_volatility, portfolio_return/portfolio_volatility]) #sharpe ratio the last element of the array

#scipy optimization function to minimize the negative sharpe ratio
def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2] #we want to maximize the sharpe ratio, so we minimize its negative value


def optimize_portfolio(weights, returns):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) #weights must sum to 1
    bounds = tuple((0, 1) for _ in range(len(stocks))) #weights must be between 0 and 1
    return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method='SLSQP', bounds=bounds, constraints=constraints)

def print_optimal_portfolio(optimum, returns):
    print("Optimal Portfolio Weights:", optimum['x'].round(3))
    print("Expected Return, Volatility, Sharpe Ratio:", statistics(optimum['x'], returns).round(4))


def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(12,8))
    plt.scatter(portfolio_vols, portfolio_rets, c=portfolio_rets/portfolio_vols, marker='o', alpha=0.5)
    plt.grid(True)
    plt.xlabel('Expected Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(opt['x'], rets)[1], statistics(opt['x'], rets)[0], 'g*', markersize=15) #plot optimal portfolio
    plt.show()


if __name__ == "__main__":
    data = download_data()
    print(data.head())
    show_data(data)
    log_daily_returns = calculate_returns(data)
    print(log_daily_returns.head())
    print(show_statistics(log_daily_returns))

    means, risks, pweights = generate_portfolios(log_daily_returns)
    show_portfolio(means, risks)
    optimum = optimize_portfolio(pweights, log_daily_returns)
    print_optimal_portfolio(optimum, log_daily_returns)
    show_optimal_portfolio(optimum, log_daily_returns, means, risks)
# Necessary imports
import pandas as pd
from scipy.stats import linregress
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats as stats


class RiskEngine:

    def __init__(self, file):
        # Load the dataset
        self.df = pd.read_csv(file)

        # Format date column as datetime
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Set date as index
        self.df.set_index('date', inplace=True)

    def annualized_performance(self, start_date, end_date):
        # Filter data for the date range
        data = self.df.loc[start_date:end_date].copy()

        # Calculate returns
        data['return'] = data['Adj Close'].pct_change()

        # Drop NaN values
        data = data.dropna()

        # Calculate the annualized performance
        annualized_performance = np.prod(1 + data['return']) ** (252 / len(data)) - 1

        return annualized_performance

    def max_drawdown(self, start_date, end_date, window):
        # Filter data for the date range
        data = self.df.loc[start_date:end_date].copy()

        # Calculate daily returns
        data['return'] = data['Adj Close'].pct_change()

        # Calculate rolling max value
        data['roll_max'] = data['Adj Close'].rolling(window, min_periods=1).max()

        # Calculate daily drawdown
        data['daily_dd'] = data['Adj Close'] / data['roll_max'] - 1.0

        # Calculate max drawdown
        max_dd = data['daily_dd'].min()

        # Get the date of max drawdown
        date_of_max_dd = data['daily_dd'].idxmin()

        return max_dd, date_of_max_dd

    def plot_prices(self, start_date, end_date):
        # Filter data for the date range
        data = self.df.loc[start_date:end_date].copy()

        # Create the plot
        plt.figure(figsize=(10,5))
        plt.plot(data['Adj Close'])
        plt.title('AAPL Adjusted Close Price')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.grid(True)
        plt.show()



    def plot_volatility(self, start_date, end_date, window=252):
        # Filter data for the date range
        data = self.df.loc[start_date:end_date].copy()

        # Calculate daily returns
        data['return'] = data['Adj Close'].pct_change()

        # Calculate rolling standard deviation
        data['volatility'] = data['return'].rolling(window).std() * np.sqrt(252)

        # Create the plot
        plt.figure(figsize=(10,5))
        plt.plot(data['volatility'])
        plt.title('AAPL Rolling Volatility')
        plt.xlabel('Date')
        plt.ylabel('Volatility')
        plt.grid(True)

    def information_ratio(self, start_date, end_date):
        # Filter data for the date range
        stock_data = self.df.loc[start_date:end_date].copy()

        # Calculate returns
        stock_data['return'] = stock_data['Adj Close'].pct_change()

        # Calculate annualized return
        annual_return = stock_data['return'].mean() * 252

        # Calculate annualized volatility
        annual_volatility = stock_data['return'].std() * np.sqrt(252)

        # Calculate the Information Ratio
        information_ratio = annual_return / annual_volatility

        return information_ratio

    def semi_deviation(self, start_date, end_date):
        # Filter data for the date range
        data = self.df.loc[start_date:end_date].copy()

        # Calculate daily returns
        data['return'] = data['Adj Close'].pct_change()

        # Filter negative returns
        negative_returns = data['return'][data['return'] < 0]

        # Calculate semi-deviation
        semi_dev = np.sqrt(negative_returns.var())

        return semi_dev

    def var_historical(self, start_date, end_date, level=5):
        # Filter data for the date range
        data = self.df.loc[start_date:end_date].copy()

        # Calculate daily returns
        data['return'] = data['Adj Close'].pct_change()

        # Drop any NaN values
        data = data.dropna()

        # Calculate historical VaR
        var = np.percentile(data['return'], level)

        return var

    def cvar_historical(self, start_date, end_date, level=5):
        # Filter data for the date range
        data = self.df.loc[start_date:end_date].copy()

        # Calculate daily returns
        data['return'] = data['Adj Close'].pct_change()

        # Drop any NaN values
        data = data.dropna()

        # Calculate historical VaR
        var = np.percentile(data['return'], level)

        # Calculate historical CVaR
        cvar = data['return'][data['return'] <= var].mean()

        return cvar

    def skewness(self, start_date, end_date):
        # Filter data for the date range
        data = self.df.loc[start_date:end_date].copy()

        # Calculate daily returns
        data['return'] = data['Adj Close'].pct_change()

        # Drop any NaN values
        data = data.dropna()

        # Calculate skewness
        skewness = stats.skew(data['return'])

        return skewness

    def kurtosis(self, start_date, end_date):
        # Filter data for the date range
        data = self.df.loc[start_date:end_date].copy()

        # Calculate daily returns
        data['return'] = data['Adj Close'].pct_change()

        # Drop any NaN values
        data = data.dropna()

        # Calculate kurtosis
        kurtosis = stats.kurtosis(data['return'])

        return kurtosis

    def omega_ratio(self, start_date, end_date, threshold=0):
        # Filter data for the date range
        data = self.df.loc[start_date:end_date].copy()

        # Calculate daily returns
        data['return'] = data['Adj Close'].pct_change()

        # Drop any NaN values
        data = data.dropna()

        # Separate returns above and below the threshold
        returns_above_threshold = data['return'][data['return'] > threshold]
        returns_below_threshold = data['return'][data['return'] <= threshold]

        # Calculate Omega ratio
        omega_ratio = returns_above_threshold.mean() / abs(returns_below_threshold.mean())

        return omega_ratio


# Usage
if __name__ == "__main__":
    risk_engine = RiskEngine('apple.csv')
    start_date = '2010-01-01'
    end_date = '2020-12-31'
    print("Annualized performance from {} to {}: {:.2%}".format(start_date, end_date,
                                                                risk_engine.annualized_performance(start_date,
                                                                                                   end_date)))
    max_dd, date_of_max_dd = risk_engine.max_drawdown(start_date, end_date, 252)
    print("Maximum drawdown from {} to {}: {:.2%} occurred on {}".format(start_date, end_date, max_dd,
                                                                         date_of_max_dd.date()))
    print("Information ratio from {} to {}: {:.2f}".format(start_date, end_date,
                                                           risk_engine.information_ratio(start_date, end_date)))
    risk_engine.plot_prices(start_date, end_date)
    risk_engine.plot_volatility(start_date, end_date)
    print("Semi-deviation from {} to {}: {:.2f}".format(start_date, end_date,
                                                        risk_engine.semi_deviation(start_date, end_date)))
    print("Historical VaR at 5% level from {} to {}: {:.2f}".format(start_date, end_date,
                                                                    risk_engine.var_historical(start_date, end_date)))
    print("Historical CVaR at 5% level from {} to {}: {:.2f}".format(start_date, end_date,
                                                                     risk_engine.cvar_historical(start_date, end_date)))
    print("Skewness from {} to {}: {:.2f}".format(start_date, end_date,
                                                  risk_engine.skewness(start_date, end_date)))
    print("Kurtosis from {} to {}: {:.2f}".format(start_date, end_date,
                                                  risk_engine.kurtosis(start_date, end_date)))
    print("Omega ratio from {} to {}: {:.2f}".format(start_date, end_date,
                                                     risk_engine.omega_ratio(start_date, end_date)))
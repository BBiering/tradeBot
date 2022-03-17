import pandas as pd
import numpy as np

def get_most_volatile(prices):
    """Return the ticker symbol for the most volatile stock.
    
    Parameters
    ----------
    prices : pandas.DataFrame
        a pandas.DataFrame object with columns: ['ticker', 'date', 'price']
    
    Returns
    -------
    ticker : string
        ticker symbol for the most volatile stock
    """
    # TODO: Fill in this function.
    A = prices.loc[prices.ticker == 'A'].price
    B = prices.loc[prices.ticker == 'B'].price
    log_return_A = A / A.shift(1)
    log_return_B = B / B.shift(1)
    volatility_A = np.std(log_return_A)
    volatility_B = np.std(log_return_B)
    
    if volatility_A > volatility_B:
        ticker = 'A'
    else:
        ticker = 'B'
    return ticker


def test_run(filename='prices.csv'):
    """Test run get_most_volatile() with stock prices from a file."""
    prices = pd.read_csv(filename, parse_dates=['date'])
    print("Most volatile stock: {}".format(get_most_volatile(prices)))


if __name__ == '__main__':
    test_run()

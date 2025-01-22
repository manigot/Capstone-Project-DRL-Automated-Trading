import yfinance as yf
import pandas as pd
# from config.config import TRAINING_DATA_FILE
# from config.config import tickers_list
test_file = "data/data_test.csv"

def get_data_from_api(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical data for a ticker
    :param ticker: The ticker symbol
    :param start_date: The start date
    :param end_date: The end date
    :return: A DataFrame containing the historical data
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    data.reset_index(inplace=True)
    data['tic'] = ticker
    data.rename(columns={
        'Adj Close': 'ajexdi',
        'Close': 'prccd',
        'Open': 'prcod',
        'High': 'prchd',
        'Low': 'prcld',
        'Volume': 'cshtrd'
    }, inplace=True)
    data = data[['Date', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    data['datadate'] = data['Date'].dt.strftime('%Y%m%d')
    data = data[['datadate', 'tic', 'prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']]
    data.columns = data.columns.get_level_values(0)
    data.columns = [col.replace('Price', '').replace('Ticker', '').strip() for col in data.columns]
    # data.to_csv(f'{TRAINING_DATA_FILE}', index=True)
    # print("Data saved to", TRAINING_DATA_FILE)
    return data


def gather_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical data for a list of tickers
    :param tickers: A list of ticker symbols
    :param start_date: The start date
    :param end_date: The end date
    :return: A DataFrame containing the historical data
    """
    # Use a generator to fetch data and create DataFrames
    print("Fetching data from API...")
    combined_data = pd.concat(
        (get_data_from_api(ticker, start_date, end_date) for ticker in tickers),
        ignore_index=True
    )
    
    # Save to CSV
    combined_data.to_csv(test_file, index=True)
    print("Data saved to", test_file)
    return combined_data
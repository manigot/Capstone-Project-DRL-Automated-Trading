import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

from config import config


def load_dataset(file_name: str) -> pd.DataFrame:
    """
    Load a CSV dataset from a file path and return as a pandas DataFrame.

    :param file_name: Path to the CSV file (str).
    :return: pandas DataFrame containing the data from the CSV file.
    """
    try:
        # Read the CSV file into a DataFrame
        _data = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is not found
    except pd.errors.EmptyDataError:
        print(f"Error: The file {file_name} is empty.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is empty
    except pd.errors.ParserError:
        print(f"Error: The file {file_name} could not be parsed.")
        return pd.DataFrame()  # Return an empty DataFrame if there is a parsing error

    return _data


def data_split(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Splits the dataset into a subset based on a date range.

    :param df: pandas DataFrame containing the data.
    :param start: Start date (inclusive) as a string (e.g., '2022-01-01').
    :param end: End date (exclusive) as a string (e.g., '2023-01-01').
    :return: Filtered pandas DataFrame sorted by 'datadate' and 'tic'.
    """
    # # Ensure 'datadate' is in datetime format
    # if df['datadate'].dtype != 'datetime64[ns]':
    #     df['datadate'] = pd.to_datetime(df['datadate'], errors='coerce')

    # Filter the dataset based on the date range
    data = df[(df['datadate'] >= start) & (df['datadate'] < end)]

    # Sort the data by 'datadate' and 'tic' columns
    data = data.sort_values(['datadate', 'tic'], ignore_index=True)

    # Re-index the data based on 'datadate'
    data.index = data['datadate'].factorize()[0]

    return data


def calculate_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate adjusted close price, open-high-low price, and volume.

    :param df: pandas DataFrame containing columns for stock data.
    :return: pandas DataFrame with calculated prices and volume.
    """
    # Create a copy of the original dataframe to avoid modifying the original data
    data = df.copy()

    # Select relevant columns
    data = data[["datadate", "tic", "prccd", "ajexdi", "prcod", "prchd", "prcld", "cshtrd"]]

    # Ensure that ajexdi is not zero (to avoid division by zero)
    data["ajexdi"] = data["ajexdi"].apply(lambda x: 1 if x == 0 else x)

    # Calculate adjusted close price and price-related columns
    data["adjcp"] = data["prccd"] / data["ajexdi"]
    data["open"] = data["prcod"] / data["ajexdi"]
    data["high"] = data["prchd"] / data["ajexdi"]
    data["low"] = data["prcld"] / data["ajexdi"]
    data["volume"] = data["cshtrd"]

    # Reorder columns and sort by ticker and date
    data = data[["datadate", "tic", "adjcp", "open", "high", "low", "volume"]]
    data = data.sort_values(["tic", "datadate"], ignore_index=True)

    return data


def add_technical_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators using the stockstats package.
    Adds MACD, RSI, CCI, and ADX indicators to the dataframe.

    :param df: pandas DataFrame containing stock data.
    :return: pandas DataFrame with added technical indicators.
    """
    # Retype the dataframe to use stockstats methods
    stock = Sdf.retype(df.copy())

    stock["close"] = stock["adjcp"]
    unique_ticker = stock.tic.unique()

    # Initialize empty lists to store the indicator values
    macd_list, rsi_list, cci_list, dx_list = [], [], [], []

    # Loop through each unique ticker and collect corresponding indicators
    for ticker in unique_ticker:
        macd_list.append(stock[stock.tic == ticker]["macd"])
        rsi_list.append(stock[stock.tic == ticker]["rsi_30"])
        cci_list.append(stock[stock.tic == ticker]["cci_30"])
        dx_list.append(stock[stock.tic == ticker]["dx_30"])

    # Concatenate all indicator lists into a single DataFrame
    macd = pd.concat(macd_list, ignore_index=True)
    rsi = pd.concat(rsi_list, ignore_index=True)
    cci = pd.concat(cci_list, ignore_index=True)
    dx = pd.concat(dx_list, ignore_index=True)

    # Add the calculated indicators to the dataframe
    df["macd"] = macd
    df["rsi"] = rsi
    df["cci"] = cci
    df["adx"] = dx

    return df


def preprocess_data() -> pd.DataFrame:
    """
    Data preprocessing pipeline that loads, filters, and processes stock data.
    - Loads the dataset.
    - Filters data after 2009.
    - Calculates adjusted prices.
    - Adds technical indicators.
    - Fills missing values.

    :return: pandas DataFrame with preprocessed data.
    """
    # Load the dataset
    df = load_dataset(file_name=config.TRAINING_DATA_FILE)

    # Filter data after 2009
    df = df[df.datadate >= 20090000]

    # Calculate adjusted price
    df_preprocess = calculate_price(df)

    # Add technical indicators
    df_final = add_technical_indicator(df_preprocess)

    # Fill missing values at the beginning
    df_final.fillna(method="bfill", inplace=True)

    return df_final


def add_turbulence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add turbulence index to the dataframe based on precalculated turbulence.

    :param df: pandas DataFrame containing stock data.
    :return: pandas DataFrame with added turbulence index.
    """
    try:
        turbulence_index = calcualte_turbulence(df)
        df = df.merge(turbulence_index, on="datadate", how="left")
        df = df.sort_values(["datadate", "tic"]).reset_index(drop=True)
    except Exception as e:
        print(f"Error adding turbulence index: {e}")
        df["turbulence"] = None  # In case of error, add an empty turbulence column

    return df


def calcualte_turbulence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the turbulence index based on historical stock prices.
    Uses the covariance matrix of historical prices to calculate turbulence.

    :param df: pandas DataFrame containing stock data.
    :return: pandas DataFrame containing turbulence index.
    """
    try:
        df_price_pivot = df.pivot(index="datadate", columns="tic", values="adjcp")
        unique_date = df.datadate.unique()

        # Start after 252 days (1 year of trading days)
        start = 252
        turbulence_index = [0] * start
        count = 0

        # Calculate turbulence index
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            hist_price = df_price_pivot[[n in unique_date[0:i] for n in df_price_pivot.index]]
            cov_temp = hist_price.cov()
            current_temp = current_price - np.mean(hist_price, axis=0)
            temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(current_temp.values.T)

            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # Avoid large outlier because the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0

            turbulence_index.append(turbulence_temp)

        turbulence_df = pd.DataFrame({"datadate": df_price_pivot.index, "turbulence": turbulence_index})
    except Exception as e:
        print(f"Error calculating turbulence index: {e}")
        turbulence_df = pd.DataFrame({"datadate": [], "turbulence": []})  # Return an empty DataFrame on error

    return turbulence_df

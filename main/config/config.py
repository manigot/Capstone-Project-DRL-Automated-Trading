import datetime
import os
import sys
# import pathlib

# import pandas as pd

# import finrl

# pd.options.display.max_rows = 10
# pd.options.display.max_columns = 10


# PACKAGE_ROOT = pathlib.Path(finrl.__file__).resolve().parent
# PACKAGE_ROOT = pathlib.Path().resolve().parent

# TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
# DATASET_DIR = PACKAGE_ROOT / "data"

# data
# TRAINING_DATA_FILE = "data/ETF_SPY_2009_2020.csv"




TRAINING_DATA_FILE = "data/dow_30_2009_2020.csv"

now = datetime.datetime.now().strftime('%Y-%m-%d %H;%M;%S')
TRAINED_MODEL_DIR = f"trained_models/{now}"
Csv_files_dir = f"results/{now}/"
os.makedirs(TRAINED_MODEL_DIR)
os.makedirs(Csv_files_dir)

TURBULENCE_DATA = "data/dow30_turbulence_index.csv"

TESTING_DATA_FILE = "test.csv"
TRAINING_DATA_TEST_FILE = "data/data_test.csv"


tickers_list_old = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GS', 'HD',
       'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
       'NKE', 'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT',
       'XOM']
tickers_list = ['AAPL', 'AXP', 'BA', 'CAT']

start_date = "2009-01-01"
end_date = "2020-12-31"
validation_date = "2015-01-01"
rebalance_window = 63
validation_window = 63

def format_date(date: str) -> int:
    """Format date to integer."""
    date = date.split("-")
    return int("".join(date))




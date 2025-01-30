# common library
import os
import sys

import pandas as pd
from model.models import run_ensemble_strategy
from preprocessing.preprocessors import add_turbulence, preprocess_data
from api.get_data import gather_data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# preprocessor
# from preprocessing.preprocessors import *
# config
# from config.config import *
# model
# from model.models_old import add_turbulence,
# preprocess_data, run_ensemble_strategy


# import numpy as np
# import time
# from stable_baselines3.common.vec_env import DummyVecEnv
def format_date(date: str) -> int:
    """Format date to integer."""
    date = date.split("-")
    return int("".join(date))

tickers_list = ['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'DIS', 'GS', 'HD',
       'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT',
       'NKE', 'PFE', 'PG', 'RTX', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT',
       'XOM']
start_date = "2009-01-01"
end_date = "2020-12-31"
validation_date = "2015-01-01"
rebalance_window = 63
validation_window = 63


def run_model(ticker_L:list = None, start_d:str = None, end_d:str = None, validation_d:str = None, 
              rebalance_w:int = rebalance_window, validation_w:int = validation_window) -> None:
    """Train the model."""

    # read and preprocess data
    preprocessed_path = "main/done_data.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        global tickers_list, start_date, end_date, validation_date
        tickers_list = ticker_L
        start_date = start_d
        end_date = end_d
        validation_date = validation_d
        gather_data(tickers=tickers_list, start_date=start_date, end_date=end_date)
        data = preprocess_data()
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)

    print(data.head())
    print(data.size)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose
    
    
    unique_trade_date = data[
        (data["datadate"] > format_date(validation_date)) & (data["datadate"] <= format_date(end_date))
    ]["datadate"].unique()
    print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model
    # and select for trading
    global rebalance_window, validation_window
    
    rebalance_window = rebalance_w
    validation_window = validation_w

    # Ensemble Strategy
    run_ensemble_strategy(
        df=data,
        unique_trade_date=unique_trade_date,
        rebalance_window=rebalance_window,
        validation_window=validation_window,
    )

    # _logger.info(f"saving model version: {_version}")


if __name__ == "__main__":
    run_model()

# common library
import os
import sys

import pandas as pd
from model.models import run_ensemble_strategy
from preprocessing.preprocessors import add_turbulence, preprocess_data
from api.get_data import gather_data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import *
from main.backtesting.backtesting import save_all_results

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

os.makedirs(TRAINED_MODEL_DIR)
os.makedirs(Csv_files_dir)
os.makedirs(Csv_files_dir + "report/")
os.makedirs(Csv_files_dir + "account_value_validation/")
os.makedirs(Csv_files_dir + "account_value_trade/")
os.makedirs(Csv_files_dir + "account_rewards_trade/")
os.makedirs(Csv_files_dir + "last_states/")

def run_model(preprocessed_data = "done_data") -> None:
    """Train the model."""
    print(date)
    # read and preprocess data
    preprocessed_path = "main/data/" + preprocessed_data + start_date + "_" + end_date + ".csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
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

    # rebalance_window is the number of day to retrain the model
    # validation_window is the number of day to validation the model
    # and select for trading

    # Ensemble Strategy
    run_ensemble_strategy(
        df=data,
        unique_trade_date=unique_trade_date,
        rebalance_window=rebalance_window,
        validation_window=validation_window,
    )

    save_all_results(
        start_date=start_date, 
        end_date=end_date, 
        validation_date=validation_date, 
        rebalance_window=rebalance_window, 
        validation_window=validation_window, 
        date=date, preprocessed_data='done_data', 
        Csv_files_dir='results/{}'.format(date))

    # _logger.info(f"saving model version: {_version}")

if __name__ == "__main__":
    run_model()
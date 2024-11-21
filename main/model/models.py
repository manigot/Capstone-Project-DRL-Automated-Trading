# Common libraries
import os
import sys
import time

# import gym
import numpy as np
import pandas as pd
from config import config

# RL models from stable-baselines3
from stable_baselines3 import A2C, DDPG, PPO  # , SAC, TD3
from stable_baselines3.common.noise import (  # NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.EnvMultipleStock_trade import StockEnvTrade
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation

# Customized environment imports
from preprocessing.preprocessors import data_split

# from stable_baselines3.common.policies import MlpPolicy


# Training functions
def train_A2C(env_train, model_name, timesteps=25000):
    """
    Trains an A2C model on the given environment.

    Args:
        env_train (gym.Env): The environment to train the model on.
        model_name (str): The name to save the trained model.
        timesteps (int, optional): The number of timesteps to train for. Default is 25000.

    Returns:
        stable_baselines3.A2C: The trained A2C model.
    """
    start = time.time()
    model = A2C("MlpPolicy", env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (A2C):", (time.time() - start) / 60, "minutes")
    return model


def train_PPO(env_train, model_name, timesteps=50000):
    """
    Trains a PPO model on the given environment.

    Args:
        env_train (gym.Env): The environment to train the model on.
        model_name (str): The name to save the trained model.
        timesteps (int, optional): The number of timesteps to train for. Default is 50000.

    Returns:
        stable_baselines3.PPO: The trained PPO model.
    """
    start = time.time()
    model = PPO("MlpPolicy", env_train, verbose=0, ent_coef=0.005)
    model.learn(total_timesteps=timesteps)
    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (PPO):", (time.time() - start) / 60, "minutes")
    return model


def train_DDPG(env_train, model_name, timesteps=10000):
    """
    Trains a DDPG model on the given environment.

    Args:
        env_train (gym.Env): The environment to train the model on.
        model_name (str): The name to save the trained model.
        timesteps (int, optional): The number of timesteps to train for. Default is 10000.

    Returns:
        stable_baselines3.DDPG: The trained DDPG model.
    """
    n_actions = env_train.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions)
    )
    start = time.time()
    model = DDPG("MlpPolicy", env_train, action_noise=action_noise, verbose=0)
    model.learn(total_timesteps=timesteps)
    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (DDPG):", (time.time() - start) / 60, "minutes")
    return model


# Prediction and validation functions
def DRL_prediction(
    df: pd.DataFrame,
    model,
    name: str,
    last_state: list,
    iter_num: int,
    unique_trade_date,
    rebalance_window,
    turbulence_threshold,
    initial,
):
    """
    Runs the trained model on the environment for a prediction and stores the last state.

    Args:
        df (pd.DataFrame): The dataset used for training and validation.
        model (stable_baselines3): The trained model to use for prediction.
        name (str): The name used for saving the last state.
        last_state (list): The state from the previous prediction.
        iter_num (int): The current iteration number.
        unique_trade_date (list): A list of unique trade dates.
        rebalance_window (int): The window for rebalancing.
        turbulence_threshold (float): The threshold for turbulence.
        initial (bool): Whether this is the first iteration.

    Returns:
        list: The last state of the environment after the prediction.
    """
    trade_data = data_split(
        df=df,
        start=unique_trade_date[iter_num - rebalance_window],
        end=unique_trade_date[iter_num],
    )
    env_trade = DummyVecEnv(
        [
            lambda: StockEnvTrade(
                df=trade_data,
                turbulence_threshold=turbulence_threshold,
                initial=initial,
                previous_state=last_state,
                model_name=name,
                iteration=iter_num,
            )
        ]
    )

    obs_trade = env_trade.reset()
    last_state = None  # Initialize
    for i in range(len(trade_data.index.unique())):
        action, _ = model.predict(obs_trade)
        _, _, _, _ = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            last_state = env_trade.envs[0].render()

    if last_state is None:
        print("Warning: last_state is still None!")

    pd.DataFrame({"last_state": last_state}).to_csv(
        f"results/last_state_{name}_{iter_num}.csv", index=False
    )
    return last_state


def DRL_validation(model, test_data, test_env, test_obs):
    """
    Runs the trained model for validation.

    Args:
        model (stable_baselines3): The trained model to use for validation.
        test_data (pd.DataFrame): The data used for testing.
        test_env (gym.Env): The environment for testing.
        test_obs (np.array): The observation for the test environment.

    This function applies the trained model to the test data and environment.
    """
    for _ in range(len(test_data.index.unique())):
        action, _ = model.predict(test_obs)
        test_obs, _, _, _ = test_env.step(action)


def get_validation_sharpe(iteration):
    """
    Calculates the Sharpe ratio based on validation results.

    Args:
        iteration (int): The current iteration for validation.

    Returns:
        float: The calculated Sharpe ratio.
    """
    df_total_value = pd.read_csv(
        f"results/account_value_validation_{iteration}.csv", index_col=0
    )
    df_total_value.columns = ["account_value_train"]
    df_total_value["daily_return"] = df_total_value.pct_change(1)
    sharpe = (
        (4**0.5)
        * df_total_value["daily_return"].mean()
        / df_total_value["daily_return"].std()
    )
    return sharpe


# Main ensemble strategy
def run_ensemble_strategy(
    df: pd.DataFrame, unique_trade_date, rebalance_window, validation_window
):
    """
    Runs the ensemble strategy by training multiple models (A2C, PPO, DDPG) and selecting the best performing model.

    Args:
        df (pd.DataFrame): The dataset used for training and validation.
        unique_trade_date (list): A list of unique trade dates.
        rebalance_window (int): The window for rebalancing.
        validation_window (int): The window for validation.

    This function trains multiple models, validates them, and selects the best performing model based on Sharpe ratio.
    It then uses the selected model for trading predictions and stores the results.
    """
    print("============Start Ensemble Strategy============")
    last_state_ensemble = []
    ppo_sharpe_list, ddpg_sharpe_list, a2c_sharpe_list, model_use = [], [], [], []

    insample_turbulence = df[
        (df["datadate"] < 20151000) & (df["datadate"] >= 20090000)
    ].drop_duplicates(subset=["datadate"])
    insample_turbulence_threshold = np.quantile(
        insample_turbulence["turbulence"].values, 0.90
    )

    start = time.time()
    for i in range(
        rebalance_window + validation_window,
        len(unique_trade_date),
        rebalance_window,
    ):
        print("============================================")
        initial = i - rebalance_window - validation_window == 0

        end_date_index = df.index[
            df["datadate"]
            == unique_trade_date[i - rebalance_window - validation_window]
        ].tolist()[-1]
        start_date_index = end_date_index - validation_window * 30 + 1
        historical_turbulence = df.iloc[
            start_date_index : (end_date_index + 1), :
        ].drop_duplicates(subset=["datadate"])
        turbulence_threshold = (
            insample_turbulence_threshold
            if np.mean(historical_turbulence.turbulence.values)
            > insample_turbulence_threshold
            else np.quantile(insample_turbulence.turbulence.values, 1)
        )
        print("turbulence_threshold:", turbulence_threshold)

        # Training environments
        train = data_split(
            df,
            start=20090000,
            end=unique_trade_date[i - rebalance_window - validation_window],
        )
        env_train = DummyVecEnv([lambda: StockEnvTrain(train)])
        validation = data_split(
            df,
            start=unique_trade_date[i - rebalance_window - validation_window],
            end=unique_trade_date[i - rebalance_window],
        )
        env_val = DummyVecEnv(
            [
                lambda: StockEnvValidation(
                    df=validation,
                    turbulence_threshold=turbulence_threshold,
                    iteration=i,
                )
            ]
        )
        obs_val = env_val.reset()

        # Training models
        print("======A2C Training========")
        model_a2c = train_A2C(env_train, f"A2C_30k_dow_{i}", 30000)
        DRL_validation(model_a2c, validation, env_val, obs_val)
        sharpe_a2c = get_validation_sharpe(i)
        print("A2C Sharpe Ratio:", sharpe_a2c)

        print("======PPO Training========")
        model_ppo = train_PPO(env_train, f"PPO_100k_dow_{i}", 100000)
        DRL_validation(model_ppo, validation, env_val, obs_val)
        sharpe_ppo = get_validation_sharpe(i)
        print("PPO Sharpe Ratio:", sharpe_ppo)

        print("======DDPG Training========")
        model_ddpg = train_DDPG(env_train, f"DDPG_10k_dow_{i}", 10000)
        DRL_validation(model_ddpg, validation, env_val, obs_val)
        sharpe_ddpg = get_validation_sharpe(i)
        print("DDPG Sharpe Ratio:", sharpe_ddpg)

        ppo_sharpe_list.append(sharpe_ppo)
        a2c_sharpe_list.append(sharpe_a2c)
        ddpg_sharpe_list.append(sharpe_ddpg)

        # Model selection based on Sharpe ratio
        if sharpe_ppo >= sharpe_a2c and sharpe_ppo >= sharpe_ddpg:
            model_ensemble = model_ppo
            model_use.append("PPO")
        elif sharpe_a2c > sharpe_ppo and sharpe_a2c > sharpe_ddpg:
            model_ensemble = model_a2c
            model_use.append("A2C")
        else:
            model_ensemble = model_ddpg
            model_use.append("DDPG")

        print(
            f"======Trading from: {unique_trade_date[i - rebalance_window]} to {unique_trade_date[i]}"
        )
        last_state_ensemble = DRL_prediction(
            df,
            model_ensemble,
            "ensemble",
            last_state_ensemble,
            i,
            unique_trade_date,
            rebalance_window,
            turbulence_threshold,
            initial,
        )

    print("Ensemble Strategy took:", (time.time() - start) / 60, "minutes")

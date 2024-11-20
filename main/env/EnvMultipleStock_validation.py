import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

matplotlib.use("Agg")
# import pickle

import matplotlib.pyplot as plt

# Constants for the environment
HMAX_NORMALIZE = 100  # Shares normalization factor: 100 shares per trade
INITIAL_ACCOUNT_BALANCE = 1000000  # Initial account balance
STOCK_DIM = 30  # Number of stocks in the portfolio
TRANSACTION_FEE_PERCENT = 0.001  # Transaction fee percentage
REWARD_SCALING = 1e-4  # Reward scaling factor

class StockEnvValidation(gym.Env):
    """
    A stock trading environment for OpenAI Gym where agents simulate stock trading.

    This environment allows agents to interact with historical stock data to take actions such as 
    buying, selling, or holding stocks. The environment includes stock prices, technical indicators, 
    and market turbulence as features for the agent to consider when making decisions.

    Attributes:
        day (int): The current day in the stock data.
        df (pandas.DataFrame): DataFrame containing historical stock data for simulation.
        action_space (gym.spaces.Box): The action space of the agent, representing buy, sell, or hold actions.
        observation_space (gym.spaces.Box): The observation space representing the state of the environment.
        state (list): The current state of the environment, including balance, stock prices, and technical indicators.
        reward (float): The reward after each action taken by the agent.
        turbulence_threshold (float): Threshold value for market turbulence.
        turbulence (float): Current turbulence value from stock data.
        cost (float): The transaction cost incurred from buying or selling stocks.
        trades (int): The number of trades made during the episode.
        asset_memory (list): History of the total asset value during the episode.
        rewards_memory (list): History of rewards earned during the episode.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, day=0, turbulence_threshold=140, iteration=""):
        """
        Initializes the stock trading environment with the given data and parameters.

        Args:
            df (pandas.DataFrame): The stock data to be used in the simulation.
            day (int, optional): The starting day in the dataset (default is 0).
            turbulence_threshold (float, optional): The threshold for market turbulence (default is 140).
            iteration (str, optional): The experiment iteration identifier (default is an empty string).
        """
        self.day = day
        self.df = df
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(181,))
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        self.state = (
            [INITIAL_ACCOUNT_BALANCE]
            + self.data["adjcp"].to_numpy().tolist()
            + [0] * STOCK_DIM
            + self.data["macd"].to_numpy().tolist()
            + self.data["rsi"].to_numpy().tolist()
            + self.data["cci"].to_numpy().tolist()
            + self.data["adx"].to_numpy().tolist()
        )
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self._seed()
        self.iteration = iteration

    def _sell_stock(self, index, action):
        """
        Executes a sell action for a specific stock based on the agent's action.

        Args:
            index (int): The index of the stock to be sold.
            action (float): The amount of stock to be sold.

        This method checks the turbulence condition and performs the sell action if applicable. 
        It updates the account balance, stock holdings, and transaction costs.
        """
        if self.turbulence < self.turbulence_threshold:
            if self.state[index + STOCK_DIM + 1] > 0:
                self.state[0] += (
                    self.state[index + 1]
                    * min(abs(action), self.state[index + STOCK_DIM + 1])
                    * (1 - TRANSACTION_FEE_PERCENT)
                )
                self.state[index + STOCK_DIM + 1] -= min(abs(action), self.state[index + STOCK_DIM + 1])
                self.cost += (
                    self.state[index + 1]
                    * min(abs(action), self.state[index + STOCK_DIM + 1])
                    * TRANSACTION_FEE_PERCENT
                )
                self.trades += 1
        else:
            if self.state[index + STOCK_DIM + 1] > 0:
                self.state[0] += (
                    self.state[index + 1]
                    * self.state[index + STOCK_DIM + 1]
                    * (1 - TRANSACTION_FEE_PERCENT)
                )
                self.state[index + STOCK_DIM + 1] = 0
                self.cost += (
                    self.state[index + 1]
                    * self.state[index + STOCK_DIM + 1]
                    * TRANSACTION_FEE_PERCENT
                )
                self.trades += 1

    def _buy_stock(self, index, action):
        """
        Executes a buy action for a specific stock based on the agent's action.

        Args:
            index (int): The index of the stock to be bought.
            action (float): The amount of stock to be bought.

        This method checks the turbulence condition and performs the buy action if applicable. 
        It updates the account balance, stock holdings, and transaction costs.
        """
        if self.turbulence < self.turbulence_threshold:
            available_amount = self.state[0] // self.state[index + 1]
            self.state[0] -= (
                self.state[index + 1]
                * min(available_amount, action)
                * (1 + TRANSACTION_FEE_PERCENT)
            )
            self.state[index + STOCK_DIM + 1] += min(available_amount, action)
            self.cost += (
                self.state[index + 1]
                * min(available_amount, action)
                * TRANSACTION_FEE_PERCENT
            )
            self.trades += 1

    def step(self, actions):
        """
        Advances the environment by one time step based on the agent's actions.

        Args:
            actions (np.ndarray): An array of actions where each value represents the amount to buy or sell for each stock.

        Returns:
            tuple: A tuple containing the updated state, reward, a boolean indicating if the episode is done, and additional information.

        This method performs the buy or sell actions, calculates rewards, and updates the environment's state.
        It ends the episode if the agent has traded through all available days.
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            plt.plot(self.asset_memory, "r")
            plt.savefig(
                "results/account_value_validation_{}.png".format(self.iteration)
            )
            plt.close()
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv(
                "results/account_value_validation_{}.csv".format(self.iteration)
            )
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (STOCK_DIM + 1)])
                * np.array(self.state[(STOCK_DIM + 1) : (STOCK_DIM * 2 + 1)])
            )
            df_total_value.columns = ["account_value"]
            df_total_value["daily_return"] = df_total_value.pct_change(1)
            sharpe = (
                (4**0.5)
                * df_total_value["daily_return"].mean()
                / df_total_value["daily_return"].std()
            )
            return self.state, self.reward, self.terminal, {}

        else:
            actions = actions * HMAX_NORMALIZE
            if self.turbulence >= self.turbulence_threshold:
                actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM)
            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (STOCK_DIM + 1)])
                * np.array(self.state[(STOCK_DIM + 1) : (STOCK_DIM * 2 + 1)])
            )
            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self._sell_stock(index, actions[index])

            for index in buy_index:
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.turbulence = self.data["turbulence"].values[0]

            self.state = (
                [self.state[0]]
                + self.data["adjcp"].to_numpy().tolist()
                + list(self.state[(STOCK_DIM + 1) : (STOCK_DIM * 2 + 1)])
                + self.data["macd"].to_numpy().tolist()
                + self.data["rsi"].to_numpy().tolist()
                + self.data["cci"].to_numpy().tolist()
                + self.data["adx"].to_numpy().tolist()
            )
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (STOCK_DIM + 1)])
                * np.array(self.state[(STOCK_DIM + 1) : (STOCK_DIM * 2 + 1)])
            )

            self.reward = (end_total_asset - begin_total_asset) * REWARD_SCALING
            self.asset_memory.append(end_total_asset)
            self.rewards_memory.append(self.reward)

            return self.state, self.reward, self.terminal, {}

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            list: The initial state of the environment, including balance, stock prices, and technical indicators.

        This method reinitializes the environment and prepares it for the next episode.
        """
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = (
            [INITIAL_ACCOUNT_BALANCE]
            + self.data["adjcp"].to_numpy().tolist()
            + [0] * STOCK_DIM
            + self.data["macd"].to_numpy().tolist()
            + self.data["rsi"].to_numpy().tolist()
            + self.data["cci"].to_numpy().tolist()
            + self.data["adx"].to_numpy().tolist()
        )
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        return self.state

    def render(self, mode="human", close=False):
        """
        Renders the current state of the environment.

        Args:
            mode (str, optional): The mode for rendering. Default is "human".
            close (bool, optional): Whether to close the rendering. Default is False.

        Returns:
            list: The current state of the environment.

        This method provides a way to visualize the current state of the environment, useful for debugging or monitoring.
        """
        return self.state

    def _seed(self, seed=None):
        """
        Seeds the environment's random number generator.

        Args:
            seed (int, optional): The seed for the random number generator (default is None).

        Returns:
            list: A list containing the seed used.

        This method allows for reproducible experiments by controlling the random state of the environment.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

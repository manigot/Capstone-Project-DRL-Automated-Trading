import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

matplotlib.use("Agg")

HMAX_NORMALIZE = 100
INITIAL_ACCOUNT_BALANCE = 1000000
STOCK_DIM = 30
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4


class StockEnvTrade(gym.Env):
    """
    A stock trading environment for OpenAI Gym.

    Attributes:
        day (int): The current trading day.
        df (pd.DataFrame): Dataframe containing stock data.
        initial (bool): Whether this is the initial state.
        previous_state (list): The state from the previous step.
        model_name (str): Name of the model for saving results.
        iteration (str): Iteration identifier for saving results.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        day=0,
        turbulence_threshold=140,
        initial=True,
        previous_state=None,
        model_name="",
        iteration="",
    ):
        """
        Initialize the StockEnvTrade environment.

        Args:
            df (pd.DataFrame): Stock data.
            day (int): Starting day index.
            turbulence_threshold (float): Threshold for turbulence.
            initial (bool): Whether it's the initial state.
            previous_state (list, optional): Previous environment state.
            model_name (str): Name of the model for saving results.
            iteration (str): Iteration identifier for saving results.
        """
        self.day = day
        self.df = df
        self.initial = initial
        self.previous_state = previous_state or []
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(181,))
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        self.state = (
            [INITIAL_ACCOUNT_BALANCE]
            + self.data["adjcp"].tolist()
            + [0] * STOCK_DIM
            + self.data["macd"].tolist()
            + self.data["rsi"].tolist()
            + self.data["cci"].tolist()
            + self.data["adx"].tolist()
        )
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self._seed()
        self.model_name = model_name
        self.iteration = iteration

    def _sell_stock(self, index, action):
        """
        Perform a sell action.

        Args:
            index (int): Index of the stock to sell.
            action (float): Sell action value.
        """
        if self.turbulence < self.turbulence_threshold:
            if self.state[index + STOCK_DIM + 1] > 0:
                self.state[0] += (
                    self.state[index + 1]
                    * min(abs(action), self.state[index + STOCK_DIM + 1])
                    * (1 - TRANSACTION_FEE_PERCENT)
                )
                self.state[index + STOCK_DIM + 1] -= min(
                    abs(action), self.state[index + STOCK_DIM + 1]
                )
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
                self.cost += (
                    self.state[index + 1]
                    * self.state[index + STOCK_DIM + 1]
                    * TRANSACTION_FEE_PERCENT
                )
                self.state[index + STOCK_DIM + 1] = 0
                self.trades += 1

    def _buy_stock(self, index, action):
        """
        Perform a buy action.

        Args:
            index (int): Index of the stock to buy.
            action (float): Buy action value.
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
        Take a step in the environment.

        Args:
            actions (np.ndarray): Actions for each stock.

        Returns:
            tuple: Updated state, reward, terminal status, and info.
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            # Final state actions
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (STOCK_DIM + 1)])
                * np.array(self.state[(STOCK_DIM + 1) : (STOCK_DIM * 2 + 1)])
            )
            return self.state, self.reward, self.terminal, {}

        actions = actions * HMAX_NORMALIZE
        if self.turbulence >= self.turbulence_threshold:
            actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM)

        # Execute buy and sell actions
        sell_index = np.argsort(actions)[: np.where(actions < 0)[0].shape[0]]
        buy_index = np.argsort(actions)[::-1][: np.where(actions > 0)[0].shape[0]]

        for index in sell_index:
            self._sell_stock(index, actions[index])
        for index in buy_index:
            self._buy_stock(index, actions[index])

        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.turbulence = self.data["turbulence"].iloc[0]
        self.state = (
            [self.state[0]]
            + self.data["adjcp"].tolist()
            + list(self.state[(STOCK_DIM + 1) : (STOCK_DIM * 2 + 1)])
            + self.data["macd"].tolist()
            + self.data["rsi"].tolist()
            + self.data["cci"].tolist()
            + self.data["adx"].tolist()
        )
        self.reward = (
            self.state[0]
            + sum(
                np.array(self.state[1 : (STOCK_DIM + 1)])
                * np.array(self.state[(STOCK_DIM + 1) : (STOCK_DIM * 2 + 1)])
            )
            - self.asset_memory[-1]
        )
        self.asset_memory.append(self.state[0])

        return self.state, self.reward * REWARD_SCALING, self.terminal, {}

    def reset(self):
        """
        Reset the environment.

        Returns:
            list: Initial state of the environment.
        """
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day]
        self.state = (
            [INITIAL_ACCOUNT_BALANCE]
            + self.data["adjcp"].tolist()
            + [0] * STOCK_DIM
            + self.data["macd"].tolist()
            + self.data["rsi"].tolist()
            + self.data["cci"].tolist()
            + self.data["adx"].tolist()
        )
        return self.state

    def render(self, mode="human"):
        """
        Render the environment.

        Args:
            mode (str): Rendering mode.

        Returns:
            np.ndarray: Current state.
        """
        return self.state

    def _seed(self, seed=None):
        """
        Seed the environment.

        Args:
            seed (int, optional): Seed value.

        Returns:
            list: Seed used.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

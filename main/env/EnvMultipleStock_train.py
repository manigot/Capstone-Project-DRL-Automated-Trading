import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

matplotlib.use("Agg")

# Constants
HMAX_NORMALIZE = 100  # Normalization factor for shares
INITIAL_ACCOUNT_BALANCE = 1_000_000  # Initial account balance
STOCK_DIM = 30  # Number of stocks in the portfolio
TRANSACTION_FEE_PERCENT = 0.001  # Transaction fee percentage
REWARD_SCALING = 1e-4  # Scaling factor for rewards


class StockEnvTrain(gym.Env):
    """
    A stock trading environment for OpenAI Gym, designed to simulate stock trading activities.
    The environment allows an agent to perform buy and sell actions on a portfolio of stocks 
    and rewards it based on the change in the total asset value.

    Attributes:
        metadata (dict): Metadata about the environment (e.g., render modes).
        action_space (gym.spaces.Box): Action space representing the proportion of stocks to buy or sell.
        observation_space (gym.spaces.Box): Observation space containing the environment's state.
        df (pd.DataFrame): Dataframe containing historical stock data (e.g., adjusted close prices, technical indicators).
        day (int): Current day in the simulation.
        state (list): Current state of the environment, including account balance and stock data.
        terminal (bool): Boolean flag indicating whether the episode has ended.
        reward (float): Reward calculated based on asset change.
        cost (float): Cost of transactions (fees).
        trades (int): Count of executed trades.
        asset_memory (list): List tracking the account balance over time.
        rewards_memory (list): List tracking the rewards over time.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, df: pd.DataFrame, day=0):
        """
        Initializes the stock trading environment with the given stock data.

        Args:
            df (pd.DataFrame): Dataframe containing historical stock data.
            day (int, optional): Starting day for the simulation. Defaults to 0.
        """
        self.day = day
        self.df = df

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,))
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(181,))

        # Initialize environment state
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.state = (
            [INITIAL_ACCOUNT_BALANCE]
            + self.data.adjcp.values.tolist()
            + [0] * STOCK_DIM
            + self.data.macd.values.tolist()
            + self.data.rsi.values.tolist()
            + self.data.cci.values.tolist()
            + self.data.adx.values.tolist()
        )
        self.reward = 0
        self.cost = 0
        self.trades = 0
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []

        self._seed()

    def _sell_stock(self, index, action):
        """
        Handles the logic for selling a stock.

        Args:
            index (int): The index of the stock to be sold.
            action (float): The amount of stock to sell.
        """
        if self.state[index + STOCK_DIM + 1] > 0:
            sell_amount = min(abs(action), self.state[index + STOCK_DIM + 1])
            self.state[0] += self.state[index + 1] * sell_amount * (1 - TRANSACTION_FEE_PERCENT)
            self.state[index + STOCK_DIM + 1] -= sell_amount
            self.cost += self.state[index + 1] * sell_amount * TRANSACTION_FEE_PERCENT
            self.trades += 1

    def _buy_stock(self, index, action):
        """
        Handles the logic for buying a stock.

        Args:
            index (int): The index of the stock to be bought.
            action (float): The amount of stock to buy.
        """
        available_amount = self.state[0] // self.state[index + 1]
        buy_amount = min(available_amount, action)
        self.state[0] -= self.state[index + 1] * buy_amount * (1 + TRANSACTION_FEE_PERCENT)
        self.state[index + STOCK_DIM + 1] += buy_amount
        self.cost += self.state[index + 1] * buy_amount * TRANSACTION_FEE_PERCENT
        self.trades += 1

    def step(self, actions):
        """
        Executes a single step in the trading environment.

        Args:
            actions (np.array): Array of actions taken by the agent (scaled between -1 and 1 for each stock).

        Returns:
            tuple: 
                - state (list): The updated state after the actions.
                - reward (float): The reward based on the change in total asset value.
                - terminal (bool): Whether the episode is over.
                - info (dict): Additional information, empty in this case.
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            self._save_results()
            return self.state, self.reward, self.terminal, {}

        actions = actions * HMAX_NORMALIZE
        begin_total_asset = self._calculate_total_asset()

        # Execute trades
        sell_index = np.argsort(actions)[:np.where(actions < 0)[0].shape[0]]
        buy_index = np.argsort(actions)[::-1][:np.where(actions > 0)[0].shape[0]]
        for index in sell_index:
            self._sell_stock(index, actions[index])
        for index in buy_index:
            self._buy_stock(index, actions[index])

        # Update environment state
        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.state = self._update_state()

        # Calculate reward
        end_total_asset = self._calculate_total_asset()
        self.reward = (end_total_asset - begin_total_asset) * REWARD_SCALING
        self.asset_memory.append(end_total_asset)
        self.rewards_memory.append(self.reward)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        """
        Resets the environment to its initial state.

        Returns:
            list: The initial state of the environment.
        """
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.state = (
            [INITIAL_ACCOUNT_BALANCE]
            + self.data.adjcp.values.tolist()
            + [0] * STOCK_DIM
            + self.data.macd.values.tolist()
            + self.data.rsi.values.tolist()
            + self.data.cci.values.tolist()
            + self.data.adx.values.tolist()
        )
        return self.state

    def render(self, mode="human"):
        """
        Renders the current state of the environment.

        Args:
            mode (str, optional): The mode in which to render the state. Defaults to "human".

        Returns:
            list: The current state of the environment.
        """
        return self.state

    def _seed(self, seed=None):
        """
        Initializes the random seed for the environment.

        Args:
            seed (int, optional): The seed for the random number generator.

        Returns:
            list: The list of seeds used.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _save_results(self):
        """
        Saves the results of the trading session, including plotting the asset value 
        and saving it to a file, and storing the daily return data as a CSV.
        """
        plt.plot(self.asset_memory, "r")
        plt.savefig("results/account_value_train.png")
        plt.close()

        df_total_value = pd.DataFrame(self.asset_memory, columns=["account_value"])
        df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
        df_total_value.to_csv("results/account_value_train.csv")

    def _calculate_total_asset(self):
        """
        Calculates the total asset value, which is the sum of the available cash 
        and the value of the owned stocks.

        Returns:
            float: The total asset value.
        """
        return self.state[0] + sum(
            np.array(self.state[1 : STOCK_DIM + 1]) * np.array(self.state[STOCK_DIM + 1 : STOCK_DIM * 2 + 1])
        )

    def _update_state(self):
        """
        Updates the environment's state with the latest stock data and financial indicators.

        Returns:
            list: The updated state of the environment.
        """
        return (
            [self.state[0]]
            + self.data.adjcp.values.tolist()
            + list(self.state[STOCK_DIM + 1 : STOCK_DIM * 2 + 1])
            + self.data.macd.values.tolist()
            + self.data.rsi.values.tolist()
            + self.data.cci.values.tolist()
            + self.data.adx.values.tolist()
        )

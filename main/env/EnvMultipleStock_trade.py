import gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

from run_DRL import tickers_list

# Shares normalization factor (100 shares per trade)
HMAX_NORMALIZE = 100
# Initial amount of money in our account
INITIAL_ACCOUNT_BALANCE = 1000000
# Total number of stocks in our portfolio
STOCK_DIM = len(tickers_list)
# Transaction fee percentage (0.1%)
TRANSACTION_FEE_PERCENT = 0.001
# Scaling factor for reward
REWARD_SCALING = 1e-4


class StockEnvTrade(gym.Env):
    """
    A stock trading environment for OpenAI Gym.

    Attributes:
        day (int): The current trading day.
        df (pd.DataFrame): DataFrame containing stock data.
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
        # Shape: 1 (balance) + 30 (prices) + 30 (shares) + 30 (MACD) + 30 (RSI) + 30 (CCI) + 30 (ADX) = 181
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(181,))

        # Load the initial slice of data
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold

        # Initialize the state
        self.state = (
            [INITIAL_ACCOUNT_BALANCE]             # current balance
            + self.data["adjcp"].tolist()         # current adjusted close prices
            + [0] * STOCK_DIM                     # shares held (all zero at start)
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
            action (float): Sell action value (negative in principle).
        """
        if self.turbulence < self.turbulence_threshold:
            # If we own shares
            if self.state[index + STOCK_DIM + 1] > 0:
                # Sell only as many shares as we have
                sell_amount = min(abs(action), self.state[index + STOCK_DIM + 1])
                # Increase balance by (price * shares * (1 - fee))
                self.state[0] += self.state[index + 1] * sell_amount * (1 - TRANSACTION_FEE_PERCENT)
                # Decrease held shares
                self.state[index + STOCK_DIM + 1] -= sell_amount
                # Increase transaction cost
                self.cost += self.state[index + 1] * sell_amount * TRANSACTION_FEE_PERCENT
                self.trades += 1
        else:
            # If turbulence goes over threshold, clear out all positions
            if self.state[index + STOCK_DIM + 1] > 0:
                sell_amount = self.state[index + STOCK_DIM + 1]
                self.state[0] += self.state[index + 1] * sell_amount * (1 - TRANSACTION_FEE_PERCENT)
                self.cost += self.state[index + 1] * sell_amount * TRANSACTION_FEE_PERCENT
                self.state[index + STOCK_DIM + 1] = 0
                self.trades += 1

    def _buy_stock(self, index, action):
        """
        Perform a buy action.

        Args:
            index (int): Index of the stock to buy.
            action (float): Buy action value (positive).
        """
        if self.turbulence < self.turbulence_threshold:
            # Maximum shares we can buy with current balance
            available_amount = self.state[0] // self.state[index + 1]
            buy_amount = min(available_amount, action)
            self.state[0] -= self.state[index + 1] * buy_amount * (1 + TRANSACTION_FEE_PERCENT)
            self.state[index + STOCK_DIM + 1] += buy_amount
            self.cost += self.state[index + 1] * buy_amount * TRANSACTION_FEE_PERCENT
            self.trades += 1
        else:
            # If turbulence is too high, do not buy
            pass

    def step(self, actions):
        """
        Take a step in the environment.

        Args:
            actions (np.ndarray): Actions for each stock.

        Returns:
            tuple: (state, reward, terminal, info)
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            # --- Terminal: Save results & compute final stats ---

            # Plot the asset memory
            plt.plot(self.asset_memory, 'r')
            plt.title("Account Value Over Time")
            plt.savefig("results/account_value_trade_{}_{}.png".format(self.model_name, self.iteration))
            plt.close()

            # Save account values to CSV
            df_total_value = pd.DataFrame(self.asset_memory, columns=['account_value'])
            df_total_value.to_csv("results/account_value_trade_{}_{}.csv".format(self.model_name, self.iteration), index=False)

            # Compute daily returns and Sharpe ratio
            df_total_value['daily_return'] = df_total_value['account_value'].pct_change(1)
            sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            print("Sharpe Ratio:", sharpe)

            # Save rewards to CSV
            df_rewards = pd.DataFrame(self.rewards_memory, columns=['reward'])
            df_rewards.to_csv("results/account_rewards_trade_{}_{}.csv".format(self.model_name, self.iteration), index=False)

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1: (STOCK_DIM + 1)]) *
                np.array(self.state[(STOCK_DIM + 1): (STOCK_DIM * 2 + 1)])
            )
            print("Previous Total Asset:", self.asset_memory[0])
            print("End Total Asset:", end_total_asset)
            print("Total Reward:", end_total_asset - self.asset_memory[0])
            print("Total Cost:", self.cost)
            print("Total Trades:", self.trades)

            return self.state, self.reward, self.terminal, {}

        # --- Not terminal: Proceed with step logic ---
        actions = actions * HMAX_NORMALIZE

        # If turbulence is high, force all sell
        if self.turbulence >= self.turbulence_threshold:
            actions = np.array([-HMAX_NORMALIZE] * STOCK_DIM)

        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1:(STOCK_DIM + 1)]) *
            np.array(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])
        )

        # Sort actions: sell (negative) from lowest, buy (positive) from highest
        sell_index = np.argsort(actions)[: np.where(actions < 0)[0].shape[0]]
        buy_index = np.argsort(actions)[::-1][: np.where(actions > 0)[0].shape[0]]

        # Execute sells
        for index in sell_index:
            self._sell_stock(index, actions[index])

        # Execute buys
        for index in buy_index:
            self._buy_stock(index, actions[index])

        self.day += 1
        self.data = self.df.loc[self.day, :]
        # Update turbulence
        self.turbulence = self.data["turbulence"].iloc[0]

        # Build next state
        self.state = (
            [self.state[0]]
            + self.data["adjcp"].tolist()
            + list(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])
            + self.data["macd"].tolist()
            + self.data["rsi"].tolist()
            + self.data["cci"].tolist()
            + self.data["adx"].tolist()
        )

        end_total_asset = self.state[0] + sum(
            np.array(self.state[1:(STOCK_DIM + 1)]) *
            np.array(self.state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])
        )
        self.asset_memory.append(end_total_asset)

        # Reward is the increase in total asset
        self.reward = end_total_asset - begin_total_asset
        self.rewards_memory.append(self.reward)

        # Scale the reward
        self.reward *= REWARD_SCALING

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        """
        Reset the environment.

        Returns:
            list: The initial state of the environment.
        """
        if self.initial:
            # Fresh start
            self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            self.rewards_memory = []

            self.state = (
                [INITIAL_ACCOUNT_BALANCE]
                + self.data["adjcp"].tolist()
                + [0] * STOCK_DIM
                + self.data["macd"].tolist()
                + self.data["rsi"].tolist()
                + self.data["cci"].tolist()
                + self.data["adx"].tolist()
            )
        else:
            # Use previous_state to resume
            previous_total_asset = (
                self.previous_state[0] +
                sum(
                    np.array(self.previous_state[1:(STOCK_DIM + 1)]) *
                    np.array(self.previous_state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)])
                )
            )
            self.asset_memory = [previous_total_asset]
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.turbulence = 0
            self.cost = 0
            self.trades = 0
            self.terminal = False
            self.rewards_memory = []

            self.state = (
                [self.previous_state[0]]
                + self.data["adjcp"].tolist()
                + self.previous_state[(STOCK_DIM + 1):(STOCK_DIM * 2 + 1)]
                + self.data["macd"].tolist()
                + self.data["rsi"].tolist()
                + self.data["cci"].tolist()
                + self.data["adx"].tolist()
            )

        return self.state

    def render(self, mode="human"):
        """
        Render the environment (simply returns the current state).

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
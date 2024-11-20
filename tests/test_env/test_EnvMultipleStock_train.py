import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add the parent directory to the Python path to find 'env' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from env.EnvMultipleStock_train import StockEnvTrain

# Mock parameters
STOCK_DIM = 30
INITIAL_ACCOUNT_BALANCE = 1000000


@pytest.fixture
def mock_data():
    """Generate mock stock data for testing."""
    return pd.DataFrame(
        {
            "adjcp": [np.array([1.1, 2.2, 3.3]) for _ in range(STOCK_DIM)],
            "macd": [np.array([0.1, 0.2, 0.3]) for _ in range(STOCK_DIM)],
            "rsi": [np.array([30, 40, 50]) for _ in range(STOCK_DIM)],
            "cci": [np.array([100, 110, 120]) for _ in range(STOCK_DIM)],
            "adx": [np.array([20, 25, 30]) for _ in range(STOCK_DIM)],
        }
    )


@pytest.fixture
def stock_env(mock_data: pd.DataFrame) -> StockEnvTrain:
    """Initialize the StockEnvTrain environment with mock data."""
    return StockEnvTrain(df=mock_data, day=0)


def test_initialization(stock_env: StockEnvTrain) -> None:
    """Test environment initialization."""
    assert stock_env.state is not None
    assert (
        len(stock_env.state) == 181
    )  # State vector length should match defined observation space
    assert stock_env.asset_memory == [INITIAL_ACCOUNT_BALANCE]


def test_reset(stock_env: StockEnvTrain) -> None:
    """Test the reset functionality."""
    initial_state = stock_env.reset()
    assert len(initial_state) == 181
    assert stock_env.day == 0
    assert stock_env.terminal is False
    assert stock_env.asset_memory == [INITIAL_ACCOUNT_BALANCE]


def test_step(stock_env: StockEnvTrain) -> None:
    """Test the step function."""
    action = np.zeros(stock_env.action_space.shape)  # No buy/sell action
    next_state, reward, done, info = stock_env.step(action)
    assert len(next_state) == 181
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert stock_env.trades == 0  # No trades for zero action


def test_step_with_trade(stock_env: StockEnvTrain) -> None:
    """Test the step function with buy and sell actions."""
    action = np.random.uniform(-1, 1, stock_env.action_space.shape)  # Random actions
    next_state, reward, done, info = stock_env.step(action)
    assert len(next_state) == 181
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert stock_env.trades > 0  # Expect at least some trades for non-zero actions


def test_terminal_state(stock_env: StockEnvTrain) -> None:
    """Test the environment behavior in terminal state."""
    stock_env.day = len(stock_env.df.index.unique()) - 1  # Force terminal condition
    action = np.zeros(stock_env.action_space.shape)
    _, _, done, _ = stock_env.step(action)
    assert done is True


def test_render(stock_env: StockEnvTrain, capsys: pytest.CaptureFixture) -> None:
    """Test the render function output."""
    stock_env.render()
    captured = capsys.readouterr()
    assert captured.out != ""  # Ensure render produces output


def test_buy_sell_logic(stock_env: StockEnvTrain) -> None:
    """Test buying and selling actions."""
    stock_env.state = (
        [INITIAL_ACCOUNT_BALANCE] + [10] * STOCK_DIM + [0] * STOCK_DIM + [0] * 90
    )
    stock_env._buy_stock(0, 5)  # Buy 5 shares
    assert stock_env.state[STOCK_DIM + 1] == 5  # Check if shares were updated
    stock_env._sell_stock(0, 3)  # Sell 3 shares
    assert stock_env.state[STOCK_DIM + 1] == 2  # Remaining shares should be 2


def test_reward_calculation(stock_env: StockEnvTrain) -> None:
    """Test reward calculation after a step."""
    initial_asset = sum(stock_env.state[1 : (STOCK_DIM + 1)])
    action = np.random.uniform(-1, 1, stock_env.action_space.shape)
    _, reward, _, _ = stock_env.step(action)
    assert isinstance(reward, float)
    assert reward != 0  # Reward should reflect changes in portfolio value

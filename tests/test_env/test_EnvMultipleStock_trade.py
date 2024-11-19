import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add the parent directory to the Python path to find 'env' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from env.EnvMultipleStock_trade import StockEnvTrade

# Fixed seed for reproducibility
SEED = 42


# Sample dataframe for testing
@pytest.fixture
def mock_data():
    np.random.seed(SEED)  # Set the seed
    num_days = 5
    stock_dim = 30
    data = {
        "adjcp": [np.random.rand(stock_dim) for _ in range(num_days)],
        "macd": [np.random.rand(stock_dim) for _ in range(num_days)],
        "rsi": [np.random.rand(stock_dim) for _ in range(num_days)],
        "cci": [np.random.rand(stock_dim) for _ in range(num_days)],
        "adx": [np.random.rand(stock_dim) for _ in range(num_days)],
        "turbulence": [np.random.rand(1) for _ in range(num_days)],
    }
    return pd.DataFrame(data)


@pytest.fixture
def stock_env(mock_data: pd.DataFrame) -> StockEnvTrade:
    """Create a StockEnvTrade instance for testing."""
    return StockEnvTrade(df=mock_data, day=0)


def test_initialization(stock_env: StockEnvTrade, mock_data: pd.DataFrame) -> None:
    """Test the environment's initialization."""
    assert stock_env.day == 0
    assert stock_env.state[0] == 1000000  # Initial account balance
    assert len(stock_env.state) == 181  # Observation space dimensions
    assert stock_env.data.equals(mock_data.loc[0, :])  # First day data loaded


def test_reset(stock_env: StockEnvTrade) -> None:
    """Test the environment's reset method."""
    initial_state = stock_env.reset()
    assert initial_state[0] == 1000000  # Reset should set initial balance
    assert len(initial_state) == 181  # Ensure correct state size
    assert stock_env.day == 0  # Reset day to 0


def test_step(stock_env: StockEnvTrade) -> None:
    """Test the step method with random actions."""
    np.random.seed(SEED)  # Set the seed for actions
    random_actions = np.random.uniform(-1, 1, size=(30,))
    state, reward, terminal, info = stock_env.step(random_actions)

    assert len(state) == 181  # Check state size
    assert isinstance(reward, float)  # Reward should be a float
    assert isinstance(terminal, bool)  # Terminal should be a boolean
    assert not terminal  # Since we are not at the last day yet


def test_terminal_state(stock_env: StockEnvTrade, mock_data: pd.DataFrame) -> None:
    """Test terminal state handling."""
    stock_env.day = len(mock_data.index.unique()) - 1  # Move to the last day
    np.random.seed(SEED)  # Set the seed for actions
    random_actions = np.random.uniform(-1, 1, size=(30,))
    _, _, terminal, _ = stock_env.step(random_actions)
    assert terminal  # Should return True when terminal state is reached


def test_render(stock_env: StockEnvTrade) -> None:
    """Test the render method."""
    state = stock_env.render()
    assert isinstance(state, list) or isinstance(
        state, np.ndarray
    )  # Ensure state is valid
    assert len(state) == 181  # State dimensions match observation space

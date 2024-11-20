import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add the parent directory to the Python path to find 'env' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from main.env.EnvMultipleStock_trade import StockEnvTrade

# Fixed seed for reproducibility
SEED = 42


# Sample dataframe for testing


# Sample dataframe for testing
@pytest.fixture
def mock_data():
    np.random.seed(SEED)  # Set the seed
    num_days = 5
    stock_dim = 30
    dates = []
    for i in range(num_days):
        for _ in range(stock_dim):
            dates.append(i)
    tics = [f"stock_{i}" for i in range(stock_dim)]
    turbulence = []
    for i in range(num_days):
        turb = np.random.rand()
        for _ in range(stock_dim):
            turbulence.append(turb)
    data = {
        "datadate": dates,
        "tic": tics * num_days,
        "adjcp": np.random.rand(num_days * stock_dim),
        "open": np.random.rand(num_days * stock_dim),
        "high": np.random.rand(num_days * stock_dim),
        "low": np.random.rand(num_days * stock_dim),
        "volume": np.random.rand(num_days * stock_dim) * 1e7,
        "macd": np.random.rand(num_days * stock_dim),
        "rsi": np.random.rand(num_days * stock_dim),
        "cci": np.random.rand(num_days * stock_dim),
        "adx": np.random.rand(num_days * stock_dim),
        "turbulence": turbulence,
    }
    df = pd.DataFrame(data)
    df.index = df["datadate"].factorize()[0]
    return df


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
    state, reward, terminal, _ = stock_env.step(random_actions)

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

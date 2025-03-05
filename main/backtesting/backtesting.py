import pandas as pd
import numpy as np

def get_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    df['daily_return']=df['account_value'].pct_change(1)
    #df=df.dropna()
    print('Sharpe: ',(252**0.5)*df['daily_return'].mean()/ df['daily_return'].std())
    return df

def backtest_strat(df: pd.DataFrame) -> pd.Series:
    strategy_ret= df.copy()
    strategy_ret['Date'] = pd.to_datetime(strategy_ret['Date'])
    strategy_ret.set_index('Date', drop = False, inplace = True)
    strategy_ret.index = strategy_ret.index.tz_localize('UTC')
    del strategy_ret['Date']
    ts = pd.Series(strategy_ret['daily_return'].values, index=strategy_ret.index)
    return ts

def get_account_value(model_name, rebalance_window, validation_window, unique_trade_date, df_trade_date, date):
    df_account_value=pd.DataFrame()
    for i in range(rebalance_window+validation_window, len(unique_trade_date)+1,rebalance_window):
        temp = pd.read_csv('results/{}/account_value_trade_{}_{}.csv'.format(date, model_name,i))
        df_account_value = pd.concat([df_account_value, temp], ignore_index=True)
    sharpe=(252**0.5)*df_account_value['account_value'].pct_change(1).mean()/df_account_value.account_value.pct_change(1).std()
    print(sharpe)
    df_account_value=df_account_value.join(df_trade_date[63:].reset_index(drop=True))
    return df_account_value


def calculate_portfolio_metrics(ensemble_account_value: pd.DataFrame, model_name: str):

    # Calculate cumulative return
    cumulative_return = (ensemble_account_value['account_value'].iloc[-1] / ensemble_account_value['account_value'].iloc[0]) - 1

    # Calculate annual return
    average_daily_return = ensemble_account_value['daily_return'].mean()
    annual_return = (1 + average_daily_return) ** 252 - 1  # Assuming 252 trading days in a year

    # Calculate annual volatility
    annual_volatility = ensemble_account_value['daily_return'].std() * np.sqrt(252)

    # Calculate Sharpe ratio (assuming a risk-free rate of 0 for simplicity)
    risk_free_rate = 0
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility

    # Calculate max drawdown
    cumulative_max = ensemble_account_value['account_value'].cummax()
    drawdown = (ensemble_account_value['account_value'] / cumulative_max) - 1
    max_drawdown = drawdown.min()

    # Return the results as a dictionary
    return {
        'model_name': model_name,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'cumulative_return': cumulative_return,
        'max_drawdown': max_drawdown
    }

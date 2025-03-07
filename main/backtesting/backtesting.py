import pandas as pd
import numpy as np
import matplotlib.dates as mdates
import yfinance as yf
from main.backtesting.backtesting import *
from main.config.config import *
import matplotlib.pyplot as plt

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
        temp = pd.read_csv('results/{}/account_value_trade/account_value_trade_{}_{}.csv'.format(date, model_name,i))
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

# Calculate the scaling factors and normalize the account values
def normalize_account_value(df, initial_value):
    scaling_factor = initial_value / df['account_value'].iloc[0]
    df['account_value_normalized'] = df['account_value'] * scaling_factor
    return df

# Function to save the portfolio vs DJIA plot
def save_portfolio_vs_djia_plot(ensemble_account_value, a2c_account_value, ppo_account_value, ddpg_account_value, file_path):

    # Convert 'datadate' to datetime if it's not already
    ensemble_account_value['datadate'] = pd.to_datetime(ensemble_account_value['datadate'])
    a2c_account_value['datadate'] = pd.to_datetime(a2c_account_value['datadate'])
    ppo_account_value['datadate'] = pd.to_datetime(ppo_account_value['datadate'])
    ddpg_account_value['datadate'] = pd.to_datetime(ddpg_account_value['datadate'])
    # Fetch DJIA data using yfinance
    djia_ticker = yf.Ticker('^DJI')
    djia_data = djia_ticker.history(start=ensemble_account_value['datadate'].min(), end=ensemble_account_value['datadate'].max())
    djia_data.reset_index(inplace=True)

    # Normalize the data
    initial_value = 1e6
    ensemble_account_value = normalize_account_value(ensemble_account_value, initial_value)
    a2c_account_value = normalize_account_value(a2c_account_value, initial_value)
    ppo_account_value = normalize_account_value(ppo_account_value, initial_value)
    ddpg_account_value = normalize_account_value(ddpg_account_value, initial_value)
    djia_scaling_factor = initial_value / djia_data['Close'].iloc[0]
    djia_data['Close_normalized'] = djia_data['Close'] * djia_scaling_factor

    # Plot the data
    plt.figure(figsize=(14, 7))
    plt.plot(ensemble_account_value['datadate'], ensemble_account_value['account_value_normalized'], label='Ensemble Portfolio Value', linestyle='--')
    plt.plot(a2c_account_value['datadate'], a2c_account_value['account_value_normalized'], label='A2C Portfolio Value', linestyle='--')
    plt.plot(ppo_account_value['datadate'], ppo_account_value['account_value_normalized'], label='PPO Portfolio Value', linestyle='--')
    plt.plot(ddpg_account_value['datadate'], ddpg_account_value['account_value_normalized'], label='DDPG Portfolio Value', linestyle='--')
    plt.plot(djia_data['Date'], djia_data['Close_normalized'], label='Dow Jones Industrial Average', color='orange', linestyle='--')

    # Add axis titles and legend
    plt.xlabel("Dates")
    plt.ylabel("Value")
    plt.title("Portfolio and DJIA Over Time (Normalized)")
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7], bymonthday=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.legend()

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()

# Function to save the combined DataFrame
def save_combined_df(ppo_sharpe_list, a2c_sharpe_list, ddpg_sharpe_list, model_use, ensemble_account_value, file_path):
    start_dates = []
    end_dates = []

    for i in range(len(ppo_sharpe_list)):
        start_date = ensemble_account_value['datadate'].iloc[i * 63]
        end_date = ensemble_account_value['datadate'].iloc[min((i + 1) * 63 - 1, len(ensemble_account_value) - 1)]
        start_dates.append(start_date)
        end_dates.append(end_date)

    ppo_sharpe_list['start_date'] = start_dates
    ppo_sharpe_list['end_date'] = end_dates
    a2c_sharpe_list['start_date'] = start_dates
    a2c_sharpe_list['end_date'] = end_dates
    ddpg_sharpe_list['start_date'] = start_dates
    ddpg_sharpe_list['end_date'] = end_dates
    model_use['start_date'] = start_dates
    model_use['end_date'] = end_dates

    ppo_sharpe_list.rename(columns={'0': 'ppo'}, inplace=True)
    a2c_sharpe_list.rename(columns={'0': 'a2c'}, inplace=True)
    ddpg_sharpe_list.rename(columns={'0': 'ddpg'}, inplace=True)
    model_use.rename(columns={'0': 'model'}, inplace=True)

    combined_df = pd.DataFrame({
        'start_date': ppo_sharpe_list['start_date'],
        'end_date': ppo_sharpe_list['end_date'],
        'ppo': ppo_sharpe_list['ppo'],
        'a2c': a2c_sharpe_list['a2c'],
        'ddpg': ddpg_sharpe_list['ddpg'],
        'model': model_use['model']
    })

    combined_df.to_csv(file_path, index=False)

# Function to save the metrics DataFrame
def save_metrics_df(ensemble_account_value, a2c_account_value, ppo_account_value, ddpg_account_value, file_path):
    ensemble_metrics = calculate_portfolio_metrics(ensemble_account_value, 'ensemble')
    a2c_metrics = calculate_portfolio_metrics(a2c_account_value, 'a2c')
    ppo_metrics = calculate_portfolio_metrics(ppo_account_value, 'ppo')
    ddpg_metrics = calculate_portfolio_metrics(ddpg_account_value, 'ddpg')

    djia_ticker = yf.Ticker('^DJI')
    djia_data = djia_ticker.history(start=ensemble_account_value['datadate'].min(), end=ensemble_account_value['datadate'].max())
    djia_data.reset_index(inplace=True)
    djia_data.rename(columns={'Date': 'datadate', 'Close': 'account_value'}, inplace=True)
    djia_data = get_daily_return(djia_data)
    dji_metrics = calculate_portfolio_metrics(djia_data, 'DJI')

    metrics_df = pd.DataFrame([ensemble_metrics, a2c_metrics, ppo_metrics, ddpg_metrics, dji_metrics])
    metrics_df = metrics_df.set_index('model_name').T
    metrics_df.to_csv(file_path, index=True)

# Final function to save all results
def save_all_results(start_date, end_date, rebalance_window, validation_window, validation_date, preprocessed_data,Csv_files_dir, date):
    preprocessed_path = "main/data/" + preprocessed_data + start_date + "_" + end_date + ".csv"
    df = pd.read_csv(preprocessed_path)

    # Format the dates
    unique_trade_date = df[(df.datadate > format_date(validation_date)) & (df.datadate <= format_date(end_date))].datadate.unique()
    df_trade_date = pd.DataFrame({'datadate': unique_trade_date})
    
    # Get account values for each model
    ensemble_account_value = get_account_value('ensemble', rebalance_window, validation_window, unique_trade_date, df_trade_date, date)
    a2c_account_value = get_account_value('a2c', rebalance_window, validation_window, unique_trade_date, df_trade_date, date)
    ppo_account_value = get_account_value('ppo', rebalance_window, validation_window, unique_trade_date, df_trade_date, date)
    ddpg_account_value = get_account_value('ddpg', rebalance_window, validation_window, unique_trade_date, df_trade_date, date)

    # Calculate daily returns for each model
    ensemble_account_value = get_daily_return(ensemble_account_value)
    a2c_account_value = get_daily_return(a2c_account_value)
    ppo_account_value = get_daily_return(ppo_account_value)
    ddpg_account_value = get_daily_return(ddpg_account_value)

    # Convert the datadate back to the original format
    ensemble_account_value['datadate'] = ensemble_account_value['datadate'].apply(invert_format_date)
    a2c_account_value['datadate'] = a2c_account_value['datadate'].apply(invert_format_date)
    ppo_account_value['datadate'] = ppo_account_value['datadate'].apply(invert_format_date)
    ddpg_account_value['datadate'] = ddpg_account_value['datadate'].apply(invert_format_date)

    # Output the results
    print("Ensemble Account Value:\n", ensemble_account_value.head())
    print("A2C Account Value:\n", a2c_account_value.head())
    print("PPO Account Value:\n", ppo_account_value.head())
    print("DDPG Account Value:\n", ddpg_account_value.head())

    ppo_sharpe_list = pd.read_csv('results/{}/ppo_sharpe_list.csv'.format(date))
    a2c_sharpe_list = pd.read_csv('results/{}/a2c_sharpe_list.csv'.format(date))
    ddpg_sharpe_list = pd.read_csv('results/{}/ddpg_sharpe_list.csv'.format(date))
    model_use = pd.read_csv('results/{}/model_use.csv'.format(date))

    save_portfolio_vs_djia_plot(ensemble_account_value, a2c_account_value, ppo_account_value, ddpg_account_value, '{}/report/portfolio_vs_djia_vs_ppo_vs_a2c_vs_ddpg.png'.format(Csv_files_dir))
    save_combined_df(ppo_sharpe_list, a2c_sharpe_list, ddpg_sharpe_list, model_use, ensemble_account_value, '{}/report/results_algos.csv'.format(Csv_files_dir))
    save_metrics_df(ensemble_account_value, a2c_account_value, ppo_account_value, ddpg_account_value, '{}/report/results_metrics.csv'.format(Csv_files_dir))

# Example usage
# Load your data here   
# save_all_results(ensemble_account_value, a2c_account_value, ppo_account_value, ddpg_account_value, ppo_sharpe_list, a2c_sharpe_list, ddpg_sharpe_list, model_use)

# def get_states(Csv_files_dir, date, preprocessed_data, start_date, end_date, validation_date, rebalance_window, validation_window):
#     preprocessed_path = "main/data/" + preprocessed_data + start_date + "_" + end_date + ".csv"
#     df = pd.read_csv(preprocessed_path)
#     df_state = pd.DataFrame()
#     # Format the dates
#     unique_trade_date = df[(df.datadate > format_date(validation_date)) & (df.datadate <= format_date(end_date))].datadate.unique()
#     for i in range(rebalance_window+validation_window, len(unique_trade_date)+1):
#         temp = pd.read_csv('results/{}/account_value_trade/account_value_trade_{}_{}.csv'.format(date, model_name,i))
#         df_state = pd.concat([df_state, temp], ignore_index=True)
import pandas as pd

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


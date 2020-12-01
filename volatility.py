# Importing external libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def get_stocks_volatility_rank(data, year=2017, top_n=10):
    """
        Get 'top_n' most and least volatile stocks by 'year'
    """
    data1 = data.get_group(year)
    final_pivot = data1.pivot(index="date", columns="Name", values="close")
    daily_volatility = final_pivot.pct_change().apply(lambda x: np.log(1+x)).std()
    weekly_volatility = daily_volatility.apply(lambda x: x*np.sqrt(5))
    WV = pd.DataFrame(weekly_volatility).reset_index()
    WV.columns = ["Name", "Volatility"]
    most_n_volatile = WV.sort_values(by="Volatility", ascending=False)[:top_n]
    least_n_volatile = WV.sort_values(by="Volatility", ascending=True)[:top_n]

    return most_n_volatile, least_n_volatile


def mavg_plot(stock_name="BHGE"):
    """
        Create moving average plot
    """
    df = df1[df1['Name'] == stock_name]
    close_px = df['close']
    mavg = close_px.rolling(window=7).mean()

    # adjusting for look-ahead bias
    pos = mavg.apply(np.sign)
    pos /= pos.abs().values

    # plot
    plt.figure(figsize=(15, 8))
    plt.plot(df['date'], pos.shift(1)*df['close'], label='BHGE')
    plt.plot(df['date'], pos.shift(1)*mavg, label='Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.title('')
    plt.legend()
    plt.show()


def plot_ret_dev(stock_name="BHGE"):
    """
        Create returns deviation plot
    """
    df = df1[df1['Name'] == stock_name]
    close_px = df['close']
    mavg = close_px.rolling(window=7).mean()

    # adjusting for look-ahead bias
    pos = mavg.apply(np.sign)
    pos /= pos.abs().values

    rets = close_px / close_px.shift(1)-1

    plt.figure(figsize=(15, 8))
    plt.plot(df['date'], pos.shift(1)*rets, label='return')


# read data from csv
df1 = pd.read_csv('data.csv')

# convert date to proper format
df1['date'] = pd.to_datetime(df1['date'])
df1['year'] = df1['date'].dt.year

# preprocessing data for calculating volatility
count_df = pd.DataFrame(df1.Name.value_counts()[:], columns=[
                        "Name", "Count"]).reset_index()
list_valid_shares = list(count_df["index"])
final_df = df1[df1.Name.isin(list_valid_shares)]
data_by_year = final_df.groupby("year")

# running function
most_volatile, least_volatile = get_stocks_volatility_rank(
    data_by_year, year=2017, top_n=10)

# print final result
print("Top 10 Most Volatile stocks:\n", most_volatile,
      "\n\nTop 10 Least Volatile stocks:\n", least_volatile)

## Analysis of one of the most volatile stock ##
# show moving average plot
mavg_plot("BHGE")

# show returns deviation plot
plot_ret_dev("BHGE")

import os
import requests
import pandas as pd
import numpy as np 
import seaborn as sns 
import hvplot.pandas
import matplotlib.pyplot as plt
#!pip install fdpf
from MCForecastTools import MCSimulation
import yfinance as yf
from finta import TA

#%matplotlib inline
#streamlit or tableau of 

import yfinance as yf
from yahoofinancials import YahooFinancials
with st.expander(stocks):
    xeroasx=pd.DataFrame()
    xeroasx = raw_data[stocks]

    # Set 9 as the lookback period of the short-term moving average
    shortterm_lookback = 9

    # Set 21 as the lookback period of the long-term moving average
    longterm_lookback = 21

    # Store short-term moving average values in the column 'ma_short'
    xeroasx['ma_short'] = xeroasx['Close'].rolling(shortterm_lookback).mean()

    # Store long-term moving average values in the column 'ma_long'
    xeroasx['ma_long'] = xeroasx['Close'].rolling(longterm_lookback).mean()

    # Create a column 'signal' to store the relative position of moving averages
    xeroasx['signal'] = np.where(xeroasx.ma_short > xeroasx.ma_long, 1, 0)

    days = 400

    # Plot the close prices of the last 100 days
    close_plot = xeroasx.Close[-days:].plot(figsize=(15, 7), color='blue')

    # Plot the signal of the last 100 days
    signal_plot = xeroasx.signal[-days:].plot(figsize=(15, 7),
                                           secondary_y=True, ax=close_plot, style='green')

    # Highlight the holding periods of the long positions
    plt.fill_between(xeroasx.Close[-days:].index, 0, 1,
                     where=(xeroasx.signal[-days:] > 0), color='green', alpha=0.1, lw=0)

    # Title of the plot
    signal_plot.set_title('Visualising The Holding Periods of The Long Positions')

    # Plot ylabels
    close_plot.set_ylabel('Price ($)')
    signal_plot.set_ylabel('Signal')

    # Legend of the plot
    plt.legend()

    # Show the plot
    st.pyplot(plt.show())

    # Calculate strategy returns
    strategy_returns = xeroasx.signal.shift(1)*xeroasx.Close.pct_change()

    # Total strategy returns
    print(f'Total strategy returns: {strategy_returns.sum():.2f} %')

    # Create a column 'long_cross_over' to store the long crossover conditions
    xeroasx['long_cross_over'] = np.where((xeroasx.ma_short.shift(
        1) < xeroasx.ma_long.shift(1)) & (xeroasx.ma_short >= xeroasx.ma_long), True, False)

    # Create a column 'exit_cross_over' to store the long exit conditions
    xeroasx['exit_cross_over'] = np.where((xeroasx.ma_short.shift(
        1) > xeroasx.ma_long.shift(1)) & (xeroasx.ma_short <= xeroasx.ma_long), True, False)
    xeroasx.tail()

    # Create a dataframe 'trade_sheet' to store the trades
    trade_sheet = pd.DataFrame()

    # Initialise the current_position as '0' ie we dont hold any assets yet 
    current_position = 0

    # Define a variable to store the long entry date
    entry_date = ''

    # Define a variable to store the long entry price
    entry_price = ''

    # Define a variable to store the long exit date
    exit_date = ''

    # Define a variable to store the long exit price
    exit_price = ''

    def backtest_trade_sheet(xeroasx, close_column, long_crossover_column, exit_crossover_column):
        # Create a dataframe 'trade_sheet' to store the trades
        trade_sheet = pd.DataFrame()

        # Initialise the current_position as '0'
        current_position = 0

        # Define a variable to store the long entry date
        entry_date = ''

        # Define a variable to store the long entry price
        entry_price = ''

        # Define a variable to store the long exit date
        exit_date = ''

        # Define a variable to store the long exit price
        exit_price = ''

        # Iterate over the dates in the dataframe 'data' 
        for current_date in xeroasx.index:

            # Define the variable 'long_crossover' that stores the long crossover condition on the current_date
            long_crossover = xeroasx.loc[current_date, long_crossover_column]

            # Define the variable 'exit_crossover' that stores the exit crossover value on date current_date
            exit_crossover = xeroasx.loc[current_date, exit_crossover_column]

            # We will enter the long position if we are not holding any position and entry condition is met
            if current_position == 0 and long_crossover == True:

                # Define the variable 'entry_date'
                entry_date = current_date

                # Extract the 'Close price' on the current_date and store in the variable 'entry price'
                entry_price = xeroasx.loc[entry_date, close_column]

                # Since a new long position is opened, change the state of current_position to '1'
                current_position = 1

            # We will exit the long position if we are holding long position and exit condition is met
            elif current_position == 1 and exit_crossover == True:

                # Define the variable 'exit_date'
                exit_date = current_date

                # Extract the 'Close price' on the current_date and store in the variable 'exit price'
                exit_price = xeroasx.loc[exit_date, close_column]

                # Append the details of this trade to the 'trades' dataframe
                trade_sheet = trade_sheet.append(
                    [(current_position, entry_date, entry_price, exit_date, exit_price)], ignore_index=True)

                # Since a new long position is closed, change the state of current_position to '0'
                current_position = 0

        # Define the names of columns in 'trades' dataframe
        trade_sheet.columns = ['Position', 'Entry Date',
                               'Entry Price', 'Exit Date', 'Exit Price']
        # Return the trades dataframe
        return trade_sheet

    # Save the trades generated in the dataframe 'crossover_trade_sheet'
    crossover_trade_sheet = backtest_trade_sheet(
        xeroasx, 'Close', 'long_cross_over', 'exit_cross_over')

    # Calculate PnL for each trade
    crossover_trade_sheet['PnL'] = (crossover_trade_sheet['Exit Price'] -
                                    crossover_trade_sheet['Entry Price']) * crossover_trade_sheet['Position']

    # Print the total profit/loss by summing up the PnL of each trade.
    print(f'The total PnL of trades generated between 2010-01-01 and 2022-05-31 is $',
          round(crossover_trade_sheet.PnL.sum(), 2))

    # Print the last 5 rows of the 'crossover_trade_sheet' dataframe
    crossover_trade_sheet.tail()

    # Create dataframe to store trade analytics
    analytics = pd.DataFrame(index=['Analyse'])

    # Calculate total PnL
    analytics['Total PnL'] = crossover_trade_sheet.PnL.sum() 

    # Print the value
    print("Total PnL: ",analytics['Total PnL'][0])

    # Number of total trades
    analytics['total_trades'] = len(crossover_trade_sheet.loc[crossover_trade_sheet.Position==1])

    # Winning trades
    analytics['Number of Winners'] = len(crossover_trade_sheet.loc[crossover_trade_sheet.PnL>0])

    # Loosing trades
    analytics['Number of Losers'] = len(crossover_trade_sheet.loc[crossover_trade_sheet.PnL<=0])

    # Winning percentage
    analytics['Win (%)'] = 100*analytics['Number of Winners']/analytics.total_trades

    # Lossing percentage
    analytics['Loss (%)'] = 100*analytics['Number of Losers']/analytics.total_trades



    analytics.T

    # Create a dataframe to store performance metrics
    performance_metrics = pd.DataFrame(index=['Outcomes'])

    # Calculate strategy returns
    xeroasx['Strategy_Returns'] = xeroasx.signal.shift(1) * xeroasx.Close.pct_change()

    # Calculate cumulative strategy returns
    xeroasx['Cumulative_Returns'] = (xeroasx['Strategy_Returns'] + 1.0).cumprod()

    # Plot the cumulative strategy returns
    (xeroasx['Cumulative_Returns'].plot(figsize=(15, 7), color='purple'))
    plt.title('Equity Curve', fontsize=14)
    plt.ylabel('Cumulative Returns')
    st.pyplot(plt.show())

    # Total number of trading days
    days = len(xeroasx['Cumulative_Returns'])

    # Calculate compounded annual growth rate
    performance_metrics['CAGR'] = "{0:.2f}%".format(
        (xeroasx.Cumulative_Returns.iloc[-1]**(252/days)-1)*100)

    # Calculate annualised volatility
    performance_metrics['Annualised Volatility'] = "{0:.2f}%".format(
        xeroasx['Strategy_Returns'].std()*np.sqrt(252) * 100)

    # Set a risk-free rate
    risk_free_rate = 0.02/365

    # Calculate Sharpe ratio
    performance_metrics['Sharpe Ratio'] = np.sqrt(252)*(np.mean(xeroasx.Strategy_Returns) -
                                                        (risk_free_rate))/np.std(xeroasx.Strategy_Returns)

    # Compute the cumulative max drawdown
    xeroasx['Peak'] = xeroasx['Cumulative_Returns'].cummax()

    # Compute the Drawdown
    xeroasx['Drawdown'] = ((xeroasx['Cumulative_Returns']-xeroasx['Peak'])/xeroasx['Peak'])

    # Compute the maximum drawdown
    performance_metrics['Maximum Drawdown'] =  "{0:.2f}%".format((xeroasx['Drawdown'].min())*100)

    performance_metrics.T

    # Plot a returns histogram
    xeroasx.Strategy_Returns.hist(bins=50, figsize=(14, 8), color='royalblue')
    plt.title('Histogram of Returns', fontsize=14)
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    st.pyplot(plt.show())

    # Define size of graph
    plt.figure(figsize=(12, 9))

    # Set the title and axis labels
    plt.title(f'Drawdowns for {stocks}', fontsize=14)
    plt.ylabel('Drawdown(%)', fontsize=12)
    plt.xlabel('Year', fontsize=12)

    # Plot max drawdown
    plt.plot(xeroasx['Drawdown'], color='red')

    # Fill in-between the drawdown
    plt.fill_between(xeroasx['Drawdown'].index, xeroasx['Drawdown'].values, color='orange')
    st.pyplot(plt.show())

    # Step-1: Create a dataframe 'trade_sheet' to store the trades
    trade_sheet = pd.DataFrame()

    # Step-2: Initialise the current_position as '0'
    current_position = 0

    #Step-3:
    # Define a variable to store the long entry date
    entry_date = ''

    # Define a variable to store the long entry price
    entry_price = ''

    # Define a variable to store the long exit date
    exit_date = ''

    # Define a variable to store the long exit price
    exit_price = ''

    # Define a variable to store the stop loss percentage
    stop_loss_percentage = 0.09

    # Define a variable to store the take profit percentage
    take_profit_percentage = 0.15

    # Iterate over the dates in the dataframe 'data'
    for current_date in xeroasx.index:

        # Define the variable 'long_crossover' that stores the long crossover condition on the current_date
        long_crossover = xeroasx.loc[current_date, 'long_cross_over']

        # Define the variable 'exit_crossover' that stores the exit crossover value on date current_date
        exit_crossover = xeroasx.loc[current_date, 'exit_cross_over']

        # Step-4: Check if we are holding a long position, we will check for stop-loss and take-profit breach
        if (current_position == 1):

            # Stop-loss = entry_price - (entry_price * stop_loss_percentage)
            stop_loss = entry_price * (1-stop_loss_percentage)

            # Take-profit = entry_price + (entry_price * take_profit_percentage)
            take_profit = entry_price * (1+take_profit_percentage)

            # stop-loss condition        
            stop_loss_breach = xeroasx.loc[current_date, 'Close'] <= stop_loss

            # take-profit condition        
            take_profit_breach = xeroasx.loc[current_date, 'Close'] >= take_profit        


            # if low is below stop-loss or high is above take-profit
            if stop_loss_breach or take_profit_breach:

                # Define the variable 'exit_date'
                exit_date = current_date

                # Extract the 'Close price' on the current_date and store in the variable 'exit price'
                exit_price = xeroasx.loc[exit_date, 'Close']

                # Exit type if 'Stop-Loss' if stop-loss was breached and 'Take-Profit' if take-profit was breached
                exit_type = "Stop-Loss" if stop_loss_breach else "Take-Profit"

                # Append the details of this trade to the 'trade_sheet' dataframe
                trade_sheet = trade_sheet.append(
                    [(current_position, entry_date, entry_price, exit_date, exit_price, exit_type)], ignore_index=True)

                # Since a new long position is closed, change the state of current_position to '0'
                current_position = 0
        # Step-5: We will enter the long position if we are not holding any position and entry condition is met
        if current_position == 0 and long_crossover == True:

            # Define the variable 'entry_date'
            entry_date = current_date

            # Extract the 'Close price' on the current_date and store in the variable 'entry price'
            entry_price = xeroasx.loc[entry_date, 'Close']

            # Since a new long position is opened, change the state of current_position to '1'
            current_position = 1

        # Step-6: We will exit the long position if we are holding long position and exit condition is met
        elif current_position == 1 and exit_crossover == True:

            # Define the variable 'exit_date'
            exit_date = current_date

            # Extract the 'Close price' on the current_date and store in the variable 'exit price'
            exit_price = xeroasx.loc[exit_date, 'Close']        

            # Exit type is 'exit_crossover'
            exit_type = 'exit_crossover'

            # Append the details of this trade to the 'trade_sheet' dataframe
            trade_sheet = trade_sheet.append(
                [(current_position, entry_date, entry_price, exit_date, exit_price, exit_type)], ignore_index=True)

            # Since a new long position is closed, change the state of current_position to '0'
            current_position = 0

    # Define the names of columns in 'trade_sheet' dataframe
    trade_sheet.columns = ['Position', 'Entry Date',
                      'Entry Price', 'Exit Date', 'Exit Price', 'Exit Type']

    trade_sheet.tail()


    def backtest_trade_sheet_sl_tp(xeroasx, close_column, long_crossover_column,exit_crossover_column, stop_loss_percentage, take_profit_percentage):
        """Function to generate trade details
        """
        # Create a dataframe  to store the trades
        crossover_trade_sheet = pd.DataFrame()

        # Initialise the current_position as '0' since we don't hold any position at the beginning of the backtest
        current_position = 0
        stop_loss_breach = False 
        take_profit_breach = False 

        # Iterate over the dates in the dataframe 'data'
        for current_date in xeroasx.index:
    # Define the variable 'long_crossover' that stores the long crossover condition on current_date
            long_crossover = xeroasx.loc[current_date, long_crossover_column]

    # Define the variable 'exit_crossover' that stores the exit crossover condition on current_date
            exit_crossover = xeroasx.loc[current_date, exit_crossover_column]

    # Check if we are holding a long position
            if (current_position == 1):

            # Stop-loss = entry_price - (entry_price * stop_loss_percentage)
                stop_loss = entry_price * (1-stop_loss_percentage)

            # Take-profit = entry_price + (entry_price * take_profit_percentage)
                take_profit = entry_price * (1+take_profit_percentage)

            # stop-loss condition        
                stop_loss_breach = xeroasx.loc[current_date, close_column] <= stop_loss

            # take-profit condition        
                take_profit_breach = xeroasx.loc[current_date, close_column] >= take_profit        


            # if low is below stop-loss or high is above take-profit
            if stop_loss_breach or take_profit_breach:

                # Define the variable 'exit_date'
                exit_date = current_date

                    # Extract exit price of the trade using the 'exit_date' and 'close_column' of dataframe 'data'
                exit_price = xeroasx.loc[exit_date, close_column]

                    # Exit type if 'Stop-Loss' if stop-loss was breached and 'Take-Profit' if take-profit was breached
                exit_type = "Stop-Loss" if stop_loss_breach else "Take-Profit"

                    # Append the details of this trade to the 'trades' dataframe
                crossover_trade_sheet = crossover_trade_sheet.append(
                        [(current_position, entry_date, entry_price, exit_date, exit_price, exit_type)], ignore_index=True)

                    # Since a new long position is closed, change the state of current_position to '0'
                current_position = 0
            # We will enter the long position if we are not holding any position and entry condition is met
            if current_position == 0 and long_crossover > 0:

                # Define the variable 'entry_date'
                entry_date = current_date

                # Extract entry price of the trade using the 'entry_date' and 'close_column' of dataframe 'data'
                entry_price = xeroasx.loc[entry_date, close_column]

                # Since a new long position is opened, change the state of current_position to '1'
                current_position = 1

            # We will exit the long position if we are holding long position and exit condition is met
            elif current_position == 1 and exit_crossover > 0:

                # Define the variable 'exit_date'
                exit_date = current_date

                # Extract exit price of the trade using the 'exit_date' and 'close_column' of dataframe 'data'
                exit_price = xeroasx.loc[exit_date, close_column]

                # Exit type is 'Squareoff'
                exit_type = 'Squareoff'

                # Append the details of this trade to the 'trades' dataframe
                crossover_trade_sheet = crossover_trade_sheet.append(
                    [(current_position, entry_date, entry_price, exit_date, exit_price, exit_type)], ignore_index=True)

                # Since a new long position is closed, change the state of current_position to '0'
                current_position = 0

        # Define the names of columns in 'trades' dataframe
        crossover_trade_sheet.columns = ['Position', 'Entry Date',
                          'Entry Price', 'Exit Date', 'Exit Price', 'Exit Type']

        # Return the xeroasxtrades dataframe
        return crossover_trade_sheet

    # Using 9% as stop_loss_percentage and 15% as take_profit_percentage
    # Save the trades generated in the dataframe 'crossover_trade_sheet'
    crossover_trade_sheet_sl_tp = backtest_trade_sheet_sl_tp(
        xeroasx, 'Close', 'long_cross_over', 'exit_cross_over',  0.09, 0.15)

    crossover_trade_sheet_sl_tp['PnL'] = (crossover_trade_sheet_sl_tp['Exit Price'] -
                         crossover_trade_sheet_sl_tp['Entry Price']) * crossover_trade_sheet_sl_tp['Position']

    # Print the total profit/loss of the trades generated over the historical time period
    print(f'The total PnL of trades generated between 2010-01-01 and 2022-05-31 is $',
          round(crossover_trade_sheet_sl_tp.PnL.sum(), 2))

    # Print the last 5 rows of the 'crossover_trade_sheet_sl_tp' dataframe
    crossover_trade_sheet_sl_tp.tail(20)

    # Brokerage fee
    brokerage = 0.03/100

    # Total brokerage cost
    broker_cost = brokerage
    broker_cost

    # Calculate strategy returns
    xeroasx['strategy_ret'] = xeroasx.signal.shift(1) * xeroasx.Close.pct_change()


    # Calculate the trading cost at square off 
    trading_cost = (broker_cost * np.abs(xeroasx.signal -
                                           xeroasx.signal.shift(1)))

    # Calculate net strategy returns
    xeroasx['strategy_returns_minus_cost'] = xeroasx['strategy_ret'] - trading_cost

    # Calculating plotting cumulative strategy return
    cum_strategy_returns = (xeroasx['strategy_returns_minus_cost']+1).cumprod()
    cum_strategy_returns.plot(figsize=(16, 8))
    
    superposition[stocks]=cum_strategy_returns

    plt.title('After cost Strategy Return', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Strategy Return', fontsize=12)
    st.pyplot(plt.show())

    # Initialize trade_type column for buys and sells
    xeroasx['trade_type'] = np.nan
    xeroasx['units_transacted'] = 0

    xeroasx["cost/proceeds"] = np.nan
    # Initialize variable to hold the previous_price
    previous_price = 0
    share_size = 500 

    # Now we implement a strategy to buy low, and sell on uptick 
    for index, row in xeroasx.iterrows():
        # buy if the previous price is 0, in other words, buy on the first day
        # set the cost/proceeds column equal to the negative value of the row close price
        # multiplied by the share_size
        if previous_price == 0:
            xeroasx.loc[index, "trade_type"] = "buy"
            xeroasx.loc[index, "cost/proceeds"] = -(row["Close"] * share_size)
            xeroasx.loc[index,"units_transacted"] = share_size

        # buy if the current day price is less than the previous day price
        # set the cost/proceeds column equal to the negative value of the row close price
        # multiplied by the share_size
        elif row["Close"] < previous_price:
            xeroasx.loc[index, "trade_type"] = "buy"
            xeroasx.loc[index, "cost/proceeds"] = -(row["Close"] * share_size)
            xeroasx.loc[index,"units_transacted"] = share_size

        # sell if the current day price is greater than the previous day price
        elif row["Close"] > previous_price:
            xeroasx.loc[index, "trade_type"] = "sell"
            xeroasx.loc[index, "cost/proceeds"] = row["Close"] * share_size
            xeroasx.loc[index,"units_transacted"] = -share_size

        # else hold if the current day price is equal to the previous day price
        else:
            xeroasx.loc[index, "trade_type"] = "hold"

        # set the previous_price variable to the close price of the current row
        previous_price = row["Close"]

        # if the index is the last index of the Dataframe, sell
        # set the cost/proceeds column equal to the row close price multiplied 
        # by the accumulated_shares
        if index == xeroasx.index[-1]:
            xeroasx.loc[index, "trade_type"] = "sell"
            xeroasx.loc[index,"units_transacted"] = -xeroasx.loc[xeroasx.index < index,"units_transacted"].sum()
            xeroasx.loc[index, "cost/proceeds"] = row["Close"] * -xeroasx.loc[index,"units_transacted"]

     # Calculate the total profit/loss for 100 share size orders
    total_profit_loss = round(xeroasx["cost/proceeds"].sum(), 2)

    # Print the profit/loss metrics
    print(f"The total profit/loss of the trading strategy is ${total_profit_loss}.")



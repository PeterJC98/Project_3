from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
import yfinance as yf
import pandas as pd
import datetime
import time
import streamlit as st

#Getting list of all tickers in index
tickers = si.tickers_dow()

#Changing string of tickers to contain a dash because that is what yfinance uses
tickers = [item.replace(".", "-") for item in tickers]

#Setting Time range to 1 month
start_date = datetime.datetime.now() - datetime.timedelta(days=30)
end_date = datetime.date.today()

#importing data
@st.cache(allow_output_mutation=True)
def get_data(tickers):
    stocks = pd.DataFrame()
    data=yf.download(tickers, start=start_date, end=end_date, group_by='tickers')
    for tickers in tickers:
        sma = [50, 150, 200]
        for x in sma:
            data["SMA_"+str(x)] = round(data[tickers]['Adj Close'].rolling(window=x).mean(), 2)
        currentClose = data[tickers]["Adj Close"][-1]
        moving_average_50 = data["SMA_50"][-1]
        moving_average_150 = data["SMA_150"][-1]
        moving_average_200 = data["SMA_200"][-1]
        low_of_52week = (min(data[tickers]["Low"][-260:]))
        #low_of_52week_date=data[data[tickers]["Low"]==low_of_52week].index.values
        high_of_52week = (max(data[tickers]["High"][-260:]))
        #high_of_52week_date=data[data[tickers]["High"]==high_of_52week].index.values
        ticker_object = yf.Ticker(tickers)
        info=ticker_object.info
        beta=info['beta']
        pegratio=info['pegRatio']
        pe=info['forwardPE']
        earningsGrowth=info['earningsGrowth']
        
        print(tickers)
        stocks = stocks.append({'Stock': tickers,
                                'Current Close':currentClose,
                                'beta':beta,
                                'pegratio':pegratio,
                                'forwardPE':pe,
                                'earningsGrowth':earningsGrowth,
                                "50 Day MA": moving_average_50, 
                                "150 Day MA": moving_average_150, 
                                "200 Day MA": moving_average_200, 
                                "52 Week Low": low_of_52week,
                                #'52 Week Low Date':low_of_52week_date, 
                                "52 week High": high_of_52week
                                #'52 Week High Date':high_of_52week_date,
                                }, ignore_index=True)
        
    stocks=stocks.set_index('Stock')
    return stocks


stocks=get_data(tickers) 

with st.form("Filters"):
    st.write("Stock Filters")
    beta= st.slider('Beta',-3.0,3.0,(-3.0,3.0), step=0.1)
    pegratio=st.slider('PEG Ratio',-3.0,3.0,(-3.0,3.0), step=0.1)
    forwardpe=st.slider('PE Ratio',-30,100,(-30,100), step=1)
    earningsgrowth=st.slider('Earnings Growth',-1.0,1.0,(-1.0,1.0), step=0.05)
    # Every form must have a submit button.
    submitted = st.form_submit_button("Apply")
Filtered_table=stocks[(stocks['beta']<=beta[1]) & (stocks['beta']>=beta[0]) 
                      & (stocks['pegratio']<=pegratio[1]) & (stocks['pegratio']>=pegratio[0])
                      & (stocks['forwardPE']<=forwardpe[1]) & (stocks['forwardPE']>=forwardpe[0])
                      & (stocks['earningsGrowth']<=earningsgrowth[1]) & (stocks['earningsGrowth']>=earningsgrowth[0])]
st.write(Filtered_table)



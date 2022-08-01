from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
import yfinance as yf
import pandas as pd
import datetime
import time
import streamlit as st
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from MCForecastTools import MCSimulation
import seaborn as sns 
from subprocess import run
finviz_url = 'https://finviz.com/quote.ashx?t='
vader = SentimentIntensityAnalyzer()
import hvplot
import holoviews as hv
import hvplot.pandas  # noqa
import hvplot.dask

#Getting list of all tickers in index
tickers = si.tickers_dow()

#Changing string of tickers to contain a dash because that is what yfinance uses
tickers = [item.replace(".", "-") for item in tickers]

#Setting Time range to 1 year
start_date = datetime.datetime.now() - datetime.timedelta(days=365*5)
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
        changes=data[tickers]['Adj Close'].pct_change()
        momentum=(data[tickers]['Adj Close']-round(data[tickers]['Adj Close'].rolling(window=10).mean(), 2))/data[tickers]['Adj Close']
        try:
            sharperatio=((changes.mean()*252))/(changes.std()*np.sqrt(252))
        except:
            sharperatio=0
        #low_of_52week_date=data[data[tickers]["Low"]==low_of_52week].index.values
        high_of_52week = (max(data[tickers]["High"][-260:]))
        #high_of_52week_date=data[data[tickers]["High"]==high_of_52week].index.values
        ticker_object = yf.Ticker(tickers)
        info=ticker_object.info
        beta=info['beta']
        pegratio=info['pegRatio']
        pe=info['forwardPE']
        currentmomentum=momentum[-1]
        try:
            marketcap=info['marketCap']/1000000
        except:
            marketcap=0
        earningsGrowth=info['earningsGrowth']
        
        print(tickers)
        stocks = stocks.append({'Stock': tickers,
                                'Current Close':currentClose,
                                'beta':beta,
                                'pegratio':pegratio,
                                'forwardPE':pe,
                                'earningsGrowth':earningsGrowth,
                                'marketcap':marketcap,
                                'sharperatio':sharperatio,
                                'Recent Momentum': currentmomentum,
                                "50 Day MA": moving_average_50, 
                                "150 Day MA": moving_average_150, 
                                "200 Day MA": moving_average_200, 
                                "52 Week Low": low_of_52week,
                                #'52 Week Low Date':low_of_52week_date, 
                                "52 week High": high_of_52week
                                #'52 Week High Date':high_of_52week_date,
                                }, ignore_index=True)
        
    stocks=stocks.set_index('Stock')
    return stocks, data


stocks, raw_data =get_data(tickers)
st.write('## Stock Screener')
op=['Beta', 'PEG Ratio', 'Forward PE','Earnings Growth', 'Market Cap', 'Sharpe Ratio', 'Recent Momentum']
#@st.cache(allow_output_mutation=True)
st.write('#### Select the filters you wish to apply')
options=st.multiselect('Filters', op)

st.write("Select the range of values for your filters")
with st.form(key='my_form'):
    if 'Beta' in options:
        beta= st.slider('Beta',-3.0,3.0,(-3.0,3.0), step=0.1)
    else:
        beta=(-np.inf,np.inf)
    if 'PEG Ratio' in options:
        pegratio=st.slider('PEG Ratio',-3.0,3.0,(-3.0,3.0), step=0.1)
    else:
        pegratio=(-np.inf,np.inf)
    if 'Forward PE' in options:
        forwardpe=st.slider('PE Ratio',-30,100,(-30,100), step=1)
    else:
        forwardpe=(-np.inf,np.inf)
    if 'Earnings Growth' in options:
        earningsgrowth=st.slider('Earnings Growth',-1.0,1.0,(-1.0,1.0), step=0.05)
    else:
        earningsgrowth=(-np.inf,np.inf)
    if 'Market Cap' in options:
        marketcap=st.slider('Market Cap (Millions)', 0, 3000000,(0,300000), step=10000)
    else:
        marketcap=(-np.inf, np.inf)
    if 'Sharpe Ratio' in options:
        sharperatio=st.slider('Sharpe Ratio',-1.0,3.0,(-1.0,3.0), step=0.1)
    else:
        sharperatio=(-np.inf, np.inf)
    if 'Recent Momentum' in options:
        moment=st.slider('Recent Momentum',-0.2,0.2,(-0.2,0.2), step=0.01)
    else:
        moment=(-np.inf, np.inf)
    # Every form must have a submit button.
    submitted = st.form_submit_button("Apply")
    
    
Filtered_table=stocks[(stocks['beta']<=beta[1]) & (stocks['beta']>=beta[0]) 
                      & (stocks['pegratio']<=pegratio[1]) & (stocks['pegratio']>=pegratio[0])
                      & (stocks['forwardPE']<=forwardpe[1]) & (stocks['forwardPE']>=forwardpe[0])
                      & (stocks['earningsGrowth']<=earningsgrowth[1]) & (stocks['earningsGrowth']>=earningsgrowth[0])
                      & (stocks['marketcap']<=marketcap[1]) & (stocks['marketcap']>=marketcap[0])
                      & (stocks['sharperatio']<=sharperatio[1]) & (stocks['sharperatio']>=sharperatio[0])
                      & (stocks['Recent Momentum']<=moment[1]) & (stocks['Recent Momentum']>=moment[0])]

st.write(Filtered_table)
with st.expander('Recent News Sentiment'):
    st.write('This may take a few minutes')
    if st.button('Run Sentiment Analysis'):
        for stocks in Filtered_table.index.values:
            news_tables={}

        for ticker in Filtered_table.index.values:
            url = finviz_url + ticker

            req = Request(url=url, headers={'user-agent': 'my-app'})
            response = urlopen(req)   
            html = BeautifulSoup(response, 'html')
            news_table=html.find(id='news-table')
            news_tables[ticker] = news_table


        parsed_data = []

        for ticker, news_table in news_tables.items():
            for row in news_table.find_all('tr'):

                title = row.a.text
                date_data = row.td.text.split(' ')

                if len(date_data) == 1:
                    time = date_data[0]
                else:
                    date = date_data[0]
                    time = date_data[1]

                parsed_data.append([ticker, date, time, title])

        df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])      
        f= lambda title: vader.polarity_scores(title)['compound']
        df['compound'] = df['title'].apply(f)
        df['date']=pd.to_datetime(df.date).dt.date
        mean_df = df.groupby(['ticker']).mean()
        #mean_df = mean_df.unstack()
        #mean_df = mean_df.xs('compound', axis='columns').transpose()
        #mean_df = mean_df.groupby(['ticker']).mean()
        nice_plot=mean_df.hvplot.bar()
        st.bokeh_chart(hv.render(nice_plot, backend='bokeh'))
with st.expander('Historical Close Prices'):
    st.set_option('deprecation.showPyplotGlobalUse', False) 
    graph=pd.DataFrame()
    slist=[]
    for stocks in Filtered_table.index.values:
        graph[stocks]=raw_data[stocks]['Close']
        slist.append(stocks)
    plt.figure(figsize=(15, 7))
    plt.plot(graph[slist])

    # Set the title
    plt.title('Historical Close Price', fontsize=16)

    # Set the labels
    plt.ylabel('Price', fontsize=15)
    plt.xlabel('Year', fontsize=15)

    # Set the tick size
    plt.tick_params(labelsize=15)
    plt.legend(slist)
    # Show the graph
    st.pyplot(plt.show())

with st.expander('Correlation'):
    corr=pd.DataFrame()
    for stocks in Filtered_table.index.values:
        corr[stocks]=raw_data[stocks]['Close']
    corrret=corr.pct_change().copy()
    defportcorr = corrret.corr()
    fig = plt.figure(figsize=(10, 4))
    sns.heatmap(defportcorr, vmin=-1, vmax=1)
    st.pyplot(fig)
st.write('## Portfolio Builder')
with st.form(key='my_form2'):
    st.write('###### Select Tickers to build a custom portfolio out of your filtered stocks')
    stocklist=[]
    for stocks in Filtered_table.index.values:
        check = st.checkbox(stocks)
        if check:
            stocklist.append(stocks)
    submitted = st.form_submit_button("Apply")
    
with st.form(key='my_form3'):
    st.write('##### Apply custom weights to your portfolio')
    weights=[]
    for stocks in stocklist:
        stock= st.slider(stocks,0.0,1.0, step=0.05)
        weights.append(stock)
    if sum(weights) != 1:
        st.write('Warning: sum is not equal to 1')
    submitted = st.form_submit_button("Apply")
#st.write(stocklist)
#st.write(weights)
shareport=pd.DataFrame()
for stocks in stocklist:
    shareport[stocks]=raw_data[stocks]['Close']
#st.write(shareport)

sharereturns_df = shareport.pct_change().copy() 
with st.expander('Correlation'):
    defportcorr1 = sharereturns_df.corr()
    fig = plt.figure(figsize=(10, 4))
    sns.heatmap(defportcorr1, vmin=-1, vmax=1)
    st.pyplot(fig)

shareport_std = sharereturns_df.std() 
with st.expander('Historical Performance'):
    plt.figure(figsize=(15, 7))
    basic_returns = shareport.pct_change()
    # Plot the cumulative returns for all stocks
    plt.plot((basic_returns+1).cumprod())

    # Set the title
    plt.title('Historical Return of Individual Stocks', fontsize=16)

    # Set the labels
    plt.ylabel('Cumulative Returns', fontsize=15)
    plt.xlabel('Year', fontsize=15)
    plt.legend(stocklist)
    # Set the tick size
    plt.tick_params(labelsize=15)

    st.pyplot(plt.show())

    #Weighted
    basic_returns['weighted'] = basic_returns.dot(weights)
    plt.figure(figsize=(15, 7))

    # Plot the equal-weighted portfolio
    plt.plot((basic_returns['weighted']+1).cumprod())

    # Set the title
    plt.title('Return of Custom Portfolio', fontsize=16)

    # Set the labels
    plt.ylabel('Cumulative Returns', fontsize=15)
    plt.xlabel('Year', fontsize=15)

    # Set the tick size
    plt.tick_params(labelsize=15)

    # Show the graph
    st.pyplot(plt.show())
a=0
bbeta=0
for i in stocklist:
    bbeta=bbeta+weights[a]*Filtered_table.loc[i]['beta']
    a=a+1
st.write('Beta of the Portfolio')
st.write(bbeta)
risk_free_rate = 0.02/252
st.write('Sharpe Ratio of the Portfolio')
sharpe=((basic_returns.mean()['weighted']*252))/(basic_returns.std()*np.sqrt(252))
st.write(sharpe['weighted'])
st.write('#### Analyse Future Performance with a Monte-Carlo Simulation')
with st.form(key='my_form4'):
    st.write('MC Simulation Parameters')
    length=st.slider('Length of time',0,40, step=5)
    inital=st.number_input('Inital Investment')
    num_simulation=st.slider('Number of Simulations',0,500, step=10)
    submitted = st.form_submit_button("Apply")
st.write('MC simulation may take a few minutes')
if st.button("Run MC Simulation"):
    MC_summitbal = MCSimulation(
    portfolio_data = raw_data[stocklist],
    weights = weights,
    
    num_simulation = num_simulation,
    num_trading_days = 252*length)
    
    # Plot simulation outcomes
    line_plot = MC_summitbal.plot_simulation()

    # Save the plot for future usage
    st.pyplot(line_plot.get_figure())
    
    
    
    MC_summitbal.calc_cumulative_return()
    
    tbl = MC_summitbal.summarize_cumulative_return()
    

    # Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes of our $20,000
    ci_lower = round(tbl[8]*inital,2)
    ci_upper = round(tbl[9]*inital,2)

    # Print results
    st.write((f"There is a 95% chance that an initial investment of \${inital} in your custom portfolio over the next {length} years will end within the range of \${ci_lower} and \${ci_upper}"))
superposition=pd.DataFrame()
st.write('#### Test a Trading Algorithm on your Custom Portfolio')
if st.button('Algo Trading'):
    for stocks in stocklist:
        with open('algo.py', 'r') as f:
            exec(f.read())

    superposition['weighted']=superposition.dot(weights)
    superposition['weighted'].plot(figsize=(16, 8))
    plt.title('After cost Strategy Return Weighted', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Strategy Return', fontsize=12)
    st.pyplot(plt.show())

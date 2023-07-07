import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import requests
import requests
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
import sklearn
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def makePredictions(name,isRelated,df):

    empty_rows = df.isnull().all(axis=1)
    non_empty_rows = ~empty_rows
    df = df.loc[non_empty_rows].reset_index(drop=True)
    df.rename(columns={'text__text__1FZLe':'Category','text__text__1FZLe 2':'Title','text__text__1FZLe href':'Link','text__text__1FZLe 3':'Date' },inplace=True)
    df = df[['Category','Title','Link','Date']]

    sia = SentimentIntensityAnalyzer()

    sentiments = []
    for title in df['Title']:
        sentiment = sia.polarity_scores(title)['compound']
        sentiments.append(sentiment)

    df['sentiment'] = sentiments

    arr = []
    words = isRelated
    for i in range(len(df)):
        val = 0
        for j in words:
            title = df.iloc[i]['Title'].lower()
            if any(keyword in title for keyword in words):
                val = 1
                break
        arr.append(val)
    df['isApple'] =arr

    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    from datetime import datetime, timedelta

    last_year_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    apple = yf.Ticker(name)

    import datetime
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=3*365)).strftime('%Y-%m-%d') 
    apple = apple.history(start=start_date, end=end_date)
    apple['Date'] = apple.index
    apple.reset_index(drop=True,inplace=True)
    start_date = str(apple.iloc[0]['Date'])[:10]
    score_start = int(start_date[:4])*365 + int(start_date[5:7])*30 + int(start_date[8:10])
    arr = [0]*len(apple)
    countArr = [1]*len(apple)

    for i in range(len(df)):
        date = str(df.iloc[i]['Date'])
        date_score = int(date[:4])*365 + int(date[5:7])*30 + int(date[8:10])
        # Check for the case where date is 30 days prior to the 1st entry in the dates
        if score_start > date_score:
            print('Article too old')
            continue
        k  =0
        score_new = score_start
        while score_new < date_score and k < len(apple)-1:
            k+=1
            start_date = str(apple.iloc[k]['Date'])[:10]
            score_new = int(start_date[:4])*365 + int(start_date[5:7])*30 + int(start_date[8:10])
        
        param = 10
        sentiment = df.iloc[i]['sentiment']
        if k+12>len(apple):
            end = len(apple)
        else:
            end = k+12
            
        if df.iloc[i]['Category']=='Business' or df.iloc[i]['Category']=='Markets' or df.iloc[i]['Category']=='European Markets' or df.iloc[i]['Category']=='Wealth' or df.iloc[i]['Category']=='Finance' or df.iloc[i]['Category']=='Technology':
            val = 2
        else:
            val = 0.7
        
        if df.iloc[i]['isApple'] ==1:
            val*= 2
            
        for j in range(k,end):
            arr[j] += param*sentiment*val
            param /= 1.5
            
    today_vals = arr[-8:]
    arr = [0] + arr[:-1]
    apple['Sentiment_score'] = arr

    # Getting the price difference
    arr = []
    for i in range(1,len(apple)):
        arr.append(((apple.iloc[i]['Close']-apple.iloc[i-1]['Close'])/apple.iloc[i]['Close'])*100)
        
    apple = apple.tail(len(apple)-1)
    apple['Diff'] = arr
    apple.drop(columns={'Dividends','Stock Splits','Volume','High','Open','Low'},inplace=True)

    arr = []
    for i in range(7,len(apple)):
        arr.append(apple['Close'][i-7:i].tolist())

    arr1 = []
    for i in range(7,len(apple)):
        arr1.append(apple['Sentiment_score'][i-7:i].tolist())

    arr2 = []
    for i in range(7,len(apple)):
        arr2.append(apple['Diff'][i-7:i].tolist())

    apple = apple.tail(len(apple)-7)

    for i in range(0,7):
        day = f'D-{i+1}'
        news_s = f'N-{i+1}'
        diff_sc = f'Diff-{i+1}'
        arr_new = []
        for j in range(len(apple)):
            arr_new.append(arr[j][i])
        apple[day] = arr_new
        arr_new = []
        for j in range(len(apple)):
            arr_new.append(arr1[j][i])
        apple[news_s] = arr_new
        arr_new = []
        for j in range(len(apple)):
            arr_new.append(arr2[j][i])
        apple[diff_sc] = arr_new
            

    for i in range(len(apple)):
        if apple.iloc[i]['Sentiment_score']!=0:
            break
    apple = apple.tail(len(apple)-i)
    apple = apple.reset_index(drop=True)
    #apple.drop(columns={'Unnamed: 0'},inplace=True)
    apple = apple.tail(529)
    apple = apple.reset_index(drop=True)
    apple_score_pred = apple[['Date','Sentiment_score','N-1','N-2','N-3','N-4','N-5','N-6','N-7','D-1','D-2','D-3','D-4','D-5','D-6','D-7','Close']]

    X = apple_score_pred.drop(columns={'Date','Close'}).values
    y = apple_score_pred['Close'].values
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2,shuffle=False)

    def analyze(model_name,model):
        ypred1 = model.predict(x_train)

        mse = mean_squared_error(y_train, ypred1)
        print("Mean Squared Error for train data:", mse)

        r2 = r2_score(y_train, ypred1)
        print("R-squared Score for train data:", r2)

        plt.figure(figsize=(12,8))
        plt.plot(apple['Date'].head(423),y_train)
        plt.plot(apple['Date'].head(423),ypred1)
        plt.title(f'Results on train data using {model_name}')
        plt.legend(['Actual Values','Predicted Values'])
        str1 = model_name + 'train.png'
        plt.savefig(str1)
        #plt.show()

        y_pred = model.predict(x_test)

        mse = mean_squared_error(y_test, y_pred)
        print("Mean Squared Error for test data:", mse)

        r2 = r2_score(y_test, y_pred)
        print("R-squared Score for test data:", r2)

        plt.figure(figsize=(12,8))
        plt.plot(apple['Date'].tail(106),y_test)
        plt.plot(apple['Date'].tail(106),y_pred)
        plt.title(f'Results on test data using {model_name}')
        plt.legend(['Actual Values','Predicted Values'])
        str2 = model_name + 'test.png'
        plt.savefig(str2)
        #plt.show()
        return r2

    
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(x_train,y_train)
    r2 = analyze('LinearRegression',model)
    
    x_final_test = []
    x_final_test += today_vals
    x_final_test +=  apple_score_pred['Close'].tolist()[-7:]
    
    x_final_test = np.array(x_final_test)
    x_final_test = x_final_test.reshape(1, -1)
    x = model.predict(x_final_test)
    return [x[0] , r2]
    
names = ['AAPL']
isRelatedTo = [['apple','iphone','mac','tim','cook','ipad','vision pro','aapl','itunes','ios','macbook','imac']]
dfs =[]
for i in range(len(names)):
    str1 = names[i] + '.xlsx'
    df = pd.read_excel(str1) 
    dfs.append(df)

for i in range(len(names)):
    print(names[i])
    y = makePredictions(names[i],isRelatedTo[i],dfs[i])
    print(f'Expected close price for {names[i]} for today is {y[0]} with {y[1]*100}% confidence')
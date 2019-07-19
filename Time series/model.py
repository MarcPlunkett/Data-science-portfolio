# Import libraries


from matplotlib.pyplot import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from pandas import Series

import matplotlib.pyplot as plt

from datetime import datetime


# Import datasets

train = pd.read_csv(
    '/Users/marc/Data-science-portfolio/Time series/dataset/Train_SU63ISt.csv', )
test = pd.read_csv(
    '/Users/marc/Data-science-portfolio/Time series/dataset/Test_0qrQsBZ.csv', )

train['source'] = 'train'
test['source'] = 'test'

# data = pd.concat([train, test], ignore_index=True)

# Data exploration
train.head()
train.tail
train.describe()
train.dtypes
train.shape
# There are 23400 total instances


# The data contains the number of passengers over the time at one hour increments
# Date time must be extracted
train['Datetime'] = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
test['Datetime'] = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')

# The time of day could also be useful as demand is likely to change throughout the day so it may be useful to extract this as a separate feature

for i in (train, test):
    i['year'] = i.Datetime.dt.year
    i['month'] = i.Datetime.dt.month
    i['day'] = i.Datetime.dt.day
    i['hour'] = i.Datetime.dt.hour

# Now we want to insert this into our dataset
# The most important variable is likely to be if the day is a weekend of now
train['day_name'] = train['Datetime'].dt.dayofweek

# If the day lies on a saturday or sunday then it's assigned a 1, otherwise a 0


def applyer(row):
    if row.day_name == 5 or row.day_name == 6:
        return 1
    else:
        return 0


temp = train['Datetime'].apply(applyer)

train['weekend'] = temp

# Finally we will plot the time series

train.index = train['Datetime']
# ID variable is not necessary
df = train.drop('ID', 1)

ts = df['Count']  # Time series Y variable is the count


plt.figure(figsize=(16, 8))
plt.plot(ts, label='Passenger count')
plt.title('Time series of passenger count over time')
plt.xlabel('Time(year-month')
plt.ylabel('Passenger count')
plt.show()

# Plot shows a general upoward trend which is promising when thinking about the goal
# There are a number of spikes which should be explored

# Exploratory analysis

# Hypothesis 1: traffic will increase over time

train.groupby('year')['Count'].mean().plot.bar()

# Exponential growth in passenger count

# Hypothesis 2:  Traffic varies with month, day and hour

train.groupby(['year', 'month'])['Count'].mean().plot.bar()

# Again we see that average passenger couynt increases month by month on an exponential basis except for 3/14 which appears to be an outlier

# Now for the daily mean

train.groupby('day')['Count'].mean().plot.bar()

# Not much insight here

train.groupby('hour')['Count'].mean().plot.bar()

# On average traffic drops between midnight until roughly 5am where is picks up. THis makes sense since most people sleep during these times

# Hypothesis: TRaffic is more during the wek

train.groupby('day_name')['Count'].mean().plot.bar()

# There is a drop at the weekend

# Now we will look at the average time series
train['Timestamp'] = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
train.index = train['Timestamp']

hourly = train.resample('H').mean()
daily = train.resample('D').mean()
weekly = train.resample('W').mean()
monthly = train.resample('M').mean()

fig, axs = plt.subplots(4, 1)
hourly.Count.plot(figsize=(15, 8), title='Hourly', fontsize=14, ax=axs[0])
daily.Count.plot(figsize=(15, 8), title='Daily', fontsize=14, ax=axs[1])
weekly.Count.plot(figsize=(15, 8), title='Weekly', fontsize=14, ax=axs[2])
monthly.Count.plot(figsize=(15, 8), title='Monthly', fontsize=14, ax=axs[3])

plt.show()

# We will use daily mean for our analysis since it gives the best insight. Hourly data is too granular whereas monthly and weekly are not

train = train.resample('D').mean()

# We will take the most revcent three months to val;idate the dataset

train_set = train.loc['2012-08-25': '2014-06-24']
train_validate = train.loc['2014-06-25': '2014-09-22']

train_set.Count.plot(figsize=(15, 8), title='Daily rides', label='Train')
train_validate.Count.plot(
    figsize=(15, 8), title='Daily rides', label='Validate')
plt.xlabel('Time(Day)')
plt.ylabel('Passenger count')
plt.legend(loc='best')
plt.show()

# Naive approach

dd = train_set.Count.values
y_hat = train_validate.copy()
y_hat['naive'] = dd[len(dd)-1]


plt.plot(train_set.index, train_set['Count'], label='Train')
plt.plot(train_validate.index, train_validate['Count'], label='Validate')
plt.plot(y_hat.index, y_hat['naive'], label='Niave')
plt.xlabel('Time(Day)')
plt.ylabel('Passenger count')
plt.legend(loc='best')
plt.show()


rms = sqrt(mean_squared_error(train_validate.Count, y_hat.naive))
rms

# Baseline for model is 103.96

# Holts linear trend method
sm.tsa.seasonal_decompose(train_set.Count).plot()
result = sm.tsa.stattools.adfuller(train.Count)
plt.show()

# Staqts model shows an upwartds trend, not to apply this to our model

y_hat_avg = train_validate
fit1 = Holt(train_set['Count'].values).fit(
    smoothing_level=0.3, smoothing_slope=0.1)
y_hat_avg['Holt_Linear'] = fit1.forecast(len(train_validate))
plt.plot(train_set.index, train_set['Count'], label='Train')
plt.plot(train_validate.index, train_validate['Count'], label='Validate')
plt.plot(y_hat.index, y_hat_avg['Holt_Linear'], label='Holt Linear')
plt.xlabel('Time(Day)')
plt.ylabel('Passenger count')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(train_validate.Count, y_hat_avg['Holt_Linear']))
rms

# Holt's model gives an improvment in rms
fit1 = ExponentialSmoothing(
    train_set['Count'].values, seasonal_periods=7, trend='add', seasonal='add').fit()

y_hat_avg['Holt_Winter'] = fit1.forecast(len(train_validate))

plt.plot(train_set.index, train_set['Count'], label='Train')
plt.plot(train_validate.index, train_validate['Count'], label='Validate')
plt.plot(y_hat.index, y_hat_avg['Holt_Winter'], label='Holt Linear')
plt.xlabel('Time(Day)')
plt.ylabel('Passenger count')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(train_validate.Count, y_hat_avg['Holt_Winter']))
rms

# LArge drop in RMS from using holt winter

# Arima model
# ARIMA can be used on a time series if
# The mean of the time series should not be a function of time. It should be constant.
# The variance of the time series should not be a function of time.
# The covariance of the ith term and the (i+m)th term should not be a function of time.

# A time series can be made stationary by removing the trend and seasonality

# To check if the dataset is starionary we can use dicky fuller method


def test_stationary(timeseries):
    rolman = pd.DataFrame.rolling(
    timeseries, window=24).mean()  # 24 hours a day
    rolstd = pd.DataFrame.rolling(timeseries, window=24).std()
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolman, color='red', label='mean')
    std = plt.plot(rolstd, color='green', label='std')
    plt.legend(loc='best')
    plt.title('Rolling mean and std')
    plt.show(block=False)

    print('Result of Dickey Fuller:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=[
                         'Test statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    print(dfoutput)


rcParams['figure.figsize'] = 20, 10
test_stationary(train['Count'])

# A trend exists where there is a long term increase or decrease in the value and is therefore not linear
# We can solve this with a simplt log transformation
# Rolling average can be used to remove the trend

train_log = np.log(train_set['Count'])
validate_log = np.log(train_validate['Count'])

moving_avg = np.log(pd.DataFrame.rolling(train_set['Count'], 24).mean())

plt.plot(train_set.index, train_log)
plt.plot(train_set.index, moving_avg, color='red')
plt.show

#Now we can substitute these values in order to make the time series stationary
train_log_moving_diff = train_log - moving_avg
train_log_moving_diff.dropna(inplace=True)
test_stationary(train_log_moving_diff)

#Test stat is a lot smaller than the crtitical value therefore we can say the trend is removed
#Now we need to stabilize the mean

train_log_diff= train_log - train_log.shift(1)
test_stationary(train_log_diff.dropna())

#Now the time series must be decomposed in order to remove the seaosnality and the residual randomness

#Removeing seasonality

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(pd.DataFrame(train_log).Count.values, freq=24)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411) 
plt.plot(train_log, label='Original') 
plt.legend(loc='best') 
plt.subplot(412) 
plt.plot(trend, label='Trend') 
plt.legend(loc='best') 
plt.subplot(413) 
plt.plot(seasonal,label='Seasonality') 
plt.legend(loc='best') 
plt.subplot(414) 
plt.plot(residual, label='Residuals') 
plt.legend(loc='best') 
plt.tight_layout() 
plt.show()

train_log_decompose = pd.DataFrame(residual)
train_log_decompose['Date'] = train_log.index
train_log_decompose.set_index('Date', inplace=True)
train_log_decompose.dropna(inplace=True)
test_stationary(train_log_decompose[0])

from statsmodels.tsa.stattools import acf, pacf
lag_acf= acf(train_log_diff.dropna(), nlags=25)
lag_pacf = pacf(train_log_diff.dropna(), nlags=25, method='ols')


plt.plot(lag_acf) 
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 
plt.title('Autocorrelation Function') 
plt.show() 
plt.plot(lag_pacf) 
plt.axhline(y=0,linestyle='--',color='gray') 
plt.axhline(y=-1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 
plt.axhline(y=1.96/np.sqrt(len(train_log_diff.dropna())),linestyle='--',color='gray') 
plt.title('Partial Autocorrelation Function') 
plt.show()

#p value is the lag value where the PACF chart crosses the upper confidence interval for the first time. It can be noticed that in this case p=1.

#q value is the lag value where the ACF chart crosses the upper confidence interval for the first time. It can be noticed that in this case q=1.

#ARIMA model
#The autoregressive model specifies that the output variable depends linearly on its own previous values.

from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(train_log,order=(2, 1, 0)) # q=0

results_ar = model.fit(disp=-1)

plt.plot(train_log_diff.dropna(), label='original') 
plt.plot(results_ar.fittedvalues, color='red', label='predictions') 
plt.legend(loc='best') 
plt.show()

AR_predict=results_ar.predict(start='2014-06-25', end='2014-09-22')
AR_predict=AR_predict.cumsum().shift().fillna(0)
AR_predict1 = pd.Series(np.ones(train_validate.shape[0])* np.log(train_validate['Count'])[0], index=train_validate.index)
AR_predict1 = AR_predict1.add(AR_predict, fill_value=0)
AR_predict= np.exp(AR_predict1)

plt.plot(train_validate['Count'], label = "Valid") 
plt.plot(AR_predict, color = 'red', label = "Predict") 
plt.legend(loc= 'best') 
plt.title('RMSE: %.4f'% (np.sqrt(np.dot(AR_predict, train_validate['Count']))/train_validate.shape[0])) 
plt.show()

#The moving-average model specifies that the output variable depends linearly on the current and various past values of a stochastic (imperfectly predictable) term.

model = ARIMA(train_log,order=(0, 1, 2)) # p=0

results_ma = model.fit(disp=-1)

plt.plot(train_log_diff.dropna(), label='original') 
plt.plot(results_ar.fittedvalues, color='red', label='predictions') 
plt.legend(loc='best') 
plt.show()

ma_predict=results_ma.predict(start='2014-06-25', end='2014-09-22')
ma_predict=ma_predict.cumsum().shift().fillna(0)
ma_predict1 = pd.Series(np.ones(train_validate.shape[0])* np.log(train_validate['Count'])[0], index=train_validate.index)
ma_predict1 = ma_predict1.add(ma_predict, fill_value=0)
ma_predict= np.exp(ma_predict1)

plt.plot(train_validate['Count'], label = "Valid") 
plt.plot(ma_predict, color = 'red', label = "Predict") 
plt.legend(loc= 'best') 
plt.title('RMSE: %.4f'% (np.sqrt(np.dot(ma_predict, train_validate['Count']))/train_validate.shape[0])) 
plt.show()

# Combined model
model = ARIMA(train_log,order=(2, 1, 2)) # p=0

results_comb = model.fit(disp=-1)

plt.plot(train_log_diff.dropna(), label='original') 
plt.plot(results_ar.fittedvalues, color='red', label='predictions') 
plt.legend(loc='best') 
plt.show()

comb_predict=results_comb.predict(start='2014-06-25', end='2014-09-22')
comb_predict=comb_predict.cumsum().shift().fillna(0)
comb_predict1 = pd.Series(np.ones(train_validate.shape[0])* np.log(train_validate['Count'])[0], index=train_validate.index)
comb_predict1 = comb_predict1.add(comb_predict, fill_value=0)
comb_predict= np.exp(comb_predict1)

plt.plot(train_validate['Count'], label = "Valid") 
plt.plot(comb_predict, color = 'red', label = "Predict") 
plt.legend(loc= 'best') 
plt.title('RMSE: %.4f'% (np.sqrt(np.dot(comb_predict, train_validate['Count']))/train_validate.shape[0])) 
plt.show()

# SARIMAX takes into account the seasonality of a dataseries 

import statsmodels.api as sm
y_hat_avg = train_validate.copy()
fit1 = sm.tsa.statespace.SARIMAX(train_set.Count, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()
y_hat_avg['SARIMAX'] = fit1.predict(start='2014-06-25', end='2014-09-22', dynamic=True)

plt.figure(figsize=(16,8)) 
plt.plot(train_set['Count'], label='Train') 
plt.plot(train_validate['Count'], label='Valid') 
plt.plot(y_hat_avg['SARIMAX'], label='SARIMA') 
plt.legend(loc='best') 
plt.show()

rms = sqrt(mean_squared_error(train_validate.Count, y_hat_avg['SARIMAX']))
rms
# Import libraries


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
    i['year']=i.Datetime.dt.year
    i['month']=i.Datetime.dt.month
    i['day']=i.Datetime.dt.day
    i['hour']=i.Datetime.dt.hour

# Now we want to insert this into our dataset
# The most important variable is likely to be if the day is a weekend of now
train['day_name'] = train['Datetime'].dt.dayofweek

#If the day lies on a saturday or sunday then it's assigned a 1, otherwise a 0

def applyer(row):
    if row.day_name == 5 or row.day_name == 6:
        return 1
    else:
        return 0

temp = train['Datetime'].apply(applyer)

train['weekend'] = temp

#Finally we will plot the time series

train.index = train['Datetime']
#ID variable is not necessary
df=train.drop('ID',1)

ts = df['Count']        #Time series Y variable is the count


plt.figure(figsize=(16,8))
plt.plot(ts, label='Passenger count')
plt.title('Time series of passenger count over time')
plt.xlabel('Time(year-month')
plt.ylabel('Passenger count')
plt.show()

#Plot shows a general upoward trend which is promising when thinking about the goal
#There are a number of spikes which should be explored

#Exploratory analysis

#Hypothesis 1: traffic will increase over time 

train.groupby('year')['Count'].mean().plot.bar()

#Exponential growth in passenger count

#Hypothesis 2:  Traffic varies with month, day and hour

train.groupby(['year','month'])['Count'].mean().plot.bar()

# Again we see that average passenger couynt increases month by month on an exponential basis except for 3/14 which appears to be an outlier

# Now for the daily mean

train.groupby('day')['Count'].mean().plot.bar()

#Not much insight here

'#For hour in day

train.groupby('hour')['Count'].mean().plot.bar()

#On average traffic drops between midnight until roughly 5am where is picks up. THis makes sense since most people sleep during these times

# Hypothesis: TRaffic is more during the wek

train.groupby('day_name')['Count'].mean().plot.bar()

#There is a drop at the weekend

#Now we will look at the average time series
train['Timestamp'] = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
train.index= train['Timestamp']

hourly = train.resample('H').mean()
daily = train.resample('D').mean()
weekly = train.resample('W').mean()
monthly = train.resample('M').mean()

fig, axs = plt.subplots(4,1)
hourly.Count.plot(figsize=(15,8), title='Hourly', fontsize=14, ax=axs[0])
daily.Count.plot(figsize=(15,8), title='Daily', fontsize=14, ax=axs[1])
weekly.Count.plot(figsize=(15,8), title='Weekly', fontsize=14, ax=axs[2])
monthly.Count.plot(figsize=(15,8), title='Monthly', fontsize=14, ax=axs[3])

plt.show()

# We will use daily mean for our analysis since it gives the best insight. Hourly data is too granular whereas monthly and weekly are not 

train = train.resample('D').mean()

# We will take the most revcent three months to val;idate the dataset

train_set = train.loc['2012-08-25': '2014-06-24']
train_validate = train.loc['2014-06-25': '2014-09-22']

train_set.Count.plot(figsize=(15,8), title='Daily rides', label='Train')
train_validate.Count.plot(figsize=(15,8), title='Daily rides', label='Validate')
plt.xlabel('Time(Day)')
plt.ylabel('Passenger count')
plt.legend(loc='best')
plt.show()


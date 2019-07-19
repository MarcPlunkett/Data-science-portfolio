# Import libraries

import numpy as np
import pandas as pd
from pandas import Series

import matplotlib.pyplot as plt

from datetime import datetime



# Import datasets

train = pd.read_csv(
    '/Users/MarcPlunkett/Data-science-portfolio/Time series/dataset/Train_SU63ISt.csv', )
test = pd.read_csv(
    '/Users/MarcPlunkett/Data-science-portfolio/Time series/dataset/Test_0qrQsBZ.csv', )

train['source'] = 'train'
test['source'] = 'test'

# data = pd.concat([train, test], ignore_index=True)

# Data exploration
train.head()
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





data.plot(train['Datetime'], train['Count'])
plot.show()


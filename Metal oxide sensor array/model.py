## Import libaraies ##
# Linear Algebra

import numpy as np

# Data processing
import pandas as pd

# Data viuslisation
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

## Import and clean the data ##
# The dataset is a time series of different sensors and measurements of the resistance of each sensor. The concentration of CO, humidity of the chamber, and temperature are controlled variables
# There are 14 sensors in total which correspond to R1-R14
# The data is formatted as 13 different time series files, each corresponding to a different experiment day

data = pd.read_csv(
    '/Users/MarcPlunkett/Data-science-portfolio/Metal oxide sensor array/Dataset/20160930_203718.csv')
data.columns
data.dtypes
data['CO (ppm)'].value_counts()

## Univariate analysis ##

plt.figure(1)
plt.subplot(121)
plt.plot(data['CO (ppm)'])
plt.subplot(122)
data['CO (ppm)'].plot.box(figsize=(16, 5))
plt.show()

plt.figure(1)
r = data.iloc[:, 6:19]
plt.plot(r)
plt.show()

matrix = data.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=0.8, square=True, cmap='BuPu')

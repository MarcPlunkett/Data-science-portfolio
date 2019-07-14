import numpy as numpy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

######## Data preprocessing and exploration ########

air_quality_monthly = pd.read_csv('/Users/marc/Data-science-portfolio/London Environment/Datasets/air-quality-london-monthly-averages.csv')
air_quality_hourly = pd.read_csv('/Users/marc/Data-science-portfolio/London Environment/Datasets/air-quality-london-time-of-day.csv')

gas_consumption = pd.read_csv('/Users/marc/Data-science-portfolio/London Environment/Datasets/gas-consumption.csv')
indexNames = gas_consumption[ gas_consumption['area'] != 'London' ].index
gas_consumption.drop(indexNames , inplace=True)
gas_consumption

# Look at number of columns establish the type of each column and get the number of nul values

air_quality_monthly.columns
air_quality_monthly.dtypes
air_quality_monthly.isnull().sum()

air_quality_hourly.columns
air_quality_hourly.dtypes
air_quality_hourly.isnull().sum()

air_quality_monthly['London Mean Background:PM2.5 Particulate (ug/m3)']

air_quality_monthly['London Mean Background:PM2.5 Particulate (ug/m3)'].convert_objects(convert_numeric=True)
air_quality_hourly.dtypes

air_quality_monthly['London Mean Background:PM2.5 Particulate (ug/m3)'].fillna(air_quality_monthly['London Mean Background:PM2.5 Particulate (ug/m3)'].median(), inplace=True)

## Univariate analysis ##
plt.plot(air_quality_monthly['London Mean Roadside:Nitrogen Dioxide (ug/m3)'])
plt.plot(air_quality_monthly['London Mean Background:Nitrogen Dioxide (ug/m3)'])
plt.show()


plt.plot(air_quality_monthly['London Mean Roadside:Ozone (ug/m3)'])
plt.plot(air_quality_monthly['London Mean Background:Ozone (ug/m3)'])
plt.show()


plt.plot(air_quality_monthly['London Mean Roadside:PM10 Particulate (ug/m3)'])
plt.plot(air_quality_monthly['London Mean Background:PM10 Particulate (ug/m3)'])
plt.show()

plt.plot(air_quality_monthly['London Mean Roadside:PM2.5 Particulate (ug/m3)'])
plt.plot(air_quality_monthly['London Mean Background:PM2.5 Particulate (ug/m3)'])
plt.show()

plt.plot(gas_consumption['commercial_and_industrial_consumers_sales_GWh'])
plt.show()


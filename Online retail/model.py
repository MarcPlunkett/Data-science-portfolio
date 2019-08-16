import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

## Import dataset

data = pd.read_csv('/Users/marc/Data-science-portfolio/Online retail/Dataset/Online Retail.csv')

## Data exploration

data.columns
data.dtypes

#8 variables, 5 of which are categorical, 2 numerical and one time stamp. Invoice
#number and stock code are unlikely to be relevant to any analysis so can be dropped

#Customer ID looks to refer to a different user
#Description refers to the product name
data.drop(['InvoiceNo', 'StockCode'], inplace=True, axis=1)
data.isnull().sum()


data['Description'].describe()
data.head()

data.describe()
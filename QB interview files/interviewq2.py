import pandas as pd
import numpy as np

n=14

data = pd.read_csv('/Users/marc/Data-science-portfolio/QB interview files/sample.csv')

data['Typical_Price'] = (data['High']+data['Low']+data['Close'])/3

data

data['Money_Flow'] = data['Typical_Price']*data['Volume']

data['Positive_Money_Flow'] = 0
data['Negative_Money_Flow'] = 0

data['Money_Ratio'] = 0
data['Money_Flow_Index'] = 0
data

for index, row in data.iterrows():
    if index>0:
        if row['Typical_Price']> data.at[index-1, 'Typical_Price']:
            data.set_value(index, 'Positive_Money_Flow', row['Money_Flow'])
        else:
            data.set_value(index, 'Negative_Money_Flow', row['Money_Flow'])

    if index >= n:
        period_slice = data['Money_Flow'][index-n:index]
        positive_sum = data['Positive_Money_Flow'][index-n:index].sum()
        negative_sum = data['Negative_Money_Flow'][index-n:index].sum()

        if negative_sum ==0.:
            negative_sum = 0.0000000001
    
        money_ratio = positive_sum/negative_sum

        money_flow_index = (money_ratio/(1+money_ratio))*100

        data.set_value(index, 'Money_Flow_Index', money_flow_index)

data.drop('Money_Flow')

data.to_csv('money_flow_index_%i'%n, index=False)



# if data['Money_Flow'].gt(data['Money_Flow'].shift()):
#     data['Positive_Money_Index'] = data['Money_Flow'] - data['Money_flow'].shift()
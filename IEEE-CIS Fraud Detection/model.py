import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
import seaborn as sns

train_identity = pd.read_csv(
    '/Users/marc/Data-science-portfolio/IEEE-CIS Fraud Detection/dataset/train_identity.csv')
train_transactions = pd.read_csv(
    '/Users/marc/Data-science-portfolio/IEEE-CIS Fraud Detection/dataset/train_transaction.csv')

test_transactions = pd.read_csv(
    '/Users/marc/Data-science-portfolio/IEEE-CIS Fraud Detection/dataset/test_transaction.csv')
test_identity = pd.read_csv(
    '/Users/marc/Data-science-portfolio/IEEE-CIS Fraud Detection/dataset/test_identity.csv')

train = train_transactions.merge(
    train_identity, on='TransactionID', how='left')

test = test_transactions.merge(test_identity, on='TransactionID', how='left')

train.set_index('TransactionID', inplace=True)
test.set_index('TransactionID', inplace=True)

del train_transactions, train_identity
del test_transactions, test_identity


gc.collect()


def downcast_df_float_columns(df):
    list_of_columns = list(df.select_dtypes(include=['float64']).columns)
    if len(list_of_columns) >= 1:
        max_string_length = max([len(col) for col in list_of_columns])
        print("Downcasting integers for: ", list_of_columns, '\n')

        for col in list_of_columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
    else:
        print('Nothing to downcast')
    gc.collect()
    print('Done')


downcast_df_float_columns(train)
downcast_df_float_columns(test)

train.info()
train.describe()


plt.bar(['train', 'test'], [train.isnull().any().sum(), test.isnull().any().sum()])
plt.title('Count of null values in dataset')
plt.show()

train.isnull().sum() / len(train)
test.isnull().sum() / len(test)

train_missing_values = train.isnull().sum().sort_values(ascending=False) / len(train)
test_missing_values = test.isnull().sum().sort_values(ascending=False) / len(test)

fig, axes = plt.subplots(2,1,figsize=(12,8))
sns.barplot(list(train_missing_values.keys()[:10]), train_missing_values[:10], ax=axes[0])
sns.barplot(list(test_missing_values.keys()[:10]), test_missing_values[:10], ax=axes[1])
plt.show()

len(train[train.isFraud==1])/len(train) * 100
len(train[train.isFraud==0])/len(train) * 100

## 3.5% of transactions are frauds so we need to be careful of overfitting

## DT is in seconds , converting to hours could give a useful insight

train['hour'] = np.floor(train["TransactionDT"] / 3600) % 24
test['hour'] = np.floor(test["TransactionDT"] / 3600) % 24

plt.plot(train.groupby('hour').mean()['isFraud'], color='r')

ax = plt.gca()
ax2 = ax.twinx()
_ = ax2.hist(train['hour'], alpha=0.3, bins=24)
ax.set_xlabel('Encoded hour')
ax.set_ylabel('Fraction of fradulent transactions')

ax2.set_ylabel('Number of transactions')

plt.show

# Number of fradulent transactions spike between roughly 5-12 hours

train["DeviceType"].value_counts(dropna=False).plot.bar()
plt.show()

# It would be interesting to analyse some characteristics of the fraudulent transactions

fraud = train[train.isFraud==1]

sns.barplot(fraud['DeviceInfo'].value_counts(dropna=False)[:15], fraud["DeviceInfo"].value_counts(dropna=False).keys()[:15])
plt.show

sns.barplot(fraud['P_emaildomain'].value_counts(dropna=False)[:15], fraud["P_emaildomain"].value_counts(dropna=False).keys()[:15])
plt.show

sns.barplot(train['P_emaildomain'].value_counts(dropna=False)[:15], train["P_emaildomain"].value_counts(dropna=False).keys()[:15])
plt.show

heatmap = train.corr() 
f, ax= plt.subplots(figsize=(9, 6))
sns.heatmap(heatmap, vmax=0.8,square=True, cmap='BuPu')


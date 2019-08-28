import pandas as pd
import numpy as np
import gc

train_identity = pd.read_csv(
    '/Users/MarcPlunkett/Data-science-portfolio/IEEE-CIS Fraud Detection/Datasets/train_identity.csv')
train_transactions = pd.read_csv(
    '/Users/MarcPlunkett/Data-science-portfolio/IEEE-CIS Fraud Detection/Datasets/train_transaction.csv')

test_transactions = pd.read_csv(
    '/Users/MarcPlunkett/Data-science-portfolio/IEEE-CIS Fraud Detection/Datasets/test_transaction.csv')
test_identity = pd.read_csv(
    '/Users/MarcPlunkett/Data-science-portfolio/IEEE-CIS Fraud Detection/Datasets/test_identity.csv')

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

import pandas as pd
import numpy as np


def data_prepare():

    df_full = pd.read_parquet('./input/train.parquet.gzip')

#     filter events only with interactions
    df = df_full.query('timespent != 0 or reaction != 0')

    test = pd.read_parquet('./input/test.parquet.gzip')
    test_df = df[df.user_id.isin(test.user_id)]

#     choose random users to validate
    np.random.seed(42)
    val_users = df[~df.user_id.isin(test.user_id)]['user_id'].unique()
    val_users = np.random.choice(val_users, size=250000)
    val_df = df[df.user_id.isin(val_users)]

#     filter train df from validation set
    mask1 = (~df.user_id.isin(test.user_id))
    mask2 = (~df.user_id.isin(val_users))
    train_df = df[mask1 & mask2]

#     devide last events as true values
    val_true = val_df.groupby('user_id').tail(1)
    val_true['target'] = 1

    val_df = val_df.merge(val_true, how='left', on=['user_id', 'item_id'])
    val_df = val_df[val_df.target.isna()]

#     save
    train_df.to_pickle('./data/train_df.pkl')
    test_df.to_pickle('./data/test_df.pkl')
    val_df.to_pickle('./data/val_df.pkl')
    val_true.to_pickle('./data/val_true.pkl')


if __name__ == '__main__':
    data_prepare()

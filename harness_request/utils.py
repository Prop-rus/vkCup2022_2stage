import pandas as pd


def select_subscribers(req_type):
    df = pd.read_pickle(f'../data/{req_type}_df.pkl')
    return df.user_id.unique()

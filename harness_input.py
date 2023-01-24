from datetime import datetime
import pandas as pd


def process_events(df):

    df['entityType'] = 'user'
    df['targetEntityType'] = 'item'
    df['entityId'] = df['entityId'].astype('str')
    df['targetEntityId'] = df['targetEntityId'].astype('int').astype('str')
    df['eventTime'] = pd.to_datetime(df['eventTime'])


def to_json(df_to_j, name):
    df_to_j = df_to_j[['eventTime',
                       'entityId',
                       'targetEntityId',
                       'event',
                       'entityType',
                       'targetEntityType']]
    return df_to_j.to_json(name,
                           orient='records',
                           date_format='iso',
                           lines=True)


def create_harness_events():
    '''
    creates needed to action ML Harness json file from events
    '''
    df = pd.read_parquet('./input/train.parquet.gzip')
    items_meta = pd.read_parquet('./input/items_meta.parquet.gzip')
    val_true = pd.read_pickle('./data/val_true.pkl')
    df = df.merge(items_meta[['item_id', 'source_id']],
                  how='inner',
                  on='item_id')

    df['eventTime'] = pd.date_range(end=datetime.now(),
                                    periods=df.index.max(),
                                    freq='100ms')

    df = df.merge(val_true[['user_id', 'item_id', 'target']],
                  how='left',
                  on=['user_id', 'item_id'])
    df = df[df.target.isna()]

    df = pd.concat((df[df.reaction == 0],
                    df[df.reaction == 1],
                    df[df.reaction == -1]))

    df['event'] = 'read-big'
    df.loc[df.timespent < 3, 'event'] = 'read-small'

    df.loc[(df.reaction == 0) & (df.timespent == 0), 'event'] = 'non_act'

    df.loc[df.reaction == 1, 'event'] = 'like'
    df.loc[df.reaction == -1, 'event'] = 'dislike'

    df.rename(columns={
        'user_id': 'entityId',
        'item_id': 'targetEntityId',
        }, inplace=True)

    df = process_events(df)

    to_json(df, 'vk_events.json')


if __name__ == '__main__':
    create_harness_events()

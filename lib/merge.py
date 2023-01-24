import pandas as pd
import settings


def merge_cands_features(req_type='test'):
    als = pd.read_pickle(f'./data/als_candidates_{req_type}.pkl')
    als = als.groupby('user_id').head(50)
    user_features = pd.read_parquet('./data/user_id_features.parquet')
    item_features = pd.read_parquet('./data/item_id_features.parquet')
    val_true = pd.read_pickle('./data/val_true.pkl')
    print('loaded')

    candidates = als
    if 'harn' in settings.candidates_to_merge:
        print('merging harn')
        harn = pd.read_parquet(f'./data/harn_cands_{req_type}.parquet')
        harn = harn.groupby('user_id').head(50)
        candidates = candidates.merge(harn,
                                      how='outer',
                                      on=['user_id', 'item_id']
                                      )
    candidates = candidates.merge(val_true,
                                  how='left',
                                  on=['user_id', 'item_id'])
    candidates = candidates.merge(user_features,
                                  how='inner',
                                  left_on='user_id',
                                  right_index=True)
    candidates = candidates.merge(item_features,
                                  how='inner',
                                  on='item_id')

    candidates.columns = [str(x) for x in candidates.columns]
    candidates.to_parquet(f'./data/candidates_{req_type}.parquet.gzip', compression='gzip')


if __name__ == '__main__':
    merge_cands_features(req_type='val')
    merge_cands_features(req_type='test')

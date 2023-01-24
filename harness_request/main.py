import io
import pandas as pd
import numpy as np

if __name__ == '__main__':
    from utils import select_subscribers
    from get_recommendations import get_recommendations_from_harness
else:
    from harness_request.utils import select_subscribers
    from harness_request.get_recommendations import get_recommendations_from_harness


def request_harness(req_type='val'):

    subscribers = select_subscribers(req_type)
    res_total = pd.DataFrame([], columns=['user_id', 'item_id', 'scores'])
    for p in np.array_split(subscribers, 4):
        print('part')
        res = get_recommendations_from_harness(p.tolist())
        res_df = pd.read_csv(io.StringIO('\n'.join(res)),
                             delimiter=';',
                             names=['user_id', 'item_id', 'scores'])
        res_total = pd.concat((res_total, res_df))
    res_total = res_total[['user_id', 'item_id']]
    res_total['item_id'] = res_total['item_id'].str.split(',')
    res_total['scores'] = res_total['scores'].str.split(',')
    tmp1 = res_total[['user_id', 'item_id']].explode('item_id')
    tmp2 = res_total[['user_id', 'scores']].explode('scores')[['scores']]
    res_total = pd.concat((tmp1, tmp2), axis=1)
    res_total = res_total[~res_total.item_id.isna()]
    res_total['user_id'] = res_total['user_id'].astype('int32')
    res_total['item_id'] = res_total['item_id'].astype('int32')
    res_total['scores'] = res_total['scores'].astype('float')

    if req_type == 'test':
        candidates = pd.read_parquet('./input/fresh_candidates.parquet.gzip')
        res_total = res_total[res_total.item_id.isin(candidates.item_id)]
    df_full = pd.read_parquet('./input/train.parquet.gzip')
    val_true = pd.read_pickle('./data/val_true.pkl')
    df_full = df_full.merge(val_true[['user_id', 'item_id', 'target']],
                            how='left',
                            on=['user_id', 'item_id'])
    df_full = df_full[df_full.target.isna()]

    res_total = res_total.merge(df_full, how='left', on=['user_id', 'item_id'])
    res_total = res_total[res_total.timespent.isna()][['user_id', 'item_id', 'scores']]
    res_total['rank_harn'] = res_total.groupby('user_id')["scores"].rank(method="first", ascending=False)
    res_total['rank_harn'] = res_total['rank_harn'].astype('int')
    res_total.rename(columns={'scores': 'scores_harn'}, inplace=True)

    res_total.to_parquet(f'./data/harn_cands_{req_type}.parquet')


if __name__ == '__main__':
    request_harness()

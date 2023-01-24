import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import settings


def gen_features(group_type, pref):
    df = pd.read_parquet('./input/train.parquet.gzip')

    timespent = df.groupby(group_type)
    timespent = timespent.agg({'timespent':
                               ['mean',
                                'median',
                                'min',
                                'max',
                                lambda x: pd.Series.mode(x).iat[0]]})
    timespent.columns = [f'{pref}_tm_' + x[1] for x in timespent.columns]
    tmp1 = df.query('timespent != 0').groupby(group_type)['timespent'].count()
    tmp2 = df.query('timespent == 0').groupby(group_type)['timespent'].count()
    timespent['perc_read'] = tmp1 / tmp2

    likes = df.query('reaction != -1').groupby(group_type)
    likes = likes.agg({'reaction':
                      ['mean',
                       'median',
                       'sum',
                       lambda x: pd.Series.mode(x).iat[0]]})
    likes.columns = [f'{pref}_lk_' + x[1] for x in likes.columns]

    dislikes = df.query('reaction != 1').groupby(group_type)
    dislikes = dislikes.agg({'reaction':
                             ['mean',
                              'median',
                              'sum',
                              lambda x: pd.Series.mode(x).iat[0]]})
    dislikes.columns = [f'{pref}_dis_' + x[1] for x in dislikes.columns]

    df_clean = df.query('timespent != 0 or reaction != 0')

    time_cl = df_clean.groupby(group_type)
    time_cl = time_cl.agg({'timespent':
                           ['mean',
                            'median',
                            'min',
                            'max',
                            lambda x: pd.Series.mode(x).iat[0]]})
    time_cl.columns = [f'{pref}_tm_cl_' + x[1] for x in time_cl.columns]
    tmp1 = df_clean.query('timespent != 0').groupby(group_type)['timespent'].count()
    tmp2 = df_clean.query('timespent == 0').groupby(group_type)['timespent'].count()
    time_cl['perc_cl_read'] = tmp1 / tmp2

    likes_cl = df_clean.query('reaction != -1')
    likes_cl = likes_cl.groupby(group_type).agg({'reaction':
                                                 ['mean',
                                                  'median',
                                                  'sum',
                                                  lambda x: pd.Series.mode(x).iat[0]]})
    likes_cl.columns = [f'{pref}_lk_cl_' + x[1] for x in likes_cl.columns]

    dislikes_cl = df_clean.query('reaction != 1')
    dislikes_cl = dislikes_cl.groupby(group_type).agg({'reaction':
                                                       ['mean',
                                                        'median',
                                                        'sum',
                                                        lambda x: pd.Series.mode(x).iat[0]]})
    dislikes_cl.columns = [f'{pref}_dis_cl_' + x[1] for x in dislikes_cl.columns]

    features = pd.concat([
        timespent,
        likes,
        dislikes,
        time_cl,
        likes_cl,
        dislikes_cl,
        ], axis=1)

    return features


def gen_all_features():
    user_features = gen_features('user_id', 'us')
    user_features.to_parquet('./data/user_id_features.parquet')

    item_features = gen_features('item_id', 'it')

#     we can shrink item features with their embeddings and factors from als
    item_meta = pd.read_parquet('./input/items_meta.parquet.gzip')
    item_factors = pd.read_pickle('./data/item_factors.pkl')
    item_factors.columns = ['item_id'] + ['f' + str(x) for x in range(settings.als_factors)]

    emb = np.array(item_meta.embeddings.tolist())
    emb = pd.DataFrame(emb)
    item_meta = pd.concat((item_meta, emb), axis=1)
    item_meta.drop(columns=['embeddings'], inplace=True)
    item_meta = pd.concat((item_meta, item_features), axis=1)

    item_meta = item_meta.merge(item_factors, how='left', on='item_id')
    item_meta = item_meta.fillna(0)

    cols = [x for x in item_meta.columns if x not in ['item_id', 'source_id']]
    pca = PCA(n_components=50)
    new_f = pca.fit_transform(item_meta[cols].values)
    new_f = pd.DataFrame(new_f)
    item_meta = pd.concat((item_meta[['item_id', 'source_id']], new_f), axis=1)
    item_meta.columns = [str(x) for x in item_meta.columns]
    item_meta.to_parquet('./data/item_id_features.parquet')


if __name__ == '__main__':
    gen_all_features()

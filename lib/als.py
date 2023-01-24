import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from utils import ItemEncoder
import settings


class AlsRecommender:

    def __init__(self,
                 item_encoder,
                 user_encoder,
                 n_factors=700,
                 n_candidates=250,
                 df_full=None):
        self.df_full = df_full
        self.df = None
        self.item_encoder = item_encoder
        self.user_encoder = user_encoder
        self.n_factors = n_factors
        self.n_candidates = n_candidates
        self.csr_data = None
        self.pred_als = None
        self.recommender = None

    def prepare_data(self, train_df, test_df, val_df, val_true):
        print('als prepare data')

        self.df_full['timedacay'] = self.df_full.index.values / self.df_full.index.max()
        self.df_full['data'] = 1 + self.df_full['timedacay'] * self.df_full['timespent']
        self.df = self.df_full[(self.df_full.timespent != 0) | (self.df_full.reaction == 1)]

        tr = self.df[self.df.user_id.isin(train_df.user_id)]
        te = self.df[self.df.user_id.isin(test_df.user_id)]
        val = self.df[self.df.user_id.isin(val_df.user_id)]
        val_tr = val.merge(val_true, on=['user_id', 'item_id'])
        val = val.merge(val_tr[['user_id', 'item_id', 'target']],
                        how='left',
                        on=['user_id', 'item_id'])
        val = val[val.target.isna()]
        self.df = pd.concat((tr, val, te))

        self.item_encoder.fit(self.df.item_id.values)
        self.item_encoder.lock()
        self.user_encoder.fit(self.df.user_id.values)
        self.user_encoder.lock()

        self.df['asset_code'] = self.item_encoder.transform(self.df['item_id'])
        self.df['subscriber_code'] = self.user_encoder.transform(self.df['user_id'])

        csr_df = csr_matrix((self.df['data'],
                             (self.df.subscriber_code, self.df.asset_code)),
                            shape=(self.df.subscriber_code.nunique(), len(self.item_encoder)))
        return csr_df

    def fit_predict(self):
        print('als fit-predict')
        self.recommender = AlternatingLeastSquares(factors=self.n_factors, iterations=100)
        self.recommender.fit(self.csr_data)

#         predict all users excluding val_true
        pred_ids, scores = self.recommender.recommend(range(self.csr_data.shape[0]),
                                                      self.csr_data, N=self.n_candidates,
                                                      filter_already_liked_items=True)

        pred_als = pd.DataFrame(
            {'users': np.ravel(np.repeat(np.arange(0, self.csr_data.shape[0]), self.n_candidates),
                               order='F'),
             'items': pred_ids.flatten(),
             'score_als': scores.flatten(),
             'rank_als': np.tile(list(range(self.n_candidates)), self.csr_data.shape[0])}
        )
        pred_als['item_id'] = self.item_encoder.inverse_transform(pred_als['items'].values)
        pred_als['user_id'] = self.user_encoder.inverse_transform(pred_als['users'].values)

        return pred_als

    def save_item_factors(self):
        #  save item_factors
        item_factors = self.recommender.item_factors
        item_factors = pd.DataFrame(item_factors).reset_index()
        item_factors['index'] = self.item_encoder.inverse_transform(item_factors['index'])
        item_factors.rename(columns={'index': 'item_id'}, inplace=True)
        item_factors.to_pickle('./data/item_factors.pkl')


def als_candidates():
    df_full = pd.read_parquet('./input/train.parquet.gzip')

    train_df = pd.read_pickle('./data/train_df.pkl')
    test_df = pd.read_pickle('./data/test_df.pkl')
    val_df = pd.read_pickle('./data/val_df.pkl')
    val_true = pd.read_pickle('./data/val_true.pkl')

    candidates_df = pd.read_parquet('./input/fresh_candidates.parquet.gzip')

    item_encoder = ItemEncoder(keep_top_n=None)
    user_encoder = ItemEncoder(keep_top_n=None)

    print('start ALS')
    als = AlsRecommender(item_encoder,
                         user_encoder,
                         n_factors=settings.als_factors,
                         n_candidates=settings.n_candidates,
                         df_full=df_full)
    als.csr_data = als.prepare_data(train_df, test_df, val_df, val_true)
    cands_als = als.fit_predict()
    als.save_item_factors()

#     make candidates to validation part
    val_cands = cands_als[cands_als.user_id.isin(val_df.user_id)]
    df_full_with_no_val_true = df_full[df_full.user_id.isin(val_df.user_id)]
    df_full_with_no_val_true = df_full_with_no_val_true.merge(val_true,
                                                              how='left',
                                                              on=['user_id', 'item_id'])
    df_full_with_no_val_true = df_full_with_no_val_true[df_full_with_no_val_true.timespent_y.isna()]
    df_full_with_no_val_true['past'] = 1
    val_cands = val_cands.merge(df_full_with_no_val_true[['user_id', 'item_id', 'past']],
                                how='left',
                                on=['user_id', 'item_id'])
    val_cands = val_cands[val_cands.past.isna()]

    val_cands[['user_id', 'item_id', 'score_als', 'rank_als']].to_pickle('./data/als_candidates_val.pkl')

#     make candidates to test part
    test_cands = cands_als[cands_als.user_id.isin(test_df.user_id)]
    test_cands = test_cands[test_cands.item_id.isin(candidates_df.item_id)]
    test_cands = test_cands.merge(df_full, how='left', on=['user_id', 'item_id'])
    test_cands = test_cands[test_cands.timespent.isna()]
    test_cands[['user_id', 'item_id', 'score_als', 'rank_als']].to_pickle('./data/als_candidates_test.pkl')


if __name__ == '__main__':
    als_candidates()

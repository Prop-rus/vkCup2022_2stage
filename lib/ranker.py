import lightgbm
from lightgbm.callback import early_stopping
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
from numba import njit, prange


def filter_zero_groups(candidates: pd.DataFrame):
    sum_targ = candidates.groupby('user_id')['target'].sum()
    sum_targ = sum_targ[sum_targ > 0]
    candidates = candidates[candidates.user_id.isin(sum_targ.index)]
    return candidates


def downsample_negative(df: pd.DataFrame, keep: float):
    np.random.seed(33)
    negatives = np.where((df['target'] == 0).values)[0]
    positives = np.where((df['target'] == 1).values)[0]
    num_negatives = int(np.ceil(len(negatives) * keep))
    np.random.shuffle(negatives)
    new_negatives = negatives[:num_negatives]
    index = np.concatenate([positives, new_negatives])
    index.sort()
    return df.iloc[index]


@njit(parallel=True)
def pmrr(preds, targets, groups):
    total = 0
    group_starts = np.cumsum(groups)

    for group_id in prange(len(groups)):
        group_end = group_starts[group_id]
        group_start = group_end - groups[group_id]
        ranks = np.argsort(preds[group_start: group_end])[::-1]
        for i in range(min(len(ranks), 100)):
            if targets[group_start + ranks[i]] == 1:
                total += 1 / (i + 1)

    return total / len(groups)


def get_group_for_lgb(df):
    groups = df.groupby('user_id')['item_id'].count().values
    return np.array(groups)


def lgb_mrr_wrapper(preds, lgb_dataset):
    metric = pmrr(preds, lgb_dataset.label, lgb_dataset.group)
    return 'mrr', metric, True


def rank_candidates():

    candidates = pd.read_parquet('./data/candidates_val.parquet.gzip')

    df_true = pd.read_pickle('./data/val_true.pkl')
    candidates = candidates.sort_values(['user_id'])
    candidates.drop(columns='target', inplace=True)
    df_true['target'] = 1
    candidates = candidates.merge(df_true[['user_id', 'item_id', 'target']],
                                  how='left',
                                  on=['user_id', 'item_id'])
    candidates.target.fillna(0, inplace=True)
    train_us, test_us = train_test_split(candidates.user_id.unique(),
                                         test_size=0.2,
                                         random_state=42)
    train_us, val_us = train_test_split(train_us, test_size=0.2)
    print('splitted')
    num_users = candidates.user_id.nunique()

    ttrain = candidates[candidates.user_id.isin(train_us)]
    ttest = candidates[candidates.user_id.isin(test_us)]
    tval = candidates[candidates.user_id.isin(val_us)]
    ttrain = ttrain.sort_values('user_id')
    ttest = ttest.sort_values('user_id')
    tval = tval.sort_values('user_id')
    
    columns = [c for c in ttrain.columns if c not in ['target',
                                                      'user_id',
                                                      'item_id',
                                                      'users',
                                                      'items',
                                                     'timespent', 'reaction']]
    # ttraind = downsample_negative(filter_zero_groups(ttrain), 0.3)
    print('downsampled')

    # X_train, y_train, train_groups = ttraind[columns], ttraind['target'], get_group_for_lgb(ttraind.user_id.values)
    X_train, y_train, train_groups = ttrain[columns], ttrain['target'], get_group_for_lgb(ttrain)
    X_val, y_val, val_groups = tval[columns], tval['target'], get_group_for_lgb(tval)
    X_test = ttest[columns]

    lgb_train = lightgbm.Dataset(
        X_train, y_train,
        group=train_groups, free_raw_data=False
    )
    lgb_eval = lightgbm.Dataset(
        X_val, y_val, reference=lgb_train,
        group=val_groups, free_raw_data=False
    )

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": 100,
        "n_estimators": 1000,
        "boosting_type": "gbdt",
        "is_unbalance": False,
        "learning_rate": 0.04,
        'lambda_l1': 0.98,
        'lambda_l2': 7.84,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 1,
    }
    print('ready to train')
    ranker = lightgbm.train(
        params,
        lgb_train,
        valid_sets=[lgb_eval],
        feval=lgb_mrr_wrapper,
        callbacks=[
            early_stopping(250),
            lightgbm.log_evaluation(10)
        ],
    )
    print('trained')
    test_pred = ranker.predict(X_test)
    ttest['pred'] = test_pred
    ttest_sorted = ttest.sort_values(['user_id', 'pred'], ascending=[True, False])
    ttest_sorted['rnk'] = ttest_sorted.groupby('user_id').cumcount()

    mrr = (1 / (ttest_sorted[ttest_sorted['target'] == 1].rnk + 1)).sum() / ttest.user_id.nunique()

    metrics = {'mrr': mrr, 'candidates_recall': candidates.target.sum() / num_users}
    print(metrics)
    pickle.dump(ranker, open('./data/ranker', 'wb'))
    print('ranking done')


if __name__ == '__main__':
    rank_candidates()

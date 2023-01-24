import pandas as pd
import numpy as np
import joblib
import settings


def make_predictions():

    # load model
    ranker = joblib.load('./data/ranker')
#     load candidates to predict
    candidates = pd.read_parquet('./data/candidates_test.parquet.gzip')

    columns = [c for c in candidates.columns if c not in ['target',
                                                          'user_id',
                                                          'item_id',
                                                          'users',
                                                          'items',
                                                          'timespent', 
                                                          'reaction']]

    chunks = np.array_split(candidates.user_id.unique(), settings.pred_chunks)

    res_df = pd.DataFrame([], columns=['user_id', 'item_id'])
    for part in chunks:
        part_df = candidates.query('user_id in @part')
        test_pred = ranker.predict(part_df[columns])
        part_df['pred'] = test_pred
        part_df = part_df.sort_values(['user_id', 'pred'],
                                      ascending=[True, False])
        part_df['rnk'] = part_df.groupby('user_id').cumcount()
        part_df = part_df.groupby('user_id').head(20)
        part_df = part_df.groupby('user_id')['item_id'].apply(list)
        part_df = part_df.to_frame().reset_index()
        res_df = pd.concat((res_df, part_df))
        print('chunk done')

    res_df.to_parquet('./output/prediction_out.parquet.gzip', compression='gzip')
    print('prediction done')


if __name__ == '__main__':
    make_predictions()

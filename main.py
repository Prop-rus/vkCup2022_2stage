from lib.data_prepare import data_prepare
from lib.als import als_candidates
from harness_request.main import request_harness
from lib.features import gen_all_features
from lib.merge import merge_cands_features
from lib.ranker import rank_candidates
from lib.make_predictions import make_predictions
import settings


def main():

    data_prepare()

    als_candidates()

    if 'harn' in settings.candidates_to_merge:
        request_harness('val')
        request_harness('test')

    print('prepare features')
    gen_all_features()

    print('merging')
    merge_cands_features('val')
    merge_cands_features('test')

    print('rank')
    rank_candidates()
    print('ranker trained')
    
    make_predictions()


if __name__ == '__main__':
    main()

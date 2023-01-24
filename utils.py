from collections import Counter
import numpy as np


class ItemEncoder:
    def __init__(self, keep_top_n):
        self.keep_top_n = keep_top_n
        self.item_to_id = None
        self.id_to_item = None
        self.counter = Counter()
        self.is_locked = False

    def fit(self, data):
        if self.is_locked:
            raise Exception("encoder has been locked")
        self.counter.update(data)

    def transform(self, data):
        if not self.is_locked:
            raise Exception("transforming with unlocked encoder is not allowed")
        return np.vectorize(lambda x: self.item_to_id.get(x, 0))(data)

    def inverse_transform(self, data):
        if not self.is_locked:
            raise Exception("transforming with unlocked encoder is not allowed")
        return np.vectorize(self.id_to_item.__getitem__)(data)

    def lock(self):
        self.id_to_item = list(map(lambda x: x[0], self.counter.most_common(self.keep_top_n)))
        self.item_to_id = {x: i for i, x in enumerate(self.id_to_item)}
        assert len(self.id_to_item) == len(self.item_to_id)
        self.is_locked = True

    def __len__(self):
        return len(self.id_to_item)

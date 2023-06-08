
import numpy as np 
import pandas as pd

def text2uint8(
    text,
):
    "transform text to a numpy array with uint8 dtype using UTF-8 encoding"
    b = bytes(text, encoding='utf8')
    return np.frombuffer(b, dtype=np.uint8)


def text2ordinals(
    text,
    dtype=np.int32,
):
    "transform text to a numpy"
    return np.array([ord(ch) for ch in text]).astype(dtype)

def ordinals2text(
    ordinals
):
    return "".join([chr(ordinal) for ordinal in ordinals])


class PermutationSampler(object):
    
    def __init__(self, indexes):
        self.indexes = np.asarray(indexes)
        self.refresh_permutation()
        self.N = len(indexes)

    @property
    def epoch(self):
        return self._pos/self.N
    
    def refresh_permutation(self):
        print('new data epoch')
        self._pos = 0
        self.permuted = self.indexes[np.random.permutation(len(self.indexes))]  
    
    def get_batch(self, batch_size):
        lb = self._pos
        ub = lb+batch_size
        self._pos = ub 
        batch_i = self.permuted[lb:ub]
        self._pos = ub
        if ub >= len(self.indexes):
            self.refresh_permutation()
        if len(batch_i) < batch_size:
            top_up = self.get_batch(batch_size-len(batch_i))
            batch_i = np.concatenate(
                [
                    batch_i,
                    top_up,
                ],
                axis=0,
            )
        return batch_i
        

class ChainedTransforms(object):
    
    def __init__(
        self, 
        transforms,
    ):
        self.transforms = transforms        
    
    def __call__(self, batch):
        for transform in self.transforms:
            batch = transform(batch)
        return batch

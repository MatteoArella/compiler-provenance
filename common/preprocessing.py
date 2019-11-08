from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np

'''
Transform assembly into abstract assembly, i.e. consider only instructions' mnemonics
'''
class AbstractedAsmTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        '''
        result = []
        for instruction in X:
            result.append(' '.join([ i.split(' ')[0] for i in instruction ]))
        return np.asarray(result)
        '''
        return [' '.join([ i.split(' ')[0] for i in instruction ]) for instruction in X]

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)

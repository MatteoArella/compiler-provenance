from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load, dump
from os import path
import numpy as np

'''
Apply TD-iDF to abstracted asm set
'''
class AbstractedAsmTfidfVectorizer(TfidfVectorizer):
    def __init__(self, vocabulary_path=None,
                 input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, analyzer='word',
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.vocabulary_path_ = 'vocabulary.joblib'
        if vocabulary_path is not None:
            self.vocabulary_path_ = vocabulary_path
        if path.exists(self.vocabulary_path_): # load vocabulary
            super().__init__(vocabulary=load(self.vocabulary_path_),
                            input=input, encoding=encoding, decode_error=decode_error,
                            strip_accents=strip_accents, lowercase=lowercase,
                            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
                            stop_words=stop_words, token_pattern=token_pattern,
                            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                            max_features=max_features, binary=binary,
                            dtype=dtype)
            print('after init load voc')
        else:
            super().__init__(input=input, encoding=encoding, decode_error=decode_error,
                                strip_accents=strip_accents, lowercase=lowercase,
                                preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
                                stop_words=stop_words, token_pattern=token_pattern,
                                ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                                max_features=max_features, vocabulary=vocabulary, binary=binary,
                                dtype=dtype)
            print('after init no load')
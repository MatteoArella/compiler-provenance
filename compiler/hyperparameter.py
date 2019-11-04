import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from joblib import dump

from pprint import pprint

class EstimatorSelectionHelper:
    def __init__(self, pipelines):
        self.pipelines_ = pipelines
        self.keys_ = pipelines.keys()
        self.grid_searches_ = {}

    def fit(self, X, y, **grid_kwargs):
        for key in self.keys_:
            print('Running GridSearchCV for %s.' % key)
            pipeline = self.pipelines_[key]['pipeline']
            params = self.pipelines_[key]['params']
            grid_search = GridSearchCV(pipeline, params, **grid_kwargs)
            grid_search.fit(X, y)
            self.grid_searches_[key] = grid_search

    def dump(self, dirpath):
        for key in self.keys_:
            grid_search = self.grid_searches_[key]
            dump(grid_search, '%s/grid_search-%s.joblib' % (dirpath, key), compress=('gzip', 6))
            dump(grid_search.best_estimator_, '%s/pipeline-%s.joblib' % (dirpath, key), compress=('gzip', 6))
    
    def score_summary(self, sort_by='mean_test_score'):
        frames = []
        for name, grid_search in self.grid_searches_.items():
            frame = pd.DataFrame(grid_search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame)*[name]
            frames.append(frame)
        df = pd.concat(frames)
        
        df = df.sort_values([sort_by], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)
        
        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator']+columns
        df = df[columns]
        return df
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from joblib import dump

class EstimatorSelectionHelper:
    def __init__(self, pipelines, **kwargs):
        self.pipelines_ = pipelines
        self.keys_ = pipelines.keys()
        self.grid_searches_ = {}
        self.compression_ = kwargs.get('compression', ('gzip', 6))

    def fit(self, X, y, **kwargs):
        dump_after_fit = kwargs.get('dump_after_fit', False)
        if dump_after_fit:
            dirpath = kwargs['dump_dirpath']
        for key in self.keys_:
            print('Running GridSearchCV for {}.'.format(key))
            pipeline = self.pipelines_[key]['pipeline']
            params = self.pipelines_[key].get('hyperparams', {})
            grid_params = { **kwargs, **self.pipelines_[key].get('grid_params', {}) }
            grid_search = GridSearchCV(pipeline, params, **grid_params)
            grid_search.fit(X, y)
            self.grid_searches_[key] = grid_search
            if dump_after_fit:
                dump(grid_search, '{}/grid_search-{}.joblib'.format(dirpath, key), compress=compression)
                dump(grid_search.best_estimator_, '{}/pipeline-{}.joblib'.format(dirpath, key), compress=compression)
    
    def dump(self, dirpath, **kwargs):
        compression = kwargs.get('compression', self.compression_)
        for key in self.keys_:
            grid_search = self.grid_searches_[key]
            dump(grid_search, '{}/grid_search-{}.joblib'.format(dirpath, key), compress=compression)
            dump(grid_search.best_estimator_, '{}/pipeline-{}.joblib'.format(dirpath, key), compress=compression)

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
from joblib import load

from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd

import sys
sys.path.append('code/common')

class ClassificationTester:
    def pipelines_load(self, pipelines_path, summary_name):
        self.summary_ = load('{}/{}'.format(pipelines_path, summary_name))
        self.target_encoder_ = LabelEncoder()
        self.classifiers_ = self.summary_['estimator'].unique()
        self.pipelines_ = [ load('{}/pipeline-{}.joblib'.format(pipelines_path, i)) for i in self.classifiers_]
        return self

    def pipelines_predict(self, X):
        self.predictions_ = [ i.predict(X) for i in self.pipelines_ ]
        self.predictions_proba_ = [ i.predict_proba(X) for i in self.pipelines_ ]
        return self
    
    def pipelines_scoring(self, y_true, scoring=f1_score, **kwargs):
        scoring_average = kwargs.get('average')
        self.y_test_ = y_true
        y = self.target_encoder_.fit_transform(y_true)
        if scoring_average is not None:
            self.scores_ = [ scoring(y, self.target_encoder_.transform(i), average=scoring_average) for i in self.predictions_ ]
        else:
            self.scores_ = [ scoring(y, self.target_encoder_.transform(i)) for i in self.predictions_ ]
        amax = np.argmax(self.scores_)
        self.best_classifier_, self.best_estimator_ = self.classifiers_[amax], self.pipelines_[amax]
        return self.best_classifier_, self.best_estimator_

    def classification_reports(self):
        return pd.DataFrame({ i: { 'report': classification_report(self.y_test_, j) }  for i, j in zip(self.classifiers_, self.predictions_)}).T

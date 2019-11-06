from joblib import load

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

#import numpy as np

class ClassificationTester:
    def __init__(self, test_set_path, **kwargs):
        self.test_set_path_ = test_set_path

    def fit(self, fitted_path):
        self.fitted_path_ = fitted_path
        test = load(self.test_set_path_)
        self.X_test_, y_test = test['instructions'], test['target']
        self.summary_ = load('{}/summary.joblib'.format(self.fitted_path_))
        self.target_encoder_ = LabelEncoder()
        self.y_test_ = self.target_encoder_.fit_transform(y_test)
        self.classifiers_ = self.summary_['estimator'].unique()
        self.pipelines_ = [ load('{}/pipeline-{}.joblib'.format(self.fitted_path_, i)) for i in self.classifiers_]
        #self.scores_ = [ roc_auc_score(self.y_test_, self.target_encoder_.transform(i.predict(self.X_test_))) for i in self.pipelines_]
        #self.best_estimator_ = self.pipelines_[np.argmax(self.scores_)]
        return self

    def predict(self):
        self.predictions_ = [ i.predict(self.X_test_) for i in self.pipelines_ ]
        return self

    def classification_reports(self):
        return [(i, classification_report(self.y_test_, self.target_encoder_.transform(j))) for i, j in zip(self.classifiers_, self.predictions_)]

    def confusion_matrixes(self):
        return [(i, confusion_matrix(self.y_test_, self.target_encoder_.transform(j))) for i, j in zip(self.classifiers_, self.predictions_)]

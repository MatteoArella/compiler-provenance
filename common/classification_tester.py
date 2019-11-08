from joblib import load

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd

import sys
sys.path.append('code/common')

def plot_confusion_matrix(confusion_matrix, classes,
                          normalize=False,
                          title=None,
                          cmap='Blues'):
    
    from sklearn.utils.multiclass import unique_labels
    import matplotlib.pyplot as plt

    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    else:
        cm = confusion_matrix

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_roc_curve(fpr, tpr, roc_auc):
    import matplotlib.pyplot as plt
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.legend(loc="lower right")

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
    
    def pipelines_scoring(self, y_true, scoring=roc_auc_score):
        self.y_test_ = y_true
        y = self.target_encoder_.fit_transform(y_true)
        self.scores_ = [ scoring(y, self.target_encoder_.transform(i)) for i in self.predictions_ ]
        amax = np.argmax(self.scores_)
        self.best_classifier_, self.best_estimator_ = self.classifiers_[amax], self.pipelines_[amax]
        return self.best_classifier_, self.best_estimator_

    def classification_reports(self):
        return pd.DataFrame({ i: { 'report': classification_report(self.y_test_, j) }  for i, j in zip(self.classifiers_, self.predictions_)})

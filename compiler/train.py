from joblib import load, dump
from sys import exit

from preprocessing import AbstractedAsmTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from hyperparameter import EstimatorSelectionHelper
from datetime import date
import os

if not (os.path.exists('train.joblib') and os.path.exists('test.joblib')):
    print('Split data before extract features')
    exit(1)

train = load('train.joblib')
X_train, y_train = train['instructions'], train['target']
test = load('test.joblib')
X_test, y_test = test['instructions'], test['target']

pipelines = {
    'SVMClassifier': {   'pipeline': Pipeline([
                                    ('asm_preproc', AbstractedAsmTransformer()),
                                    ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True)),
                                    ('clf', SVC(gamma='scale', probability=True))
                                ]),
                        'params': {
                            'tfidf__ngram_range': [(1,2), (1,3), (1,4), (1,5),
                                                   (2,2), (3,3), (4,4), (5,5)],
                            'clf__kernel': ['linear', 'poly', 'rbf'],
                            'clf__C': [0.01, 0.1, 1, 10, 100],
                        },
    },
    'MultinomialNVClassifier': { 'pipeline': Pipeline([
                                    ('asm_preproc', AbstractedAsmTransformer()),
                                    ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True)),
                                    ('clf', MultinomialNB())
                                ]),
                                'params': {
                                    'tfidf__ngram_range': [(1,2), (1,3), (1,4), (1,5),
                                                            (2,2), (3,3), (4,4), (5,5)],
                                }
    }
}

dirpath = date.today().strftime('%d-%m-%Y')

estimator = EstimatorSelectionHelper(pipelines)
estimator.fit(X_train, y_train, cv=5, n_jobs=-1, iid=False, refit=True, verbose=10)
summary = estimator.score_summary()
print(summary)

if not os.path.exists(dirpath):
    os.makedirs(dirpath)

estimator.dump(dirpath)
dump(summary, '%s/summary.joblib' % (dirpath), compress=('gzip', 6))

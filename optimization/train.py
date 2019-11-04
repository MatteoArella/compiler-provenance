from hyperparameter import EstimatorSelectionHelper
from preprocessing import AbstractedAsmTransformer, OptLevelLabelEncoder
from joblib import load, dump
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import date
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

def lr_cv(splits, X, Y, pipeline, average_method):
    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=777)
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for train, test in kfold.split(X, Y):
        lr_fit = pipeline.fit(X[train], Y[train])
        prediction = lr_fit.predict(X[test])
        scores = lr_fit.score(X[test],Y[test])
        accuracy.append(scores * 100)
        precision.append(precision_score(Y[test], prediction, average=average_method)*100)
        print('              negative    neutral     positive')
        print('precision:',precision_score(Y[test], prediction, average=None))
        recall.append(recall_score(Y[test], prediction, average=average_method)*100)
        print('recall:   ',recall_score(Y[test], prediction, average=None))
        f1.append(f1_score(Y[test], prediction, average=average_method)*100)
        print('f1 score: ',f1_score(Y[test], prediction, average=None))
        print('-'*50)
    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))

train = load('train.joblib')
test = load('test.joblib')

X_train, y_train = train['instructions'], train['target']
X_test, y_test = test['instructions'], test['target']

encoder = LabelEncoder()
encoder.fit(y_train)

#sns.countplot('target', data = train)
#plt.show()
'''
# USE NAYVE BAYES
pipeline = Pipeline([
    ('asm_prep', AbstractedAsmTransformer()),
    ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True, ngram_range=(1, 5))),
    ('clf', MultinomialNB())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('AUC score: %f' % roc_auc_score(encoder.transform(y_test), encoder.transform(y_pred)))
'''
'''
# USE RANDOM FOREST
# 0.855795 AUC score
pipeline = Pipeline([
    ('asm_prep', AbstractedAsmTransformer()),
    ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True, ngram_range=(1, 5))),
    ('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('AUC score: %f' % roc_auc_score(encoder.transform(y_test), encoder.transform(y_pred)))
'''
'''
# OVER SAMPLE
from imblearn.over_sampling import RandomOverSampler, SMOTE
import imblearn.pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import *
from sklearn.svm import LinearSVC, SVC

encoder = LabelEncoder()
encoder.fit(y_train)

# 0.861823 AUC score
pipeline1 = imblearn.pipeline.Pipeline([
    ('asm_prep', AbstractedAsmTransformer()),
    ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True, ngram_range=(1, 3))),
    ('sampler', RandomOverSampler(random_state=777)),
    ('clf', LinearSVC())
])

# 0.862955 auc score
pipeline2 = imblearn.pipeline.Pipeline([
    ('asm_prep', AbstractedAsmTransformer()),
    ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True, ngram_range=(1, 3))),
    ('sampler', SMOTE(random_state=777)),
    ('clf', LinearSVC())
])

pipeline1.fit(X_train, y_train)
y_pred = pipeline1.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('AUC score: %f' % roc_auc_score(encoder.transform(y_test), encoder.transform(y_pred)))
print()
pipeline2.fit(X_train, y_train)
y_pred = pipeline2.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('AUC score: %f' % roc_auc_score(encoder.transform(y_test), encoder.transform(y_pred)))
'''

'''
# DOWN SAMPLE
from imblearn.under_sampling import RandomUnderSampler
import imblearn.pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# 0.869642 AUC score
pipeline = imblearn.pipeline.Pipeline([
    ('asm_prep', AbstractedAsmTransformer()),
    ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True, ngram_range=(1, 5))),
    ('sampler', RandomUnderSampler(random_state=777)),
    ('clf', LinearSVC())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('AUC score: %f' % roc_auc_score(encoder.transform(y_test), encoder.transform(y_pred)))
'''
'''
# OVER SAMPLING - DOWN SAMPLING
from imblearn.combine import SMOTETomek
import imblearn.pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# 0.873983 AUC score
pipeline = imblearn.pipeline.Pipeline([
    ('asm_prep', AbstractedAsmTransformer()),
    ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True, ngram_range=(1, 5))),
    ('sampler', SMOTETomek(ratio='auto')),
    ('clf', LinearSVC())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('AUC score: %f' % roc_auc_score(encoder.transform(y_test), encoder.transform(y_pred)))
'''

'''
'MultinomialNaiveBayesClassifier': {
        'pipeline': Pipeline([
                        ('asm_preproc', AbstractedAsmTransformer()),
                        ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True)),
                        ('clf', MultinomialNB())
                    ]),
        'hyperparams': {
                    'tfidf__ngram_range': [(1, 2), (1, 3), (1, 4), (1, 5),
                                           (2, 2), (3, 3), (4, 4), (5, 5)]
        }
    },
    'OverSamplerMultinomialNaiveBayesClassifier': {
        'pipeline': ImblearnPipeline([
                        ('asm_prep', AbstractedAsmTransformer()),
                        ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True)),
                        ('sampler', RandomOverSampler(random_state=777)),
                        ('clf', MultinomialNB())
                    ]),
        'hyperparams': {
                    'tfidf__ngram_range': [(1, 2), (1, 3), (1, 4), (1, 5),
                                           (2, 2), (3, 3), (4, 4), (5, 5)]
        }
    },
'''

random_state=25

asmTransformer = AbstractedAsmTransformer()
overSampler = RandomOverSampler(random_state=random_state)
SMOTESampler = SMOTE(random_state=random_state)
underSampler = RandomUnderSampler(random_state=random_state)

pipelines = {
    'ROSLogisticRegression': { #over-sampling
        'pipeline': ImblearnPipeline([
                        ('asm_prep', asmTransformer),
                        ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True)),
                        ('sampler', overSampler),
                        ('clf', LogisticRegression(n_jobs=-1))
                    ]),
        'hyperparams': {
                    'tfidf__ngram_range': [(1, 2), (1, 3), (1, 4), (1, 5),
                                           (2, 2), (3, 3), (4, 4), (5, 5)],
                    'clf__solver': ['sag', 'saga'],
                    'clf__C': [0.1, 1, 10, 100]
        }
    },
    'SMOTELogisticRegression': {
        'pipeline': ImblearnPipeline([
                        ('asm_prep', asmTransformer),
                        ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True)),
                        ('clf', LogisticRegression(n_jobs=-1))
                    ]),
        'hyperparams': {
                    'tfidf__ngram_range': [(1, 2), (1, 3), (1, 4), (1, 5),
                                           (2, 2), (3, 3), (4, 4), (5, 5)],
                    'clf__solver': ['sag', 'saga'],
                    'clf__C': [0.1, 1, 10, 100]
        }
    },
    'RUSLogisticRegression': { # under-sampling
        'pipeline': ImblearnPipeline([
                        ('asm_prep', asmTransformer),
                        ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True)),
                        ('sampler', underSampler),
                        ('clf', LogisticRegression(n_jobs=-1))
                    ]),
        'hyperparams': {
                    'tfidf__ngram_range': [(1, 2), (1, 3), (1, 4), (1, 5),
                                           (2, 2), (3, 3), (4, 4), (5, 5)],
                    'clf__solver': ['sag', 'saga'],
                    'clf__C': [0.1, 1, 10, 100]
        }
    },
    'BalancedLogisticRegression': { # balanced regression
        'pipeline': ImblearnPipeline([
                        ('asm_prep', asmTransformer),
                        ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True)),
                        ('clf', LogisticRegression(class_weight='balanced', n_jobs=-1))
                    ]),
        'hyperparams': {
                    'tfidf__ngram_range': [(1, 2), (1, 3), (1, 4), (1, 5),
                                           (2, 2), (3, 3), (4, 4), (5, 5)],
                    'clf__solver': ['sag', 'saga'],
                    'clf__C': [0.1, 1, 10, 100]
        }
    },
    'RandomForestClassifier': {
        'pipeline': ImblearnPipeline([
                        ('asm_prep', asmTransformer),
                        ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True)),
                        ('clf', RandomForestClassifier(n_estimators=500, class_weight='balanced', n_jobs=-1))
                    ]),
        'hyperparams': {
                    'tfidf__ngram_range': [(1, 2), (1, 3), (1, 4), (1, 5),
                                           (2, 2), (3, 3), (4, 4), (5, 5)],
                    'clf__criterion': ['gini', 'entropy']
        }
    }
}

dirpath = date.today().strftime('%d-%m-%Y')

estimator = EstimatorSelectionHelper(pipelines)
estimator.fit(X_train, y_train, scoring='roc_auc', cv=5, n_jobs=1, pre_dispatch=2, verbose=10)
#estimator.fit(X_train, y_train, cv=5, n_jobs=-1)
summary = estimator.score_summary()
print(summary)

if not os.path.exists(dirpath):
    os.makedirs(dirpath)

estimator.dump(dirpath)
dump(summary, '%s/summary.joblib' % (dirpath), compress=('gzip', 6))
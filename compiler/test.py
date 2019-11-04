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
import numpy as np
from pprint import pprint

test = load('test.joblib')
X_test, y_test = test['instructions'], test['target']

gridBayes = load('31-10-2019/grid_search-MultinomialNVClassifier.joblib')
gridSVM = load('31-10-2019/grid_search-SVMClassifier.joblib')
pipelineBayes = load('31-10-2019/pipeline-MultinomialNVClassifier.joblib')
pipelineSVM = load('31-10-2019/pipeline-SVMClassifier.joblib')

for i in range(0,len(gridBayes.cv_results_['params'])):
    print("[%2d] params: %s  \tscore: %.3f +/- %.3f" %(i,
        gridBayes.cv_results_['params'][i],
        gridBayes.cv_results_['mean_test_score'][i],
        gridBayes.cv_results_['std_test_score'][i] ))

a = np.argmax(gridBayes.cv_results_['mean_test_score'])
bestparams = gridBayes.cv_results_['params'][a]
bestscore = gridBayes.cv_results_['mean_test_score'][a]

print("Best configuration [%d] %r  %.3f" %(a,bestparams,bestscore))
pprint(bestparams)

for i in range(0,len(gridSVM.cv_results_['params'])):
    print("[%2d] params: %s  \tscore: %.3f +/- %.3f" %(i,
        gridSVM.cv_results_['params'][i],
        gridSVM.cv_results_['mean_test_score'][i],
        gridSVM.cv_results_['std_test_score'][i] ))

a = np.argmax(gridSVM.cv_results_['mean_test_score'])
bestparams = gridSVM.cv_results_['params'][a]
bestscore = gridSVM.cv_results_['mean_test_score'][a]

print("Best configuration [%d] %r  %.3f" %(a,bestparams,bestscore))
pprint(bestparams)
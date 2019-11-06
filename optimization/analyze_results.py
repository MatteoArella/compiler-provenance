from joblib import load, dump
import json
import matplotlib.pyplot as plt
import numpy as np
from pandas.io.json import json_normalize
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from pprint import pprint
import pandas as pd

from sklearn.tree import export_graphviz
from subprocess import call
from test import ClassificationTester

data = load('../train_dataset.joblib')
test = load('test.joblib')
X_test, y = test['instructions'], test['target']

tester = ClassificationTester(test_set_path='test.joblib').fit(fitted_path='04-11-2019').predict()
for classifier, report in tester.classification_reports():
    print(classifier)
    print(report)

for classifier, cm in tester.confusion_matrixes():
    print(classifier)
    print(cm)

# plot classes count
counts = data['opt'].value_counts(normalize=True).apply(lambda x: x*100)
ax = counts.plot(kind='bar', rot=0)
plt.ylim(0, 100)
for i in ax.patches:
    ax.text(i.get_x()+.08, i.get_height()/2, '%.2f%%' % i.get_height(), fontsize=16, color='white')

plt.xlabel('Classes')
plt.ylabel('Classes count (%)')
plt.savefig('images/classes-count.pdf', bbox_inches='tight')

# get best n-gram range
print('Best N-gram range: {}'.format(summary.iloc[0]['params']['tfidf__ngram_range']))

# plot decision tree graph (show only first 5 levels)
pipeline = load('04-11-2019/pipeline-RandomForestClassifier.joblib')
tfidf = pipeline['tfidf']
clf = pipeline['clf']

export_graphviz(clf.estimators_[2], out_file='tree-big.dot', 
                feature_names = tfidf.get_feature_names(),
                class_names = ['H', 'L'],
                rounded = True, proportion = False,
                precision = 2, filled = True)

call(['dot', '-Tpdf', 'tree-big.dot', '-o', 'tree-big.pdf'])

# plot important features
'''
pipeline = load('04-11-2019/pipeline-RandomForestClassifier.joblib')
tfidf = pipeline['tfidf']
clf = pipeline['clf']
indices = np.argsort(clf.feature_importances_)[::-1][:10]
features = [tfidf.get_feature_names()[i] for i in indices]

plt.title('Feature Importances')
plt.barh(features, clf.feature_importances_[indices], color='b', align='center')

plt.show()
'''
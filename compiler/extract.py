from os import path
from sys import exit
from time import time
from pprint import pprint
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

from pandas.io.json import json_normalize
from pandas import DataFrame

from joblib import dump, load # model persistance

from preprocessing import AbstractedAsmTransformer
from feature_extraction import AbstractedAsmTfidfVectorizer
'''
def top_tfidf_feats(row, features, top_n=25):
    topn_ids = np.argsort(row)[:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    df.label = 'Much frequent instructions'
    return df

def plot_tfidf_classfeats_h(dfs):
    fig = plt.figure(figsize=(12, 9), facecolor="w")
    x = np.arange(len(dfs[0]))
    for i, df in enumerate(dfs):
        ax = fig.add_subplot(1, len(dfs), i+1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_frame_on(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=14)
        ax.set_title("label = " + str(df.label), fontsize=16)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        ax.bar(x, df.tfidf, align='center', color='#3F5D7D')
        ax.set_yticks(x)
        ax.set_ylim([0, 2])
        yticks = ax.set_xticklabels(df.feature)
        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)
    plt.show()
'''



train = load('train.joblib')
X_train, y_train = train['instructions'], train['target']
test = load('test.joblib')
X_test, y_test = test['instructions'], test['target']

pipeline = Pipeline([
    ('asm_preproc', AbstractedAsmTransformer()),
    ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True)),
    ('clf', svm.SVC(gamma='scale', probability=True))
])

parameters = {
    'tfidf__ngram_range': ((2,2), (3,3), (4,4), (5,5)),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__kernel': ('linear', 'rbf'),
    'clf__C': (0.1, 1, 10, 100),
}

grid_search = GridSearchCV(pipeline, parameters, cv=5,
                               n_jobs=-1)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()
grid_search.fit(X_train, y_train)
print("done in %0.3fs" % (time() - t0))
print()
#dump(grid_search, 'grid_search2.joblib', compress=('gzip', 6))
#dump(pipeline, 'pipeline.joblib', compress=('gzip', 6))

pipeline.fit(X_train, y_train) # IMPORTANT: always fit pipeline before dump or vocabulary will not be dumped
dump(pipeline, 'pipeline.joblib', compress=('gzip', 6))

'''
#grid_search = load('grid_search.joblib')
#besttfidf = load('tfidf.joblib')
for i in range(0,len(grid_search.cv_results_['params'])):
    print("[%2d] params: %s  \tscore: %.3f +/- %.3f" %(i,
        grid_search.cv_results_['params'][i],
        grid_search.cv_results_['mean_test_score'][i],
        grid_search.cv_results_['std_test_score'][i] ))

a = np.argmax(grid_search.cv_results_['mean_test_score'])
bestparams = grid_search.cv_results_['params'][a]
bestscore = grid_search.cv_results_['mean_test_score'][a]

print("Best configuration [%d] %r  %.3f" %(a,bestparams,bestscore))
print("Best kernel: %s" %(bestparams['clf__kernel']))
print("Best C: %s" %(bestparams['clf__C']))

besttfidf = TfidfVectorizer(analyzer='word', ngram_range=bestparams['tfidf__ngram_range'], use_idf=True)
besttfidf.fit(X_train)
#print([besttfidf.get_feature_names()[i] for i in np.argsort(besttfidf.idf_)[:6]])
b = top_tfidf_feats(besttfidf.idf_, besttfidf.get_feature_names(), 5)
plot_tfidf_classfeats_h([b])
'''
#dump(besttfidf, 'besttfidf.joblib')
#X_all = besttfidf.fit_transform(abstracted_asm)
'''
bestclf = svm.SVC(kernel=bestparams['clf__kernel'], C=bestparams['clf__C'], gamma='scale', probability=True)
clf = bestclf.fit(X_train, y_train)

dump(bestclf, 'bestmodel.joblib')

bestclf = load('bestmodel.joblib')
y_pred = bestclf.predict(X_test)

acc = bestclf.score(X_test, y_test)    
print("Accuracy %.3f" %acc)
print(classification_report(y_test, y_pred, labels=None, target_names=['gcc', 'icc', 'clang'], digits=3))
cm = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm)
'''

# Best configuration C:10, kernel: rbf






















'''
images = np.asarray([np.frombuffer(bytes('\n'.join(i['instructions']), 'ascii'), 'u1') for i in data])
y_all = np.asarray([i['compiler'] for i in data])

# solo una immagine grande: invece di max fare una media e poi tagliare al centro se immagine troppo grande o paddare con zero al centro se troppo piccola
images_len = np.mean([len(i) for i in images])
width = int(images_len**0.5) + 1
max_len = width*width
print('Max len = %d, width = %d'%(max_len, width))

count = 0
bar = progressbar.ProgressBar(maxval=images.shape[0], \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()
for image in images:
    image_len = len(image)
    image_width = int(image_len**0.5)
    image_diff = max_len - image_width*image_width
    z = np.zeros(max_len, dtype='u1')
    half = image_len//2 if image_diff > 0 else max_len//2 # z bigger else image bigger
    z[:half] = image[:half]
    z[max_len-half:] = image[half:] # error here
    g = np.reshape(z, (width, width))
    imageio.imwrite('images/reshaped-%d.png' % (count), g)
    bar.update(count)
    if count < 10:
        plt.imshow(g, cmap='gray')
        plt.show()
    count += 1
bar.finish()

#reshaped_images = np.asarray([np.zeros((width, width), dtype='u1') for i in range(images.shape[0])])
#print(reshaped_images[0])
#for i in data:
    #print(i['compiler'])
    #print(np.frombuffer(bytes('\n'.join(data[i]['instructions']), 'ascii'), 'u1').shape)
'''
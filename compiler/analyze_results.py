if __name__ == '__main__':
    from joblib import load, dump
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, confusion_matrix
    from pprint import pprint
    import pandas as pd
    from sys import argv, exit

    from code.common.classification_tester import ClassificationTester, plot_confusion_matrix

    if len(argv) < 2:
        print('usage: {} <path>'.format(argv[0]))
        exit(1)
    path = argv[1]

    test = load('code/compiler/test.joblib')
    X_test, y_test = test['instructions'], test['target']
    '''
    tester = ClassificationTester() \
                        .pipelines_load(pipelines_path='code/compiler/{}'.format(path), \
                                        summary_name='summary.joblib') \
                        .pipelines_predict(X_test)

    best_classifier_name, best_pipeline = tester.pipelines_scoring(y_test, scoring=accuracy_score)
    dump(tester, 'code/compiler/tester.joblib', compress=('gzip', 6))
    df = tester.classification_reports()
    dump(df, 'code/compiler/reports.joblib', compress=('gzip', 6))
    dump(best_pipeline, 'code/compiler/bestmodel.joblib', compress=('gzip', 6))
    '''
    
    tester = load('code/compiler/tester.joblib')
    reports = load('code/compiler/reports.joblib').T

    '''
    # get classification reports
    for (i, val), score in zip(reports.stack().iteritems(), tester.scores_):
        print('{}: {}'.format(i[0], score))
        print(val)
        print()
    '''

    import scikitplot as skplt
    for classifier, predict, proba in zip(tester.classifiers_, tester.predictions_, tester.predictions_proba_):
        skplt.metrics.plot_confusion_matrix(y_test, predict, title='', normalize=True)
        plt.savefig('report/images/comp/{}-cm.pdf'.format(classifier), bbox_inches='tight')
    '''
    #print('Best classificator: {}'.format(best_classifier_name))
    #print('Best pipeline: {}'.format(best_pipeline))
    '''
    
    # plot classes count
    data = load('code/train_dataset.joblib')
    counts = data['compiler'].value_counts(normalize=True).apply(lambda x: x*100)
    ax = counts.plot(kind='bar', rot=0)
    plt.ylim(0, 100)
    for i in ax.patches:
        ax.text(i.get_x()+.08, i.get_height()/2, '%.2f%%' % i.get_height(), fontsize=16, color='white')

    plt.xlabel('Classes')
    plt.ylabel('Classes count (%)')
    plt.savefig('report/images/comp/classes-count.pdf', bbox_inches='tight')

    # get best n-gram range
    summary = load('code/compiler/{}/summary.joblib'.format(path))
    print('Best N-gram range: {}'.format(summary.iloc[0]['params']['tfidf__ngram_range']))

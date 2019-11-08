if __name__ == '__main__':
    from joblib import load, dump
    import json
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
    from pprint import pprint
    import pandas as pd

    from code.common.classification_tester import ClassificationTester

    data = load('code/train_dataset.joblib')
    test = load('code/compiler/test.joblib')
    X_test, y_test = test['instructions'], test['target']
    '''
    tester = ClassificationTester() \
                        .pipelines_load(pipelines_path='code/compiler/31-10-2019', \
                                        summary_name='summary.joblib') \
                        .pipelines_predict(X_test)

    best_classifier_name, best_pipeline = tester.pipelines_scoring(y_test, scoring=accuracy_score)
    dump(tester, 'code/compiler/tester.joblib', compress=('gzip', 6))
    df = tester.classification_reports()
    dump(df, 'code/compiler/reports.joblib', compress=('gzip', 6))
    '''
    
    tester = load('code/optimization/tester.joblib')
    reports = load('code/compiler/reports.joblib').T
    #reports['report'].apply(lambda x: print(x))
    for i, val in reports.stack().iteritems():
        print(i)
        print(val)
        print()

    '''
    import scikitplot as skplt
    for classifier, predict, proba in zip(tester.classifiers_, tester.predictions_, tester.predictions_proba_):
        skplt.metrics.plot_roc(y_test, proba, title='')
        plt.savefig('report/images/opt/curves/{}-roc-curve.pdf'.format(classifier), bbox_inches='tight')
        skplt.metrics.plot_precision_recall(y_test, proba, title='')
        plt.savefig('report/images/opt/curves/{}-prec-recall-curve.pdf'.format(classifier), bbox_inches='tight')
        skplt.metrics.plot_confusion_matrix(y_test, predict, title='', normalize=True)
        plt.savefig('report/images/opt/{}-cm.pdf'.format(classifier), bbox_inches='tight')
    '''
    '''
    #print('Best classificator: {}'.format(best_classifier_name))
    #print('Best pipeline: {}'.format(best_pipeline))
    '''
    '''
    # plot classes count
    counts = data['opt'].value_counts(normalize=True).apply(lambda x: x*100)
    ax = counts.plot(kind='bar', rot=0)
    plt.ylim(0, 100)
    for i in ax.patches:
        ax.text(i.get_x()+.08, i.get_height()/2, '%.2f%%' % i.get_height(), fontsize=16, color='white')

    plt.xlabel('Classes')
    plt.ylabel('Classes count (%)')
    plt.savefig('report/images/opt/classes-count.pdf', bbox_inches='tight')

    # get best n-gram range
    print('Best N-gram range: {}'.format(summary.iloc[0]['params']['tfidf__ngram_range']))

    # plot decision tree graph (show only first 5 levels)
    pipeline = load('code/optimization/07-11-2019/pipeline-RandomForestClassifier.joblib')
    tfidf = pipeline['tfidf']
    clf = pipeline['clf']

    export_graphviz(clf.estimators_[2], out_file='tree-big.dot', 
                    feature_names = tfidf.get_feature_names(),
                    class_names = ['H', 'L'],
                    rounded = True, proportion = False,
                    precision = 2, filled = True)

    call(['dot', '-Tpdf', 'tree-big.dot', '-o', 'tree-big.pdf'])
    '''
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

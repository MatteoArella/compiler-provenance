if __name__ == '__main__':
    from joblib import load, dump
    import matplotlib.pyplot as plt
    import scikitplot as skplt
    from sklearn.metrics import accuracy_score
    from sys import argv, exit

    from code.common.classification_tester import ClassificationTester

    if len(argv) < 2:
        print('usage: {} <path>'.format(argv[0]))
        exit(1)
    path = argv[1]

    test = load('code/compiler/test.joblib')
    X_test, y_test = test['instructions'], test['target']

    tester = ClassificationTester() \
                        .pipelines_load(pipelines_path='code/compiler/{}'.format(path), \
                                        summary_name='summary.joblib') \
                        .pipelines_predict(X_test)

    best_classifier_name, best_pipeline = tester.pipelines_scoring(y_test, scoring=accuracy_score)
    dump(tester, 'code/compiler/tester.joblib', compress=('gzip', 6))
    reports = tester.classification_reports()
    dump(reports, 'code/compiler/reports.joblib', compress=('gzip', 6))
    dump(best_pipeline, 'code/compiler/bestmodel.joblib', compress=('gzip', 6))

    # get classification reports
    for (i, val), score in zip(reports.stack().iteritems(), tester.scores_):
        print('{}: {}'.format(i[0], score))
        print(val)
        print()
    
    for classifier, predict, proba in zip(tester.classifiers_, tester.predictions_, tester.predictions_proba_):
        skplt.metrics.plot_roc(y_test, proba, title='')
        plt.savefig('report/images/comp/curves/{}-roc-curve.pdf'.format(classifier), bbox_inches='tight')
        skplt.metrics.plot_precision_recall(y_test, proba, title='')
        plt.savefig('report/images/comp/curves/{}-prec-recall-curve.pdf'.format(classifier), bbox_inches='tight')
        skplt.metrics.plot_confusion_matrix(y_test, predict, title='', normalize=True)
        plt.savefig('report/images/comp/{}-cm.pdf'.format(classifier), bbox_inches='tight')

    # get best n-gram range
    summary = load('code/compiler/{}/summary.joblib'.format(path))
    print('Best N-gram range: {}'.format(summary.iloc[0]['params']['tfidf__ngram_range']))

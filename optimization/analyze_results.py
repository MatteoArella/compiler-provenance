if __name__ == '__main__':
    from joblib import load, dump
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import f1_score

    from sklearn.tree import export_graphviz
    from subprocess import call
    import scikitplot as skplt
    from code.common.classification_tester import ClassificationTester
    from sys import argv, exit

    if len(argv) < 2:
        print('usage: {} <path>'.format(argv[0]))
        exit(1)
    path = argv[1]

    test = load('code/optimization/test.joblib')
    X_test, y_test = test['instructions'], test['target']

    tester = ClassificationTester() \
                        .pipelines_load(pipelines_path='code/optimization/{}'.format(path), \
                                        summary_name='summary.joblib') \
                        .pipelines_predict(X_test)

    best_classifier_name, best_pipeline = tester.pipelines_scoring(y_test, scoring=f1_score, average='weighted')
    dump(tester, 'code/optimization/tester.joblib', compress=('gzip', 6))
    reports = tester.classification_reports()
    dump(reports, 'code/optimization/reports.joblib', compress=('gzip', 6))
    dump(best_pipeline, 'code/optimization/bestmodel.joblib', compress=('gzip', 6))

    for (i, val), score in zip(reports.stack().iteritems(), tester.scores_):
        print('{}: {}'.format(i[0], score))
        print(val)
        print()

    # plot ROC curve, precision-recall curve, confusion matrix
    for classifier, predict, proba in zip(tester.classifiers_, tester.predictions_, tester.predictions_proba_):
        skplt.metrics.plot_roc(y_test, proba, title='')
        plt.savefig('report/images/opt/curves/{}-roc-curve.pdf'.format(classifier), bbox_inches='tight')
        skplt.metrics.plot_precision_recall(y_test, proba, title='')
        plt.savefig('report/images/opt/curves/{}-prec-recall-curve.pdf'.format(classifier), bbox_inches='tight')
        skplt.metrics.plot_confusion_matrix(y_test, predict, title='', normalize=True)
        plt.savefig('report/images/opt/{}-cm.pdf'.format(classifier), bbox_inches='tight')

    # plot classes count
    data = load('code/train_dataset.joblib')
    counts = data['opt'].value_counts(normalize=True).apply(lambda x: x*100)
    ax = counts.plot(kind='bar', rot=0)
    plt.ylim(0, 100)
    for i in ax.patches:
        ax.text(i.get_x()+.08, i.get_height()/2, '%.2f%%' % i.get_height(), fontsize=16, color='white')
    plt.xlabel('Classes')
    plt.ylabel('Classes count (%)')
    plt.savefig('report/images/opt/classes-count.pdf', bbox_inches='tight')

    # get best n-gram range
    summary = load('code/optimization/{}/summary.joblib'.format(path))
    print('Best N-gram range: {}'.format(summary.iloc[0]['params']['tfidf__ngram_range']))

    # plot decision tree graph (show only until depth 4)
    pipeline = load('code/optimization/{}/pipeline-RandomForestClassifier.joblib'.format(path))
    tfidf = pipeline['tfidf']
    clf = pipeline['clf']

    export_graphviz(clf.estimators_[2], out_file='decision-tree.dot', 
                    feature_names = tfidf.get_feature_names(),
                    class_names = ['H', 'L'], max_depth=4,
                    rounded = True, proportion = False,
                    precision = 2, filled = True)

    call(['dot', '-Tpdf', 'decision-tree.dot', '-o', 'report/images/opt/decision-tree.pdf'])

    # plot important features
    pipeline = load('code/optimization/{}/pipeline-RandomForestClassifier.joblib'.format(path))
    tfidf = pipeline['tfidf']
    clf = pipeline['clf']
    indices = np.argsort(clf.feature_importances_)[::-1][:10]
    features = [tfidf.get_feature_names()[i] for i in indices]

    plt.barh(features, clf.feature_importances_[indices], color='b', align='center')
    plt.savefig('report/images/opt/feature-importances.pdf', bbox_inches='tight')

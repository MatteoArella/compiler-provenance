if __name__ == '__main__'
    from code.common.hyperparameter import EstimatorSelectionHelper
    from code.common.preprocessing import AbstractedAsmTransformer
    from joblib import load, dump
    from imblearn.pipeline import Pipeline as ImblearnPipeline
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import KFold
    from datetime import date
    import os

    train = load('code/optimization/train.joblib')
    test = load('code/optimization/test.joblib')

    X_train, y_train = train['instructions'], train['target']
    X_test, y_test = test['instructions'], test['target']

    random_state=25

    asmTransformer = AbstractedAsmTransformer()

    pipelines = {
        'SVMClassifier': {   'pipeline': Pipeline([
                                        ('asm_preproc', asmTransformer),
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
                                        ('asm_preproc', asmTransformer),
                                        ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True)),
                                        ('clf', MultinomialNB())
                                    ]),
                                    'params': {
                                        'tfidf__ngram_range': [(1,2), (1,3), (1,4), (1,5),
                                                                (2,2), (3,3), (4,4), (5,5)],
                                    }
        }
    }

    dirpath = date.today().strftime('code/compiler/%d-%m-%Y')

    estimator = EstimatorSelectionHelper(pipelines)
    estimator.fit(X_train, y_train, cv=5, n_jobs=-1, iid=False, verbose=10)
    summary = estimator.score_summary()
    print(summary)

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    estimator.dump(dirpath)
    dump(summary, '%s/summary.joblib' % (dirpath), compress=('gzip', 6))
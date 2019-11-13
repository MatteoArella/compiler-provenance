if __name__ == '__main__':
    from compiler-provenance.common.hyperparameter import EstimatorSelectionHelper
    from compiler-provenance.common.preprocessing import AbstractedAsmTransformer
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

    train = load('compiler-provenance/optimization/train.joblib')
    test = load('compiler-provenance/optimization/test.joblib')

    X_train, y_train = train['instructions'], train['target']
    X_test, y_test = test['instructions'], test['target']

    random_state=25

    asmTransformer = AbstractedAsmTransformer()
    kf = KFold(n_splits=5, random_state=random_state, shuffle=False)

    pipelines = {
        'ROSLogisticRegression': { #over-sampling
            'pipeline': ImblearnPipeline([
                            ('asm_prep', asmTransformer),
                            ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True)),
                            ('sampler', RandomOverSampler(random_state=random_state)),
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
                            ('sampler', SMOTE(random_state=random_state)),
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
                            ('sampler', RandomUnderSampler(random_state=random_state)),
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
        },
        'BalancedSVMClassifier': {
            'pipeline': ImblearnPipeline([
                            ('asm_prep', asmTransformer),
                            ('tfidf', TfidfVectorizer(analyzer='word', use_idf=True)),
                            ('clf', SVC(gamma='scale', class_weight='balanced', probability=True))
                        ]),
            'hyperparams': {
                        'tfidf__ngram_range': [(1,2), (1,3), (1,4), (1,5),
                                               (2,2), (3,3), (4,4), (5,5)],
                        'clf__kernel': ['linear', 'poly', 'rbf'],
                        'clf__C': [0.01, 0.1, 1, 10, 100],
            }
        }
    }

    dirpath = date.today().strftime('compiler-provenance/optimization/%d-%m-%Y')

    estimator = EstimatorSelectionHelper(pipelines)
    estimator.fit(X_train, y_train, scoring='f1_weighted', cv=kf, n_jobs=1, verbose=10)
    summary = estimator.score_summary()
    print(summary)

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    estimator.dump(dirpath)
    dump(summary, '%s/summary.joblib' % (dirpath), compress=('gzip', 6))
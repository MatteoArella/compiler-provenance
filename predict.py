if __name__ == '__main__':
    from joblib import load, dump
    import pandas as pd

    blind_set = pd.read_json('compiler_provenance/test_dataset_blind.jsonl', lines=True)
    X_blind = blind_set['instructions']

    best_comp_model = load('compiler_provenance/compiler/bestmodel.joblib')
    best_opt_model = load('compiler_provenance/optimization/bestmodel.joblib')

    y_comp = best_comp_model.predict(X_blind)
    y_opt = best_opt_model.predict(X_blind)

    df = pd.DataFrame(data={'compiler': y_comp, 'opt': y_opt}, columns=['compiler', 'opt'], index=None)
    df.to_csv('compiler_provenance/predictions.csv', index=False, header=False, sep=',')
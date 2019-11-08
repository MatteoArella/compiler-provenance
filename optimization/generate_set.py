if __name__ == '__main__':
    from code.common.split_data import DataShuffleSplit
    DataShuffleSplit(dataset_path='code/train_dataset.jsonl',
            train_output='code/optimization/train.joblib',
            test_output='code/optimization/test.joblib',
            target_label='opt',
            train_description='Train dataset for compiler optimization level provenance problem',
            test_description='Test dataset for compiler optimization level provenance problem',
            force=False,
            random_state=25)
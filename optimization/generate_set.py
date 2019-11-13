if __name__ == '__main__':
    from compiler_provenance.common.split_data import DataShuffleSplit
    DataShuffleSplit(dataset_path='compiler_provenance/train_dataset.jsonl',
            train_output='compiler_provenance/optimization/train.joblib',
            test_output='compiler_provenance/optimization/test.joblib',
            target_label='opt',
            train_description='Train dataset for compiler optimization level provenance problem',
            test_description='Test dataset for compiler optimization level provenance problem',
            force=False,
            random_state=25)
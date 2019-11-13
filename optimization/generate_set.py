if __name__ == '__main__':
    from compiler-provenance.common.split_data import DataShuffleSplit
    DataShuffleSplit(dataset_path='compiler-provenance/train_dataset.jsonl',
            train_output='compiler-provenance/optimization/train.joblib',
            test_output='compiler-provenance/optimization/test.joblib',
            target_label='opt',
            train_description='Train dataset for compiler optimization level provenance problem',
            test_description='Test dataset for compiler optimization level provenance problem',
            force=False,
            random_state=25)
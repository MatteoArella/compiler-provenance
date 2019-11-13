if __name__ == '__main__':
    from compiler_provenance.common.split_data import DataShuffleSplit
    DataShuffleSplit(dataset_path='compiler_provenance/train_dataset.jsonl',
            train_output='compiler_provenance/compiler/train.joblib',
            test_output='compiler_provenance/compiler/test.joblib',
            target_label='compiler',
            train_description='Train dataset for compiler provenance problem',
            test_description='Test dataset for compiler provenance problem',
            force=False,
            random_state=25)

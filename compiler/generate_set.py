if __name__ == '__main__':
    from compiler-provenance.common.split_data import DataShuffleSplit
    DataShuffleSplit(dataset_path='compiler-provenance/train_dataset.jsonl',
            train_output='compiler-provenance/compiler/train.joblib',
            test_output='compiler-provenance/compiler/test.joblib',
            target_label='compiler',
            train_description='Train dataset for compiler provenance problem',
            test_description='Test dataset for compiler provenance problem',
            force=False,
            random_state=25)

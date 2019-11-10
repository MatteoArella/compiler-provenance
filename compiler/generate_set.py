if __name__ == '__main__':
    from code.common.split_data import DataShuffleSplit
    DataShuffleSplit(dataset_path='code/train_dataset.jsonl',
            train_output='code/compiler/train.joblib',
            test_output='code/compiler/test.joblib',
            target_label='compiler',
            train_description='Train dataset for compiler provenance problem',
            test_description='Test dataset for compiler provenance problem',
            force=False,
            random_state=25)

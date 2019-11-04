import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from pandas.io.json import json_normalize
from pandas import DataFrame

from joblib import dump # model persistance

class DataSplit:
    def __init__(self, test_size=0.2, **kwargs):
        self.input_file = '../../train_dataset.jsonl'
        self.train_output = kwargs.get('train_output', 'train.joblib')
        self.test_output = kwargs.get('test_output', 'test.joblib')
        self.force = kwargs.get('force', False)
        self.test_size = test_size
        self.random_state = kwargs.get('random_state', 15)

        with open(self.input_file, 'r', encoding='utf-8') as jsonl_file:
            data = np.asarray([json.loads(jline.strip()) for jline in jsonl_file])

        data = json_normalize(data)
        all_instructions = data['instructions']
        y_all = data['opt']

        X_all, y_all = shuffle(all_instructions, y_all)

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                test_size=self.test_size, random_state=self.random_state)

        self.train = DataFrame(data=X_train)
        self.train['target'] = y_train
        self.train['description'] = 'Train dataset for compiler optimization level provenance problem'
        self.test = DataFrame(data=X_test)
        self.test['target'] = y_test
        self.train['description'] = 'Test dataset for compiler optimization level provenance problem'

        dump(self.train, self.train_output, compress=('gzip', 6))
        dump(self.test, self.test_output, compress=('gzip', 6))

if __name__ == '__main__':
    DataSplit(random_state=25)
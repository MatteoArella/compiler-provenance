import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from pandas.io.json import json_normalize
from pandas import DataFrame

from os import path
from joblib import dump

class DataShuffleSplit:
    def __init__(self, dataset_path, train_output, test_output, target_label, test_size=0.2, **kwargs):
        self.input_file = dataset_path
        self.train_output = train_output
        self.test_output = test_output
        self.force = kwargs.get('force', False)
        self.test_size = test_size
        self.random_state = kwargs.get('random_state', 15)
        self.compress = kwargs.get('compress', ('gzip', 6))

        if not self.force and (path.exists(self.train_output) or path.exists(self.test_output)):
            return

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
        self.train['description'] = kwargs.get('train_description', '')
        self.test = DataFrame(data=X_test)
        self.test['target'] = y_test
        self.test['description'] = kwargs.get('test_description', '')

        dump(self.train, self.train_output, compress=self.compress)
        dump(self.test, self.test_output, compress=self.compress)

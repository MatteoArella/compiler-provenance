import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from pandas.io.json import json_normalize
from pandas import DataFrame

from joblib import dump # model persistance
from os import path
from sys import argv, exit
import getopt
import logging

def usage(prog_name):
    logging.warning('%s' % prog_name)

def main(prog_args):
    try:
        opts, args = getopt.getopt(prog_args[1:], 'hfi:vs', ['help', 'force', 'input=', 'test-output=', 'train-output=', 'verbose', 'size=']) # size of test/train sets split (percentage)
    except getopt.GetoptError as err:
        logging.error(str(err))
        usage(prog_args[0])
        exit(1)

    input_file = '../../train_dataset.jsonl'
    train_output = 'train.joblib'
    test_output = 'test.joblib'
    verbose = False
    force = False
    test_size = 0.2

    for o, a in opts:
        if o == '-v':
            verbose = True
        elif o in ('-h', '--help'):
            usage(prog_args[0])
            exit(0)
        elif o in ('-f', '--force'):
            force = True
        elif o in ('-i', '--input'):
            input_file = a
        elif o == '--test-output':
            test_output = a
        elif o == '--train-output':
            train_output = a
        elif o in ('-s', '--size'):
            test_size = float(a)
        else:
            assert False, "unhandled option"

    if verbose:
        logging.getLogger().setLevel(level=logging.INFO)

    if not force and path.exists(train_output) and path.exists(test_output):
        logging.info('Data are already splitted')
        exit(0)

    logging.info('Extracting data from file "%s"' % input_file)
    with open(input_file, 'r', encoding='utf-8') as jsonl_file:
        data = np.asarray([json.loads(jline.strip()) for jline in jsonl_file])

    logging.info('Normalizing json')
    df = json_normalize(data)

    all_instructions = df['instructions']
    y_all = df['compiler']

    logging.info('Shuffling data')
    X_all, y_all = shuffle(all_instructions, y_all)

    logging.info('Splitting data into train set (%d%%) and test set (%d%%)' % ((1.0 - test_size)*100, test_size*100))
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
            test_size=test_size, random_state=15)

    train = DataFrame(data=X_train)
    train['target'] = y_train
    train['description'] = 'Train dataset for compiler provenance problem'
    test = DataFrame(data=X_test)
    test['target'] = y_test
    test['description'] = 'Test dataset for compiler provenance problem'

    logging.info('Saving train set and test set into files "%s" and "%s"' % (train_output, test_output))
    dump(train, train_output, compress=('gzip', 6))
    dump(test, test_output, compress=('gzip', 6))
    logging.info('Done.')

if __name__ == "__main__":
    logging.basicConfig(format='%(message)s')
    main(argv)
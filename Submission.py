import pandas as pd
import numpy as np


def create_submission_file(ids, pred, filename='submission1.csv'):
    '''Create a .csv submission file to be uploaded to Kaggle'''
    sub_df = pd.DataFrame(dict(ID=ids, target=pred))
    sub_df.to_csv(filename, header=True, index=False)
    print('Submission file created succesfully.')


def merge_leaky_and_ML_sub(leaky_sub, ML_sub, filename='submission1.csv'):
    '''Merge a leaky and ML submission'''
    sub_leak = pd.read_csv(leaky_sub)
    ML_sub = pd.read_csv(ML_sub)
    sub_leak[sub_leak['target'] == 0] = ML_sub[sub_leak['target'] == 0]
    sub_leak.to_csv(filename, header=True, index=False)
    print('Submission file created succesfully.')


def blend_submissions(list_of_subs, filename='submission1.csv'):
    '''Average submissions'''
    for i, path in enumerate(list_of_subs):
        if not i:
            sub = pd.read_csv(path)
        else:
            sub['target'] += pd.read_csv(path)['target']
    sub['target'] /= len(list_of_subs)
    sub.to_csv(filename, header=True, index=False)
    print('Submission file created succesfully.')


def load_submissions_as_data_for_ensembling(list_of_subs, train_y=None):
    '''Load submissions and prepare for training a stack model'''
    for i, path in enumerate(list_of_subs):
        if not i:
            X = np.expand_dims(pd.read_csv(path).target, axis=1)
        else:
            tmp = np.expand_dims(pd.read_csv(path).target, axis=1)
            X = np.concatenate([X, tmp], axis=1)

    if train_y is not None:
        if isinstance(train_y, np.ndarray):
            y = train_y
        else:
            y = np.load(train_y)
        return X, y

    else:
        return X

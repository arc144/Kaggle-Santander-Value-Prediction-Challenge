import pandas as pd


def create_submission_file(ids, pred, filename='submission1.csv'):
    '''Create a .csv submission file to be uploaded to Kaggle'''
    sub_df = pd.DataFrame(dict(ID=ids, target=pred))
    sub_df.to_csv(filename, header=True, index=False)
    print('Submission file created succesfully.')


def merge_leaky_and_ML_sub(leaky_sub, ML_sub, filename='submission1.csv'):
    sub_leak = pd.read_csv(leaky_sub)
    ML_sub = pd.read_csv(ML_sub)
    sub_leak[sub_leak['target'] == 0] = ML_sub[sub_leak['target'] == 0]
    sub_leak.to_csv(filename, header=True, index=False)
    print('Submission file created succesfully.')

import pandas as pd


def create_submission_file(ids, pred, filename='submission1.csv'):
    '''Create a .csv submission file to be uploaded to Kaggle'''
    sub_df = pd.DataFrame(dict(ID=ids, target=pred))
    sub_df.to_csv(filename, header=True, index=False)
    print('Submission file created succesfully.')

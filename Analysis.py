import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

########################################################################
# ###################### Single dataset functions ######################
########################################################################


def count_constant_features(df, name='dataset'):
    '''Count number of constant feature over all rows'''
    count = 0
    for col in df.columns:
        if col != 'ID' and col != 'target':
            if df[col].std() == 0:
                count += 1

    print('{} constant features found in {}'.format(count, name))


def check_for_missing_values(df, name='dataset'):
    '''Check df for missing values in features'''
    nulls = df.isnull().sum()
    cols_with_null = df.columns[nulls != 0]
    count = cols_with_null.size
    print('{} missing values found in {}'.format(count, name))
    print('Missing feature columns are: {}'.format(cols_with_null))


########################################################################
# #############  ###### Multiple dataset functions #####################
########################################################################

def plot_feature_stats(train_df, test_df):
    '''Plot mean and std for each feature in dataset'''
    target = train_df.pop('target')

    sns.set()
    mn_train = train_df.mean()
    std_train = train_df.std()

    mn_test = test_df.mean()
    std_test = test_df.std()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax = sns.distplot(mn_train, kde=False, norm_hist=False, ax=ax)
    sns.distplot(mn_test, kde=False, norm_hist=False, ax=ax)
    ax.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_title('Distribution of the mean value of train/test features.')
    ax.set_xlabel(r'Mean value ($\mu$)')
    ax.set_ylabel('Number of features')
    ax.legend(['train', 'test'])

    ax = axes[1]
    sns.distplot(std_train, kde=False, norm_hist=False, ax=ax)
    sns.distplot(std_test, kde=False, norm_hist=False, ax=ax)
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_title('Distribution of std value of train/test features.')
    ax.set_xlabel(r'Standard Deviation ($\sigma^2$)')
    ax.set_ylabel('Number of features')
    ax.legend(['Train', 'Test'])

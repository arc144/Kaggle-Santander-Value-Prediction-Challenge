import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD, SparsePCA, FactorAnalysis
from sklearn.random_projection import SparseRandomProjection
from scipy.stats import ks_2samp


def load_df_from_path(path):
    df = pd.read_csv(path, index_col=0)
    return df


def compute_row_aggregates(df):
    '''Add series of aggreagates to dataset rowise'''
    agg_df = pd.DataFrame(index=df.index)
    for index, row in df.iterrows():
        non_zero_values = row.iloc[row.nonzero()]

        agg_df.at[index, 'non_zero_mean'] = non_zero_values.mean()
        agg_df.at[index, 'non_zero_max'] = non_zero_values.max()
        agg_df.at[index, 'non_zero_min'] = non_zero_values.min()
        agg_df.at[index, 'non_zero_std'] = np.std(non_zero_values.values)
        agg_df.at[index, 'non_zero_sum'] = non_zero_values.sum()
        agg_df.at[index, 'non_zero_count'] = non_zero_values.count()
        agg_df.at[index, 'non_zero_fraction'] = \
            non_zero_values.count() / row.count()
    return agg_df


class KaggleDataset():
    '''Class used to load Kaggle's official datasets'''

    def __init__(self, train_path, test_path=None, join_dfs=False,
                 verbose=True):
        self.train_path = train_path
        self.test_path = test_path
        self.verbose = verbose
        # Default settings, to be overidden if required
        self.normalize = False
        self.reduce_dim_nb = 0
        self.reduce_dim_method = 'svd'

        self.scaler = MaxAbsScaler(copy=False)

        # Load datasets
        self.train_df = load_df_from_path(self.train_path)
        if test_path is not None:
            self.test_df = load_df_from_path(self.test_path)
        else:
            self.test_df = None

        # If joint_dfs, all dfs are joint in a single joint_df
        if join_dfs:
            self.joint_df = pd.concat([self.train_df, self.test_df], axis=0)

        # Pre-compute aggregates
        self.train_agg = compute_row_aggregates(
            self.train_df.drop(["target"], axis=1))
        self.test_agg = compute_row_aggregates(self.test_df)

    def get_train_data(self, logloss=True, normalize=False, n_components=None,
                       reduce_dim_nb=0, use_aggregates=True, reduce_dim_method='svd'):
        '''Convert train_df to train array'''
        # Save settings to proccess test data later on
        self.normalize = normalize
        self.reduce_dim_nb = reduce_dim_nb
        self.reduce_dim_method = reduce_dim_method
        self.use_aggregates = use_aggregates
        # Get trainning data and labels from dataframe
        x = self.train_df.drop(["target"], axis=1).values
        if logloss:
            y = np.log1p(self.train_df["target"].values)
        else:
            y = self.train_df["target"].values

        # Preprocess if required
        if reduce_dim_nb or n_components is not None:
            x = self.reduce_dimensionality(x,
                                           n_components=n_components,
                                           red_num=reduce_dim_nb,
                                           method=reduce_dim_method,
                                           verbose=self.verbose)
        if normalize:
            x = self.normalize_data(x, fit=True, verbose=self.verbose)

        # Compute aggregates if required
        if use_aggregates:
            x = np.concatenate([x, self.train_agg.values], axis=-1)
        return x, y

    def get_test_data(self):
        '''Convert test_df to array using the same preprocess
         as trainning data'''
        x = self.test_df.values
        # Preprocess if required
        if self.reduce_dim_nb:
            x = self.reduce_dimensionality(x, self.reduce_dim_nb,
                                           method=self.reduce_dim_method,
                                           fit=False,
                                           verbose=self.verbose)
        if self.normalize:
            x = self.normalize_data(x, fit=False, verbose=self.verbose)

        # Compute aggregates if required
        if self.use_aggregates:
            x = np.concatenate([x, self.test_agg.values], axis=-1)

        return x

    def remove_constant_features(self, verbose=True):
        '''Remove features that are constant for all train set entries'''
        col_list = []
        count = 0
        for col in self.train_df.columns:
            if col != 'ID' and col != 'target':
                if self.train_df[col].std() == 0:
                    col_list.append(col)
                    count += 1

        # Remove feature in both train and test sets
        self.train_df.drop(col_list, axis=1, inplace=True)
        if self.test_df is not None:
            self.test_df.drop(col_list, axis=1, inplace=True)
        if verbose:
            print('{} constant features removed from datasets'.format(count))

    def remove_different_distribution_features(self,
                                               pvalue_threshold=0.01,
                                               stat_threshold=0.2,
                                               verbose=True):
        '''Remove features that have different distribuition in
         train and test sets'''
        diff_cols = []
        for col in self.train_df.drop(["target"], axis=1).columns:
            statistic, pvalue = ks_2samp(
                self.train_df[col].values, self.test_df[col].values)
            if pvalue <= pvalue_threshold and \
                    np.abs(statistic) > stat_threshold:
                diff_cols.append(col)

        for col in diff_cols:
            if col in self.train_df.columns:
                self.train_df.drop(col, axis=1, inplace=True)
                self.test_df.drop(col, axis=1, inplace=True)
        if verbose:
            print('{} features removed.'.format(len(diff_cols)))

    def normalize_data(self, x, fit=True, verbose=True):
        '''Normalize data taking sparsity into account'''
        if fit:
            self.scaler.fit(x)
        x = self.scaler.transform(x)
        if verbose:
            print('Data normalized.')
        return x

    def reduce_dimensionality(self, x, n_components=None, red_num=None,
                              method='svd', fit=True, verbose=True):
        '''Reduce #red_num of features from the dataset'''
        assert method in ['svd', 'srp', 'fa']
        if n_components is None:
            n_components = x.shape[0] - red_num
        elif n_components == -1:
            n_components = 'auto'
            red_num = 'Unkown'
        else:
            red_num = x.shape[0] - n_components

            # When reducing test data fit must be False
        if fit:
            if method == 'svd':
                self.reductor = TruncatedSVD(n_components=n_components)
            elif method == 'srp':
                self.reductor = SparseRandomProjection(
                    n_components=n_components)
            elif method == 'fa':
                self.reductor = FactorAnalysis(n_components=n_components)

            self.reductor.fit(x)

        x = self.reductor.transform(x)

        if verbose:
            print(red_num, ' less important features removed.')
            # print('Importance ratios are: ',
            #       self.reductor.explained_variance_ratio_)
            print('Data new shape: ', x.shape)
        return x

if __name__ == '__main__':
    train_path = './train.csv'
    test_path = './test.csv'

    dataset = KaggleDataset(train_path, test_path=test_path)
    print(dataset.train_df.describe(), dataset.test_df.describe())

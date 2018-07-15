import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD, SparsePCA, FactorAnalysis
from sklearn.random_projection import SparseRandomProjection
from scipy.stats import ks_2samp
from scipy.stats import kurtosis, skew, mode
from Models import LightGBM
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


def geo_mean_overflow(iterable):
    a = np.log(iterable)
    return np.exp(a.sum() / len(a))


def load_df_from_path(path):
    df = pd.read_csv(path, index_col=0)
    return df


def compute_row_aggregates(df, prefix=''):
    '''Add series of aggreagates to dataset rowise'''
    agg_df = pd.DataFrame(index=df.index)
    for index, row in df.iterrows():
        non_zero_values = row.iloc[row.nonzero()]
        if non_zero_values.empty:
            continue

        non_zero_values = non_zero_values.values
        agg_df.at[index, '{}_non_zero_mean'.format(
            prefix)] = non_zero_values.mean()
        agg_df.at[index, '{}_non_zero_max'.format(
            prefix)] = non_zero_values.max()
        agg_df.at[index, '{}_non_zero_min'.format(
            prefix)] = non_zero_values.min()
        agg_df.at[index, '{}_non_zero_std'.format(
            prefix)] = np.std(non_zero_values)
        agg_df.at[index, '{}_non_zero_sum'.format(
            prefix)] = non_zero_values.sum()
        agg_df.at[index, '{}non_zero_median'.format(prefix)] = \
            np.median(non_zero_values)
        agg_df.at[index, '{}non_zero_q1'.format(prefix)] = \
            np.percentile(non_zero_values, q=25)
        agg_df.at[index, '{}non_zero_q3'.format(prefix)] = \
            np.percentile(non_zero_values, q=75)
        agg_df.at[index, '{}_non_zero_gmean'.format(prefix)] = \
            geo_mean_overflow(non_zero_values)

        mode_ = mode(np.around(non_zero_values, decimals=4))
        agg_df.at[index, '{}_non_zero_mode'.format(prefix)] = mode_[
            0] if mode_[1] > 1 else 0
        # agg_df.at[index, '{}_non_zero_mode_count'.format(prefix)] = mode_[1]
        # agg_df.at[index, '{}_non_zero_skewness'.format(prefix)] = \
        #     skew(non_zero_values)
        # agg_df.at[index, '{}_non_zero_kurtosis'.format(prefix)] = \
        #     kurtosis(non_zero_values)

        # LOG AGGREGATES
        agg_df.at[index, '{}non_zero_log_mean'.format(prefix)] = \
            np.log1p(non_zero_values).mean()
        agg_df.at[index, '{}non_zero_log_max'.format(prefix)] = \
            np.log1p(non_zero_values).max()
        agg_df.at[index, '{}non_zero_log_min'.format(prefix)] = \
            np.log1p(non_zero_values).min()
        agg_df.at[index, '{}non_zero_log_std'.format(prefix)] = \
            np.log1p(np.std(non_zero_values))
        agg_df.at[index, '{}non_zero_log_sum'.format(prefix)] = \
            np.log1p(non_zero_values).sum()
        agg_df.at[index, '{}non_zero_log_median'.format(prefix)] = \
            np.median(np.log1p(non_zero_values))
        agg_df.at[index, '{}non_zero_log_q1'.format(prefix)] = \
            np.percentile(np.log1p(non_zero_values), q=25)
        agg_df.at[index, '{}non_zero_log_q3'.format(prefix)] = \
            np.percentile(np.log1p(non_zero_values), q=75)
        agg_df.at[index, '{}non_zero_log_gmean'.format(prefix)] = \
            geo_mean_overflow(np.log1p(non_zero_values))
        # agg_df.at[index, '{}non_zero_log_skewness'.format(prefix)] = \
        #     skew(np.log1p(non_zero_values))
        # agg_df.at[index, '{}non_zero_log_kurtosis'.format(prefix)] = \
        #     kurtosis(np.log1p(non_zero_values))

        agg_df.at[index, '{}_non_zero_count'.format(
            prefix)] = np.count_nonzero(~np.isnan(non_zero_values))
        agg_df.at[index, '{}_non_zero_fraction'.format(prefix)] = \
            np.count_nonzero(~np.isnan(non_zero_values)) / \
            np.count_nonzero(~np.isnan(row))
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

        # Create aggregates dfs
        self.train_agg = pd.DataFrame(index=self.train_df.index)
        if test_path is not None:
            self.test_agg = pd.DataFrame(index=self.test_df.index)

    def get_train_data(self, logloss=True, normalize=False, n_components=None,
                       reduce_dim_nb=0, use_aggregates=True,
                       reduce_dim_method='svd'):
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
        if normalize:
            x = self.normalize_data(x, fit=True, verbose=self.verbose)

        if reduce_dim_nb or n_components is not None:
            x = self.reduce_dimensionality('train',
                                           n_components=n_components,
                                           red_num=reduce_dim_nb,
                                           method=reduce_dim_method,
                                           verbose=self.verbose)

        # Compute aggregates if required
        if use_aggregates:
            x = np.concatenate([x, self.train_agg.values], axis=-1)
        return x, y

    def get_test_data(self):
        '''Convert test_df to array using the same preprocess
         as trainning data'''
        x = self.test_df.values
        # Preprocess if required
        if self.normalize:
            x = self.normalize_data(x, fit=False, verbose=self.verbose)

        if self.reduce_dim_nb:
            x = self.reduce_dimensionality('test', self.reduce_dim_nb,
                                           method=self.reduce_dim_method,
                                           fit=False,
                                           verbose=self.verbose)

        # Compute aggregates if required
        if self.use_aggregates:
            x = np.concatenate([x, self.test_agg.values], axis=-1)

        return x

    def get_aggregates_as_data(self, dataset, logloss=True):
        '''Get aggregates as np data'''
        if dataset == 'train':
            x = self.train_agg.values
            if logloss:
                y = np.log1p(self.train_df["target"].values)
            else:
                y = self.train_df["target"].values
            return x, y

        elif dataset == 'test':
            x = self.test_agg.values
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

    def remove_duplicated_features(self, verbose=True):
        '''Remove features that have duplicated values'''
        colsToRemove = []
        columns = self.train_df.columns
        for i in range(len(columns) - 1):
            v = self.train_df[columns[i]].values
            for j in range(i + 1, len(columns)):
                if np.array_equal(v, self.train_df[columns[j]].values):
                    colsToRemove.append(columns[j])
        # Remove feature in both train and test sets
        self.train_df.drop(colsToRemove, axis=1, inplace=True)
        if self.test_df is not None:
            self.test_df.drop(colsToRemove, axis=1, inplace=True)
        if verbose:
            print('{} duplicated features removed from datasets'.format(
                len(colsToRemove)))

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

    def keep_only_selected_features(self, dataset='both', return_only=False):
        '''Remove all columns except for the hand picked ones'''
        features = [
            'f190486d6', 'c47340d97', 'eeb9cd3aa', '66ace2992', 'e176a204a',
            '491b9ee45', '1db387535', 'c5a231d81', '0572565c2', '024c577b9',
            '15ace8c9f', '23310aa6f', '9fd594eec', '58e2e02e6', '91f701ba2',
            'adb64ff71', '2ec5b290f', '703885424', '26fc93eb7', '6619d81fc',
            '0ff32eb98', '70feb1494', '58e056e12', '1931ccfdd', '1702b5bf0',
            '58232a6fb', '963a49cdc', 'fc99f9426', '241f0f867', '5c6487af1',
            '62e59a501', 'f74e8f13d', 'fb49e4212', '190db8488', '324921c7b',
            'b43a7cfd5', '9306da53f', 'd6bb78916', 'fb0f5dbfe', '6eef030c1'
        ]
        if dataset == 'test' or dataset == 'both':
            new_test_df = self.test_df.filter(items=features)
            if not return_only:
                self.test_df = new_test_df
        if dataset == 'train' or dataset == 'both':
            features.append('target')
            new_train_df = self.train_df.filter(items=features)
            if not return_only:
                self.train_df = new_train_df

        if return_only:
            if dataset == 'train':
                return new_train_df
            elif dataset == 'test':
                return new_test_df
            elif dataset == 'both':
                return new_train_df, new_test_df

    def to_sparse(self, dataset='both', verbose=True):
        '''Transform datasets to sparse by removing zeros'''
        if dataset == 'train' or dataset == 'both':
            if verbose:
                print('Dense memory usage: train = {}mb'.format(
                    self.train_df.memory_usage().sum() / 1024 / 1024))

            self.train_df = self.train_df.replace(0, np.nan)

            if verbose:
                print('Sparse memory usage: train = {}mb'.format(
                    self.train_df.memory_usage().sum() / 1024 / 1024))

        if dataset == 'test' or dataset == 'both':
            if verbose:
                print('Dense memory usage: test = {}mb'.format(
                    self.test_df.memory_usage().sum() / 1024 / 1024))

            self.test_df = self.test_df.replace(0, np.nan)

            if verbose:
                print('Sparse memory usage: test = {}mb'.format(
                    self.test_df.memory_usage().sum() / 1024 / 1024))

    def normalize_data(self, x, fit=True, verbose=True):
        '''Normalize data taking sparsity into account'''
        if fit:
            self.scaler.fit(x)
        x = self.scaler.transform(x)
        if verbose:
            print('Data normalized.')
        return x

    def reduce_dimensionality(self, dataset, n_components=None, red_num=None,
                              method='svd', fit=True, normalize=False,
                              verbose=True):
        '''Reduce #red_num of features from the dataset'''
        assert method in ['svd', 'srp', 'fa']
        if dataset == 'train':
            x = self.train_df.drop(["target"], axis=1).values
            if normalize:
                x = self.normalize_data(x, fit=True)
        elif dataset == 'test':
            x = self.test_df.values
            if normalize:
                x = self.normalize_data(x, fit=False)

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
            print('Data new shape: ', x.shape)
        return x

    def add_decomposition_as_features(self, dataset='both', n_components=None,
                                      method='svd', comp_stats=False,
                                      verbose=True, normalize=False):
        '''Perform feature decomposition and add as an aggregate'''
        if dataset == 'train' or dataset == 'both':
            train_agg = self.reduce_dimensionality(
                dataset='train', n_components=n_components,
                method=method, fit=True, normalize=normalize, verbose=verbose)
            train_agg = pd.DataFrame(train_agg, index=self.train_df.index)
            self.train_agg = pd.concat([self.train_agg, train_agg], axis=1)
            if comp_stats:
                train_agg = compute_row_aggregates(train_agg, prefix='dec')
                self.train_agg = pd.concat([self.train_agg, train_agg], axis=1)

        if dataset == 'test' or dataset == 'both':
            test_agg = self.reduce_dimensionality(
                dataset='test', n_components=n_components,
                method=method, fit=False, normalize=normalize, verbose=verbose)
            test_agg = pd.DataFrame(test_agg, index=self.test_df.index)
            self.test_agg = pd.concat([self.test_agg, test_agg], axis=1)
            if comp_stats:
                test_agg = compute_row_aggregates(test_agg, prefix='dec')
                self.test_agg = pd.concat([self.test_agg, test_agg], axis=1)

    def get_most_important_features(self, num=50, importance_type='split',
                                    random_seed=43):
        '''Get the column names for the most important features'''
        assert importance_type in ['split', 'gain']
        LightGBM_params = dict(num_leaves=53, lr=0.005, bagging_fraction=0.67,
                               max_depth=8, min_sum_hessian_in_leaf=1e-1,
                               feature_fraction=0.35, bagging_freq=3,
                               min_data_in_leaf=12,
                               use_missing=True, zero_as_missing=True,
                               lambda_l1=0.1, lambda_l2=1,
                               device='cpu', num_threads=8)

        model = LightGBM(**LightGBM_params)

        x, y = self.get_train_data(use_aggregates=False)
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, test_size=0.2, random_state=random_seed)
        model.fit(x_train, y_train, x_val, y_val, verbose=0)
        most_important = model.model.feature_importance(
            importance_type=importance_type)
        index = np.argsort(most_important)[-num:]
        return index

    def compute_aggregates_for_all_features(self, dataset):
        '''Compute aggregates for all features'''
        if dataset == 'train' or dataset == 'both':
            train_agg = compute_row_aggregates(
                self.train_df.drop(["target"], axis=1), prefix='global')
            # Add to aggregates
            self.train_agg = pd.concat([self.train_agg, train_agg], axis=1)

        if dataset == 'test' or dataset == 'both':
            test_agg = compute_row_aggregates(
                self.test_df, prefix='global')
            # Add to aggregates
            self.test_agg = pd.concat([self.test_agg, test_agg], axis=1)

    def compute_aggregates_for_selected_features(self, dataset):
        '''Compute aggregate features for the hand selected features'''
        if dataset == 'train' or dataset == 'both':
            df = self.keep_only_selected_features('train', return_only=True)
            train_agg = compute_row_aggregates(
                df.drop('target', axis=1), prefix='hand_picked')
            # Add to aggregates
            self.train_agg = pd.concat([self.train_agg, train_agg], axis=1)

        if dataset == 'test' or dataset == 'both':
            df = self.keep_only_selected_features('test', return_only=True)
            test_agg = compute_row_aggregates(
                df, prefix='hand_picked')
            # Add to aggregates
            self.test_agg = pd.concat([self.test_agg, test_agg], axis=1)

    def compute_aggregates_for_most_important(self, dataset, num=50,
                                              importance_type='split',
                                              random_seed=43):
        '''Compute aggregate features for the most important features'''
        index = self.get_most_important_features(num,
                                                 importance_type,
                                                 random_seed)

        if dataset == 'train' or dataset == 'both':
            features = self.train_df.drop('target', axis=1).values[:, index]
            df = pd.DataFrame(features, index=self.train_df.index)
            train_agg = compute_row_aggregates(
                df, prefix='{}_most_important'.format(num))
            # Add to aggregates
            self.train_agg = pd.concat([self.train_agg, train_agg], axis=1)

        if dataset == 'test' or dataset == 'both':
            features = self.test_df.values[:, index]
            df = pd.DataFrame(features, index=self.test_df.index)
            test_agg = compute_row_aggregates(
                df, prefix='{}_most_important'.format(num))
            # Add to aggregates
            self.test_agg = pd.concat([self.test_agg, test_agg], axis=1)

    def compute_meta_aggregates(self, dataset='both'):
        '''Compute aggregate features for the existing aggregate features'''
        if dataset == 'train' or dataset == 'both':
            train_agg = compute_row_aggregates(
                self.train_agg, prefix='meta')
            self.train_agg = pd.concat([self.train_agg, train_agg], axis=1)

        if dataset == 'test' or dataset == 'both':
            test_agg = compute_row_aggregates(
                self.test_agg, prefix='meta')
            self.test_agg = pd.concat([self.test_agg, test_agg], axis=1)

    def compute_cluster_features(self, dataset='both',
                                 iter_cluster=range(2, 11)):
        '''Compute cluster centers using K-means'''
        for n_cluster in iter_cluster:
            if dataset == 'train' or dataset == 'both':
                features = self.compute_Kmeans('train', n_cluster, fit=True)
                train_agg = pd.DataFrame(
                    {'clusterIndex{}'.format(n_cluster): features},
                    index=self.train_df.index)
                self.train_agg = pd.concat([self.train_agg, train_agg], axis=1)
            if dataset == 'test' or dataset == 'both':
                features = self.compute_Kmeans('test', n_cluster, fit=False)
                test_agg = pd.DataFrame(
                    {'clusterIndex{}'.format(n_cluster): features},
                    index=self.test_df.index)
                self.test_agg = pd.concat([self.test_agg, test_agg], axis=1)

    def compute_Kmeans(self, dataset, n_cluster, fit=True, verbose=True):
        '''Compute K_means algorithm on data'''
        if dataset == 'train':
            X = self.train_df.drop('target', axis=1).values
        elif dataset == 'test':
            X = self.test_df.values

        if fit:
            self.clusterizer = KMeans(n_clusters=n_cluster, n_jobs=-2)
            self.clusterizer.fit(X)

        cluster_index = self.clusterizer.predict(X)
        # cluster_centers = self.clusterizer.cluster_centers_
        # cluster_feat = np.array([cluster_centers[x, :] for x in
        # cluster_index])

        if verbose:
            print('Clusters inertia: ', self.clusterizer.inertia_)

        return cluster_index

    def create_feature_as_targets(self):
        '''Create new target using arg of the feat closer to target'''
        for index, row in self.train_df.iterrows():
            target = row.pop('target')
            dists = abs(row.values - target)
            self.train_df.at[index, 'target'] = np.argmin(dists)


if __name__ == '__main__':
    train_path = './train.csv'
    test_path = './test.csv'

    dataset = KaggleDataset(train_path, test_path=test_path)
    print(dataset.train_df.describe(), dataset.test_df.describe())

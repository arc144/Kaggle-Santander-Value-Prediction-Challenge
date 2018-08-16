import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD, SparsePCA, FactorAnalysis
from sklearn.random_projection import SparseRandomProjection
from scipy.stats import ks_2samp
from scipy.stats import kurtosis, skew, mode
from Models import LightGBM
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from math import sqrt


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
        agg_df.at[index, '{}_non_zero_mode_count'.format(prefix)] = mode_[1]
        agg_df.at[index, '{}_non_zero_skewness'.format(prefix)] = \
            skew(non_zero_values)
        agg_df.at[index, '{}_non_zero_kurtosis'.format(prefix)] = \
            kurtosis(non_zero_values)

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
        agg_df.at[index, '{}non_zero_log_skewness'.format(prefix)] = \
            skew(np.log1p(non_zero_values))
        agg_df.at[index, '{}non_zero_log_kurtosis'.format(prefix)] = \
            kurtosis(np.log1p(non_zero_values))

        agg_df.at[index, '{}_non_zero_count'.format(
            prefix)] = np.count_nonzero(~np.isnan(non_zero_values))
        agg_df.at[index, '{}_non_zero_fraction'.format(prefix)] = \
            np.count_nonzero(~np.isnan(non_zero_values)) / \
            np.count_nonzero(~np.isnan(row))
    return agg_df


def find_fillers(df):
    '''Find the filler values in cols'''
    u, c = np.unique(df.loc['6726fff18', :].values, return_counts=True)
    arg = np.argsort(c)[::-1]
    fillers = u[arg]
    return fillers


class KaggleDataset():
    '''Class used to load Kaggle's official datasets'''

    def __init__(self, train_path, test_path=None, join_dfs=False,
                 verbose=True):
        self.train_path = train_path
        self.test_path = test_path
        self.verbose = verbose
        # Default settings, to be overidden if required
        self.normalize = False
        self.use_aggregates = False
        self.reduce_dim_nb = 0
        self.reduce_dim_method = 'svd'

        self.scaler = MaxAbsScaler(copy=False)

        # Load datasets
        self.train_df = load_df_from_path(self.train_path)
        # self.train_df[self.train_df == 1563411.76] = 0
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

    def remove_fillers_from_data(self, dataset='both', n_fillers=47):
        '''Substitute filler values in dataset for 0'''
        fillers = find_fillers(self.train_df)[:n_fillers]
        print(fillers)
        target = self.train_df['target']
        for filler in fillers:
            if dataset == 'test' or dataset == 'both':
                self.test_df[self.test_df == filler] = 0
            if dataset == 'train' or dataset == 'both':
                self.train_df[self.train_df == filler] = 0
        self.train_df['target'] = target

    def get_train_data(self, logloss=True, round_targets=False,
                       normalize=False, n_components=None,
                       reduce_dim_nb=0, use_aggregates=True,
                       get_leaky_data=None, reduce_dim_method='svd'):
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

        if round_targets:
            y = np.around(y,
                          decimals=round_targets)

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

        # Get leaky rows for training
        if get_leaky_data is not None:
            x_lk, y_lk = self.get_validation_set_from_leaky_test(
                get_leaky_data, logloss=logloss)
            x = np.concatenate([x, x_lk], axis=0)
            y = np.concatenate([y, y_lk], axis=0)

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

    def get_aggregates_as_data(self, dataset, logloss=True,
                               round_targets=False):
        '''Get aggregates as np data'''
        if dataset == 'train':
            x = self.train_agg.values
            if logloss:
                y = np.log1p(self.train_df["target"].values)
            else:
                y = self.train_df["target"].values
            if round_targets:
                y = np.around(y,
                              decimals=round_targets)
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
        # features = [
        #     'f190486d6', 'c47340d97', 'eeb9cd3aa', '66ace2992', 'e176a204a',
        #     '491b9ee45', '1db387535', 'c5a231d81', '0572565c2', '024c577b9',
        #     '15ace8c9f', '23310aa6f', '9fd594eec', '58e2e02e6', '91f701ba2',
        #     'adb64ff71', '2ec5b290f', '703885424', '26fc93eb7', '6619d81fc',
        #     '0ff32eb98', '70feb1494', '58e056e12', '1931ccfdd', '1702b5bf0',
        #     '58232a6fb', '963a49cdc', 'fc99f9426', '241f0f867', '5c6487af1',
        #     '62e59a501', 'f74e8f13d', 'fb49e4212', '190db8488', '324921c7b',
        #     'b43a7cfd5', '9306da53f', 'd6bb78916', 'fb0f5dbfe', '6eef030c1'
        # ]
        features = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98', '6469d38ef', '22ff4e1b8', 'cf4a89033', '2d7f256c6', 'e98dd91da', 'bce33962d', 'cdaf3e576', '78b2cf944', 'b3dbb1222', 'a2d707f4e', 'a5565ec7d', '26b423c42', '2731a96a2', 'c24634d56', 'dc61677e5', '62c319fd1', 'fbe583de5', 'ab7534e84', '92eb376f1', '7def2d7d3', '657d07d19', 'df08959aa', '48130b04b', 'e0cad27c5', '0b139c060', '99df18cb2', '32c6f9aa6', '7ff7d351b', 'e49fea26e', '403e7e48d', '2c5809e51', 'fe1160758', 'd83991b80', 'ce999e374', '29e58696d', '212226b7b', '950b9cf2d', 'adc70e02f', '7792cebe0', '22f0deffc', 'a3e023f65', '9126049d8', '6eaea198c', '5244415dd', '0616154cc', '2165c4b94', 'fc436be29', '1834f29f5', '9d5af277d', 'c6850e7db', '6b241d083', '56f619761', '45319105a', 'fcda960ae', '07746dcda', 'c906cd268', 'c24ea6548', '829fb34b8', '89ebc1b76', '22c019a2e', '1e16f11f3', '94072d7a3', '59dfc16da', '9886b4d22', '0b1741a7f', 'a682ef110', 'e26299c3a', '5c220a143', 'ac0493670', '8d8bffbae', '68c7cf320', '3cea34020', 'e9a8d043d', 'afb6b8217', '5780e6ffa', '26628e8d8', '1de4d7d62', '4c53b206e', '99cc87fd7', '593cccdab', 'ccc7609f4', 'ca7ea80a3', 'e509be270', '3b8114ab0', 'a355497ac', '27998d0f4', 'fa05fd36e', '81aafdb57', '4e22de94f', 'f0d5ffe06', '9af753e9d', 'f1b6cc03f', '567d2715c', '857020d0f', '99fe351ec', '3e5dab1e3', '001476ffa', '5a5eabaa7', 'cb5587baa', '32cab3140', '313237030', '0f6386200', 'b961b0d59', '9452f2c5f', 'bcfb439ee', '04a22f489', '7e58426a4', 'a4c9ea341', 'ffdc4bcf8', '1a6d866d7', 'd7334935b', '298db341e', '08984f627', '8367dfc36', '5d9f43278', '7e3e026f8', '37c10d610', '5a88b7f01', '324e49f36', '99f466457', '48b839509', '2b8851e90', '28f75e1a5', '0e3ef9e8f', '37ac53919', '7ca10e94b', '4b6c549b1', '467aa29ce', '74c5d55dc', '0700acbe1', '44f3640e4', 'e431708ff', '097836097', 'd1fd0b9c2', 'a0453715a', '9e3aea49a', '899dbe405', '525635722', '87a2d8324', 'faf024fa9', 'd421e03fd', '1254b628a', 'a19b05919', '34a4338bc', '08e89cc54', 'a29c9f491', 'a0a8005ca', '62ea662e7', '5fe6867a4', '8b710e161', '7ab926448', 'd04e16aed', '4e5da0e96', 'ff2c9aa8f', 'b625fe55a', '7124d86d9', '215c4d496', 'b6fa5a5fd', '55a7e0643', '0a26a3cfe', '87ffda550', '63c094ba4', '2e103d632', '1c71183bb', 'd5fa73ead', 'e078302ef', 'a6b6bc34a', 'f6eba969e', '0d51722ca', 'ce3d7595b', '6c5c8869c', 'dfd179071', '122c135ed', 'b4cfe861f', 'b7c931383', '44d5b820f', '4bcf15776', '51d4053c7', '1fe5d56b9', 'ea772e115', 'ad009c8b9', '68a945b18', '62fb56487', 'c10f31664', 'cbb673163', 'c8d582dd2', '8781e4b91', 'bd6da0cca', 'ca2b906e8', '11e12dbe8', 'bb0ce54e9', 'c0d2348b7', '77deffdf0', 'f97d9431e', 'a09a238d0', '935ca66a9', '9de83dc23', '861076e21', 'f02ecb19c', '166008929', '920a04ee2', '93efdb50f', '15ea45005', '78c57d7cd', '91570fb11', 'c5dacc85b', '145c7b018', '590b24ab1', 'c283d4609', 'e8bd579ae', '7298ca1ef', 'ce53d1a35', 'a8f80f111', '2a9fed806', 'feb40ad9f', 'cfd255ee3', '31015eaab', '303572ae2', 'cd15bb515', 'cb5161856', 'a65b73c87', '71d64e3f7', 'ec5fb550f', '4af2493b6', '18b4fa3f5', '3d655b0ed', '5cc9b6615', '88c0ec0a6', '8722f33bb', '5ed0c24d0', '54f26ee08', '04ecdcbb3', 'ade8a5a19', 'd5efae759', 'ac7a97382', 'e1b20c3a6', 'b0fcfeab8', '438b8b599', '43782ef36', 'df69cf626', '9d4428628', '37f11de5d', '39549da61', 'ceba761ec', '4c60b70b8', '304ebcdbc', '823ac378c', '4e21c4881', '5ee81cb6e', 'eb4a20186', 'f6bdb908a', '6654ce6d8', '65aa7f194', '00f844fea', 'c4de134af', 'a240f6da7', '168c50797', '13d6a844f', '7acae7ae9', '8c61bede6', '45293f374', 'feeb05b3f', 'a5c62af4a', '22abeffb6', '1d0aaa90f', 'c46028c0f', '337b3e53b', 'd6af4ee1a', 'cde3e280a', 'c83fc48f2', 'f99a09543', '85ef8a837', 'a31ba11e6', '64cabb6e7', '93521d470', '46c525541', 'cef9ab060', '375c6080e', '3c4df440f', 'e613715cc', '5030aed26', 'b850c3e18', '212efda42', '9e7c6b515', '2d065b147', '49ca7ff2e', '37c85a274', 'ea5ed6ff7', 'deabe0f4c', 'bae4f747c', 'ca96df1db', '05b0f3e9a', 'eb19e8d63', '235b8beac', '85fe78c6c', 'cc507de6c', 'e0bb9cf0b', '80b14398e', '9ca0eee11', '4933f2e67', 'fe33df1c4', 'e03733f56', '1d00f511a', 'e62cdafcf', '3aad48cda', 'd36ded502', '92b13ebba', 'f30ee55dd', '1f8754c4e', 'db043a30f', 'e75cfcc64', '5d8a55e6d', '6e29e9500', 'c5aa7c575', 'c2cabb902', 'd251ee3b4', '73700eaa4', '8ab6f5695', '54b1c1bc0', 'cbd0256fb', '81de0d45e', '18562fc62', '543c24e33', '0256b6714', 'd6006ff44', '6a323434b', 'e3a38370e', '7c444370b', '8d2d050a2', '9657e51e1', '13f3a3d19', 'b5c839236', '70f3033c6', 'f4b374613', '849125d91', '16b532cdc', '88219c257', '74fb8f14c', 'fd1102929', '699712087', '22501b58e', '9e9274b24', '2c42b0dce', '2c95e6e31', '5263c204d', '526ed2bec', '01f7de15d', 'cdbe394fb', 'adf357c9b', 'd0f65188c', 'b8a716ebf', 'ef1e1fac8', 'a3f2345bf', '110e4132e', '586b23138', '680159bab', 'f1a1562cd', '9f2f1099b', 'bf0e69e55', 'af91c41f0', 'a5f8c7929', '330006bce', 'b22288a77', 'de104af37', '8d81c1c27', 'd7285f250', '123ba6017', '3c6980c42', '2d3296db7', '95cdb3ab7', '05527f031', '65753f40f', '45a400659', '1d5df91e2', '233c7c17c', '2a879b4f7', 'c3c633f64', 'fdae76b2c', '05d17ab7a', 'c25078fd7', 'e209569b2', '3fd2b9645', '268b047cd', '3d350431d', '5fb9cabb1', 'b70c76dff', '3f6246360', '89e7dcacc', '12122f265', 'fcc17a41d', 'c5a742ee4', '9e711a568', '597d78667', '0186620d7', '4c095683e', '472cd130b', 'b452ba57e', '2ce2a1cdb', '50c7ea46a', '2761e2b76', 'ced6a7e91', '9df4daa99', '83c3779bf', 'edc84139a', 'f1e0ada11', '73687e512', 'aa164b93b', '342e7eb03', 'cd24eae8a', '8f3740670', '2b2a10857', 'a00adf70e', '3a48a2cd2', 'a396ceeb9', '9280f3d04', 'fec5eaf1a', '5b943716b', '22ed6dba3', '5547d6e11', 'e222309b0', '5d3b81ef8', '1184df5c2', '2288333b4', 'f39074b55', 'a8b721722', '13ee58af1', 'fb387ea33', '4da206d28', 'ea4046b8d', 'ef30f6be5', 'b85fa8b27', '2155f5e16', '794e93ca6', '070f95c99', '939f628a7', '7e814a30d', 'a6e871369', '0dc4d6c7d', 'bc70cbc26', 'aca228668', 'd0d340214', '34d3715d5', '9c404d218', 'c624e6627', 'a1b169a3a', 'c144a70b1', 'b36a21d49', 'dfcf7c0fa', 'c63b4a070', '43ebb15de', '1f2a670dd', '3f07a4581', '0b1560062', 'e9f588de5', '65d14abf0', '9ed0e6ddb', '0b790ba3a', '9e89978e3', 'ee6264d2b', 'c86c0565e', '4de164057', '87ba924b1', '4d05e2995', '2c0babb55', 'e9375ad86', '8988e8da5', '8a1b76aaf', '724b993fd', '654dd8a3b', 'f423cf205', '3b54cc2cf', 'e04141e42', 'cacc1edae', '314396b31', '2c339d4f2', '3f8614071', '16d1d6204', '80b6e9a8b', 'a84cbdab5', '1a6d13c4a', '1d9078f84', '64e483341', 'a75d400b8', '4fe8154c8', '29ab304b9', '20604ed8f', 'bd8f989f1', 'c1b9f4e76', '4824c1e90', '4ead853dc', 'b599b0064', 'd26279f1a', '58ed8fb53', 'ff65215db', '402bb0761', '74d7998d4', 'c7775aabf', '9884166a7', 'beb7f98fd', 'fd99c18b5', 'd83a2b684', '18c35d2ea', '0c8063d63', '400e9303d', 'c976a87ad', '8a088af55', '5f341a818', '5dca793da', 'db147ffca', '762cbd0ab', 'fb5a3097e', '8c0a1fa32', '01005e5de', '47cd6e6e4', 'f58fb412c', 'a1db86e3b', '50e4f96cf', 'f514fdb2e', '7a7da3079', 'bb1113dbb', 'f3cf9341c', 'fa11da6df', 'd47c58fe2', '0d5215715', '555f18bd3', '134ac90df', '716e7d74d', 'c00611668', '1bf8c2597', '1f6b2bafa', '174edf08a', 'f1851d155', '5bc7ab64f', 'a61aa00b0', 'b2e82c050', '26417dec4', '53a550111', '51707c671', 'e8d9394a0', 'cbbc9c431', '6b119d8ce', 'f296082ec', 'be2e15279', '698d05d29', '38e6f8d32', '93ca30057', '7af000ac2', '1fd0a1f2a', '41bc25fef', '0df1d7b9a', '88d29cfaf', '2b2b5187e', 'bf59c51c3', 'cfe749e26', 'ad207f7bb', '11114a47a', '341daa7d1', 'a8dd5cea5', '7b672b310', 'b88e5de84', 'c928b4b74', '8e4d0fe45', '6c0e0801a', '02861e414', 'aac52d8d9', '041c5d0c9', 'd7875bb6c', 'e7c0cfd0f', 'd48c08bda', '0c9462c08', '57dd44c29', 'a93118262', '850027e38', 'db3839ab0', '27461b158', '32174174c', '9306da53f', '95742c2bf', '5831f4c76', '1e6306c7c', '06393096a', '13bdd610a', 'd7d314edc', '9a07d7b1f', '4d2671746', '822e49b95', '3c8a3ced0', '83635fb67', '1857fbccf', 'c4972742d', 'b6c0969a2', 'e78e3031b', '36a9a8479', 'e79e5f72c', '092271eb3', '74d7f2dc3', '277ef93fc', 'b30e932ba', '8f57141ec', '350473311', '51c141e64', '0e348d340', '64e010722', '55a763d90', '13b54db14', '01fdd93d3', '1ec48dbe9', 'cf3841208', 'd208491c8', '90b0ed912', '633e0d42e', '9236f7b22', '0824edecb', '71deb9468', '1b55f7f4d', '377a76530', 'c47821260', 'bf45d326d', '69f20fee2', 'd6d63dd07', '5ab3be3e1', '93a31829f', '121d8697e', 'f308f8d9d', '0e44d3981', 'ecdef52b2', 'c69492ae6', '58939b6cc', '3132de0a3', 'a175a9aa4', '7166e3770', 'abbde281d', '23bedadb2', 'd4029c010', 'fd99222ee', 'bd16de4ba', 'fb32c00dc', '12336717c', '2ea42a33b', '50108b5b5', '1ad24da13', '8c5025c23', 'f52a82e7f', 'c0b22b847', 'd75793f21', '4cffe31c7', '6c2d09fb1', 'fb42abc0d', '206ba1242', '62f61f246', '1389b944a', 'd15e80536', 'fa5044e9e', 'a0b0a7dbf', '1ff6be905', '4e06c5c6d', '1835531cd', '68b647452', 'c108dbb04', '58e8e2c82', 'f3bfa96d5', 'f2db09ac3', '4e8196700', '8cd9be80e', '83fc7f74c', 'dbc48d37c', '2028e022d', '17e160597', 'eb8cbd733', 'addb3f3eb', '460744630', '9108ee25c', 'b7950e538', 'a7da4f282', '7f0d863ba', 'b7492e4eb', '24c41bd80', 'fd7b0fc29', '621f71db3', '26f222d6d', '4302b67ec', '75b663d7d', 'fc4a873e0', '1e9bdf471', '86875d9b0', '8f76eb6e5', '3d71c02f0', '05c9b6799', '26df61cc3', '27a7cc0ca', '9ff21281c', '3ce93a21b', '9f85ae566', '3eefaafea', 'afe8cb696', '72f9c4f40', 'be4729cb7', '8c94b6675', 'ae806420c', '63f493dba', '5374a601b', '5291be544', 'acff85649', '3690f6c26', '26c68cede', '12a00890f', 'dd84964c8', 'a208e54c7', 'fb06e8833', '7de39a7eb', '5fe3acd24', 'e53805953', '3de2a9e0d', '2954498ae', '6c3d38537', '86323e98a', 'b719c867c', '1f8a823f2', '9cc5d1d8f', 'd3fbad629', '63be1f619', '36a56d23e', '9e2040e5b', 'a00a63886', '4edc3388d', '5f11fbe33', '26e998afd', 'f7faf2d9f', '992b5c34d', 'f7f553aea', '7e1c4f651', 'f5538ee5c', '711c20509', '55338de22', '374b83757', 'f41f0eb2f', 'bf10af17e', 'e2979b858', 'd3ed79990', 'fe0c81eff', '5c0df6ac5', '82775fc92', 'f1c20e3ef', 'fa9d6b9e5', 'a8b590c6e', 'b5c4708ad', 'c9aaf844f', 'fe3fe2667', '50a6c6789', '8761d9bb0', 'b6403de0b', '2b6f74f09', '5755fe831', '91ace30bd', '84067cfe0', '15e4e8ee5', 'd01cc5805', '870e70063', '2bd16b689', '8895ea516', '7ba58c14d', '1fe02bc17', '4672a8299', '8794c72c8', 'cca45417f', '55dbd6bcb', 'e6e2c3779', '3cae817df', '973663d14', 'e8dfb33d5', '9281abeea', '11c01e052', '1520de553', 'edddb1ba5', 'c18b41ac3', '00e87edf2', 'ae72cba0a', 'eb4f2651e', '300398f1c', '6c05550b8', '9b26736c3', '24744410a', '26faf1b2e', '44f09b92d', '19975f6ff',
                    '1bf6240eb', 'e438105db', 'cdc36a26a', '087e01c14', '828b327a6', 'cc62f0df8', '9370aa48d', 'd4815c074', '18321c252', '22fbf6997', 'feed9d437', 'f6c9661fc', '55f2b3d34', '69fe81b64', '1074273db', '2135fa05a', 'e8a3423d6', '90a438099', '7ad6b38bd', '60e45b5ee', '2b9b1b4e2', 'd6c82cd68', '923114217', 'b361f589e', '04be96845', 'ee0b53f05', '21467a773', '47665e3ce', 'a6229abfb', '9666bfe76', '7dcc40cda', '17be6c4e7', 'a89ab46bb', '9653c119c', 'cc01687d0', '60e9cc05b', 'ffcec956f', '51c250e53', '7344de401', 'a15b2f707', 'a8e607456', 'dbb8e3055', '2a933bcb8', 'b77bc4dac', '58d9f565a', '17068424d', '7453eb289', '027a2206a', '343042ed9', 'c8fb3c2d8', '29eddc376', '1c873e4a6', '588106548', '282cfe2ad', '358dc07d0', 'b6daeae32', '3bdee45be', '3d6d38290', '5a1589f1a', '961b91fe7', '29c059dd2', 'cfc1ce276', '0a953f97e', '30b3daec2', 'fb5f5836e', 'c7525612c', '6fa35fbba', '72d34a148', 'dcc269cfe', 'bdf773176', '469630e5c', '23db7d793', 'dc10234ae', '5ac278422', '6cf7866c1', 'a39758dae', '45f6d00da', '251d1aa17', '84d9d1228', 'b98f3e0d7', '66146c12d', 'd6470c4ce', '3f4a39818', 'f16a196c6', 'b8f892930', '6f88afe65', 'ed8951a75', '371da7669', '4b9540ab3', '230a025ca', 'f8cd9ae02', 'de4e75360', '540cc3cd1', '7623d805a', 'c2dae3a5a', 'fec5644cf', 'caa9883f6', '9437d8b64', '68811ba58', 'ef4b87773', 'ff558c2f2', '8d918c64f', '0b8e10df6', '2d6565ce2', '0fe78acfa', 'b75aa754d', '2ab9356a0', '4e86dd8f3', '348aedc21', 'd7568383a', '856856d94', '69900c0d1', '02c21443c', '5190d6dca', '20551fa5b', '79cc300c7', '8d8276242', 'da22ed2b8', '89cebceab', 'f171b61af', '3a07a8939', '129fe0263', 'e5b2d137a', 'aa7223176', '5ac7e84c4', '9bd66acf6', '4c938629c', 'e62c5ac64', '57535b55a', 'a1a0084e3', '2a3763e18', '474a9ec54', '0741f3757', '4fe8b17c2', 'd5754aa08', '0f8d7b98e', 'c30ff7f31', 'ac0e2ebd0', '24b2da056', 'bd308fe52', '476d95ef1', '202acf9bd', 'dbc0c19ec', '06be6c2bb', 'd8296080a', 'f977e99dc', '2191d0a24', '7db1be063', '1bc285a83', '9a3a1d59b', 'c4d657c5b', 'a029667de', '21bd61954', '16bf5a9a2', '0e0f8504b', '5910a3154', 'ba852cc7a', '685059fcd', '21d6a4979', '78947b2ad', '1435ecf6b', '3839f8553', 'e9b5b8919', 'fa1dd6e8c', '632586103', 'f016fd549', 'c25ea08ba', '7da54106c', 'b612f9b7e', 'e7c0a50e8', '29181e29a', '395dbfdac', '1beb0ce65', '04dc93c58', '733b3dc47', 'a1cd7b681', '9b490abb3', 'b10f15193', '05f54f417', 'a7ac690a8', 'ed6c300c2', 'd0803e3a1', 'b1bb8eac3', 'bd1c19973', 'a34f8d443', '84ec1e3db', '24018f832', '82e01a220', '4c2064b00', '0397f7c9b', 'ba42e41fa', '22d7ad48d', '9abffd22c', 'dbfa2b77f', '2c6c62b54', '9fa38def3', 'ecb354edf', '9c3154ae6', '2f26d70f4', '53102b93f', 'a36b95f78', '1fa0f78d0', '19915a6d3', 'c944a48b5', '482b04cba', '2ce77a58f', '86558e595', 'c3f400e36', '20305585c', 'f8ccfa064', 'dd771cb8e', '9aa27017e', 'cd7f0affd', '236cc1ff5', 'a3fc511cd', '5b465f819', 'a2aa0e4e9', '944e05d50', '4f8b27b6b', 'a498f253f', 'c73c31769', '025dea3b3', '616c01612', 'f3316966c', '83ea288de', '2dbeac1de', '47b7b878f', 'b4d41b335', '686d60d8a', '6dcd9e752', '7210546b2', '78edb3f13', '7f9d59cb3', '30992dccd', '26144d11f', 'a970277f9', '0aea1fd67', 'dc528471e', 'd51d10e38', 'efa99ed98', '48420ad48', '7f38dafa6', '1af4ab267', '3a13ed79a', '73445227e', '971631b2d', '57c4c03f6', '7f91dc936', '0784536d6', 'c3c3f66ff', '052a76b0f', 'ffb34b926', '9d4f88c7b', '442b180b6', '948e00a8d', 'df838756c', '2cb73ede7', '4dcf81d65', '61c1b7eb6', 'a9f61cf27', '1af4d24fa', 'e13b0c0aa', 'b9ba17eb6', '796c218e8', '37f57824c', 'd1e0f571b', 'f9e3b03b7', 'a3ef69ad5', 'e16a20511', '04b88be38', '99e779ee0', '9f7b782ac', '1dd7bca9f', '2eeadde2b', '6df033973', 'cdfc2b069', '031490e77', '5324862e4', '467bee277', 'a3fb07bfd', '64c6eb1cb', '8618bc1fd', '6b795a2bc', '956d228b9', '949ed0965', 'a4511cb0b', 'b64425521', '2e3c96323', '191e21b5f', 'bee629024', '1977eaf08', '5e645a169', '1d04efde3', '8675bec0b', '8337d1adc', 'c13ee1dc9', 'abb30bd35', 'd2919256b', '66728cc11', 'eab8abf7a', 'cc03b5217', '317ee395d', '38a92f707', '467c54d35', 'e8f065c9d', '2ac62cba5', '6495d8c77', '94cdda53f', '13f2607e4', '1c047a8ce', '28a5ad41a', '05cc08c11', 'b0cdc345e', '38f49406e', '773180cf6', '1906a5c7e', 'c104aeb2e', '8e028d2d2', '0dc333fa1', '28a785c08', '03ee30b8e', '8e5a41c43', '67102168f', '8b5c0fb4e', '14a22ab1a', '9fc776466', '4aafb7383', '8e1dfcb94', '55741d46d', '8f940cb1b', '758a9ab0e', 'fd812d7e0', '4ea447064', '6562e2a2c', '343922109', '8f6514df0', '6679fe54f', '5e62457b7', 'f17ff4efd', 'ec7f7017f', 'c02ab7d25', '8c309c553', 'e0b968d7b', '22b980fc8', '3b6b46221', '3e4a6796d', 'c680e9350', '834fb292d', 'e3d33877c', '4052a9419', 'b95be4138', '16517c8b0', '219e051b5', 'a6fbe0987', '37d7af8ad', 'b84b2f72d', '775577e6f', '4f0c5f900', 'a68b83290', '2a2832b07', 'ce1f5b02a', 'a6c9347a7', '82c9b4fcd', '7f78a36f7', 'f49ff3269', '89cffafe9', 'aeb3a6ccf', 'c7753cbfc', '4d6a1439e', '2123a4f36', '5c56fccf1', '03bfe48b2', '6beb0b35d', '9fb38aabe', 'ae141696e', 'e20edfcb8', '842415efb', '300d6c1f1', '720f83290', '069a2c70b', '87a91f998', '611151826', '74507e97f', '504e4b156', 'baa95693d', 'cb4f34014', '5239ceb39', '81e02e0fa', 'dfdf4b580', 'fc9d04cd7', 'fe5d62533', 'bb6260a44', '08d1f69ef', 'b4ced4b7a', '98d90a1d1', 'b6d206324', '6456250f1', '96f5cf98a', 'f7c8c6ad3', 'cc73678bf', '5fb85905d', 'cb71f66af', '212e51bf6', 'd318bea95', 'b70c62d47', '11d86fa6a', '3988d0c5e', '42cf36d73', '9f494676e', '1c68ee044', 'a728310c8', '612bf9b47', '105233ed9', 'c18cc7d3d', 'f08c20722', '266525925', '4b6dfc880', '2cff4bf0c', 'a3382e205', '6488c8200', '547d3135b', 'b46191036', '453128993', '2599a7eb7', '2fc60d4d9', '009319104', 'de14e7687', 'aa31dd768', '2b54cddfd', 'a67d02050', '37aab1168', '939cc02f5', '31f72667c', '6f951302c', '54723be01', '4681de4fd', '8bd53906a', '435f27009', 'f82167572', 'd428161d9', '9015ac21d', 'ec4dc7883', '22c7b00ef', 'd4cc42c3d', '1351bf96e', '1e8801477', 'b7d59d3b5', 'a459b5f7d', '580f5ff06', '39b3c553a', '1eec37deb', '692c44993', 'ce8ce671e', '88ef1d9a8', 'bf042d928', '0d7692145', '62071f7bc', 'ab515bdeb', 'c30c6c467', 'eab76d815', 'b6ee6dae6', '49063a8ed', '4cb2946ce', '6c27de664', '772288e75', 'afd87035a', '44f2f419e', '754ace754', 'e803a2db0', 'c70f77ef2', '65119177e', '3a66c353a', '4c7768bff', '9e4765450', '24141fd90', 'dc8b7d0a8', 'ba499c6d9', '8b1379b36', '5a3e3608f', '3be3c049e', 'a0a3c0f1b', '4d2ca4d52', '457bd191d', '6620268ab', '9ad654461', '1a1962b67', '7f55b577c', '989d6e0f5', 'bc937f79a', 'e059a8594', '3b74ac37b', '555265925', 'aa37f9855', '32c8b9100', 'e71a0278c', '2d60e2f7a', '11ad148bd', '54d3e247f', 'c25438f10', 'e6efe84eb', '964037597', '0196d5172', '47a8de42e', '6f460d92f', '0656586a4', '22eb11620', 'c3825b569', '6aa919e2e', '086328cc6', '9a33c5c8a', 'f9c3438ef', 'c09edaf01', '85da130e3', '2f09a1edb', '76d34bbee', '04466547a', '3b52c73f5', '1cfb3f891', '704d68890', 'f45dd927f', 'aba01a001', 'c9160c30b', '6a34d32d6', '3e3438f04', '038cca913', '504c22218', '56c679323', '002d634dc', '1938873fd', 'd37030d36', '162989a6d', 'e4dbe4822', 'ad13147bd', '4f45e06b3', 'ba480f343', '7f72c937f', '79e55ef6c', '408d86ce9', '7a1e99f69', '736513d36', '0f07e3775', 'eb5a2cc20', '2b0fc604a', 'aecd09bf5', '91de54e0a', '66891582e', '20ef8d615', '8d4d84ddc', 'dfde54714', '2be024de7', 'd19110e37', 'e637e8faf', '2d6bd8275', 'f3b4de254', '5cebca53f', 'c4255588c', '23c780950', 'bc56b26fd', '55f4891bb', '020a817ab', 'c4592ac16', '542536b93', '37fb8b375', '0a52be28f', 'bd7bea236', '1904ce2ac', '6ae9d58e0', '5b318b659', '25729656f', 'f8ee2386d', '589a5c62a', '64406f348', 'e157b2c72', '0564ff72c', '60d9fc568', '9fa984817', '3d23e8abd', '1b681c3f0', '3be4dad48', 'dcfcddf16', 'b25319cb3', 'b14026520', 'c5cb7200e', 'ede70bfea', 'e5ddadc85', '07cb6041d', 'df6a71cc7', 'dc60842fb', '3a90540ab', '6bab7997a', 'c87f4fbfb', '21e0e6ae3', '9b39b02c0', '5f5cfc3c0', '35da68abb', 'f0aa40974', '625525b5d', 'd7978c11c', '2bbcbf526', 'bc2bf3bcd', '169f6dda5', '4ceef6dbd', '9581ec522', 'd4e8dd865', 'bf8150471', '542f770e5', 'b05eae352', '3c209d9b6', 'b2e1308ae', '786351d97', 'e5a8e9154', '2b85882ad', 'dc07f7e11', '14c2463ff', '14a5969a6', '86cefbcc0', '717eff45b', '7d287013b', '8d7bfb911', 'aecaa2bc9', '193a81dce', '8dc7f1eb9', 'c5a83ecbc', '60307ab41', '3da5e42a7', 'd8c61553b', '072ac3897', '1a382b105', 'f3a4246a1', '4e06e4849', '962424dd3', 'a3da2277a', '0a69cc2be', '408d191b3', '98082c8ef', '96b66294d', 'cc93bdf83', 'ffa6b80e2', '226e2b8ac', '678b3f377', 'b56f52246', '4fa02e1a8', '2ef57c650', '9aeec78c5', '1477c751e', 'a3c187bb0', '1ce516986', '080cd72ff', '7a12cc314', 'ead538d94', '480e78cb0', '737d43535', 'a960611d7', '4416cd92c', 'd5e6c18b0', '9a2b0a8be', '856225035', 'f9db72cff', '709573455', '616be0c3e', '19a67cb97', '9d478c2ae', 'cf5b8da95', '9c502dcd9', '2f7b0f5b5', 'd50798d34', '56da2db09', 'c612c5f8f', '08c089775', '7aaefdfd7', '59cb69870', '37c0a4deb', 'fb9a4b46d', 'b4eaa55ea', '304633ac8', '99f22b12d', '65000b269', '4bffaff52', '4c536ffc0', '93a445808', 'e8b513e29', 'a2616a980', '97d5c39cf', '71aae7896', '62d0edc4f', 'c2acc5633', 'c8d5efceb', 'f0317ca4f', '402b0d650', '7e78d546b', '2ad744c57', '47abb3cb4', '71ac2b961', '5b8c88c94', '293e2698e', '4bdeca0d2', '2ef8b7f4f', 'c380056bb', '2488e17f5', '20442bac4', '8e8736fc8', '8a4c53d3e', '62c547c8e', '86f13324d', 'da52febdb', '64e38e7a2', 'b0310a768', '0d866c3d7', '34a2f580b', '24bcc2f15', 'e1e8947d8', '05f11f48f', '8c8616b62', '79e0c374a', 'ad1466df8', 'f642213a6', 'f8405f8b9', '1ae0db9d5', '9dbb6b717', '0f7ae26ce', '81ec47b4c', 'ad4e33a4c', 'a78f85d49', '8de6fcbf1', '3ecc09859', '844df03d7', '2e7f340f2', '22f95560c', '2a3c59733', 'a1d11c496', '1e403019b', 'e429ad370', 'a165f5761', '6ab79c2fe', 'afac06058', '735ea6729', '95ba53cf8', '9685f5e16', '6879db4be', '227ac0d56', '5da2e6220', 'dc5a8f1d8', '89ca53693', 'dd0491aad', '98d0d2971', '324aaa96f', '3d4a6baed', '2715b2d4a', 'b7f26c1f7', 'b0385cee8', '007d71f12', 'be448d5b9', 'e871db27b', '69918e0c1', '9d2dea573', '43a1bf3e9', 'adc721d55', 'db1da2c31', 'ec1425047', 'cc462dc0b', 'b96c4256a', 'cb5329038', '3391c0af1', 'd8e951a15', '320c3880a', '37bd241bb', 'ef6fab1bc', '39d765b49', 'c0f2800fc', '9401c82a9', '6f9d256a9', 'f967f827b', '798a8e92e', '1e991f09b', '77854931e', '0024cd760', 'ab7b12083', '7b8ac394f', 'd8aa77bcc', '54b3d6afc', '4c6a1accd', '0a7ef2f9b', 'dc3b4460b', '83ce5d80d', '7e293fbaf', '8d468092c', 'dc0118d47', '9f8df01bd', '3f02111a8', 'bf0195c11', '28bc37378', '05410a84f', 'ecdc88e4b', '8d9263a24', '53aa182a2', '4e92107c6', '295408598', 'b76bf3f19', '3305c8063', 'd3a116347', 'ac5260727', '199caef5d', '97ea72529', '1d4d5cd4a', '8fc7efaf0', '225fa9d61', '94f3dcaee', '4634c8fae', '660fdbc58', '052f633c1', '657dec16b', '7fa5bc19f', '7207afb67', 'cda277b2a', 'e9a473fbb', '3eac9a76e', '1c554649c', '86ffb104c', 'b14d5014b', '8348ea8d3', 'e3a4596f9', '49db469f5', 'f928893ca', 'aa610feec', 'fa2a340da', '652142369']
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

    def get_gibas_pred(self, dataset='train'):
        # Need more features!!! Note that if we use
        if dataset == 'train':
            data = self.train_df
        elif dataset == 'test':
            data = self.test_df

        features = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec',
                    '6eef030c1', '15ace8c9f', 'fb0f5dbfe', '58e056e12',
                    '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5',
                    '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501',
                    '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992',
                    'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7',
                    '1931ccfdd', '703885424', '70feb1494', '491b9ee45',
                    '23310aa6f', 'e176a204a', '6619d81fc', '1db387535']
        d1 = data[features[:-2]
                  ].apply(tuple, axis=1).to_frame().rename(columns={0: 'key'})
        d2 = data[features[2:]].apply(
            tuple, axis=1).to_frame().rename(columns={0: 'key'})
        d2['pred'] = data[features[0]]
        d2 = d2[d2['pred'] != 0]  # Keep?
        d3 = d2[~d2.duplicated(['key'], keep='first')]  # Need more features!
        d = d1.merge(d3, how='left', on='key')

        pred = d.pred.fillna(0)
        log_pred = np.log1p(d.pred).fillna(0)
        have_data = log_pred != 0

        if dataset == 'train':
            error = sqrt(mean_squared_error(
                np.log1p(self.train_df.reset_index().target[have_data]),
                log_pred[have_data]))
            print(
                f'Score={error} on {have_data.sum()} out of {self.train_df.shape[0]} training samples')
        elif dataset == 'test':
            print(f'Have predictions for {have_data.sum()} out of {self.test_df.shape[0]} test samples')

        return pred

    def get_data_as_time_series(self, dataset='train',
                                logloss=True, round_targets=False):
        '''Get data as time-series using selected features'''
        cols = ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec',
                '6eef030c1', '15ace8c9f', 'fb0f5dbfe', '58e056e12',
                '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5',
                '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501',
                '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992',
                'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7',
                '1931ccfdd', '703885424', '70feb1494', '491b9ee45',
                '23310aa6f', 'e176a204a', '6619d81fc', '1db387535',
                'fc99f9426', '91f701ba2', '0572565c2', '190db8488',
                'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'
                ]

        if dataset == 'train':
            x = self.train_df.drop(["target"], axis=1)[cols].values
            if logloss:
                y = np.log1p(self.train_df["target"].values)
            else:
                y = self.train_df["target"].values
            if round_targets:
                y = np.around(y,
                              decimals=round_targets)
            return x, y

        elif dataset == 'test':
            x = self.test_df[cols].values
            return x

    def add_IsTargetAvaliable_as_feature(self, params=None, test=True,
                                         verbose=True, random_seed=43,
                                         threshold='soft',
                                         calc_on_selected_feat=False):
        '''Use a LightGBM model to predict if the target is one of the cols in the row'''
        if not params:
            params = {
                'objective': 'binary',
                'num_leaves': 20,
                'min_data_in_leaf': 20,
                'bagging_fraction': 0.6143, 'bagging_freq': 3,
                'feature_fraction': 0.6453,
                'min_split_gain': np.power(10, -2.5988),
                'lambda_l1': np.power(10, -2.2887),
                'lambda_l2': np.power(10, 1.7570),
                'min_child_weight': np.power(10, -0.1477),
                'verbose': -1,
                'seed': 3,
                'boosting_type': 'gbdt',
                'max_depth': -1,
                'learning_rate': 0.05,  # 0.05
                'metric': 'rmse',
                'device': 'cpu',
                'num_threads': 8
            }
        model = LightGBM(**params)
        if calc_on_selected_feat:
            train_df = self.keep_only_selected_features(
                'train', return_only=True)
            x = train_df.drop('target', axis=1).values
            x = np.concatenate([x, self.train_agg.values], axis=-1)
            target = train_df['target'].values
            y = np.array([target[i] in x[i]
                          for i in range(x.shape[0])]).astype(np.int)
        else:
            x, target = self.get_train_data(
                use_aggregates=True, logloss=False)
            y = np.array([target[i] in x[i]
                          for i in range(x.shape[0])]).astype(np.int)

        if test:
            if calc_on_selected_feat:
                test_df = self.keep_only_selected_features(
                    'test', return_only=True)
                x_test = test_df.values
            else:
                x_test = self.get_test_data()

            test_feat = model.cv_predict(x, y, x_test,
                                         random_seed=random_seed,
                                         logloss=False,
                                         oof_pred=False)
        else:
            model.cv(x, y,
                     random_seed=random_seed,
                     oof_pred=False)

        # Add to aggregates
        train_agg = pd.DataFrame(dict(is_label=y), index=self.train_df.index)
        self.train_agg = pd.concat([self.train_agg, train_agg], axis=1)

        if test:
            if threshold == 'hard':
                thresh = 0.5
            elif threshold == 'soft':
                thresh = 1572 / (2887 + 1572)
            else:
                raise (('Invalid threshold param, must be',
                        'either "hard" or "soft"'))

            test_feat[test_feat <= thresh] = 0
            test_feat[test_feat > thresh] = 1
            test_agg = pd.DataFrame(
                dict(is_label=test_feat), index=self.test_df.index)
            self.test_agg = pd.concat([self.test_agg, test_agg], axis=1)

    def get_validation_set_from_leaky_test(self, sub_path,
                                           logloss=False, return_index=False):
        '''Use leaky rows from test as validation set'''
        val_df = load_df_from_path(sub_path)
        idx = list(val_df[val_df.target > 0].index)
        val_df = val_df.target.values
        X_test = self.get_test_data()[val_df > 0]
        Y_test = val_df[val_df > 0]
        if logloss:
            Y_test = np.log1p(Y_test)

        if return_index:
            return idx, X_test, Y_test
        else:
            return X_test, Y_test

    # def get_train_data_plus_leaky_test_rows(self, **kwargs):
    #     '''Get leaky test rows as additional training data'''
    #     get_train_data

if __name__ == '__main__':
    train_path = './train.csv'
    test_path = './test.csv'

    dataset = KaggleDataset(train_path, test_path=test_path)
    print(dataset.train_df.describe(), dataset.test_df.describe())

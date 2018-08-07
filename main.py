import numpy as np
from Dataset import KaggleDataset
from Models import LightGBM, RNN_LSTM
from Submission import create_submission_file

LOAD_TEST = False
# Define paths and anything related to OS
train_path = './train.csv'
if LOAD_TEST:
    test_path = './test.csv'
else:
    test_path = None
# Load and preprocess Dataset
dataset = KaggleDataset(train_path, test_path=test_path)

dataset.compute_aggregates_for_all_features('both' if LOAD_TEST else 'train')

# dataset.compute_aggregates_for_selected_features(
#    'both' if LOAD_TEST else 'train')
#
# dataset.keep_only_selected_features('both' if LOAD_TEST else 'train')

# dataset.create_feature_as_targets()

# dataset.compute_aggregates_for_most_important('both' if LOAD_TEST else 'train',
#                                              num=75, importance_type='gain')

dataset.add_decomposition_as_features('both' if LOAD_TEST else 'train',
                                      n_components=50, method='fa',
                                      comp_stats=False,
                                      normalize=False)

dataset.add_decomposition_as_features('both' if LOAD_TEST else 'train',
                                      n_components=50, method='svd',
                                      comp_stats=False,
                                      normalize=False)

# dataset.compute_cluster_features('both' if LOAD_TEST else 'train',
#                                 iter_cluster=range(2, 7))

# dataset.compute_meta_aggregates('both' if LOAD_TEST else 'train')
# dataset.to_sparse(dataset='both' if LOAD_TEST else 'train')
# dataset.remove_constant_features()
# dataset.remove_duplicated_features()
# dataset.remove_different_distribution_features()

#dataset.add_IsTargetAvaliable_as_feature(test=True if LOAD_TEST else False,
#                                         threshold='soft',
#                                         verbose=True)

# %% Get data for trainning
TIME_SERIES = False
LOGLOSS = True
NORMALIZE = False
AGGREGATES = True
ONLY_AGGREGATES = False
ROUND_TARGETS = False
# DIM_TO_REDUCE = 50
# DIM_TO_KEEP = 50

if ONLY_AGGREGATES:
    X, Y = dataset.get_aggregates_as_data('train', logloss=LOGLOSS,
                                          round_targets=ROUND_TARGETS)

elif TIME_SERIES:
    X, Y = dataset.get_data_as_time_series('train', logloss=LOGLOSS,
                                           round_targets=ROUND_TARGETS)
else:
    X, Y = dataset.get_train_data(normalize=NORMALIZE, logloss=LOGLOSS,
                                  round_targets=ROUND_TARGETS,
                                  use_aggregates=AGGREGATES,
                                  # n_components=DIM_TO_KEEP,
                                  # reduce_dim_nb=0,
                                  reduce_dim_method='fa')
if LOAD_TEST:
    if ONLY_AGGREGATES:
        X_test = dataset.get_aggregates_as_data('test')
    elif TIME_SERIES:
        X_test = dataset.get_data_as_time_series('test')
    else:
        X_test = dataset.get_test_data()

# %% Split to train and val data
RANDOM_SEED = 143
NFOLD = 3
BAGGING = True
# Train model on KFold
MODEL_TYPE = 'LightGBM'     # Either LightGBM, XGBoost, CatBoost or LSTM


if MODEL_TYPE == 'LightGBM':
    LightGBM_params = dict(boosting='gbdt',
                           num_leaves=53, lr=0.005, bagging_fraction=0.71,
                           max_depth=8,
                           max_bin=255,
                           feature_fraction=0.23, bagging_freq=3,
                           min_data_in_leaf=12,  # 12
                           min_sum_hessian_in_leaf=1e-1,
                           use_missing=True, zero_as_missing=False,
                           lambda_l1=1e-1, lambda_l2=1,
                           device='cpu', num_threads=8)
    
    fit_params = dict(nfold=NFOLD,  ES_rounds=100,
                      steps=50000, random_seed=RANDOM_SEED,
                      bootstrap=BAGGING, bagging_size_ratio=1)
    
    model = LightGBM(**LightGBM_params)

elif MODEL_TYPE == 'LSTM':
    LSTM_params = dict(units=10,
                       layers=1,
                       lr=0.001,
                       lr_decay=0,
                       in_shape=(40, 1),
                       out_shape=1)
    
    fit_params = dict(nfold=NFOLD,  epochs=100,
                      mb_size=10, random_seed=RANDOM_SEED,
                      bootstrap=BAGGING, bagging_size_ratio=1,
                      scale_data=True, early_stop=True)
    
    model = RNN_LSTM(**LSTM_params)

if LOAD_TEST:
    pred = model.cv_predict(X, Y, X_test, **fit_params)

else:
    pred = model.cv(X, Y, **fit_params)


# %%Create submission file
if LOAD_TEST:
    create_submission_file(dataset.test_df.index, pred)

# %% Optimize Hyper params
OPTIMIZE = True
if OPTIMIZE:
    param_grid = {
#            'num_leaves': np.arange(8, 10, 1), 
#            'min_data_in_leaf': np.arange(5, 7, 1),
#            'max_depth': np.arange(1, 10, 2),
#                'bagging_fraction': np.arange(0.5, 0.9, 0.1),
             'bagging_freq': np.arange(1, 5, 1), 
#             'lambda_l1': [np.power(10, x) for x in np.arange(0.9, 1.2, 0.1).astype(float)],
#             'lambda_l2': [np.power(10, x) for x in np.arange(0.9, 1.2, 0.1).astype(float)],
#        'max_bin': range(100, 300, 50),
#                'feature_fraction': np.arange(0.6, 0.65, 0.025),
        #        'min_child_weight'
        #        'reg_lambda'
        #        'reg_alpha'
        #        'min_child_samples'

    }
    model.optmize_hyperparams(param_grid, X, Y, cv=3, verbose=2)

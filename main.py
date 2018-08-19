import numpy as np
from Dataset import KaggleDataset
from Models import LightGBM, RNN_LSTM, Ensembler, CatBoost, XGBoost
from Submission import create_submission_file

LOAD_TEST = True
# Define paths and anything related to OS
train_path = './train.csv'
if LOAD_TEST:
    test_path = './test.csv'
else:
    test_path = None
# Load and preprocess Dataset
dataset = KaggleDataset(train_path, test_path=test_path)

dataset.compute_aggregates_for_all_features('both' if LOAD_TEST else 'train')

dataset.compute_time_series_aggregates('both' if LOAD_TEST else 'train')

#dataset.remove_fillers_from_data('both' if LOAD_TEST else 'train', 20)

#dataset.compute_aggregates_for_selected_features(
#    'both' if LOAD_TEST else 'train')
#
#dataset.keep_only_selected_features('both' if LOAD_TEST else 'train')

# dataset.create_feature_as_targets()

# dataset.compute_aggregates_for_most_important('both' if LOAD_TEST else 'train',
#                                              num=75, importance_type='gain')

#dataset.add_IsTargetAvaliable_as_feature(test=True if LOAD_TEST else False,
#                                         threshold='soft',
#                                         verbos#dataset.add_decomposition_as_features('both' if LOAD_TEST else 'train',
#                                      n_components=50, method='fa',
#                                      comp_stats=False,
#                                      normalize=False)
#
#dataset.add_decomposition_as_features('both' if LOAD_TEST else 'train',
#                                      n_components=50, method='svd',
#                                      comp_stats=False,
#                                      normalize=False)e=True,
#                                         calc_on_selected_feat=False)
###


# dataset.compute_cluster_features('both' if LOAD_TEST else 'train',
#                                 iter_cluster=range(2, 7))

# dataset.compute_meta_aggregates('both' if LOAD_TEST else 'train')
# dataset.to_sparse(dataset='both' if LOAD_TEST else 'train')
# dataset.remove_constant_features()
# dataset.remove_duplicated_features()
# dataset.remove_different_distribution_features()

# %% Get data for trainning
VAL_FROM_LEAKY_TEST_ROWS = True
TRAIN_WITH_LEAKY_ROWS = False
LEAKY_TEST_SUB_PATH = 'baseline_sub_lag_37.csv'
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
                                  reduce_dim_method='fa',
                                  get_leaky_data=LEAKY_TEST_SUB_PATH if TRAIN_WITH_LEAKY_ROWS else None)
if LOAD_TEST:
    if ONLY_AGGREGATES:
        X_test = dataset.get_aggregates_as_data('test')
    elif TIME_SERIES:
        X_test = dataset.get_data_as_time_series('test')
    else:
        X_test = dataset.get_test_data()

if VAL_FROM_LEAKY_TEST_ROWS:
    X_val, Y_val = dataset.get_validation_set_from_leaky_test(
        LEAKY_TEST_SUB_PATH, logloss=LOGLOSS)

# %% Split to train and val data
RANDOM_SEED = 143
NFOLD = 5
BAGGING = False
# Train model on KFold
MODEL_TYPE = 'LightGBM'     # Either LightGBM, XGBoost, CatBoost or LSTM


if MODEL_TYPE == 'LightGBM':
    LightGBM_params = dict(boosting='gbdt',
                           num_leaves=53, lr=0.0039, bagging_fraction=0.71,
                           max_depth=8, 
                           max_bin=201,
                           feature_fraction=0.23, bagging_freq=3,
                           min_data_in_leaf=12,  # 12
                           min_sum_hessian_in_leaf=1e-1,
                           use_missing=True, zero_as_missing=False,
                           lambda_l1=1e-1, lambda_l2=1,
                           device='cpu', num_threads=4)

    fit_params = dict(nfold=NFOLD,  ES_rounds=100,
                      steps=50000, random_seed=RANDOM_SEED,
                      bootstrap=BAGGING, bagging_size_ratio=1)

    model = LightGBM(**LightGBM_params)

if MODEL_TYPE == 'XGBoost':
    XGBoost_params = dict(booster='gbtree',
                          silent=True,
                          max_leaves=20,
                          lr=0.003,
                          subsample=0.5,
                          max_depth=20, 
                          max_bin=256,
                          colsample_bytree=0.15,
                          colsample_bylevel = 0.5, 
                          gamma = 1.5,
                          min_child_weight=57,
                          reg_alpha=1e-1, reg_lambda=1,
                          tree_method='auto', nthread=11)

    fit_params = dict(nfold=NFOLD,  ES_rounds=10,
                      steps=50000, random_seed=RANDOM_SEED,
                      bootstrap=BAGGING, bagging_size_ratio=1)

    model = XGBoost(**XGBoost_params)
    
elif MODEL_TYPE == 'CatBoost':
    CatBoost_params = dict(objective='RMSE', eval_metric='RMSE',
                           iterations=10000,  random_seed=RANDOM_SEED,
                           l2_leaf_reg=3.5,
                           bootstrap_type='Bayesian', bagging_temperature=16,
                           rsm=0.51,
                           border_count=15,
                           learning_rate=0.03,
                           nan_mode='Min',
                           use_best_model=True,
                           max_depth=6,
                           task_type='CPU', thread_count=11,
                           verbose=True)

    fit_params = dict(nfold=NFOLD,  ES_rounds=30,
                      random_seed=RANDOM_SEED,
                      bootstrap=BAGGING, bagging_size_ratio=1,
                      verbose=100)

    model = CatBoost(**CatBoost_params)

elif MODEL_TYPE == 'LSTM':
    LSTM_params = dict(units=100,
                       layers=10,
                       lr=0.001,
                       lr_decay=0,
                       in_shape=(40, 81),
                       out_shape=1)

    fit_params = dict(nfold=NFOLD,  epochs=1000,
                      mb_size=1000, random_seed=RANDOM_SEED,
                      bootstrap=BAGGING, bagging_size_ratio=1,
                      scale_data=True, early_stop=True)

    model = RNN_LSTM(**LSTM_params)

if LOAD_TEST:
    if VAL_FROM_LEAKY_TEST_ROWS:
        _, pred, oof_pred = model.fit_predict(X, Y, X_test,
                                              val_X=X_val, val_y=Y_val,
                                              logloss=True,
                                              return_oof_pred=True)
    else:
        pred, oof_pred = model.cv_predict(X, Y, X_test,
                                          logloss=True, return_oof_pred=True,
                                          **fit_params)

else:
    if VAL_FROM_LEAKY_TEST_ROWS:
        _, pred, oof_pred = model.fit(X, Y,
                                      val_X=X_val, val_y=Y_val,
                                      logloss=True,
                                      return_oof_pred=True)
    else:
        oof_pred = model.cv(X, Y, 
                            return_oof_pred=True, **fit_params)


# %%Create submission file
if LOAD_TEST:
    NAME = '5FoldFull_CatBoost_GAgg_TSAgg_1pt337'
    create_submission_file(dataset.test_df.index, pred, './Stacking/Test/Test_{}.csv'.format(NAME))
    create_submission_file(range(len(oof_pred)), oof_pred, './Stacking/Train/Train_{}.csv'.format(NAME))

# %% Optimize Hyper params
OPTIMIZE = False
if OPTIMIZE:
    param_grid = {
        #            'num_leaves': np.arange(8, 10, 1),
        #            'min_data_in_leaf': np.arange(5, 7, 1),
        #            'max_depth': np.arange(1, 10, 2),
        'bagging_fraction': np.arange(0.78, 0.82, 0.01),
        #             'bagging_freq': np.arange(1, 5, 1),
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

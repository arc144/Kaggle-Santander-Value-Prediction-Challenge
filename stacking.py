import numpy as np
import pandas as pd
import os
from Dataset import KaggleDataset
from Models import LightGBM, RNN_LSTM, Ensembler, CatBoost, LiRegression
from Submission import create_submission_file, load_submissions_as_data_for_ensembling, merge_leaky_and_ML_sub

TRAIN_NAMES = [
     'Train_5FoldFull_CatBoost_GAgg_1pt351.csv',
     'Train_5FoldFull_CatBoost_GAgg_isLabel_1pt346.csv',
#     'Train_5FoldFull_LightGBM_GAgg_TSAgg_1pt326.csv',
     'Train_5FoldFull_LightGBM_GAgg_1pt336.csv',
     'Train_5FoldFull_LightGBM_GAgg_50svd50FA_isLabel_1pt331.csv',
     'Train_5FoldFull_LightGBM_OnlySelected_GAgg4Selected_isLabel_1pt335.csv',
     'Train_5FoldFull_LightGBM_raw_1pt402.csv',
     'Train_5FoldFull_LightGBM_GAgg_TSAgg_1pt327.csv'
    ]

TEST_NAMES = [
     'Test_5FoldFull_CatBoost_GAgg_1pt351.csv',
     'Test_5FoldFull_CatBoost_GAgg_isLabel_1pt346.csv',  
#     'Test_5FoldFull_LightGBM_GAgg_TSAgg_1pt326.csv',
     'Test_5FoldFull_LightGBM_GAgg_1pt336.csv',
     'Test_5FoldFull_LightGBM_GAgg_50svd50FA_isLabel_1pt331.csv',
     'Test_5FoldFull_LightGBM_OnlySelected_GAgg4Selected_isLabel_1pt335.csv',
     'Test_5FoldFull_LightGBM_raw_1pt402.csv',
     'Test_5FoldFull_LightGBM_GAgg_TSAgg_1pt327.csv'
              ]

# =============================================================================
# TRAIN_NAMES = [
#      'Train_CatBoost_GAgg_1pt377.csv',
#      'Train_CatBoost_OnlySelected_GAgg4Selected_1pt390.csv',
#      'Train_LightGBM_GAgg_1pt375.csv',
#      'Train_LightGBM_GAgg_50FA50tSVD_isLabel_1pt381.csv',
#      'Train_LightGBM_GAgg_TSAgg_1pt354.csv',
#      'Train_LightGBM_raw_1pt430.csv',
#      ]
# 
# 
# TEST_NAMES = [
#      'Test_CatBoost_GAgg_1pt377.csv',
#      'Test_CatBoost_OnlySelected_GAgg4Selected_1pt390.csv',
#      'Test_LightGBM_GAgg_1pt375.csv',
#      'Test_LightGBM_GAgg_50FA50tSVD_isLabel_1pt381.csv',
#      'Test_LightGBM_GAgg_TSAgg_1pt354.csv',
#      'Test_LightGBM_raw_1pt430.csv',
#       ]
# =============================================================================
              
     
TRAIN_PATH = './Stacking/Train/'
TEST_PATH = './Stacking/Test/'
TRAIN_LIST = [os.path.join(TRAIN_PATH, name) for name in TRAIN_NAMES]
TEST_LIST = [os.path.join(TEST_PATH, name) for name in TEST_NAMES]

#Y_TRAIN = './Stacking/Train/leaky_rows_Y.npy'
Y_TRAIN = './Stacking/Train/fullTrain_Y.npy'
LEAKY_SUB_NAME = 'baseline_sub_lag_37.csv'

LOAD_TEST = True
USE_LEAK = True

X, y = load_submissions_as_data_for_ensembling(TRAIN_LIST, Y_TRAIN)
if LOAD_TEST:
    X_test = load_submissions_as_data_for_ensembling(TEST_LIST)
    X_test = np.log1p(X_test)

# %% Split to train and val data
RANDOM_SEED = 143
NFOLD = 3
BAGGING = True
# Train model on KFold
MODEL_TYPE = 'LiRegression'     # Either LightGBM, XGBoost, CatBoost, LiRegression


if MODEL_TYPE == 'LightGBM':
    LightGBM_params = dict(boosting='gbdt',
                           num_leaves=5, lr=0.0039, bagging_fraction=0.6,
                           max_depth=1,
                           max_bin=201,
                           feature_fraction=0.6, bagging_freq=3,
                           min_data_in_leaf=50,
                           min_sum_hessian_in_leaf=10,
                           use_missing=True, zero_as_missing=False,
                           lambda_l1=10, lambda_l2=10,
                           device='gpu', num_threads=11)

    fit_params = dict(nfold=NFOLD,  ES_rounds=100,
                      steps=50000, random_seed=RANDOM_SEED,
                      bootstrap=BAGGING, bagging_size_ratio=1)

    model = LightGBM(**LightGBM_params)

elif MODEL_TYPE == 'CatBoost':
    CatBoost_params = dict(objective='RMSE', eval_metric='RMSE',
                           iterations=10000,  random_seed=RANDOM_SEED,
                           l2_leaf_reg=3.4,
                           bootstrap_type='Bayesian', bagging_temperature=1,
                           rsm=0.6,
                           border_count=30,
                           learning_rate=0.03,
                           nan_mode='Min',
                           use_best_model=True,
                           max_depth=8,
                           task_type='CPU', thread_count=11,
                           verbose=True)

    fit_params = dict(nfold=NFOLD,  ES_rounds=30,
                      random_seed=RANDOM_SEED,
                      bootstrap=BAGGING, bagging_size_ratio=1,
                      verbose=100)

    model = CatBoost(**CatBoost_params)

elif MODEL_TYPE == 'LiRegression':
    LiRegression_params = dict(normalize=True)
    
    fit_params = dict(nfold=NFOLD,
                      random_seed=RANDOM_SEED,
                      bootstrap=BAGGING, bagging_size_ratio=1,
                      verbose=100)
    
    model = LiRegression(**LiRegression_params)
    
if LOAD_TEST:
    pred = model.cv_predict(X, y, X_test, logloss=True, **fit_params)

else:
    pred = model.cv(X, y, **fit_params)


# %%Create submission file
if LOAD_TEST:
    SUB_NAME = 'sub_stacking'
    test_index = pd.read_csv(TEST_LIST[0]).ID
    create_submission_file(test_index, pred, '{}.csv'.format(SUB_NAME))
    if USE_LEAK:
        merge_leaky_and_ML_sub(LEAKY_SUB_NAME,
                               '{}.csv'.format(SUB_NAME),
                               '{}_with_leak.csv'.format(SUB_NAME))

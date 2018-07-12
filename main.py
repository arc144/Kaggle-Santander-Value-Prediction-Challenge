import numpy as np
from Dataset import KaggleDataset
from Models import LightGBM
from Submission import create_submission_file

LOAD_TEST = True
# Define paths and anything related to OS
train_path = './train.csv'
if LOAD_TEST:
    test_path = './test.csv'
else:
    test_path = None
# Load and preprocess Dataset
dataset = KaggleDataset(train_path, test_path=test_path, aggregates=True)
dataset.compute_aggregates_for_most_important('both' if LOAD_TEST else 'train',
                                              num=75, importance_type='gain')

dataset.add_decomposition_as_features('both' if LOAD_TEST else 'train',
                                      n_components=75, method='fa',
                                      comp_stats=False,
                                      normalize=False)

dataset.add_decomposition_as_features('both' if LOAD_TEST else 'train',
                                      n_components=50, method='svd',
                                      comp_stats=False,
                                      normalize=False)


# dataset.compute_meta_aggregates('both' if LOAD_TEST else 'train')
# dataset.to_sparse(dataset='both' if LOAD_TEST else 'train')
# dataset.remove_constant_features()
# dataset.remove_duplicated_features()
# dataset.remove_different_distribution_features()

#%% Get data for trainning
NORMALIZE = False
AGGREGATES = True
ONLY_AGGREGATES = True
# DIM_TO_REDUCE = 50
# DIM_TO_KEEP = 50

if ONLY_AGGREGATES:
    X, Y = dataset.get_aggregates_as_data('train')
else:
    X, Y = dataset.get_train_data(normalize=NORMALIZE, logloss=True,
                              use_aggregates=AGGREGATES,
#                              n_components=DIM_TO_KEEP,
                              #reduce_dim_nb=0,
                              reduce_dim_method='fa')
if LOAD_TEST:
    if ONLY_AGGREGATES:
        X_test = dataset.get_aggregates_as_data('test')
    else:
        X_test = dataset.get_test_data()

#%% Split to train and val data
RANDOM_SEED = 143
NFOLD = 3
BAGGING = True
# Train model on KFold
MODEL_TYPE = 'LightGBM'     # Either LightGBM, XGBoost or CatBoost


LightGBM_params = dict(boosting='gbdt',
                       num_leaves=53, lr=0.005, bagging_fraction=0.67,
                       max_depth=8,
                       feature_fraction=0.35, bagging_freq=3,
                       min_data_in_leaf=12,
                       min_sum_hessian_in_leaf=1e-1,
                       use_missing=True, zero_as_missing=False,
                       lambda_l1=1e-1, lambda_l2=1,
                       device='cpu', num_threads=8)

if MODEL_TYPE == 'LightGBM':
    model = LightGBM(**LightGBM_params)

if LOAD_TEST:
    pred = model.cv_predict(X, Y, X_test, nfold=NFOLD,  ES_rounds=100,
                            steps=5000, random_seed=RANDOM_SEED,
                            bootstrap=BAGGING, bagging_size_ratio=1)
  
else:
    pred = model.cv(X, Y, nfold=NFOLD,  ES_rounds=100,
                    steps=5000, random_seed=RANDOM_SEED,
                    bootstrap=BAGGING, bagging_size_ratio=1)


# %%Create submission file
if LOAD_TEST:
    create_submission_file(dataset.test_df.index, pred)

# %% Optimize Hyper params
OPTIMIZE = False
if OPTIMIZE:
    param_grid = {
        'bagging_freq': range(0, 5),
    }
    model.optmize_hyperparams(param_grid, X, Y, cv=4, verbose=2)
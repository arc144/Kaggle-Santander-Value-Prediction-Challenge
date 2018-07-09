import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from Dataset import KaggleDataset
from Models import LightGBM
from lightgbm import LGBMModel
from Submission import create_submission_file

# Define paths and anything related to OS
train_path = './train.csv'
test_path = './test.csv'
# Load and preprocess Dataset
dataset = KaggleDataset(train_path, test_path=None)
#dataset.to_sparse(dataset='train')
#dataset.remove_constant_features()
#dataset.remove_duplicated_features()
# dataset.remove_different_distribution_features()
#%% Get data for trainning
NORMALIZE = False
AGGREGATES = True
# DIM_TO_REDUCE = 50
#DIM_TO_KEEP = 50
X, Y = dataset.get_train_data(normalize=NORMALIZE, logloss=True,
                              use_aggregates=AGGREGATES,
                              #                              n_components=DIM_TO_KEEP,
                              #                              reduce_dim_nb=0,
                              reduce_dim_method='fa')
# low_rep_X = dataset.reduce_dimensionality(X,
#                                          n_components=DIM_TO_KEEP,
#                                          method='fa',
#                                          fit=True)
#
#X = np.concatenate([X, low_rep_X], axis=-1)

#%% Split to train and val data
RANDOM_SEED = 143
NFOLD = 5

# Train model on KFold
MODEL_TYPE = 'LightGBM'     # Either LightGBM, XGBoost or CatBoost

LightGBM_params = dict(num_leaves=53, lr=0.005, bagging_fraction=0.67,
                       feature_fraction=0.35, bagging_frequency=6,
                       min_data_in_leaf=21,
                       use_missing=True, zero_as_missing=False,
                       lambda_l1=0.1, lambda_l2=10,
                       device='cpu', num_threads=8)

if MODEL_TYPE == 'LightGBM':
    model = LightGBM(**LightGBM_params)

model.cv(X, Y, nfold=NFOLD,  ES_rounds=100,
         steps=5000, RANDOM_SEED=RANDOM_SEED,
         bootstrap=True, split_rate=0.8)

# %% GRIDSEARCHCV
param_test1 = {
    'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100],
}
params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 53,
    "min_child_samples ": 21,
    "learning_rate": 0.05,
    "subsample ": 0.67,
    "colsample_bytree ": 0.12,
    "subsample_freq ": 6,
    "bagging_seed": 42,
    "verbosity": -1,
    "seed": 42,
    "device": 'cpu',
    "gpu_platform_id":  0,
    "gpu_device_id":  0,
    "silent": False,
    "n_jobs": 8,
}
gsearch1 = GridSearchCV(estimator=LGBMModel(**params),
                        param_grid=param_test1, scoring='neg_mean_squared_error', n_jobs=1, iid=False, cv=4)
gsearch1.fit(X, Y)
gsearch1.grid_scores_, gsearch1.best_params_, np.sqrt(-gsearch1.best_score_)

# %%Re-train model with only 1% of data for validation
x_train, x_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.01, random_state=RANDOM_SEED)
evals_result = model.fit(train_X=x_train, train_y=y_train,
                         val_X=x_val, val_y=y_val,
                         steps=1000)

# %%Create submission file
X_test = dataset.get_test_data()
# low_rep_X_test = dataset.reduce_dimensionality(X_test,
#                                               n_components=DIM_TO_KEEP,
#                                               method='fa',
#                                               fit=False)
#
#X_test = np.concatenate([X_test, low_rep_X_test], axis=-1)

pred = model.predict(X_test, logloss=True)
create_submission_file(dataset.test_df.index, pred)

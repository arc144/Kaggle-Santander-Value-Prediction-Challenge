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
# MUST BE CHANGED FOR SUBMISSION
dataset = KaggleDataset(train_path, test_path=None)
dataset.remove_constant_features()
# Get data for trainning
NORMALIZE = True
DIM_TO_REDUCE = 0
X, Y = dataset.get_train_data(normalize=NORMALIZE, logloss=True,
                              reduce_dim_nb=DIM_TO_REDUCE,
                              reduce_dim_method='svd')
#%% Split to train and val data
RANDOM_SEED = 143
K = 10
kf = KFold(n_splits=K, shuffle=True, random_state=RANDOM_SEED)

# Train model on KFold
MODEL_TYPE = 'LightGBM'     # Either LightGBM, XGBoost or CatBoost
LightGBM_params = dict(num_leaves=40, lr=0.05, bagging_fraction=0.7,
                       feature_fraction=0.6, bagging_frequency=6,
                       min_data_in_leaf=21, device='cpu')

kFold_results = []
for train_index, val_index in kf.split(X, y=Y):
    x_train = X[train_index]
    y_train = Y[train_index]
    x_val = X[val_index]
    y_val = Y[val_index]

    if MODEL_TYPE == 'LightGBM':
        model = LightGBM(**LightGBM_params)

    evals_result = model.fit(train_X=x_train, train_y=y_train,
                             val_X=x_val, val_y=y_val,
                             ES_rounds=100)
    kFold_results.append(np.array(evals_result['valid_1']['rmse']).min())
    
kFold_results = np.array(kFold_results)
print('K-Fold val error: ', kFold_results)
print('Mean val error: ', kFold_results.mean())

# %% GRIDSEARCHCV
param_test1 = {
 'subsample': np.arange(0.4, 1, 0.2),
 'subsample_freq': range(0, 10, 2)
}
params = {
    "objective": "regression",
    "metric": "rmse",
    "num_leaves": 40,
    "min_child_samples ": 21,
    "learning_rate": 0.05,
    "subsample ": 0.7,
    "colsample_bytree ": 0.1,
    "subsample_freq ": 6,
    "bagging_seed": 42,
    "verbosity": -1,
    "seed": 42,
    "device": 'cpu',
    "gpu_platform_id":  0,
    "gpu_device_id":  0,
    "silent": 0,
        }
gsearch1 = GridSearchCV(estimator=LGBMModel(**params), 
 param_grid = param_test1, scoring='neg_mean_squared_error', n_jobs=1, iid=False, cv=4)
gsearch1.fit(X, Y)
gsearch1.grid_scores_, gsearch1.best_params_, np.sqrt(gsearch1.best_score_)

# %%Re-train model with only 1% of data for validation
x_train, x_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.01, random_state=RANDOM_SEED)
evals_result = model.fit(train_X=x_train, train_y=y_train,
                         val_X=x_val, val_y=y_val,
                         steps=1000)

# %%Create submission file
X = dataset.get_test_data()
pred = model.predict(X, logloss=True)
create_submission_file(dataset.test_df.index, pred)

import numpy as np
import lightgbm as lgb
# import xgboost as xgb
from catboost import CatBoostRegressor


class LightGBM():
    '''Microsoft LightGBM class wrapper'''

    def __init__(self, num_leaves=40, lr=0.005, bagging_fraction=0.7,
                 feature_fraction=0.6, bagging_frequency=6, device='gpu',
                 **kwargs):
        self.params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": num_leaves,
            "learning_rate": lr,
            "bagging_fraction": bagging_fraction,
            "feature_fraction": feature_fraction,
            "bagging_frequency": bagging_frequency,
            "bagging_seed": 42,
            "verbosity": -1,
            "seed": 42,
            "device": device,
            "gpu_platform_id":  0,
            "gpu_device_id":  0,
        }
        for key, value in kwargs.items():
            self.params[key] = value

    def fit(self, train_X, train_y, val_X, val_y, ES_rounds=100, steps=5000):
        # Train LGB model
        lgtrain = lgb.Dataset(train_X, label=train_y)
        lgval = lgb.Dataset(val_X, label=val_y)
        evals_result = {}
        self.model = lgb.train(self.params, lgtrain,
                               num_boost_round=steps,
                               valid_sets=[lgtrain, lgval],
                               early_stopping_rounds=ES_rounds,
                               verbose_eval=150,
                               evals_result=evals_result)
        return evals_result

    def predict(self, test_X, logloss=True):
        '''Predict using a fitted model'''
        pred_y = self.model.predict(test_X, num_iteration=self.model.best_iteration)
        if logloss:
            pred_y = np.expm1(pred_y)
        return pred_y

    def fit_predict(self, train_X, train_y, val_X, val_y,
                    test_X, logloss=True):
        evals_result = self.fit(train_X, train_y, val_X, val_y)
        pred_y = self.predict(test_X, logloss)
        return evals_result, pred_y

import numpy as np
import lightgbm as lgb
# import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold, train_test_split, GridSearchCV


def generate_bagging_splits(n_size, nfold, bagging_size_ratio=1, random_seed=143):
    '''Generate random bagging splits'''
    np.random.seed(random_seed)
    ref = range(n_size)
    out_size = int(bagging_size_ratio * n_size)

    splits = []
    for _ in range(nfold):
        t_index = np.random.randint(0, n_size, size=out_size)
        v_index = [j for j in ref if j not in t_index]
        splits.append((t_index, v_index))

    return splits


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

    def fit(self, train_X, train_y, val_X, val_y, ES_rounds=100, steps=5000,
            verbose=150):
        # Train LGB model
        lgtrain = lgb.Dataset(train_X, label=train_y)
        lgval = lgb.Dataset(val_X, label=val_y)
        evals_result = {}
        self.model = lgb.train(self.params, lgtrain,
                               num_boost_round=steps,
                               valid_sets=[lgtrain, lgval],
                               early_stopping_rounds=ES_rounds,
                               verbose_eval=verbose,
                               evals_result=evals_result)
        return evals_result

    def cv(self, X, Y, nfold=5,  ES_rounds=100, steps=5000, random_seed=143,
           bootstrap=False, bagging_size_ratio=1):
        # Train LGB model using CV
        if bootstrap:
            splits = generate_bagging_splits(
                X.shape[0], nfold,
                bagging_size_ratio=bagging_size_ratio,
                random_seed=random_seed)

        else:
            kf = KFold(n_splits=nfold, shuffle=True, random_state=random_seed)
            splits = kf.split(X, y=Y)

        kFold_results = []
        for train_index, val_index in splits:
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]

            evals_result = self.fit(train_X=x_train, train_y=y_train,
                                    val_X=x_val, val_y=y_val,
                                    ES_rounds=100,
                                    steps=10000)
            kFold_results.append(
                np.array(evals_result['valid_1']['rmse']).min())

        kFold_results = np.array(kFold_results)
        print('Mean val error: {}, std {} '.format(
            kFold_results.mean(), kFold_results.std()))

    def cv_predict(self, X, Y, test_X, nfold=5,  ES_rounds=100, steps=5000,
                   random_seed=143, logloss=True,
                   bootstrap=False, bagging_size_ratio=1):
        '''Fit model using CV and predict test using the average
         of all folds'''
        if bootstrap:
            splits = generate_bagging_splits(
                X.shape[0], nfold,
                bagging_size_ratio=bagging_size_ratio,
                random_seed=random_seed)

        else:
            kf = KFold(n_splits=nfold, shuffle=True, random_state=random_seed)
            splits = kf.split(X, y=Y)

        kFold_results = []
        for i, (train_index, val_index) in enumerate(splits):
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]

            evals_result = self.fit(train_X=x_train, train_y=y_train,
                                    val_X=x_val, val_y=y_val,
                                    ES_rounds=100,
                                    steps=10000)
            kFold_results.append(
                np.array(evals_result['valid_1']['rmse']).min())

            # Get predictions
            if not i:
                pred_y = self.predict(test_X, logloss=logloss)
            else:
                pred_y += self.predict(test_X, logloss=logloss)

        kFold_results = np.array(kFold_results)
        print('Mean val error: {}, std {} '.format(
            kFold_results.mean(), kFold_results.std()))

        # Divide pred by the number of folds and return
        return pred_y / nfold

    def predict(self, test_X, logloss=True):
        '''Predict using a fitted model'''
        pred_y = self.model.predict(
            test_X, num_iteration=self.model.best_iteration)
        if logloss:
            pred_y = np.expm1(pred_y)
        return pred_y

    def fit_predict(self, train_X, train_y, val_X, val_y,
                    test_X, logloss=True):
        evals_result = self.fit(train_X, train_y, val_X, val_y)
        pred_y = self.predict(test_X, logloss)
        return evals_result, pred_y

    def optmize_params(self, param_grid, X, Y, cv=4, verbose=1):
        '''Use GridSearchCV to optimize models params'''
        param_test1 = {
            'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100],
        }

        gsearch1 = GridSearchCV(estimator=lgb.LGBMModel(**self.params),
                                param_grid=param_test1,
                                scoring='neg_mean_squared_error',
                                n_jobs=1,
                                iid=False,
                                cv=4)
        gsearch1.fit(X, Y)
        scores = gsearch1.grid_scores_
        best_params = gsearch1.best_params_
        best_score = np.sqrt(-gsearch1.best_score_)
        if verbose > 0:
            if verbose > 1:
                print('Scores are: ', scores)
            print('Best params: ', best_params)
            print('Best score: ', best_score)

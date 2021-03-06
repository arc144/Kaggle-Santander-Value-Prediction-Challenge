import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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


def assert_time_series_dims(data):
    '''Assert TS dims are of the format (rows, time-steps, features)'''
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    elif len(data.shape) == 3:
        pass
    else:
        raise(('Invalid data dims, expected' +
               '2 or 3, but got {}'.format(data.shape)))
    return data


class NonFittedError(Exception):

    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors


class Ensembler():
    '''Class responsible to ensemble several classifiers'''

    def __init__(self, list_of_models):
        self.list_of_models = list_of_models

    def fit(self, train_X, train_y, val_X, val_y, ES_rounds=100, steps=5000,
            verbose=150, oof_pred=False):
        # Fit an ensembler model using by averaging predictions
        # Train individual models
        oof_predictions = []
        for model in self.list_of_models:
            _, oof_pred = model.fit(train_X, train_y, val_X, val_y,
                                    ES_rounds=100, steps=5000,
                                    verbose=150, oof_pred=True)
            oof_predictions.append(oof_pred)

        pred_y = np.average(oof_predictions, axis=0)
        rmse = np.sqrt(mean_squared_error(val_y, pred_y))
        print('Root mean squared error ', rmse)
        return pred_y

    def cv(self, X, Y, ensembler_model=None, nfold=5, ES_rounds=100, steps=5000, random_seed=143,
           bootstrap=False, bagging_size_ratio=1, oof_pred=False):
        # Use CV or Bagging to train the Ensembler model
        #  First get oof predictions
        oof_predictions = []
        for model in self.list_of_models:
            oof_pred = model.cv(X, Y, nfold=5, ES_rounds=100, steps=5000, random_seed=random_seed,
                                bootstrap=False, bagging_size_ratio=1, shuffle=False, oof_pred=True)
            oof_predictions.append(oof_pred)
        oof_predictions = np.array(oof_predictions)
        # Perform average
        if ensembler_model is None:
            pred_y = np.average(oof_predictions, axis=0)
            rmse = np.sqrt(mean_squared_error(Y, pred_y))
            print('Root mean squared error ', rmse)
        # else:
            # ens_train_X = oof_predictions.T
            # Train ensembler model on individual predictions
            # if bootstrap:
            #     splits = generate_bagging_splits(
            #         X.shape[0], nfold,
            #         bagging_size_ratio=bagging_size_ratio,
            #         random_seed=random_seed)

            # else:
            #     kf = KFold(n_splits=nfold, shuffle=True, random_state=random_seed)
            #     splits = kf.split(X, y=Y)

            # kFold_results = []
            # oof_results = []
            # for train_index, val_index in splits:
            #     x_train = X[train_index]
            #     y_train = Y[train_index]
            #     x_val = X[val_index]
            #     y_val = Y[val_index]

            #     evals_result, oof_prediction = self.fit(train_X=x_train, train_y=y_train,
            #                                             val_X=x_val, val_y=y_val,
            #                                             ES_rounds=100,
            #                                             steps=10000,
            #                                             oof_pred=oof_pred)
            #     if oof_pred:
            #         oof_results.extend(oof_prediction)
            #     if evals_result:
            #         kFold_results.append(
            #             np.array(
            #                 self.get_best_metric(
            #                     evals_result['valid_1'][self.params['metric']])))

            # kFold_results = np.array(kFold_results)
            # if kFold_results.size > 0:
            #     print('Mean val error: {}, std {} '.format(
            #         kFold_results.mean(), kFold_results.std()))
            # if oof_pred:
            #     return np.array(oof_results)

    def predict(self, test_X, logloss=True):
        '''Predict using a fitted model'''
        predictions = []
        for model in self.list_of_models:
            predictions.append(model.predict(test_X, logloss=logloss))
        if self.ensembler_model is not None:
            pred_y = self.ensembler_model.predict(
                np.array(predictions), logloss=False)
        else:
            pred_y = np.average(predictions, axis=0)

        return pred_y


class LightGBM():
    '''Microsoft LightGBM class wrapper'''

    def __init__(self, objective='regression', metric='rmse',
                 num_leaves=40, lr=0.005, bagging_fraction=0.7,
                 feature_fraction=0.6, bagging_frequency=6, device='gpu',
                 **kwargs):
        self.params = {
            "objective": objective,
            "metric": metric,
            "num_leaves": num_leaves,
            "learning_rate": lr,
            "bagging_fraction": bagging_fraction,
            "feature_fraction": feature_fraction,
            "bagging_frequency": bagging_frequency,
            "bagging_seed": 42,
            "verbosity": -1,
            "seed": 42,
            "device": device,
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
        }
        for key, value in kwargs.items():
            self.params[key] = value

        if self.params['metric'] in ['auc', 'binary_logloss', 'multi_logloss']:
            self.get_best_metric = max
        else:
            self.get_best_metric = min

    def fit(self, train_X, train_y, val_X, val_y, ES_rounds=100, steps=5000,
            verbose=150, return_oof_pred=True, **kwargs):
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
        if return_oof_pred:
            pred = self.predict(val_X, logloss=False)
        else:
            pred = None
        return evals_result, pred

    def cv(self, X, Y, nfold=5, ES_rounds=100, steps=5000, random_seed=143,
           bootstrap=False, bagging_size_ratio=1, shuffle=True,
           stacking_mode='mean', return_oof_pred=False, splits=None):
        # Train LGB model using CV
        assert stacking_mode in ['mean', 'LinearRegression']
        if splits is None:
            if bootstrap:
                splits = generate_bagging_splits(
                    X.shape[0], nfold,
                    bagging_size_ratio=bagging_size_ratio,
                    random_seed=random_seed)

            else:
                kf = KFold(n_splits=nfold, shuffle=shuffle,
                           random_state=random_seed)
                splits = kf.split(X, y=Y)

        oof_results = []
        y_true = []
        for train_index, val_index in splits:
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]

            _, oof_prediction = self.fit(train_X=x_train, train_y=y_train,
                                         val_X=x_val, val_y=y_val,
                                         ES_rounds=100,
                                         steps=10000,
                                         return_oof_pred=True)

            oof_results.extend(oof_prediction)
            y_true.extend(y_val)

        if stacking_mode == 'mean':
            y_fold_pred = np.mean(oof_prediction, axis=0)
            results = np.sqrt(mean_squared_error(y_true, y_fold_pred))

        print('Mean val error: {}'.format(results))

        if return_oof_pred:
            return np.array(oof_results)

    def cv_predict(self, X, Y, test_X, nfold=5, ES_rounds=100, steps=5000,
                   random_seed=143, logloss=False,
                   bootstrap=False, bagging_size_ratio=1,
                   return_oof_pred=False, splits=None):
        '''Fit model using CV and predict test using the average
         of all folds'''
        if splits is None:
            if bootstrap:
                splits = generate_bagging_splits(
                    X.shape[0], nfold,
                    bagging_size_ratio=bagging_size_ratio,
                    random_seed=random_seed)

            else:
                kf = KFold(n_splits=nfold, shuffle=True,
                           random_state=random_seed)
                splits = kf.split(X, y=Y)

        kFold_results = []
        oof_results = []
        for i, (train_index, val_index) in enumerate(splits):
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]

            evals_result, oof_prediction = self.fit(train_X=x_train, train_y=y_train,
                                                    val_X=x_val, val_y=y_val,
                                                    ES_rounds=100,
                                                    steps=10000,
                                                    return_oof_pred=return_oof_pred)
            if return_oof_pred:
                oof_results.extend(oof_prediction)
            if evals_result:
                kFold_results.append(
                    np.array(
                        self.get_best_metric(
                            evals_result['valid_1'][self.params['metric']])))

            # Get predictions
            if not i:
                pred_y = self.predict(test_X, logloss=logloss)
            else:
                pred_y += self.predict(test_X, logloss=logloss)

        kFold_results = np.array(kFold_results)
        if kFold_results.size > 0:
            print('Mean val error: {}, std {} '.format(
                kFold_results.mean(), kFold_results.std()))

        # Divide pred by the number of folds and return
        if return_oof_pred:
            return pred_y / nfold, np.array(oof_results)
        return pred_y / nfold

    def multi_seed_cv_predict(self, X, Y, test_X, nfold=5, ES_rounds=100,
                              steps=5000,
                              random_seed=[143, 135, 138], logloss=True,
                              bootstrap=False, bagging_size_ratio=1):
        '''Perform cv_predict for multiple seeds and avg them'''
        for i, seed in enumerate(random_seed):
            if not i:
                pred = self.cv_predict(X, Y, test_X, nfold=nfold,
                                       ES_rounds=ES_rounds, steps=steps,
                                       random_seed=seed, logloss=logloss,
                                       bootstrap=bootstrap,
                                       bagging_size_ratio=bagging_size_ratio)
            else:
                pred += self.cv_predict(X, Y, test_X, nfold=nfold,
                                        ES_rounds=ES_rounds, steps=steps,
                                        random_seed=seed, logloss=logloss,
                                        bootstrap=bootstrap,
                                        bagging_size_ratio=bagging_size_ratio)

        return pred / len(random_seed)

    def predict(self, test_X, logloss=False):
        '''Predict using a fitted model'''
        pred_y = self.model.predict(
            test_X, num_iteration=self.model.best_iteration)
        if logloss:
            pred_y = np.expm1(pred_y)
        return pred_y

    def fit_predict(self, train_X, train_y, test_X, val_X=None, val_y=None,
                    logloss=True, return_oof_pred=False, **kwargs):
        evals_result, oof_pred = self.fit(
            train_X, train_y, val_X, val_y, return_oof_pred=return_oof_pred)
        pred_y = self.predict(test_X, logloss)
        if return_oof_pred:
            return evals_result, pred_y, oof_pred
        else:
            return evals_result, pred_y

    def optmize_hyperparams(self, param_grid, X, Y,
                            cv=4, scoring='neg_mean_squared_error',
                            verbose=1):
        '''Use GridSearchCV to optimize models params'''
        params = self.params
        params['learning_rate'] = 0.05
        params['n_estimators'] = 1000
        gsearch1 = GridSearchCV(estimator=lgb.LGBMModel(**params),
                                param_grid=param_grid,
                                scoring=scoring,
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


class CatBoost():
    '''Yandex catboost class wrapper
    Reference values:
         objective='rmse', eval_metric='rmse',
         iterations=1000,  random_seed=143,
         l2_leaf_reg=3,
         bootstrap_type='Bayesian', bagging_temperature=1,
         subsample=0.66, sampling_frequency='PerTreeLevel',
         rsm=1,
         lr=0.03,
         nan_mode='Min',
         use_best_model='True',
         max_depth=6,
         ignored_features=None, one_hot_max_size=2,
         task_type='CPU', verbose=True'''

    def __init__(self, **kwargs):

        self.params = {}
        self.is_fitted = False
        for key, value in kwargs.items():
            self.params[key] = value
        self.model = cb.CatBoost(self.params)

        if self.params['objective'] in ['auc', 'binary_logloss', 'multi_logloss']:
            self.get_best_metric = max
        else:
            self.get_best_metric = min

    def fit(self, train_X, train_y, val_X=None, val_y=None, ES_rounds=100,
            verbose=150, use_best_model=True, return_oof_pred=True):
        # Train Catboost model
        # val_set = np.array([(x, y) for x, y in zip(val_X, val_y)])
        self.model.fit(X=train_X, y=train_y,
                       eval_set=(val_X, val_y) if val_X is not None else None,
                       early_stopping_rounds=ES_rounds,
                       verbose_eval=verbose)
        self.is_fitted = True
        if return_oof_pred:
            pred = self.predict(val_X, logloss=False)
            oof_result = np.sqrt(mean_squared_error(val_y, pred))
        else:
            pred = None
            oof_result = None
        return oof_result, pred

    def cv(self, X, Y, nfold=5, ES_rounds=100, random_seed=143,
           bootstrap=False, bagging_size_ratio=1, splits=None,
           shuffle=True, return_oof_pred=False, verbose=100):
        # Train LGB model using CV
        if splits is None:
            if bootstrap:
                splits = generate_bagging_splits(
                    X.shape[0], nfold,
                    bagging_size_ratio=bagging_size_ratio,
                    random_seed=random_seed)

            else:
                kf = KFold(n_splits=nfold, shuffle=shuffle,
                           random_state=random_seed)
                splits = kf.split(X, y=Y)

        kFold_results = []
        oof_results = []
        for train_index, val_index in splits:
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]

            evals_result, oof_prediction = self.fit(
                train_X=x_train, train_y=y_train,
                val_X=x_val, val_y=y_val,
                ES_rounds=ES_rounds,
                return_oof_pred=True,
                verbose=verbose)

            if return_oof_pred:
                oof_results.extend(oof_prediction)
            if evals_result is not None:
                kFold_results.append(evals_result)

        kFold_results = np.array(kFold_results)
        if kFold_results.size > 0:
            print('Mean val error: {}, std {} '.format(
                kFold_results.mean(), kFold_results.std()))
        if return_oof_pred:
            return np.array(oof_results)

    def cv_predict(self, X, Y, test_X, nfold=5, ES_rounds=100,
                   random_seed=143, shuffle=True, return_oof_pred=False,
                   bootstrap=False, bagging_size_ratio=1,
                   logloss=False, verbose=100, splits=None):
        '''Fit model using CV and predict test using the average
         of all folds'''
        if splits is None:
            if bootstrap:
                splits = generate_bagging_splits(
                    X.shape[0], nfold,
                    bagging_size_ratio=bagging_size_ratio,
                    random_seed=random_seed)

            else:
                kf = KFold(n_splits=nfold, shuffle=True,
                           random_state=random_seed)
                splits = kf.split(X, y=Y)

        kFold_results = []
        oof_results = []
        for i, (train_index, val_index) in enumerate(splits):
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]

            evals_result, oof_prediction = self.fit(
                train_X=x_train, train_y=y_train,
                val_X=x_val, val_y=y_val,
                ES_rounds=ES_rounds,
                return_oof_pred=True,
                verbose=verbose)

            if return_oof_pred:
                oof_results.extend(oof_prediction)
            if evals_result is not None:
                kFold_results.append(evals_result)

            # Get predictions
            if not i:
                pred_y = self.predict(test_X, logloss=logloss)
            else:
                pred_y += self.predict(test_X, logloss=logloss)

        kFold_results = np.array(kFold_results)
        if kFold_results.size > 0:
            print('Mean val error: {}, std {} '.format(
                kFold_results.mean(), kFold_results.std()))

        # Divide pred by the number of folds and return
        if return_oof_pred:
            return pred_y / nfold, np.array(oof_results)
        return pred_y / nfold

    def predict(self, test_X, logloss=False):
        '''Predict using a fitted model'''
        if not self.is_fitted:
            raise NonFittedError(('Model has not been fitted.',
                                  ' First fit the model before predicting.'))
        pred_y = self.model.predict(test_X)
        if logloss:
            pred_y = np.expm1(pred_y)
        return pred_y

    def fit_predict(self, train_X, train_y, test_X, val_X=None, val_y=None,
                    ES_rounds=100, verbose=150, use_best_model=True,
                    return_oof_pred=True, logloss=False):
        evals_result, oof_pred = self.fit(train_X, train_y, val_X, val_y,
                                          ES_rounds=ES_rounds, verbose=verbose,
                                          use_best_model=use_best_model,
                                          return_oof_pred=return_oof_pred)
        pred_y = self.predict(test_X, logloss=logloss)
        if return_oof_pred:
            return evals_result, pred_y, oof_pred
        else:
            return evals_result, pred_y

    def optmize_hyperparams(self, param_grid, X, Y,
                            cv=4, scoring='neg_mean_squared_error',
                            verbose=1):
        '''Use GridSearchCV to optimize models params'''
        params = self.params
        params['learning_rate'] = 0.05
        params['n_estimators'] = 1000
        gsearch1 = GridSearchCV(estimator=lgb.LGBMModel(**params),
                                param_grid=param_grid,
                                scoring=scoring,
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


class RNN_LSTM():
    '''LSTM class for time-series'''

    def __init__(self, units, layers, in_shape, out_shape,
                 lr=0.001, lr_decay=0):
        self.units = units
        self.layers = layers
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.lr = lr
        self.lr_decay = lr_decay

        self.scaler = MinMaxScaler()

    def create_model(self):
        '''Create NN graph'''
        self.model = Sequential()
        for i in range(self.layers):
            if not i:
                self.model.add(
                    LSTM(self.units,
                         input_shape=self.in_shape,
                         return_sequences=False if self.layers == 1 else True))
            elif i == self.layers - 1:
                self.model.add(LSTM(self.units,
                                    return_sequences=False))
            else:
                self.model.add(LSTM(self.units,
                                    return_sequences=True))
        self.model.add(Dense(self.out_shape))
        optimizer = Adam(lr=self.lr, decay=self.lr_decay)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def fit(self, train_X, train_y, val_X, val_y, epochs=5, mb_size=10,
            scale_data=True, early_stop=True):
        '''Train LSTM model'''
        self.scale_data = scale_data
        if scale_data:
            train_X = self.scaler.fit_transform(
                np.reshape(train_X, (-1, np.prod(self.in_shape))))
            val_X = self.scaler.transform(
                np.reshape(val_X, (-1, np.prod(self.in_shape))))
            train_X = np.reshape(
                train_X, (-1, self.in_shape[0], self.in_shape[1]))
            val_X = np.reshape(
                val_X, (-1, self.in_shape[0], self.in_shape[1]))
        # Assert and correct data dims
        train_X = assert_time_series_dims(train_X)
        val_X = assert_time_series_dims(val_X)
        # Train model
        if early_stop:
            callbacks = [EarlyStopping(patience=1)]
        else:
            callbacks = None

        self.create_model()
        history = self.model.fit(train_X, train_y,
                                 validation_data=(val_X, val_y),
                                 epochs=epochs, batch_size=mb_size,
                                 callbacks=callbacks)
        return history

    def predict(self, test_X, logloss=True):
        '''Predict using a fitted model'''
        if self.scale_data:
            test_X = self.scaler.transform(test_X)
        # Assert and correct data dims
        test_X = assert_time_series_dims(test_X)
        # Predict
        pred_y = self.model.predict(test_X)

        if logloss:
            pred_y = np.expm1(pred_y)
        return pred_y

    def fit_predict(self, train_X, train_y, val_X, val_y,
                    test_X, pochs=100, mb_size=10,
                    scale_data=True, early_stop=True, logloss=True):
        history = self.fit(train_X, train_y, val_X, val_y,
                           epochs=100, mb_size=10,
                           scale_data=True,
                           early_stop=early_stop)
        pred_y = self.predict(test_X, logloss)
        return history, pred_y

    def cv(self, X, Y, nfold=5, epochs=5, mb_size=10, random_seed=143,
           scale_data=True, early_stop=True,
            bootstrap=False, bagging_size_ratio=1, **kwargs):
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

            history = self.fit(x_train, y_train, x_val, y_val,
                               epochs=epochs, mb_size=mb_size,
                               scale_data=scale_data,
                               early_stop=early_stop)
            if history:
                kFold_results.append(
                    np.array(
                        history.history).min())

        kFold_results = np.array(kFold_results)
        if kFold_results.size > 0:
            print('Mean val error: {}, std {} '.format(
                kFold_results.mean(), kFold_results.std()))

    def cv_predict(self, X, Y, test_X, nfold=5, epochs=5, mb_size=10,
                   random_seed=143, logloss=True,
                   scale_data=True, early_stop=True,
                   bootstrap=False, bagging_size_ratio=1, **kwargs):
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

            history = self.fit(x_train, y_train, x_val, y_val,
                               epochs=epochs, mb_size=mb_size,
                               scale_data=scale_data,
                               early_stop=early_stop)

            # if history:
            #     kFold_results.append(
            #         np.array(
            #             history.history).min())

            # Get predictions
            if not i:
                pred_y = self.predict(test_X, logloss=logloss)
            else:
                pred_y += self.predict(test_X, logloss=logloss)

        # kFold_results = np.array(kFold_results)
        # if kFold_results.size > 0:
        #     print('Mean val error: {}, std {} '.format(
        #         kFold_results.mean(), kFold_results.std()))

        # Divide pred by the number of folds and return
        return pred_y / nfold


class LiRegression():
    '''Linear regression class wrapper'''

    def __init__(self, normalize):
        self.normalize = normalize
        self.is_fitted = False
        self.model = LinearRegression(normalize)

    def fit(self, train_X, train_y, val_X=None, val_y=None,
            verbose=150, return_oof_pred=True):
        self.model.fit(X=train_X, y=train_y)
        self.is_fitted = True
        if return_oof_pred:
            pred = self.predict(val_X, logloss=False)
            oof_result = np.sqrt(mean_squared_error(val_y, pred))
        else:
            pred = None
            oof_result = None
        return oof_result, pred

    def cv(self, X, Y, nfold=5, random_seed=143,
           bootstrap=False, bagging_size_ratio=1,
           shuffle=True, oof_pred=False, verbose=100):
        # Train LGB model using CV
        if bootstrap:
            splits = generate_bagging_splits(
                X.shape[0], nfold,
                bagging_size_ratio=bagging_size_ratio,
                random_seed=random_seed)

        else:
            kf = KFold(n_splits=nfold, shuffle=shuffle,
                       random_state=random_seed)
            splits = kf.split(X, y=Y)

        kFold_results = []
        oof_results = []
        for train_index, val_index in splits:
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]

            evals_result, oof_prediction = self.fit(
                train_X=x_train, train_y=y_train,
                val_X=x_val, val_y=y_val,
                return_oof_pred=True)

            if oof_pred:
                oof_results.extend(oof_prediction)
            if evals_result is not None:
                kFold_results.append(evals_result)

        kFold_results = np.array(kFold_results)
        if kFold_results.size > 0:
            print('Mean val error: {}, std {} '.format(
                kFold_results.mean(), kFold_results.std()))
        if oof_pred:
            return np.array(oof_results)

    def cv_predict(self, X, Y, test_X, nfold=5,
                   random_seed=143, shuffle=True, oof_pred=False,
                   bootstrap=False, bagging_size_ratio=1,
                   logloss=False, verbose=100):
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
        oof_results = []
        for i, (train_index, val_index) in enumerate(splits):
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]

            evals_result, oof_prediction = self.fit(
                train_X=x_train, train_y=y_train,
                val_X=x_val, val_y=y_val,
                return_oof_pred=True)

            if oof_pred:
                oof_results.extend(oof_prediction)
            if evals_result is not None:
                kFold_results.append(evals_result)

            # Get predictions
            if not i:
                pred_y = self.predict(test_X, logloss=logloss)
            else:
                pred_y += self.predict(test_X, logloss=logloss)

        kFold_results = np.array(kFold_results)
        if kFold_results.size > 0:
            print('Mean val error: {}, std {} '.format(
                kFold_results.mean(), kFold_results.std()))

        # Divide pred by the number of folds and return
        if oof_pred:
            return pred_y / nfold, np.array(oof_results)
        return pred_y / nfold

    def predict(self, test_X, logloss=False):
        '''Predict using a fitted model'''
        if not self.is_fitted:
            raise NonFittedError(('Model has not been fitted.',
                                  ' First fit the model before predicting.'))
        pred_y = self.model.predict(test_X)
        if logloss:
            pred_y = np.expm1(pred_y)
        return pred_y

    def fit_predict(self, train_X, train_y, test_X, val_X=None, val_y=None,
                    verbose=150, return_oof_pred=True, logloss=False):
        evals_result, oof_pred = self.fit(train_X, train_y, val_X, val_y,
                                          return_oof_pred=return_oof_pred)
        pred_y = self.predict(test_X, logloss=logloss)
        if return_oof_pred:
            return evals_result, pred_y, oof_pred
        else:
            return evals_result, pred_y


class XGBoost():
    '''XGBoost wrapper class'''

    def __init__(self, objective='reg:linear', metric='rmse', device='gpu',
                 **kwargs):
        self.params = {
            "objective": objective,
            "eval_metric": metric,
            "bagging_seed": 42,
            "verbosity": -1,
            "seed": 42,
            "device": device,
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
        }
        for key, value in kwargs.items():
            self.params[key] = value

        if self.params['eval_metric'] in ['auc', 'binary_logloss', 'multi_logloss']:
            self.get_best_metric = max
        else:
            self.get_best_metric = min

    def fit(self, train_X, train_y, val_X, val_y, ES_rounds=20, steps=5000,
            verbose=25, return_oof_pred=True, **kwargs):
        # Train LGB model
        xgbtrain = xgb.DMatrix(train_X, label=train_y)
        xgbval = xgb.DMatrix(val_X, label=val_y)
        evals_result = {}
        self.model = xgb.train(self.params, xgbtrain,
                               num_boost_round=steps,
                               evals=[(xgbtrain, 'train'),
                                      (xgbval, 'valid_1')],
                               early_stopping_rounds=ES_rounds,
                               verbose_eval=verbose,
                               evals_result=evals_result)

        if return_oof_pred:
            pred = self.predict(val_X, logloss=False)
        else:
            pred = None
        return evals_result, pred

    def cv(self, X, Y, nfold=5, ES_rounds=100, steps=5000, random_seed=143,
           bootstrap=False, bagging_size_ratio=1, shuffle=True, oof_pred=False):
        # Train LGB model using CV
        if bootstrap:
            splits = generate_bagging_splits(
                X.shape[0], nfold,
                bagging_size_ratio=bagging_size_ratio,
                random_seed=random_seed)

        else:
            kf = KFold(n_splits=nfold, shuffle=shuffle,
                       random_state=random_seed)
            splits = kf.split(X, y=Y)

        kFold_results = []
        oof_results = []
        for train_index, val_index in splits:
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]

            evals_result, oof_prediction = self.fit(train_X=x_train, train_y=y_train,
                                                    val_X=x_val, val_y=y_val,
                                                    ES_rounds=100,
                                                    steps=10000,
                                                    return_oof_pred=oof_pred)
            if oof_pred:
                oof_results.extend(oof_prediction)
            if evals_result:
                kFold_results.append(
                    np.array(
                        self.get_best_metric(
                            evals_result['valid_1'][self.params['eval_metric']])))

        kFold_results = np.array(kFold_results)
        if kFold_results.size > 0:
            print('Mean val error: {}, std {} '.format(
                kFold_results.mean(), kFold_results.std()))
        if oof_pred:
            return np.array(oof_results)

    def cv_predict(self, X, Y, test_X, nfold=5, ES_rounds=100, steps=5000,
                   random_seed=143, logloss=False,
                   bootstrap=False, bagging_size_ratio=1, oof_pred=False):
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
        oof_results = []
        for i, (train_index, val_index) in enumerate(splits):
            x_train = X[train_index]
            y_train = Y[train_index]
            x_val = X[val_index]
            y_val = Y[val_index]

            evals_result, oof_prediction = self.fit(train_X=x_train, train_y=y_train,
                                                    val_X=x_val, val_y=y_val,
                                                    ES_rounds=100,
                                                    steps=10000,
                                                    return_oof_pred=oof_pred)
            if oof_pred:
                oof_results.extend(oof_prediction)
            if evals_result:
                kFold_results.append(
                    np.array(
                        self.get_best_metric(
                            evals_result['valid_1'][self.params['eval_metric']])))

            # Get predictions
            if not i:
                pred_y = self.predict(test_X, logloss=logloss)
            else:
                pred_y += self.predict(test_X, logloss=logloss)

        kFold_results = np.array(kFold_results)
        if kFold_results.size > 0:
            print('Mean val error: {}, std {} '.format(
                kFold_results.mean(), kFold_results.std()))

        # Divide pred by the number of folds and return
        if oof_pred:
            return pred_y / nfold, np.array(oof_results)
        return pred_y / nfold

    def multi_seed_cv_predict(self, X, Y, test_X, nfold=5, ES_rounds=100,
                              steps=5000,
                              random_seed=[143, 135, 138], logloss=True,
                              bootstrap=False, bagging_size_ratio=1):
        '''Perform cv_predict for multiple seeds and avg them'''
        for i, seed in enumerate(random_seed):
            if not i:
                pred = self.cv_predict(X, Y, test_X, nfold=nfold,
                                       ES_rounds=ES_rounds, steps=steps,
                                       random_seed=seed, logloss=logloss,
                                       bootstrap=bootstrap,
                                       bagging_size_ratio=bagging_size_ratio)
            else:
                pred += self.cv_predict(X, Y, test_X, nfold=nfold,
                                        ES_rounds=ES_rounds, steps=steps,
                                        random_seed=seed, logloss=logloss,
                                        bootstrap=bootstrap,
                                        bagging_size_ratio=bagging_size_ratio)

        return pred / len(random_seed)

    def predict(self, test_X, logloss=False):
        '''Predict using a fitted model'''
        xgbtest = xgb.DMatrix(test_X)
        pred_y = self.model.predict(xgbtest)
        if logloss:
            pred_y = np.expm1(pred_y)
        return pred_y

    def fit_predict(self, train_X, train_y, test_X, val_X=None, val_y=None,
                    logloss=True, return_oof_pred=False, **kwargs):
        evals_result, oof_pred = self.fit(
            train_X, train_y, val_X, val_y,
            return_oof_pred=return_oof_pred, **kwargs)
        pred_y = self.predict(test_X, logloss)
        if return_oof_pred:
            return evals_result, pred_y, oof_pred
        else:
            return evals_result, pred_y

    def optmize_hyperparams(self, param_grid, X, Y,
                            cv=4, scoring='neg_mean_squared_error',
                            verbose=1):
        '''Use GridSearchCV to optimize models params'''
        params = self.params
        params['learning_rate'] = 0.05
        params['n_estimators'] = 1000
        gsearch1 = GridSearchCV(estimator=lgb.LGBMModel(**params),
                                param_grid=param_grid,
                                scoring=scoring,
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

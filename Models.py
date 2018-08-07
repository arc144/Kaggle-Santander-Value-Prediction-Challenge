import numpy as np
import lightgbm as lgb
# import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold, GridSearchCV
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
            "gpu_platform_id":  0,
            "gpu_device_id":  0,
        }
        for key, value in kwargs.items():
            self.params[key] = value

        if self.params['metric'] in ['auc', 'binary_logloss', 'multi_logloss']:
            self.get_best_metric = max
        else:
            self.get_best_metric = min

    def fit(self, train_X, train_y, val_X, val_y, ES_rounds=100, steps=5000,
            verbose=150, oof_pred=False):
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
        if oof_pred:
            pred = self.predict(val_X, logloss=False)
        else:
            pred = None
        return evals_result, pred

    def cv(self, X, Y, nfold=5,  ES_rounds=100, steps=5000, random_seed=143,
           bootstrap=False, bagging_size_ratio=1, oof_pred=False):
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
                                                    oof_pred=oof_pred)
            if oof_pred:
                oof_results.extend(oof_prediction)
            if evals_result:
                kFold_results.append(
                    np.array(
                        self.get_best_metric(
                            evals_result['valid_1'][self.params['metric']])))

        kFold_results = np.array(kFold_results)
        if kFold_results.size > 0:
            print('Mean val error: {}, std {} '.format(
                kFold_results.mean(), kFold_results.std()))
        if oof_pred:
            return np.array(oof_results)

    def cv_predict(self, X, Y, test_X, nfold=5,  ES_rounds=100, steps=5000,
                   random_seed=143, logloss=True,
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
                                                    oof_pred=oof_pred)
            if oof_pred:
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
        if oof_pred:
            return pred_y / nfold, np.array(oof_results)
        return pred_y / nfold

    def multi_seed_cv_predict(self, X, Y, test_X, nfold=5,  ES_rounds=100,
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
            train_X = self.scaler.fit_transform(train_X)
            val_X = self.scaler.transform(val_X)
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

    def cv(self, X, Y, nfold=5,  epochs=5, mb_size=10, random_seed=143,
           scale_data=True, early_stop=True,
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

    def cv_predict(self, X, Y, test_X, nfold=5,  epochs=5, mb_size=10,
                   random_seed=143, logloss=True,
                   scale_data=True, early_stop=True,
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

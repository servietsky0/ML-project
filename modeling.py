import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from preprocessing import str_to_int
import optuna

df = str_to_int()

def set_model():
    params = {
            'learning_rate': 0.0921459262053085,
            'depth': 10,
            'subsample': 0.617993504291168,
            'colsample_bylevel': 0.8692370930849208,
            'min_data_in_leaf': 84,
        }
    return params

def train_model(params, df):
    y = np.array(df['Price']).reshape(-1, 1)
    X = np.array(df.drop(['Price'], axis=1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = CatBoostRegressor(**params, silent=True)
    model.fit(X_train, y_train, verbose=100)
    
    predictions = model.predict(X_test)
    print(model.score(X_test, y_test))

    y_pred = model.predict(X_test)
    print(mean_absolute_error(y_test, y_pred))


df.to_csv('datasett.csv', index=False)

def model(trial):
    y = np.array(df['Price']).reshape(-1, 1)
    X = np.array(df.drop(['Price'], axis=1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    model = CatBoostRegressor(**params, silent=True)
    model.fit(X_train, y_train, verbose=100)
    
    predictions = model.predict(X_test)
    print(model.score(X_test, y_test))

    y_pred = model.predict(X_test)
    print(mean_absolute_error(y_test, y_pred))
    
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return rmse


study = optuna.create_study(direction='minimize')
study.optimize(model, n_trials=200)

print('Best hyperparameters:', study.best_params)
print('Best RMSE:', study.best_value)



# df.to_csv('datasett.csv', index=False)




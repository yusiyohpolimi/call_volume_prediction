import preprocess
import plot
import pickle
import argparse
import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.model_selection import TimeSeriesSplit



def select_features(model, df, n_features=None, target='calls'):
    """
    Implements Recursive Feature Elimination and returns selected feature 
    indices.

    Parameters
    ----------
        model: XGBRegressor() instance
        df: pandas.DataFrame
        n_features: int
            Number of features to keep.
        target: str
            Target variable column name in df.        

    Returns
    -------
        indices: array
            Selected feature indices to filter the dataset.
    """
    rfe = RFE(model, n_features_to_select=n_features, step=1, verbose=0, 
              importance_getter='auto')

    features = df.drop(columns=[target]).columns
    
    X_train = df[features]
    y_train = df[target]

    # using recursive feature elimination to select best features
    rfe.fit(X_train, y_train)
    indices = rfe.support_

    return indices


def train(df, target=None, cv=False, select_feat=False):
    """
    Trains XGBRegressor() instance. Supports k-fold cross-validation, but for
    time series, utilizing TimeSeriesSplit from sklearn, and Recursive Feature
    Elimination (RFE) algorithm for feature selection.

    Parameters
    ----------
        df: pandas.DataFrame
        target: str
            Target variable column name in df.
        cv: boolean
            Set True for using k-fold cross-validation.
        select_feat: boolean
            Set True for using RFE algorithm.

    Returns
    -------
        model: trained XGBRegressor() instance
        y_preds, preds: array
            Predictions of single test set (y_preds), or predictions for each 
            fold if cv=True (preds).
        scores: array
            Scores for each fold if cv=True.
        test: pandas.DataFrame
            Test/validation set that is used for validation RMSE calculation.    
    """
    features = df.drop(columns=[target]).columns
    # XGB will be used for regression
    model = xgb.XGBRegressor(n_estimators=1000)

    if cv:      # cross-validation for time series 
        # setting test size to 1 year hours
        splitter = TimeSeriesSplit(n_splits=5, test_size=24*365)

        fold = 0
        preds = list()
        scores = list()
        for train_idx, val_idx in splitter.split(df):
            train = df.iloc[train_idx]
            test = df.iloc[val_idx]

            X_train = train[features]
            y_train = train[target]
            X_test = test[features]   
            y_test = test[target]
            
            model.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    early_stopping_rounds=50,
                    verbose=100)

            y_pred = model.predict(X_test)
            preds.append(y_pred)
            score = np.sqrt(mean_squared_error(y_test, y_pred))
            scores.append(score)

        print(f'Score across folds {np.mean(scores):0.4f}')
        print(f'Fold scores:{scores}')

        return model, preds, scores

    else:
        # training without cross-validation, split 1 train/test
        splitter = TimeSeriesSplit(n_splits=2, test_size=24*365)        
        _, indices = splitter.split(df)

        train = df.iloc[indices[0]]
        test = df.iloc[indices[1]]
                
        if select_feat:     # to use RFE to select best features
            indices = select_features(model, df)
            features = features[indices]

        X_train = train[features]
        y_train = train[target]
        X_test = test[features]   
        y_test = test[target]


        model.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                early_stopping_rounds=50,
                verbose=100)

        y_pred = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f'RMSE: {score}')
    
        return model, y_pred, test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='2008-09-15', 
                        help='start date to trim dataset')
    parser.add_argument('--end', type=str, default='2022-09-15',
                         help='end date to trim dataset')
    parser.add_argument('--download', action='store_true', 
                        help='download dataset from the source')
    parser.add_argument('--rfe', action='store_true', 
                        help='use Recursive Feature Elimination')
    parser.add_argument('--cv', action='store_true', 
        help='use 5-fold cross-validation for time series, need 5+ years of data')
    args = parser.parse_args()

    # start = input("Provide a start date to trim the dataset: ")
    # end = input("Provide an end date to trim the dataset: ")    
    # download = input("Do you want to download the data? Skip for loading from" +
    #                  " current folder: ")
    # rfe = input("Do you want to use Recursive Feature Elimination?")    

    train_df = preprocess.get_data_ready(args.start, args.end, 
                                         download=args.download)

    print("Saving processed dataset for later uses, i.e., testing ....")
    train_df.to_csv("dataset.csv", index_label='stamp')

    print(f"Using {train_df.shape[0]} samples for training ***")
    train_df.drop(columns='windspeedKmph', inplace=True)
    model, preds, test_data= train(train_df, target='calls', cv=args.cv, 
                                   select_feat=args.rfe)

    if not args.cv:
        print("Plotting the results ....")
        plot.plot_results(test_data.calls, preds)

    print("Saving the model (and features too) for using it in test or predict" + 
        " functions ....")
    file_name = "xgb_model.pkl"
    # using pickle to save also the features
    pickle.dump(model, open(file_name, "wb"))

    print("All done! You can use xgb_model.pkl to load model and test/predict!")

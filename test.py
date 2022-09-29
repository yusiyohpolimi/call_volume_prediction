import pickle
import sys
import plot
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def test(model, test_data, target=None): 
    """
    Predicts call volume, for subset of the dataset.

    Parameters
    ----------
    model : XGBRegressor() instance
    test_data: pandas.DataFrame
        Should include features and target variable.
    target: str
        Target variable column name in the test_data.    
    
    Returns
    -------
    y_pred : array
        Call volume predictions.
    score: float
        RMSE for the given test set.
    """ 

    if target:
        features = model.get_booster().feature_names
        x_test = test_data[features]
        y_test = test_data[target]
    else:
        print("Use predict.py if you do not provide target variable.")
        return
    
    y_pred = model.predict(x_test)

    score = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE: {score}')
    
    return y_pred, score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='2021-09-15', 
                        help='start date to trim dataset')
    parser.add_argument('--end', type=str, default='2022-09-15',
                         help='end date to trim dataset')
    args = parser.parse_args()

    print("Loading xgb_model.pkl from the current folder ...")
    file_name = 'xgb_model.pkl'

    try:
        model = pickle.load(open(file_name, "rb"))
        df = pd.read_csv("dataset.csv")
        df.index = df.stamp

    except OSError as err:
        print(err)
        print("Please train the model first and save it as xgb_model.pkl !!")
        sys.exit(0)

    if args.start and args.end:
        test_data = df[(df.index >= args.start) & (df.index < args.end)]

    y_pred, score = test(model, test_data, target='calls')
    
    plot.plot_results(test_data.calls, y_pred)





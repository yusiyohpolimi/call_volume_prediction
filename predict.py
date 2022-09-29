import pickle
import sys
import argparse
import plot
import pandas as pd
from preprocess import DataSet


def predict(model, start_date, end_date, freq='H'):  
    """
    Predicts a future call volume, for a date/period out of the dataset.

    Parameters
    ----------
    model : XGBRegressor() instance
    start_date : str
        Date in %Y-%M-%D format.
    end_date : str
        Date in %Y-%M-%D format.
    freq: str
        Any frequency from ['H', 'D', 'W', 'M']
    
    Returns
    -------
    y_pred : pandas.DataFrame
        Call volume predictions with date index.
    """
    
    assert end_date > start_date, "Provide end date which is LATER than start!"
    assert freq in ['H', 'D', 'W', 'M'], \
        "Frequency can be 'H', 'D', 'W', 'M' !!"

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    indexer = pd.date_range(start=start_date, end=end_date, freq='H')

    features = model.get_booster().feature_names
    df = pd.DataFrame(columns=features)
    df = df.reindex(indexer)
    df.index.name = 'stamp'
    df['stamp'] = df.index

    test_ds = DataSet(df)
    test_ds.add_time_features()
    test_df = test_ds.df.drop(columns=['hour_only'])
    test_df = test_df.astype('float32')
    
    y_pred = model.predict(test_df[features])
    y_pred = pd.DataFrame(y_pred ,index=test_df.index)
    y_pred.columns = ['Calls']
    
    if freq == 'D':
        y_pred = y_pred.resample('1D').sum()
    elif freq == 'W':
        y_pred = y_pred.resample('7D').sum()
    elif freq == 'M':
        y_pred = y_pred.resample('30D').sum()
    
    return y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='2022-09-15', 
                        help='start date to make prediciton')
    parser.add_argument('--end', type=str, default='2022-09-20',
                         help='end date to make prediction')
    parser.add_argument('--freq', type=str, default='H',
                         help='resolution of the prediction')
    args = parser.parse_args()

    print("Loading xgb_model.pkl from the current folder ...")
    file_name = 'xgb_model.pkl'

    try:
        model = pickle.load(open(file_name, "rb"))
        print("Model is ready to predict!!")
    except OSError as err:
        print(err)
        print("Please train the model first and save it as xgb_model.pkl !!")
        sys.exit(0)
    
    preds = predict(model, args.start, args.end, args.freq)
    
    # since there is no label for future prediction
    if args.freq == 'H':
        plot.plot_results(preds.Calls, preds.Calls)

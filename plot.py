import pandas as pd
import matplotlib 
from matplotlib import pyplot as plt

def plot_results(test_data, pred):
    """
    Plots the actual and prediction data, both hourly and daily.

    Parameters
    ----------
    test_data: pandas.DataFrame
        Should include features and target variable.
    pred: array
        Predictions of the model.
    """ 
    
    hourly_pred_lastyear = pd.DataFrame(
        {"Calls": test_data, "BestPrediction": pred}
    )
    
    plt.style.use('ggplot')
    matplotlib.rcParams['figure.dpi'] = 120

    fig, axs = plt.subplots(3, 1, figsize=(12,9), sharex=True)
    axs[0].plot(hourly_pred_lastyear, lw=1)
    axs[0].set_ylabel('Hourly Call Count', fontsize=7)
    axs[0].legend(['Actual', 'Prediction'])

    axs[1].plot(hourly_pred_lastyear.rolling(24).sum(), lw=1)
    axs[1].set_ylabel('Daily Call Count', fontsize=7)
    axs[1].legend(['Actual', 'Prediction'])

    axs[2].plot(hourly_pred_lastyear.rolling(24*7).sum(), lw=1)
    axs[2].set_ylabel('Weekly Call Count', fontsize=7)
    axs[2].legend(['Actual', 'Prediction'])

    [ax.tick_params(axis='both', which='major', labelsize=6) for ax in axs];
    plt.savefig('Hourly_daily_weekly_results.png')
    print("Figures are saved to Hourly_daily_weekly_results.png")
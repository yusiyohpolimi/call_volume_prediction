# Call Volume Prediction for Seattle Fire Department

The project is developed by Yusuf Can Simsek, as an answer to the request from
niologic GmbH. 

## Installation

First, clone the repository and install the requirements:

```
git clone https://github.com/yusiyohpolimi/call_volume_prediction.git

pip install -r requirements.txt
```

## Training  
  
To train the model (XGBRegressor), use train.py script

```
usage: train.py [-h] [--start START] [--end END] [--download] [--rfe] [--cv]

optional arguments:
  -h, --help     show this help message and exit
  --start START  start date to trim dataset
  --end END      end date to trim dataset
  --download     download dataset from the source
  --rfe          use Recursive Feature Elimination
  --cv           use 5-fold cross-validation for time series, need 5+ years of
                 data
```

## Testing and Future Prediction

Use test.py script to test the trained model with a subset of the dataset. 
Other script, predict.py, is used to make a future prediction, out of the dataset.

```
usage: test.py [-h] [--start START] [--end END]

optional arguments:
  -h, --help     show this help message and exit
  --start START  start date to trim dataset
  --end END      end date to trim dataset

usage: predict.py [-h] [--start START] [--end END] [--freq FREQ]

optional arguments:
  -h, --help     show this help message and exit
  --start START  start date to make prediciton
  --end END      end date to make prediction
  --freq FREQ    resolution of the prediction
```
 
## Notes
 
*	Check the notebook for the details of the project, and the followed steps.
*	Downloading weather data takes VERY long time, prefer using from the repository.
*	Covid pandemic seems to have a strong effect on call data trend.
*	Weather data other than temperature does not yield satisfactory results for
    this weather dataset. However, other services require subscription.


 

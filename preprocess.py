import os
import pandas as pd
from sodapy import Socrata
from wwo_hist import retrieve_hist_data


class DataSet:
    """Dataset class with several methods to process pandas.DataFrame."""

    def __init__(self, df):
        """
        Constructs DataSet with pandas.DataFrame attribute.

        Parameters
        ----------
            df: pandas.DataFrame
                Dataframe to be processed.
        """
        self.df =  df

    def add_lags(self, target):
        """
        Adds lag features to the call data, using moving window average.

        Parameters
        ----------
            target: str
                Target variable column name in call dataframe.
        """
        self.df["12h_mean"] = self.df[target].rolling(12).mean()
        self.df["24h_mean"] = self.df[target].rolling(24).mean()
        self.df["36h_mean"] = self.df[target].rolling(36).mean()
        self.df["w_mean"] = self.df[target].rolling(7*24).mean()
        self.df["m_mean"] = self.df[target].rolling(30*24).mean()
        self.df["y_mean"] = self.df[target].rolling(365*24).mean()
        
    def clear_dups(self):
        """
        Clears the duplicates from the call data. 
        
        Note that, there are some calls with same time stamp, however they are 
        not discarded since the adress information is different for each.
        """
        self.df.drop(columns=['report_location'], inplace=True, errors='ignore')
        # no duplicates in the dataset        
        if self.df[self.df.duplicated(keep=False)].count().sum() :
            self.df.drop_duplicates()

    def drop_unrelated(self):
        """
        Drops unrelated variables from the dataframe such as lat/lon/adress.
        """
        self.df.drop(
            columns=['latitude', 'longitude', 
                    'incident_number', 'type', 'address'], 
            inplace=True,
            errors='ignore'
        )
              
    def add_time_features(self):
        """
        Adds time features to the dataframe utilizing `pandas` datetime
        attributes.
        """
        # changing datetime name for convinience
        self.df.rename(columns={"datetime":"stamp"}, inplace=True)
        self.df.stamp = pd.to_datetime(self.df.stamp, infer_datetime_format=True)

        # time series features, useful as training features
        self.df['hour'] = self.df['stamp'].dt.hour
        self.df['dayofweek'] = self.df['stamp'].dt.dayofweek
        self.df['quarter'] = self.df['stamp'].dt.quarter
        self.df['month'] = self.df['stamp'].dt.month
        self.df['year'] = self.df['stamp'].dt.year
        self.df['dayofyear'] = self.df['stamp'].dt.dayofyear
        self.df['dayofmonth'] = self.df['stamp'].dt.day
        self.df['weekofyear'] = self.df['stamp'].dt.isocalendar().week.astype('int64')

        # adding hour resolution 
        self.df.loc[:, "hour_only"] = pd.to_datetime(
            self.df.stamp.dt.strftime("%Y-%m-%d %H:00:00")
        )

        self.df.set_index('stamp', inplace=True)
        self.df.sort_index(inplace=True)       
 
    def process_weather(self):
        """Processes weather data to make it ready to merge with call data."""

        self.df.rename(columns={"date_time":"stamp"}, inplace=True)
        self.df.stamp = pd.to_datetime(self.df.stamp, infer_datetime_format=True)
        self.df = self.df.astype(
            {'windspeedKmph':'float32', 'tempC':'float32'}
        )
        self.df.set_index('stamp', inplace=True)
        self.df.sort_index(inplace=True)

    def check_missing_hours(self):
        """
        Checks missing hours in the data and fills the gaps using linear
        interpolation.
        """
        N_hours = pd.date_range(start=self.df.index.min(), 
                                end=self.df.index.max(), freq='H')
        df_reindexed = self.df.reindex(N_hours)
        n_hours = N_hours.difference(self.df.index.unique()).values.shape[0]
        if n_hours > 0:
            print(f"Missing {n_hours} hours of data. Filling the gaps ....")
            self.df = df_reindexed.interpolate(method = 'linear')
    

def download_data(start=None, end=None):
    """
    Downloads 911 Call data and weather data for Seattle.
    
    Parameters
    ----------
        start: str
            Should be in %Y-%M-%D format.
        end: str
            Should be in %Y-%M-%D format.

    Returns
    -------
        call_df: pandas.DataFrame 
            Dataframe of the 911 call data.
        weather_df: pandas.DataFrame
            Dataframe of the weather data of Seattle.    
    """    
    # Downloading both call data and weather data from respective sources
    client = Socrata("data.seattle.gov", None)
    if start and end:
        print("Downloading partial 911 Call data from data.seattle.gov .....")
        query = f"datetime BETWEEN '{start}' AND '{end}'"
        call_data = client.get("kzjm-xkqj", where=query, limit=int(2e6))  

    else:
        print("Downloading full 911 Call data from data.seattle.gov .....")
        call_data = client.get("kzjm-xkqj", limit=int(2e6))   

    print("911 Call Data is downloaded!")

    # convert to DataFrame
    call_df = pd.DataFrame.from_records(call_data).iloc[:, :7]
    # saving data to ease usage
    call_df.to_csv("seattle_call.csv", index=False)

    # downloading weather data
    if not os.path.exists("weather_data.csv"):                
        api_key = 'e63769263db649d7af9180013222309'
        freq = 1
        start_date = '2008-07-01'   # data available from this date on
        end_date = str(pd.Timestamp.now().date())
        location_list = ['seattle']
        print("Downloading weather data from WorldWeatherOnline .....")

        if start and end:
            print('Setting start and end dates for weather data ....')
            start_date = start
            end_date = end

        wwo_df = retrieve_hist_data(
            api_key,
            location_list,
            start_date,
            end_date,
            freq,
            location_label = False,
            export_csv = False,
            store_df = True
        );      

        weather_df = wwo_df[0][['date_time', 'tempC', 'windspeedKmph']]

        print("Weather data is downloaded!")
        # saving data to ease usage
        weather_df.to_csv("weather_data.csv", index=False)

    else:
        print("Fortunately, weather data is found! Loading ...")
        weather_df = pd.read_csv("weather_data.csv")
        weather_df = weather_df[['date_time', 'tempC', 'windspeedKmph']]

    return call_df, weather_df


def get_data_ready(start=None, end=None, download=False):
    """
    Processes data by using DataSet class methods and returns a DataFrame.

    Parameters
    ----------
        start: str
            Should be in %Y-%M-%D format.
        end: str
            Should be in %Y-%M-%D format.
        download: boolean
            Set True to downlaod the call and weather data. Note that it takes
            quite long time.

    Returns
    -------
        train_df: pandas.DataFrame
            DataFrame ready to use for training/validation.
    """
    
    if not download:
        # reading data since downloading takes very long time
        print("Loading data from current folder ....")
        try:
            call_df = pd.read_csv('seattle_call.csv') 

            weather_df = pd.read_csv(
                'weather_data.csv', 
                usecols=['date_time', 'tempC', 'windspeedKmph']
            )   
        except OSError as err:
            print(err)
            print('Downloading call and weather data ....')
            call_df, weather_df = download_data(start, end)

    else:     
        print('Downloading call and weather data ....')
        call_df, weather_df = download_data(start, end)   
         
    # Preprocessing the data with DataSet class methods
    call_ds = DataSet(call_df)
    call_ds.clear_dups()
    call_ds.drop_unrelated()
    call_ds.add_time_features()

    weather_ds = DataSet(weather_df)
    weather_ds.clear_dups()
    weather_ds.process_weather()

    hourly_calls = call_ds.df.hour_only.value_counts().astype('float32').sort_index()
    # daily_calls = call_ds.df.day_only.value_counts().astype('float32').sort_index()

    merged_df = call_ds.df\
        .reset_index()\
        .merge(weather_ds.df, left_on='hour_only', right_on='stamp', how='left')
    merged_df.set_index('stamp', inplace=True)

    target = 'calls'
    hourly_calls.name = target

    train_ds = DataSet(merged_df)
    train_ds.df = train_ds.df\
        .merge(hourly_calls, left_on='hour_only', right_index=True)\
        .drop_duplicates()

    train_ds.add_lags(target)
    train_ds.df.set_index('hour_only', inplace=True)        
    train_ds.check_missing_hours()        
    train_df = train_ds.df

    if start and end:   # when the full data available but trim required
        train_df = train_df[(train_df.index >= start) & (train_df.index < end)]

    return train_df

# python -m pip install pythonnet --user -U
import clr
import datetime
import joblib
import json
import math
import numpy as np
import os
import pandas as pd
import pickle
#import PIconnect as PI
import re
import requests
import System
import traceback
import warnings


from pandas import Timestamp
from requests.auth import HTTPBasicAuth
from scipy.optimize import fsolve, minimize, curve_fit
from urllib.parse import urlparse

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set the option to opt-in to the future behavior
pd.set_option('future.no_silent_downcasting', True)
PI_AF_SERVER = 'AWSENT-AF'
PI_AF_DATABASE = 'Shenzi Ramp-Up'

# PI Web API connection settings 
PI_WEB_API_URL = "https://piwebapi.osipi.np.woodside/piwebapi/"
USERNAME = "svc.svc.piwebapi"  # Replace with actual username
PASSWORD = "X9v&k#2ZpR8mU@d4Lq$Y1!oA7%TfWuXQ"  # Replace with actual password
BASE_PATH = r"GOMPU\Shenzi\Subsea\A. Wells\1.1 Oil Producers"

element_template = 'Oil Well'
OSI_AF_ATTRIBUTE_TAG = 'Press.Ramp.PFiltered'
OSI_AF_DATABASE = PI_AF_DATABASE
OSI_AF_ELEMENT = rf'{BASE_PATH}\SN102'

WELL_NAMES = ['B101', 'B102', 'B103', 'B104', 'B201', 'B202', 'B203', 'B204', 'C102', 'C103', 'C104', 'G101', 'G102', 'G103', 'G104', 'H101', 'H102', 'H103', 'SN101', 'SN102']
DEBUG = False
if DEBUG:
    WELL_NAMES = ['SN102']

piwebapi_url = PI_WEB_API_URL
af_server_name = PI_AF_SERVER
piwebapi_user = USERNAME
piwebapi_password = PASSWORD
piwebapi_security_method = 'Basic'
piwebapi_security_method = piwebapi_security_method.lower()
verify_ssl_string = 'N'

if (verify_ssl_string.upper() == "N"):
    #print('Not verifying certificates poses a security risk and is not recommended')
    verify_ssl = False
else:
    verify_ssl = True
    
class WellRampDM:
    def __init__(self):
        self.wellName = ""
        self.ramp_up_limit = 0.0
        self.ramp_hihi_limit = 0.0
        self.ramp_hi_limit = 0.0
        self.press_ramp_pfiltered = pd.Series()
        self.delta_press_ramp_pfiltered = pd.Series()
        self.pos_pcv = pd.Series()
        self.qnt_pcv = pd.Series()
        self.delta_pressure_choke = pd.Series()
        self.df = pd.DataFrame()
        self.Model_params_dP = []
        self.Model_params_nextP = []
        self.Model_params_next_dpChoke = []

    def print_variables(self):
        print("wellName:", self.wellName)
        print("ramp_up_limit:", self.ramp_up_limit)
        print("ramp_hihi_limit:", self.ramp_hihi_limit)
        print("ramp_hi_limit:", self.ramp_hi_limit)
        print("press_ramp_pfiltered: ", self.press_ramp_pfiltered)
        print("delta_press_ramp_pfiltered:", self.delta_press_ramp_pfiltered)
        print("pos_pcv:", self.pos_pcv)
        print("qnt_pcv:", self.qnt_pcv)
        print("delta_pressure_choke:", self.delta_pressure_choke)
        print("df:", self.df)
        print("Model_params_dP:", self.Model_params_dP)
        print("Model_params_nextP:", self.Model_params_nextP)
        print("Model_params_next_dpChoke:", self.Model_params_next_dpChoke)

def delta_series(value_series):
    delta_value_series = value_series.diff()
    return delta_value_series

# Function to filter out non-numeric values
def filter_numeric_values(series):
    return series[pd.to_numeric(series, errors='coerce').notnull()]

alpha_value = 0.3
# Define a function for polynomial curve fitting
def curve_fit_polynomial(series, column_names, degree=3):
    if series.name in column_names:  # Check if the column name is in the specified column_names list
        non_nan_series = series.dropna()  # Drop NaN values

        if not non_nan_series.empty:  # Check if non_nan_series is not empty
            x_data = (non_nan_series.index - non_nan_series.index[0]).total_seconds().values.astype(np.float64)
            y_data = non_nan_series.values.astype(np.float64)

            if len(x_data) > 1:  # Only perform curve fitting if sufficient non-NaN data points
                coeffs = np.polyfit(x_data, y_data, degree)  # Perform polynomial curve fitting
                fitted_curve = np.poly1d(coeffs)  # Create fitted curve function

                x_all = (series.index - series.index[0]).total_seconds().values.astype(np.float64)
                y_filled = fitted_curve(x_all)
                series_filled = series.copy()
                series_filled[series.isnull()] = y_filled[series.isnull()]  # Update NaN values with fitted curve values
            else:
                series_filled = series.copy()  # If not enough non-NaN values for curve fitting, just copy the series
        else:
            series_filled = series.copy()  # If non_nan_series is empty, just copy the series
    else:
        series_filled = series.copy()  # For columns not in column_names list, just copy the series

    return series_filled

def handle_duplicates(series):
    return series.groupby(level=0).first()  # You can adjust this logic based on your data

def convert_to_timezone_naive(timestamp):
    if timestamp.tz is not None:
        return timestamp.tz_localize(None)
    else:
        return timestamp

# Convert timestamp to millisecond resolution
def convert_to_millisecond_resolution(ts):
    return ts.floor('L')

# PI Web API helper functions
def read_attribute_snapshot(piwebapi_url, asset_server, user_name, user_password,
                            piwebapi_security_method, verify_ssl, BASE_PATH, element_name, attribute_tag):
    """ Read a single value
        @param piwebapi_url string: The URL of the PI Web API
        @param asset_server string: Name of the Asset Server
        @param user_name string: The user's credentials name
        @param user_password string: The user's credentials password
        @param piwebapi_security_method string: Security method: basic or kerberos
        @param verify_ssl: If certificate verification will be performed
    """
    #print('readAttributeSnapshot')

    #  create security method - basic or kerberos
    security_method = call_security_method(
        piwebapi_security_method, user_name, user_password)

    element_path = rf'{BASE_PATH}\{element_name}'
    #  Get the sample tag
    request_url = '{}/attributes?path=\\\\{}\\{}\\{}|{}'.format(
        piwebapi_url, asset_server, OSI_AF_DATABASE, element_path, attribute_tag)
    response = requests.get(request_url, auth=security_method, verify=verify_ssl)

    #  Only continue if the first request was successful
    if response.status_code == 200:
        #print(response.text)
        #  Deserialize the JSON Response
        data = json.loads(response.text)

        url = urlparse(piwebapi_url + '/streams/' + data['WebId'] + '/value')
        # Validate URL
        assert url.scheme == 'https'
        assert url.geturl().startswith(piwebapi_url)

        #  Read the single stream value
        response = requests.get(url.geturl(),
                                auth=security_method, verify=verify_ssl)

        if response.status_code != 200:
            print(response.status_code, response.reason, response.text)
    else:
        print(response.status_code, response.reason, response.text)
    return response

def call_headers(include_content_type):
    """ Create API call headers
        @includeContentType boolean: Flag determines whether or not the
        content-type header is included
    """
    if include_content_type is True:
        header = {
            'content-type': 'application/json',
            'X-Requested-With': 'XmlHttpRequest'
        }
    else:
        header = {
            'X-Requested-With': 'XmlHttpRequest'
        }

    return header


def call_security_method(security_method, user_name, user_password):
    """ Create API call security method
        @param security_method string: Security method to use: basic or kerberos
        @param user_name string: The user's credentials name
        @param user_password string: The user's credentials password
    """
    from requests.auth import HTTPBasicAuth

    security_auth = HTTPBasicAuth(user_name, user_password)

    return security_auth

def read_attribute_stream(piwebapi_url, asset_server, user_name, user_password,
                          piwebapi_security_method, verify_ssl, BASE_PATH, element_name, attribute_tag, time_back: str = '*-2d'):
    """ Read a set of values
        @param piwebapi_url string: The URL of the PI Web API
        @param asset_server string: Name of the Asset Server
        @param user_name string: The user's credentials name
        @param user_password string: The user's credentials password
        @param piwebapi_security_method string: Security method: basic or kerberos
        @param verify_ssl: If certificate verification will be performed
    """
    #print('readAttributeStream')

    #  create security method - basic or kerberos
    security_method = call_security_method(
        piwebapi_security_method, user_name, user_password)

    element_path = rf'{BASE_PATH}\{element_name}'
    #  Get the sample tag
    request_url = '{}/attributes?path=\\\\{}\\{}\\{}|{}'.format(
        piwebapi_url, asset_server, OSI_AF_DATABASE, element_path, attribute_tag)

    url = urlparse(request_url)
    # Validate URL
    assert url.scheme == 'https'
    assert url.geturl().startswith(piwebapi_url)

    response = requests.get(url.geturl(), auth=security_method, verify=verify_ssl)

    #  Only continue if the first request was successful
    if response.status_code == 200:
        #  Deserialize the JSON Response
        data = json.loads(response.text)

        url = urlparse(piwebapi_url + '/streams/' + data['WebId'] +
                       f'/recorded?startTime={time_back}')
        # Validate URL
        assert url.scheme == 'https'
        assert url.geturl().startswith(piwebapi_url)

        #  Read the set of values
        response = requests.get(
            url.geturl(), auth=security_method, verify=verify_ssl)

        if response.status_code != 200:
            print(response.status_code, response.reason, response.text)
    else:
        print(response.status_code, response.reason, response.text)
    return response

def json_to_dataframe(json_data):
    data = json.loads(json_data)
    df = pd.DataFrame(data['Items'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    df.rename(columns={'Value': 'Press.Ramp.PFiltered Values'}, inplace=True)
    
    return df

def json_to_series(json_data):
    """
    Converts JSON data into a Pandas Series with datetime index.

    Parameters:
    - json_data (str): JSON string containing the data.

    Returns:
    - pd.Series: Series with timestamps as index and corresponding values.
    """
    # Load the JSON data
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON data: {e}")

    items = data.get('Items', [])
    
    if not items:
        raise ValueError("No 'Items' found in the JSON data.")
    
    # Lists to store timestamps and values
    timestamps = []
    values = []
    
    # List to collect problematic timestamps
    problematic_timestamps = []
    
    for idx, item in enumerate(items):
        ts = item.get('Timestamp')
        value = item.get('Value')
        
        if not ts:
            print(f"Warning: Missing 'Timestamp' in item at index {idx}: {item}")
            problematic_timestamps.append(ts)
            continue  # Skip items without a timestamp
        
        ts = ts.strip()  # Remove any leading/trailing whitespace
        
        # Replace 'Z' with '+00:00' to standardize timezone
        if ts.endswith('Z'):
            ts = ts[:-1] + '+00:00'
        
        # If timestamp contains fractional seconds, limit to 6 digits
        if '.' in ts:
            # Use regex to match the fractional seconds and limit to 6 digits
            # Example: "2024-11-19T20:35:03.4740142+00:00" => "2024-11-19T20:35:03.474014+00:00"
            ts = re.sub(
                r'\.(\d{1,6})\d*([+-]\d{2}:\d{2})$',
                lambda m: f".{m.group(1).ljust(6, '0')}{m.group(2)}",
                ts
            )
        
        # Append the cleaned timestamp and its value
        timestamps.append(ts)
        values.append(value)
    
    # For debugging: print all cleaned timestamps before parsing
    #print("Cleaned timestamps:", timestamps)
    
    # Convert the list of timestamps to datetime objects
    datetime_index = pd.to_datetime(timestamps, utc=True, errors='coerce')
    
    # Identify any timestamps that couldn't be parsed
    mask = datetime_index.isnull()
    if mask.any():
        failed_timestamps = [ts for ts, invalid in zip(timestamps, mask) if invalid]
        #print("The following timestamps could not be parsed:")
        #for ts in failed_timestamps:
        #    print(f" - {ts}")
        
        # Remove problematic entries
        valid_indices = ~mask
        timestamps = [ts for ts, valid in zip(timestamps, valid_indices) if valid]
        values = [v for v, valid in zip(values, valid_indices) if valid]
        datetime_index = datetime_index[valid_indices]
    
    if not datetime_index.empty:
        # Create the Pandas Series
        series_data = pd.Series(values, index=datetime_index)
        return series_data
    else:
        raise ValueError("No valid timestamps found in the data.")


def read_attribute_selected_fields(piwebapi_url, asset_server, user_name, user_password,
                                   piwebapi_security_method, verify_ssl):
    """ Read sampleTag values with selected fields to reduce payload size
        @param piwebapi_url string: The URL of the PI Web API
        @param asset_server string: Name of the Asset Server
        @param user_name string: The user's credentials name
        @param user_password string: The user's credentials password
        @param piwebapi_security_method string: Security method: basic or kerberos
        @param verify_ssl: If certificate verification will be performed
    """
    print('readAttributeSelectedFields')

    #  create security method - basic or kerberos
    security_method = call_security_method(
        piwebapi_security_method, user_name, user_password)

    #  Get the sample tag
    request_url = '{}/attributes?path=\\\\{}\\{}\\{}|{}'.format(
        piwebapi_url, asset_server, OSI_AF_DATABASE, OSI_AF_ELEMENT, OSI_AF_ATTRIBUTE_TAG)
    response = requests.get(request_url,
                            auth=security_method, verify=verify_ssl)

    #  Only continue if the first request was successful
    if response.status_code == 200:
        #  Deserialize the JSON Response
        data = json.loads(response.text)

        url = urlparse(piwebapi_url + '/streams/' + data['WebId'] +
                       '/recorded?startTime=*-2d&selectedFields=Items.Timestamp;Items.Value')
        # Validate URL
        assert url.scheme == 'https'
        assert url.geturl().startswith(piwebapi_url)

        #  Read a set of values and return only the specified columns
        response = requests.get(url.geturl(),
                                auth=security_method, verify=verify_ssl)
        if response.status_code == 200:
            print('SampleTag Values with Selected Fields')
            print(json.dumps(json.loads(response.text), indent=4, sort_keys=True))
        else:
            print(response.status_code, response.reason, response.text)
    else:
        print(response.status_code, response.reason, response.text)
    return response

def read_attribute_snapshot_value(piwebapi_url, asset_server, user_name, user_password,
                        piwebapi_security_method, verify_ssl, BASE_PATH, element_name, attribute_tag):
    response = read_attribute_snapshot(piwebapi_url, asset_server, user_name, user_password,
                        piwebapi_security_method, verify_ssl, BASE_PATH, element_name, attribute_tag)

    data = json.loads(response.text)
    return data['Value']

# Create and populate Well classes
def load_oil_wells(start_time: str = '*-2d'):
    WellDMs = []
    for well_name in WELL_NAMES:
        print(f'Getting data for well: {well_name}')
        try:
            well_ramp_dm = WellRampDM()
            well_ramp_dm.wellName = well_name
        
            attribute_tagname = 'Ramp Up Limit (psi/hr)'
            if DEBUG:
                print(f'gettin attribute" {attribute_tagname}')
            value = read_attribute_snapshot_value(piwebapi_url, af_server_name, piwebapi_user, piwebapi_password,
                                piwebapi_security_method, verify_ssl, BASE_PATH, well_name, attribute_tagname)
            well_ramp_dm.ramp_up_limit = value
        
            attribute_tagname = 'Ramp HiHi Limit %'
            if DEBUG:
                print(f'gettin attribute" {attribute_tagname}')
            value = read_attribute_snapshot_value(piwebapi_url, af_server_name, piwebapi_user, piwebapi_password,
                                piwebapi_security_method, verify_ssl, BASE_PATH, well_name, attribute_tagname)
            well_ramp_dm.ramp_hihi_limit = value
        
            attribute_tagname = 'Ramp Hi Limit %'
            if DEBUG:
                print(f'gettin attribute" {attribute_tagname}')
            value = read_attribute_snapshot_value(piwebapi_url, af_server_name, piwebapi_user, piwebapi_password,
                                piwebapi_security_method, verify_ssl, BASE_PATH, well_name, attribute_tagname)
            well_ramp_dm.ramp_hi_limit = value    
    
            attribute_tagname = 'Press.Ramp.PFiltered'
            if DEBUG:
                print(f'gettin attribute" {attribute_tagname}')
            value = read_attribute_stream(piwebapi_url, af_server_name, piwebapi_user, piwebapi_password,
                            piwebapi_security_method, verify_ssl, BASE_PATH, well_name, attribute_tagname, start_time)
            series_data = json_to_series(value.text)    
            well_ramp_dm.press_ramp_pfiltered = series_data   
            well_ramp_dm.delta_press_ramp_pfiltered = well_ramp_dm.press_ramp_pfiltered.diff().abs()
            
            attribute_tagname = 'Pos.PCV'
            if DEBUG:
                print(f'gettin attribute" {attribute_tagname}')
            value = read_attribute_stream(piwebapi_url, af_server_name, piwebapi_user, piwebapi_password,
                            piwebapi_security_method, verify_ssl, BASE_PATH, well_name, attribute_tagname, start_time)
            series_data = json_to_series(value.text)        
            well_ramp_dm.pos_pcv = series_data
    
            attribute_tagname = 'Qnt.PCV'
            if DEBUG:
                print(f'gettin attribute" {attribute_tagname}')
            value = read_attribute_stream(piwebapi_url, af_server_name, piwebapi_user, piwebapi_password,
                            piwebapi_security_method, verify_ssl, BASE_PATH, well_name, attribute_tagname,start_time)
            series_data = json_to_series(value.text)        
            well_ramp_dm.qnt_pcv = series_data         
    
            attribute_tagname = 'DPress.Choke'
            if DEBUG:
                print(f'gettin attribute" {attribute_tagname}')
            value = read_attribute_stream(piwebapi_url, af_server_name, piwebapi_user, piwebapi_password,
                            piwebapi_security_method, verify_ssl, BASE_PATH, well_name, attribute_tagname, start_time)
            series_data = json_to_series(value.text)        
            well_ramp_dm.delta_pressure_choke = series_data
            well_ramp_dm.delta_press_ramp_pfiltered = well_ramp_dm.press_ramp_pfiltered.diff().abs()
    
            attribute_tagname = 'Opt.Valve.Response.Model'
            if DEBUG:
                print(f'gettin attribute" {attribute_tagname}')
            value = read_attribute_snapshot_value(piwebapi_url, af_server_name, piwebapi_user, piwebapi_password,
                                piwebapi_security_method, verify_ssl, BASE_PATH, well_name, attribute_tagname)
            well_ramp_dm.Model_params_dP = value      
    
            attribute_tagname = 'Opt.Valve.Response.Model1'
            if DEBUG:
                print(f'gettin attribute" {attribute_tagname}')
            value = read_attribute_snapshot_value(piwebapi_url, af_server_name, piwebapi_user, piwebapi_password,
                                piwebapi_security_method, verify_ssl, BASE_PATH, well_name, attribute_tagname)
            well_ramp_dm.Model_params_nextP = value        
    
            attribute_tagname = 'Opt.Valve.Response.SP'
            if DEBUG:
                print(f'gettin attribute" {attribute_tagname}')
            value = read_attribute_snapshot_value(piwebapi_url, af_server_name, piwebapi_user, piwebapi_password,
                                piwebapi_security_method, verify_ssl, BASE_PATH, well_name, attribute_tagname)
            well_ramp_dm.Model_params_next_dpChoke = value   

            # feature engineering logic
            # Reindex each Series to a common index
            # Collect unique timestamps from all series
            if DEBUG:
                print(f'process all_series_timestamps')            
            all_series_timestamps = set()
            for s in [well_ramp_dm.press_ramp_pfiltered, well_ramp_dm.delta_press_ramp_pfiltered, well_ramp_dm.pos_pcv, well_ramp_dm.qnt_pcv, well_ramp_dm.delta_pressure_choke]:
                if not s.empty:
                    all_series_timestamps.update(s.index)

            # Get the minimum and maximum timestamps from the unique series timestamps
            min_timestamp = min(all_series_timestamps, default=Timestamp.max)
            max_timestamp = max(all_series_timestamps, default=Timestamp.min)

            # Create a new DatetimeIndex with milliseconds frequency
            if DEBUG:
                print(f'Create a new DatetimeIndex with milliseconds frequency')            
            common_index = pd.DatetimeIndex(list(sorted(all_series_timestamps)))
        
            # Handle duplicates for Series objects
            if DEBUG:
                print(f'Handle duplicates for Series objects')   
            press_ramp_pfiltered = handle_duplicates(well_ramp_dm.press_ramp_pfiltered)
            delta_press_ramp_pfiltered = handle_duplicates(well_ramp_dm.delta_press_ramp_pfiltered)
            delta_pressure_choke = handle_duplicates(well_ramp_dm.delta_pressure_choke)
            pos_pcv = handle_duplicates(well_ramp_dm.pos_pcv)
            qnt_pcv = handle_duplicates(well_ramp_dm.qnt_pcv)
        
            press_ramp_pfiltered = press_ramp_pfiltered.reindex(common_index)
            delta_press_ramp_pfiltered = delta_press_ramp_pfiltered.reindex(common_index)
            pos_pcv = pos_pcv.reindex(common_index)
            qnt_pcv = qnt_pcv.reindex(common_index)
            delta_pressure_choke = delta_pressure_choke.reindex(common_index)

            # Concatenate the reindexed Series into a DataFrame
            if DEBUG:
                print(f'Concatenate the reindexed Series into a DataFrame')              
            df = pd.concat([press_ramp_pfiltered, delta_press_ramp_pfiltered, pos_pcv, qnt_pcv, delta_pressure_choke], axis=1)
        
            # Rename the columns
            df.columns = ['press_ramp_pfiltered', 'delta_press_ramp_pfiltered', 'pos_pcv', 'qnt_pcv', 'delta_pressure_choke']
        
            # Define the column names for which curve fitting should be applied
            selected_columns = ['press_ramp_pfiltered', 'delta_pressure_choke', 'delta_press_ramp_pfiltered'] 
        
            # Apply curve fitting function to selected columns in the DataFrame
            if DEBUG:
                print(f'Apply curve fitting function to selected columns in the DataFrame')                  
            df = df.apply(curve_fit_polynomial, args=(selected_columns,), axis=0)

            # Fill NaN values in 'pos_pcv' column by forward fill and backward fill
            df['pos_pcv'] = df['pos_pcv'].ffill().bfill()
            # Fill NaN values in 'qnt_pcv' column by forward fill and backward fill
            df['qnt_pcv'] = df['qnt_pcv'].ffill().bfill()

            well_ramp_dm.df = df            
            
            WellDMs.append(well_ramp_dm)
        except:
            print(f'Error adding well: {well_name}')

    return WellDMs

def optimize_wellRampUp(well_ramp_dm, delta_press_target, delta_press_adjust, Cv_text, current_press_ramp_pfiltered, current_delta_pressure_choke):
    Cv_initial = 0
    timeToNextChange = 0
    try:
        # Maximum iterations and minimum delta_press_ramp_pfiltered
        max_iterations = 10000
    
        # Initialize the iteration count
        iteration = 0
    
        # The DataFrame operations are used here to: filter the data, handle missing values, split the data into dependent and independent variables, and preprocess the data for model fitting.
        # Cleanup data and add features
        df = well_ramp_dm.df[well_ramp_dm.df['delta_press_ramp_pfiltered'] > 1].copy()  # Ensure we are working with a copy
        # Convert datatype to float and replace non-numeric values with NaN, if any exist.
        df['press_ramp_pfiltered'] = pd.to_numeric(df['press_ramp_pfiltered'], errors='coerce')
        df['delta_press_ramp_pfiltered'] = pd.to_numeric(df['delta_press_ramp_pfiltered'], errors='coerce')
        df['pos_pcv'] = pd.to_numeric(df['pos_pcv'], errors='coerce')
        df['qnt_pcv'] = pd.to_numeric(df['qnt_pcv'], errors='coerce')
        df['delta_pressure_choke'] = pd.to_numeric(df['delta_pressure_choke'], errors='coerce')
        df['next_press_ramp_pfiltered'] = df['press_ramp_pfiltered'].shift(-1)
        df['next_delta_pressure_choke'] = df['delta_pressure_choke'].shift(-1)
        # keep only rows where pressure is declining
        df = df[df['next_press_ramp_pfiltered'] <= df['press_ramp_pfiltered']]
        # Remove rows having any NaN values
        df.dropna(inplace=True)
        # Remove rows having any NaN values
        df.dropna(inplace=True)
        # Ensure no division by zero or negative values
        valid_indices = df['delta_press_ramp_pfiltered'] > 0
        df = df[valid_indices]
        # Now re-run the sqrt operation.
        df['Q'] = df[Cv_text] * np.sqrt(df['delta_pressure_choke'] / df['delta_press_ramp_pfiltered'])
        df = df[df['Q'] >= 1]
        #df = df[df['next_press_ramp_pfiltered'] < df['press_ramp_pfiltered']]
        df_high = df[df['delta_press_ramp_pfiltered'] > 10]
        df_choke = df[df['delta_pressure_choke'] > df['next_delta_pressure_choke']]

        if(df['delta_press_ramp_pfiltered'].iloc[-1] < 20):
            print(f'Well {well_ramp_dm.wellName} is not in startup mode')
            return Cv_initial, timeToNextChange
        else:
            print(f"Well {well_ramp_dm.wellName} is in startup mode. delta_press_ramp_pfiltered = {df['delta_press_ramp_pfiltered'].iloc[-1]}")

    
    
        #define equations to fit and use
        # The objective_function returns current Cv which it got as parameter. In this script this is used in optimization function later. Optimizer will use this function in order to minimize its returned value (optimization goal).
        def objective_function(Cv):
            return Cv
    
        initial_params_5 = [1.0, 1.0, 1.0, 1.0, 1.0]  
        # Define the model function
        def model_func_delta_press(xdata, A, B, C, D, E):
            Pres, dPchoke, Cv = xdata  # Unpack the tuple of independent variables
            return ((A * Pres ** 2 + B * Pres + C + D * dPchoke ** 2 + E * dPchoke) * Cv)
    
        def model_next_press(X, A, B, C, D, E):
            P, dPchoke, Cv = X  # Unpack the input array into P and Cv
            return (A * P ** 2 + B * P + C + D * dPchoke + E * dPchoke ** 2)
    
        # equation to predict next_press_ramp_pfiltered
        def model_next_press(X, A, B, C, D, E):
            P, dPchoke, Cv = X  # Unpack the input array into P and Cv
            return (A * P ** 2 + B * P + C + D * dPchoke + E * dPchoke ** 2)
        
        # Extract the coefficients
        A_optimal_dp, B_optimal_dp, C_optimal_dp, D_optimal_dp, E_optimal_dp = well_ramp_dm.Model_params_dP
        model_array_dp = [A_optimal_dp, B_optimal_dp, C_optimal_dp, D_optimal_dp, E_optimal_dp]    
        
        A_optimal_nextP, B_optimal_nextP, C_optimal_nextP, D_optimal_nextP, E_optimal_nextP = well_ramp_dm.Model_params_nextP
        model_array_nextP = [A_optimal_nextP, B_optimal_nextP, C_optimal_nextP, D_optimal_nextP, E_optimal_nextP]
    
        A_optimal_next_dpChoke, B_optimal_next_dpChoke, C_optimal_next_dpChoke, D_optimal_next_dpChoke, E_optimal_next_dpChoke = well_ramp_dm.Model_params_next_dpChoke
        model_array_next_dpChoke = [A_optimal_next_dpChoke, B_optimal_next_dpChoke, C_optimal_next_dpChoke, D_optimal_next_dpChoke, E_optimal_next_dpChoke]
    
        
        # Initialize the list
        cv_list = []
        press_ramp_list = []
        iteration = 0  # Ensure iteration is initialized before use
    
        # Initialize current_press_ramp_pfiltered_iter
        current_press_ramp_pfiltered_iter = current_press_ramp_pfiltered
    
        Cv_prev = 0
        # Constraint where delta_press_ramp_pfiltered should be equal to 150
        Cv_initial = 1
        current_press_ramp_pfiltered_iter = current_press_ramp_pfiltered
        current_delta_pressure_choke_iter = current_delta_pressure_choke
        constraints = ({
            "type": "eq",
            "fun": lambda Cv: model_func_delta_press(
                (current_press_ramp_pfiltered_iter, current_delta_pressure_choke, Cv),
                A_optimal_dp, B_optimal_dp, C_optimal_dp, D_optimal_dp, E_optimal_dp
            ) - delta_press_target
        })
        # Apply minimization with constraints
        result = minimize(
            objective_function,
            Cv_initial,
            method='SLSQP',
            constraints=constraints,
            options={"disp": False})
        
        # Extract optimal Cv_initial from result
        Cv_initial = result.x[0]
    
        # Determine time to next control valve change requirement
        while iteration < max_iterations:
            # get the next pressure
            X = (current_press_ramp_pfiltered_iter, current_delta_pressure_choke_iter, Cv_initial)
            current_press_ramp_pfiltered_iter = model_next_press(X, A_optimal_nextP, B_optimal_nextP, C_optimal_nextP, D_optimal_nextP, E_optimal_nextP)
            #print(f"current_press_ramp_pfiltered_iter: {current_press_ramp_pfiltered_iter}")
    
            # get the next dp Choke Valve
            X = (current_press_ramp_pfiltered_iter, current_delta_pressure_choke_iter, Cv_initial)
            current_delta_pressure_choke_iter = model_next_press(X, A_optimal_next_dpChoke, B_optimal_next_dpChoke, C_optimal_next_dpChoke, D_optimal_next_dpChoke, E_optimal_next_dpChoke)
            #print(f"current_delta_pressure_choke_iter: {current_delta_pressure_choke_iter}")
    
            # get the delta P to check for changes
            xdata = (current_press_ramp_pfiltered_iter, current_delta_pressure_choke, Cv_initial)
            dp_model = model_func_delta_press(xdata, A_optimal_dp, B_optimal_dp, C_optimal_dp, D_optimal_dp, E_optimal_dp)
            #print(f"dp_model: {dp_model}")
    
            # Simulation stopping criteria
            if dp_model >= delta_press_adjust:
                print("Simulation Terminated: control valve change criteria met.")
                break
    
            iteration += 1
    
        #print(f"iteration: {iteration}")
        average_timestamp_diff = df_choke.index.to_series().diff().median()
        timeToNextChange = (iteration + 1) * average_timestamp_diff

        write_attribute_value(PI_WEB_API_URL, PI_AF_SERVER, PI_AF_DATABASE, USERNAME, PASSWORD,
                             piwebapi_security_method, verify_ssl, BASE_PATH,
                             well_ramp_dm.wellName, 'Press.Ramp.PyForecast.5min', Cv_initial)
    
        total_minutes = timeToNextChange.total_seconds() / 60
        rounded_minutes = round(total_minutes, 2)
        write_attribute_value(PI_WEB_API_URL, PI_AF_SERVER, PI_AF_DATABASE, USERNAME, PASSWORD,
                             piwebapi_security_method, verify_ssl, BASE_PATH,
                             well_ramp_dm.wellName, 'Press.Ramp.PyForecast.60min', rounded_minutes)      
    except Exception as e:
        print(f'Error calculating setpoint info for well: {well_ramp_dm.wellName}. Exception: {str(e)}')

    return Cv_initial, timeToNextChange

def update_attribute_value(webid, value):
    """Update the attribute value in PI AF using the PI Web API."""
    url = f"{PI_WEB_API_URL}/attributes/{webid}/value"
    data = {
        "Timestamp": datetime.datetime.now().isoformat(),
        "Value": value
    }
    response = requests.put(url, json=data, auth=HTTPBasicAuth(USERNAME, PASSWORD), verify=False)
    if response.status_code == 204:
        print(f"Successfully updated attribute with WebID {webid}.")
    else:
        print(f"Failed to update attribute with WebID {webid}: {response.status_code}, {response.text}")
        
def write_attribute_value(piwebapi_url, asset_server, pi_af_database, user_name, user_password,
                         piwebapi_security_method, verify_ssl, BASE_PATH,
                         element_name, attribute_tag, value, timestamp=None):
    """
    Write a value to a PI AF attribute.

    @param piwebapi_url string: The URL of the PI Web API
    @param asset_server string: Name of the Asset Server
    @param user_name string: The user's credentials name
    @param user_password string: The user's credentials password
    @param piwebapi_security_method string: Security method: basic or kerberos
    @param verify_ssl: If certificate verification will be performed
    @param BASE_PATH string: Base path of the AF hierarchy
    @param element_name string: Name of the AF Element
    @param attribute_tag string: Tag of the AF Attribute
    @param value: The value to write to the attribute
    @param timestamp string (optional): The timestamp for the value in ISO 8601 format
    @return response: The HTTP response object
    """
    # Create security method - basic or kerberos
    security_method = call_security_method(
        piwebapi_security_method, user_name, user_password)

    # Construct the full element path
    element_path = rf'{BASE_PATH}\{element_name}'

    # Define the AF Database name (replace 'OSI_AF_DATABASE' with your actual AF Database name)
    OSI_AF_DATABASE = pi_af_database 

    # Get the attribute's WebId
    request_url = '{}/attributes?path=\\\\{}\\{}\\{}|{}'.format(
        piwebapi_url, asset_server, OSI_AF_DATABASE, element_path, attribute_tag)
    response = requests.get(request_url, auth=security_method, verify=verify_ssl)

    # Only continue if the first request was successful
    if response.status_code == 200:
        # Deserialize the JSON Response
        data = response.json()

        web_id = data['WebId']
        print(f'web_id: {web_id}')
        
        # Construct the URL for writing the value
        write_url = f"{piwebapi_url}/streams/{web_id}"

        update_attribute_value(web_id, value)

    else:
        print(f"Failed to retrieve attribute WebId: {response.status_code} {response.reason}")
        print(response.text)

    return response
    
WellDMs = load_oil_wells('*-30d')
for well_ramp_dm in WellDMs:
    # get initial values
    delta_press_target = well_ramp_dm.ramp_up_limit*well_ramp_dm.ramp_hihi_limit/100
    delta_press_adjust = well_ramp_dm.ramp_up_limit*well_ramp_dm.ramp_hi_limit/100
    print(f'Rampup limit: {delta_press_target}')
    print(f'Rampup limit adjust: {delta_press_adjust}')
    Cv_text = "qnt_pcv"

    current_press_ramp_pfiltered = 15000
    current_delta_pressure_choke = 3000

    max_iterations = 10000
    min_delta_press = 10
    #iteration = 0
    
    Cv_initial, timeToNextChange = optimize_wellRampUp(well_ramp_dm, delta_press_target, delta_press_adjust, Cv_text, current_press_ramp_pfiltered, current_delta_pressure_choke)
    print(f'Cv_initial: {Cv_initial}/t timeToNextChange: {timeToNextChange}')
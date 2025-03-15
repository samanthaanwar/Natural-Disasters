import os
import requests
import pandas as pd
import datetime
from datetime import datetime, timedelta
import xarray as xr # x-array
import numpy as np # numpy

def load_tracks():
    path = 'data/NA_data.nc'
    if os.path.exists(path):
        tks = xr.open_dataset(path, engine="netcdf4", decode_times=False)
        return tks

    # IBTrACS.NA.v04r00.nc presents data from 1842-10-25 through 2023-06-07 
    url = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/netcdf/IBTrACS.NA.v04r00.nc'

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()  # Raise an error for HTTP codes >= 400
        with open(path, 'wb') as f:
            f.write(response.content)
        print("File downloaded successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    tks = xr.open_dataset('data/NA_data.nc', engine="netcdf4", decode_times=False)
    return tks


def load_clusters():
    data = pd.read_csv('data/storm_landfall_times.csv')
    return data.groupby('spatmoment_label')


def storm_time_to_datetime(storm_time):
    initial_day = '1858-11-17 00:00:00'
    initial_day = datetime.strptime(initial_day, '%Y-%m-%d %H:%M:%S')
    new_datetime = initial_day + timedelta(days=storm_time)
    return datetime(
        year=new_datetime.year,
        month=new_datetime.month,
        day=new_datetime.day,
        hour=new_datetime.hour,
    )

def datetime_to_storm_time(datetime_):
    initial_day = datetime.strptime('1858-11-17 00:00:00', '%Y-%m-%d %H:%M:%S')

    delta = (datetime_ - initial_day)
    return delta.days + delta.seconds / 3600 / 24

def date_str_to_storm_time(date_str):
    initial_day = datetime.strptime('1858-11-17 00:00:00', '%Y-%m-%d %H:%M:%S')
    date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    delta = (date - initial_day)
    return delta.days + delta.seconds / 3600 / 24

def get_intensity(storm):
    #wmo_wind is the max sustained wind speed 
    wind_speed = storm.wmo_wind.values
    #filter for over land
    #landfall = storm.landfall.values
    #wind_speed = wind_speed[~landfall]
    wind_speed = wind_speed[~np.isnan(wind_speed)]
    #wind_spped = np.average(wind_speed)
    #wind_speed = np.max(wind_speed)

    if (len(wind_speed) == 0):
        return -1

    return np.max(wind_speed)

state_dict = {'Alabama':'AL', 'Arkansas': 'AR', 'Connecticut': 'CT',
              'Delaware': 'DE', 'Florida': 'FL', 'Kentucky': 'KY',
              'Louisiana':'LA', 'Maine':'ME', 'Maryland':'MD',
              'Massachusetts':'MA', 'Mississippi':'MS',
              'Missouri':'MO', 'New Hampshire':'NH', 'New Jersey': 'NJ',
              'New York':'NY', 'North Carolina':'NC', 'Oklahoma':'OK',
              'Rhode Island':'RI', 'South Carolina':'SC', 'Tennessee':'TN',
              'Texas': 'TX', 'Virginia':'VA', 'West Virginia':'WV',
              'Georgia':'GA'}

def get_intensity_time(storm): # return intensity at every timestep rather than max per storm
    wind_speed = storm.wmo_wind
    if (len(wind_speed) == 0):
        return -1
    return wind_speed

def get_storm_data(storms_xr, storm_name):
    storms_xr['name'] = storms_xr['name'].astype(str)
    storm = storms_xr.where(storms_xr['name'] == storm_name.upper(), drop=True)
    storm = storm.to_dataframe().reset_index()

    return storm

def get_timestamps(storms_xr, storm_name):
    initial_day = datetime(1858, 11, 17)
    storm = get_storm_data(storms_xr, storm_name)
    storm_times = storm.time

    timestamps, dates, hours = [], [], []

    for time in storm_times:
        if np.isnan(time):
            timestamps.append(np.nan)
            dates.append(np.nan)
            hours.append(np.nan)
        else:
            x = initial_day + timedelta(days=time)
            timestamps.append(pd.Timestamp(x))
            dates.append(x.date())
            hours.append(x.hour)

    return timestamps, dates, hours

def outages_bystate(df, year):
    data = df[year].drop(columns='hour')
    data = data.groupby(by=['date', 'fips_code', 'county', 'state']).max().reset_index()
    data = data.drop(columns=['fips_code', 'county'])
    data = data.groupby(by=['state', 'date']).sum().reset_index()
    data['state_abbr'] = data['state'].map(state_dict)
    return data

def storm_df(storms_xr, storm_name):

    storm_choice = get_storm_data(storms_xr, storm_name)
    
    storm_dict = {'lat': storm_choice.lat, 
                  'lon':storm_choice.lon,
                  'date': get_timestamps(storms_xr, storm_name)[1], 
                  'intensity': get_intensity_time(storm_choice)}

    storm_df = pd.DataFrame(storm_dict)
    storm_df = storm_df.dropna()

    # max lat lon and intensity each date
    pivoted_storm = storm_df.groupby(by= 'date').max().reset_index()

    return pivoted_storm


def filter_data(data, lower_year, upper_year, longitude_boundary, latitude_boundary):
    # Filter for storms that actually make landfall
    landfall_mask = data.groupby('storm').map(lambda x: (x.landfall == 0).any())
    storms_with_landfall = landfall_mask.storm[landfall_mask]
    data_landfall = data.sel(storm=storms_with_landfall)
    
    # Filter for years that have blackout data
    def passes_year(storm):
        return ((storm.season <= upper_year) & (storm.season >= lower_year)).any()
    
    year_mask = data_landfall.groupby('storm').map(passes_year)
    storms_within_year = year_mask.storm[year_mask]
    data_landfall_year = data_landfall.sel(storm=storms_within_year)
    
    # Filter for storms within the geographical boundaries of the continental US
    def passes_boundary(storm):
        return ((storm.lon <= longitude_boundary) & (storm.lat >= latitude_boundary)).any()
    
    boundary_mask = data_landfall_year.groupby('storm').map(passes_boundary)
    storms_within_boundary = boundary_mask.storm[boundary_mask]
    data_landfall_year_US = data_landfall_year.sel(storm=storms_within_boundary)
    
    return data_landfall_year_US

def get_landfall_lon_lat(storm):
    """Returns the longitude and latitude values where the landfall variable is equal to zero."""
    # Filter the storm data where landfall is equal to zero
    filtered_storm = storm.where(storm.landfall == 0, drop=True)
    
    lon_lst = filtered_storm.lon.values
    lat_lst = filtered_storm.lat.values
    
    lon_lst = lon_lst[~np.isnan(lon_lst)]
    lat_lst = lat_lst[~np.isnan(lat_lst)]
    
    return lon_lst, lat_lst

def get_landfall_moments(storm):
  lon_lst, lat_lst = get_landfall_lon_lat(storm)
  # If the track only has one point, there is no point in calculating the moments
  if len(lon_lst)<= 1: return None
      
  # M1 (first moment = mean). 
  lon_weighted, lat_weighted = np.mean(lon_lst), np.mean(lat_lst)
    
  # M2 (second moment = variance of lat and of lon / covariance of lat to lon
  cv = np.ma.cov([lon_lst, lat_lst])
    
  return [lon_weighted, lat_weighted, cv[0, 0], cv[1, 1], cv[0, 1]]
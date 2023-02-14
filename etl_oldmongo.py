"""
Takes the smart meter csv files from ../mongo/csv, 
then fixes data scaling issues, replaces missing values and exports only the full days in a final
csv file, which it saves as an artifact
"""
from configparser import ConfigParser
import os
from cmath import nan

import pandas as pd

from os.path import abspath
dir_path = os.path.abspath('')
import json
import math
import warnings

import click
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from numpy import nan as nan
from scipy import stats

from utils import none_checker, truth_checker

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
#os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().('backend','mlflow_tracking_uri')
os.environ["MLFLOW_TRACKING_URI"] = 'http://131.154.97.48:5000'
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

warnings.filterwarnings('ignore')

@click.command(help="Takes the smart meter csv  ../oldmongo/csv, "
                    "then fixes data scaling issues, replaces missing values and exports only the full days in a final"
                    "csv file ")
@click.option("--mongo_folder", type=str, default="oldmongo/")
@click.option("--smart_meter_description_csv", type=str, default="smart_meter_description.csv")
@click.option("--resolution", type=str, default='60')    
def main(mongo_folder, smart_meter_description_csv, resolution):
    all_meter_data, dict, meter_description = loadto_df_and_dict(mongo_folder, smart_meter_description_csv)
    data_preprocessing(all_meter_data, dict, meter_description, resolution)

def only_full_days(meter_data): 
    meter_data['date'] = meter_data.index.date
    meter_data['time'] = meter_data.index.time
    full_days = meter_data.pivot(index='date', columns ='time', values='isnull')
    full_days = full_days.loc[~full_days[full_days.columns].any(True)]
    meter_data = meter_data[meter_data['date'].isin(full_days.index)]
    return(meter_data)

def loadto_df_and_dict(mongo_folder, smart_meter_description_csv):
    mongo_folder = none_checker(mongo_folder)
    mongo_folder = abspath(mongo_folder)
    smart_meter_description_csv = none_checker(smart_meter_description_csv)
    smart_meter_description_csv = abspath(smart_meter_description_csv)

    #list of smart meter ids:
    meter_description = pd.read_csv(smart_meter_description_csv, engine='python')
    meter_description.index = meter_description.id
    meters = meter_description.id.tolist()
    
    all_meter_data = pd.DataFrame()

    for meter in meters:
        file_path = mongo_folder + '/csv/' + str(meter) + '.csv'
        df = pd.read_csv(file_path, index_col=0)
        #reformating timestamps to python datetimes
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        all_meter_data = pd.concat([all_meter_data,df])
    
    all_meter_data['timestamp'] = pd.to_datetime(all_meter_data['timestamp'])
    all_meter_data = all_meter_data[all_meter_data.timestamp.dt.year != 1970]
    all_meter_data = all_meter_data[all_meter_data.timestamp.dt.year != 2040]
    all_meter_data = all_meter_data.reset_index()
    all_meter_data.to_csv('test.csv')

    #creating a dictionary with a dataframe for each ID:
    IDs = pd.unique(all_meter_data['id'])
    dict = {}
    for id in IDs:
        df = all_meter_data[all_meter_data['id']==id]
        df.index = df.timestamp
        dict[id] = df

    return(all_meter_data, dict, meter_description)

#code to format titles for the graphs to contain info on consumption, generation, use
def id_title(id, description):
    title = id + "(C,"
    if(not np.isnan(description.loc[id]['Production (kW)']) and description.loc[id]['Production (kW)']>0.0): title = title + "P,"
     
    title = title + str(description.loc[id]['Type'])

    return(title + ")")


def fill_missing_periods(meter_data, filling_window: int, fill_night: bool, id, meter_description, resolution):
    #creating Total Energy field for the energy(Wh) field so that it can be used for interpolation, to fill in missing data:
    meter_data['Active_Energy_+_Total_end'] = meter_data['Active_Energy_+_Total_end'].interpolate()
    meter_data['Active_Energy_-_Total_end'] = meter_data['Active_Energy_-_Total_end'].interpolate()

    meter_data['energy_plus'] = meter_data['Active_Energy_+_Total_end'].diff()
    meter_data['energy_minus'] = meter_data['Active_Energy_-_Total_end'].diff()
    meter_data['energy'] = meter_data['energy_plus'] - meter_data['energy_minus']

    #for smart meters that have production, fill in missing values during the night with 0
    production_list = meter_description[meter_description['Production (kW)']!=0]['id']
    if (id in production_list):
        meter_data['hour'] = meter_data.index.hour
        null_generation = meter_data[meter_data['isnull']==True][meter_data['hour'].isin([0,1,2,3,4,5,6])]
        null_generation['energy'] = 0.0
        null_generation['isnull'] = False
        meter_data.update(null_generation)
    

    if(fill_night):
        #impute every null value that is in the 00:00h to 00:06h interval and has consecutive_null < 24 using energy_difference 
        meter_data['hour'] = meter_data.index.hour
        missing_night = meter_data[meter_data['consecutive_null'].between(1,24)][meter_data['hour'].isin([0,1,2,3,4,5,6])]
        missing_night['isnull'] =  False
        meter_data.update(missing_night)
    
    #impute every null value that isn't more than (filling_window) hours away from the last real value using energy_difference
    missing_lessthan_window = meter_data[meter_data['consecutive_null'].isin(list(range(1, int(filling_window*(60/resolution)))))]
    missing_lessthan_window['isnull'] =  False
    meter_data.update(missing_lessthan_window)

    return(meter_data)


#to fix the scaling issues that many smart meters had when it comes to the total active energy fields:
def fix_scaling(dict):
    #for these smart meters multiply total active energy values before 2019-05-14T10:00:00Z by 10^3 and drop all readings before 2019-04-29 13:00:00
    simple = ['BBB6004', 'BBB6017', 'BBB6020', 'BBB6025', 'BBB6065', 'BBB6100', 'BBB6103', 'BBB6105', 'BBB6133', 'BBB6140', 'BBB6170', 'BBB6173', 'BBB6197'] + ['BBB6168', 'BBB6169', 'BBB6171', 'BBB6177', 'BBB6178', 'BBB6179', 'BBB6180', 'BBB6186', 'BBB6191', 'BBB6198']
    for id in simple:
        df = dict[id]
        df2 = df[df.timestamp < '2019-05-14 10:00:00']   
        df2['Active_Energy_+_Total_ini'] = df2['Active_Energy_+_Total_ini'] * 1000
        df2['Active_Energy_+_Total_end'] = df2['Active_Energy_+_Total_end'] * 1000
        df2['Active_Energy_-_Total_ini'] = df2['Active_Energy_-_Total_ini'] * 1000
        df2['Active_Energy_-_Total_end'] = df2['Active_Energy_-_Total_end'] * 1000
        df.update(df2)
        df = df.drop(df[df.timestamp < '2019-04-29 13:00:00'].index)
        df = df.drop(df[df.timestamp > '2023-04-29 13:00:00'].index)
        df = df.drop(df[df['Active_Energy_+_Total_end']<1000].index)
        if(id == 'BBB6020' or id == 'BBB6025'): df = df.drop(df[df.timestamp < '2019-04-30 13:00:00'].index)
        dict[id] = df

    #for these smart meters we assume after ... is correct (except for BBB6036 where that magnitude is multiplied by 10)
    for id in ['BBB6036']:
        df = dict[id]
        df2 = df[df.timestamp < '2019-05-30 11:00:00'] 
        df2['Active_Energy_+_Total_ini'] = df2['Active_Energy_+_Total_ini']/60000
        df2['Active_Energy_+_Total_end'] = df2['Active_Energy_+_Total_end']/60000
        df2['Active_Energy_-_Total_ini'] = df2['Active_Energy_-_Total_ini']/60000
        df2['Active_Energy_-_Total_end'] = df2['Active_Energy_-_Total_end']/60000
        df.update(df2)
        df2 = df2[df2.timestamp < '2019-05-14 10:00:00'] 
        df2['Active_Energy_+_Total_ini'] = df2['Active_Energy_+_Total_ini'] * 1000 
        df2['Active_Energy_+_Total_end'] = df2['Active_Energy_+_Total_end'] * 1000 
        df2['Active_Energy_-_Total_ini'] = df2['Active_Energy_-_Total_ini'] * 1000 
        df2['Active_Energy_-_Total_end'] = df2['Active_Energy_-_Total_end'] * 1000 
        df.update(df2)
        
        
        df2 = df[df.timestamp > '2019-05-30 10:45:00']
        df2['Active_Energy_+_Total_ini'] = df2['Active_Energy_+_Total_ini'] * 10 
        df2['Active_Energy_+_Total_end'] = df2['Active_Energy_+_Total_end'] * 10
        df2['Active_Energy_-_Total_ini'] = df2['Active_Energy_-_Total_ini'] * 10
        df2['Active_Energy_-_Total_end'] = df2['Active_Energy_-_Total_end'] * 10 
        df.update(df2)
        
        dict[id] = df

    for id in ['BBB6029']:
        df = dict[id]
        df2 = df[df.timestamp < '2019-05-24 13:00:00'] 
        df2['Active_Energy_+_Total_ini'] = df2['Active_Energy_+_Total_ini']/60000
        df2['Active_Energy_-_Total_ini'] = df2['Active_Energy_-_Total_ini']/60000
        df.update(df2)

        df2 = df[df.timestamp < '2019-05-24 12:45:00'] 
        df2['Active_Energy_+_Total_end'] = df2['Active_Energy_+_Total_end']/60000
        df2['Active_Energy_-_Total_end'] = df2['Active_Energy_-_Total_end']/60000
        df.update(df2)
    
        df2 = df2[df2.timestamp < '2019-05-14 10:00:00'] 
        df2['Active_Energy_+_Total_ini'] = df2['Active_Energy_+_Total_ini'] * 1000 
        df2['Active_Energy_+_Total_end'] = df2['Active_Energy_+_Total_end'] * 1000 
        df2['Active_Energy_-_Total_ini'] = df2['Active_Energy_-_Total_ini'] * 1000 
        df2['Active_Energy_-_Total_end'] = df2['Active_Energy_-_Total_end'] * 1000 
        df.update(df2)
        dict[id] = df



    for id in ['BBB6055']:
        df = dict[id]
        df2 = df[df.timestamp < '2019-05-29 13:15:00'] 
        df2['Active_Energy_+_Total_ini'] = df2['Active_Energy_+_Total_ini']/30000
        df2['Active_Energy_+_Total_end'] = df2['Active_Energy_+_Total_end']/30000
        df2['Active_Energy_-_Total_ini'] = df2['Active_Energy_-_Total_ini']/30000
        df2['Active_Energy_-_Total_end'] = df2['Active_Energy_-_Total_end']/30000
        df.update(df2)

        df2 = df2[df2.timestamp < '2019-05-14 10:00:00'] 
        df2['Active_Energy_+_Total_ini'] = df2['Active_Energy_+_Total_ini'] * 1000 
        df2['Active_Energy_+_Total_end'] = df2['Active_Energy_+_Total_end'] * 1000 
        df2['Active_Energy_-_Total_ini'] = df2['Active_Energy_-_Total_ini'] * 1000 
        df2['Active_Energy_-_Total_end'] = df2['Active_Energy_-_Total_end'] * 1000 
        df.update(df2)
        
        #df2 = df2[df2.timestamp < '2019-05-30 10:30:00'] 
        #df2['Active_Energy_+_Total_ini'] = df2['Active_Energy_+_Total_ini'] * 1.25
        #df2['Active_Energy_+_Total_end'] = df2['Active_Energy_+_Total_end'] * 1.25 
        #df2['Active_Energy_-_Total_ini'] = df2['Active_Energy_-_Total_ini'] * 1.25 
        #df2['Active_Energy_-_Total_end'] = df2['Active_Energy_-_Total_end'] * 1.25 
        #df.update(df2)
        dict[id] = df

    for id in ['BBB6063', 'BBB6064']:
        df = dict[id]
        df2 = df[df.timestamp < '2019-05-30 15:00:00'] 
        df2['Active_Energy_+_Total_ini'] = df2['Active_Energy_+_Total_ini']*2
        df2['Active_Energy_-_Total_ini'] = df2['Active_Energy_-_Total_ini']*2
        df.update(df2)

        df2 = df[df.timestamp < '2019-05-30 14:45:00'] 
        df2['Active_Energy_+_Total_end'] = df2['Active_Energy_+_Total_end']*2
        df2['Active_Energy_-_Total_end'] = df2['Active_Energy_-_Total_end']*2
        df.update(df2)

        df2 = df2[df2.timestamp < '2019-05-14 10:00:00'] 
        df2['Active_Energy_+_Total_ini'] = df2['Active_Energy_+_Total_ini'] * 1000 
        df2['Active_Energy_+_Total_end'] = df2['Active_Energy_+_Total_end'] * 1000 
        df2['Active_Energy_-_Total_ini'] = df2['Active_Energy_-_Total_ini'] * 1000 
        df2['Active_Energy_-_Total_end'] = df2['Active_Energy_-_Total_end'] * 1000 
        df.update(df2)

        dict[id] = df


    for id in ['BBB6030']:
        df = dict[id]
        df2 = df[df['Active_Energy_+_Total_ini'] > 8000000000.0]
        df2['Active_Energy_+_Total_ini'] = df2['Active_Energy_+_Total_ini'] / 30
        df2['Active_Energy_-_Total_ini'] = df2['Active_Energy_-_Total_ini'] / 30
        df.update(df2)
        df2 = df[df['Active_Energy_+_Total_end'] > 8000000000.0]
        df2['Active_Energy_+_Total_end'] = df2['Active_Energy_+_Total_end'] / 30
        df2['Active_Energy_-_Total_end'] = df2['Active_Energy_-_Total_end'] / 30
        df.update(df2)

        df2 = df[df['Active_Energy_+_Total_ini'] > 288354648000.0]
        df2['Active_Energy_+_Total_ini'] = df2['Active_Energy_+_Total_ini'] / 1000
        df2['Active_Energy_-_Total_ini'] = df2['Active_Energy_-_Total_ini'] / 1000    
        df.update(df2)
        df2 = df[df['Active_Energy_+_Total_end'] > 288354648000.0]
        df2['Active_Energy_+_Total_end'] = df2['Active_Energy_+_Total_end'] / 1000
        df2['Active_Energy_-_Total_end'] = df2['Active_Energy_-_Total_end'] / 1000
        df.update(df2)
        
        dict[id] = df

    for id in ['BBB6052']:
        df = dict[id]
        df2 = df[df.timestamp < '2019-05-14 10:00:00']   
        df2['Active_Energy_+_Total_ini'] = df2['Active_Energy_+_Total_ini'] * 1000
        df2['Active_Energy_+_Total_end'] = df2['Active_Energy_+_Total_end'] * 1000
        df2['Active_Energy_-_Total_ini'] = df2['Active_Energy_-_Total_ini'] * 1000
        df2['Active_Energy_-_Total_end'] = df2['Active_Energy_-_Total_end'] * 1000
        df.update(df2)

        df2 = df[df['Active_Energy_+_Total_ini'] < 300000000.0]
        df2['Active_Energy_+_Total_ini'] = df2['Active_Energy_+_Total_ini'] * 2
        df2['Active_Energy_-_Total_ini'] = df2['Active_Energy_-_Total_ini'] * 2   
        df.update(df2)
        df2 = df[df['Active_Energy_+_Total_end'] < 300000000.0]
        df2['Active_Energy_+_Total_end'] = df2['Active_Energy_+_Total_end'] * 2
        df2['Active_Energy_-_Total_end'] = df2['Active_Energy_-_Total_end'] * 2
        df.update(df2)
        
        dict[id] = df
    
    return(dict)


def data_preprocessing(all_meter_data, dict, meter_description, resolution):
    resolution = none_checker(resolution)
    resolution = int(resolution)
    with mlflow.start_run(run_name='etl_oldmongo', nested=True) as mlrun:
        fix_scaling(dict)
        #resample:
        IDs = pd.unique(all_meter_data['id'])
        for id in IDs:
            one_meter = dict[id].copy()
            one_meter = one_meter.resample(str(resolution)+'min').agg({'Active_Energy_+_Total_end':np.max, 'Active_Energy_-_Total_end':np.max})
            dict[id] = one_meter
        
        to_export = pd.DataFrame()
        for id in IDs:
            one_meter = dict[id].copy()
            one_meter['isnull'] = one_meter['Active_Energy_+_Total_end'].isnull()
            one_meter['consecutive_null'] = one_meter['Active_Energy_+_Total_end'].isnull().astype(int).groupby(one_meter['Active_Energy_+_Total_end'].notnull().astype(int).cumsum()).cumsum()


            #filling missing periods:
            one_meter = fill_missing_periods(one_meter, 2, True, id, meter_description, resolution)

            #now select all days that have no null values after imputation
            one_meter = only_full_days(one_meter)

            one_meter = one_meter[['energy']]
            one_meter['date'] = one_meter.index.date
            one_meter['time'] = one_meter.index.time
            one_meter = one_meter.pivot(index='date', columns='time', values='energy')
            one_meter['id'] = id

            to_export = pd.concat([to_export,one_meter])

        to_export = to_export.reset_index()
        #exporting to CSV:
        to_export.to_csv('clean_oldmongo.csv')
        mlflow.log_artifact(dir_path+'/clean_oldmongo.csv', "clean_oldmongo")
        mlflow.set_tag('clean_oldmongo_csv', f'{mlrun.info.artifact_uri}/clean_oldmongo/clean_oldmongo.csv')

    



if __name__ == '__main__':
    main()

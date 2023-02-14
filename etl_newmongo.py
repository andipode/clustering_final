from configparser import ConfigParser
import os

import pandas as pd

from os.path import abspath
dir_path = os.path.abspath('')
import math

import click
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from scipy import stats

from utils import none_checker, truth_checker

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
#os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().('backend','mlflow_tracking_uri')
os.environ["MLFLOW_TRACKING_URI"] = 'http://131.154.97.48:5000'
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

@click.command(help="Given a path to an SQL folder with atxt file for every smart meter "
                    "and the smart_meter_description.csv file, it takes data from the SQL folder, runs necessary preproccessing and saves it in "
                    "a single CSV file.")
@click.option("--load_new", type=str, default="False")
@click.option("--mongo_csv", type=str, default="newmongo.csv")
@click.option("--smart_meter_description_csv", type=str, default="smart_meter_description.csv")
@click.option("--resolution", type=str, default='60')
def main(load_new, mongo_csv, smart_meter_description_csv, resolution):
    load_new = truth_checker(load_new)
    all_meter_data, dict, meter_description = loadto_df_and_dict(load_new, mongo_csv, smart_meter_description_csv, resolution)
    if(load_new == True):
        data_preprocessing(all_meter_data, dict, meter_description, resolution)



def loadto_df_and_dict(load_new, mongo_csv, smart_meter_description_csv, resolution):
    if (load_new == False):
        with mlflow.start_run(run_name='etl_newmongo', nested=True) as mlrun:
            mlflow.log_artifact(dir_path+'/clean_newmongo.csv', "clean_newmongo")
            mlflow.set_tag('clean_newmongo_csv', f'{mlrun.info.artifact_uri}/clean_newmongo/clean_newmongo.csv')
            return(0, 0, 0)
    else:
        mongo_csv = none_checker(mongo_csv)
        mongo_csv = abspath(mongo_csv)
        smart_meter_description_csv = none_checker(smart_meter_description_csv)
        smart_meter_description_csv = abspath(smart_meter_description_csv)
        resolution = none_checker(resolution)
        resolution = int(resolution)


        #list of smart meter names:
        meter_description = pd.read_csv(smart_meter_description_csv, engine='python')
        meter_description.index = meter_description.id
        meters = meter_description.id.tolist()

        all_meter_data = pd.DataFrame()

        raw_data = pd.read_csv(mongo_csv, index_col=[0])
        negative = raw_data[raw_data['energy_type']=='negative_active']
        raw_data = raw_data[raw_data['energy_type']=='positive_active']

        for meter in meters:
            df = raw_data[raw_data['id']==meter[1:]].copy()
            df = df[df['id']==meter[1:]]
            df = df.drop(columns = {'id', 'energy_type'})
            df = pd.melt(df, id_vars='date', var_name='time', value_name = 'energy')

            df.index = pd.to_datetime(df['date'] + ' ' + df['time'], format='%Y-%m-%d %H:%M:%S')
            df = df.drop(columns={'date','time'})
            df = df.sort_index()

            df['isnull'] = df['energy']
            #df['energy'] = df['energy'].fillna(0)
            

            df2 = negative[negative['id']==meter[1:]].copy()
            df2 = df2[df2['id']==meter[1:]]

            df2 = df2.drop(columns = {'id', 'energy_type'})
            df2 = pd.melt(df2, id_vars='date', var_name='time', value_name = 'energy')
            df2.index = pd.to_datetime(df2['date'] + ' ' + df2['time'], format='%Y-%m-%d %H:%M:%S')
            df2 = df2.drop(columns={'date','time'})
            df2 = df2.sort_index()

            df2['isnull'] = df2['energy']

            minus = df2.copy()
            minus['energy'] = minus['energy'] * -1
            df2.update(minus)

            
            
            #resample data in 15 minute intervals
            df = df.resample(str(60)+'min').agg({'energy':np.sum, 'isnull':np.max})
            #create new column to indicate if there was any data during this 1 hour interval
            df['isnull'] = df['isnull'].isnull()
            df.loc[df['isnull'] == True, 'energy'] = None
            #creating column with the meter id:  
            df['id'] = meter
            df = df.rename(columns={"index": "timestamp"})

            #resetting the index (not datetime index, so that in the dataframe with all the smart meters every index is unique)
            #df = df.reset_index()
            ##df = df.rename(columns={"index": "timestamp"})

            #resample data in 15 minute intervals
            df2 = df2.resample(str(60)+'min').agg({'energy':np.sum, 'isnull':np.max})
            #create new column to indicate if there was any data during this 1 hour interval
            df2['isnull'] = df2['isnull'].isnull()
            df2.loc[df2['isnull'] == True, 'energy'] = None
            #creating column with the meter id:  
            df2['id'] = meter

            #resetting the index (not datetime index, so that in the dataframe with all the smart meters every index is unique)
            #df2 = df2.reset_index()
            df2 = df2.rename(columns={"index": "timestamp"})

            #df = df[df['isnull']==False]
            #df2 = df2[df2['isnull']==False]

            ##df = pd.DataFrame({'energy': pd.concat([df.energy, df2.energy], axis=1).sum(axis=1, skipna=False)})
            ##df = df.resample(str(60)+'min').agg({'energy':np.sum})
            df = pd.DataFrame({'energy': pd.concat([df.energy, df2.energy], axis=1).sum(axis=1, min_count=1)})
            df = df.sort_index()
            df['isnull'] = df['energy']
            df['isnull'] = df['isnull'].isnull()
            df['id'] = meter
            ##df = df.drop(columns={'null_indicator'})
            #   
            df = df.rename(columns={"index": "timestamp"})
            df['timestamp'] = df.index
            df['energy'] = df['energy'].fillna(0)
            


            all_meter_data = pd.concat([all_meter_data,df])
            ##all_meter_data = all_meter_data.reset_index()

        all_meter_data = all_meter_data.rename(columns={"index": "timestamp"})
        all_meter_data['timestamp'] = pd.to_datetime(all_meter_data['timestamp'])
        #all_meter_data = all_meter_data[all_meter_data.timestamp >= '2022-07-01 00:00:00']
        all_meter_data.reset_index()
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


def reject_outliers(meter_data, contractual_power, contractual_production, resolution):
    if(math.isnan(contractual_power) or contractual_power==0): contractual_power = 200.0    
    if(math.isnan(contractual_production)): contractual_production = 200.0

    meter_data['consecutive_null_shift'] = meter_data.shift(1)['consecutive_null']
    meter_data.loc[meter_data['consecutive_null_shift']==0, 'consecutive_null_shift'] = 1.0

    meter_data.loc[meter_data['energy']>meter_data['consecutive_null_shift']*contractual_power*1000/(60/resolution), 'isnull'] = True
    meter_data.loc[meter_data['energy']>meter_data['consecutive_null_shift']*contractual_power*1000/(60/resolution), 'energy'] = 0.0
    meter_data.loc[meter_data['energy']<meter_data['consecutive_null_shift']*contractual_production*1000*(-1)/(60/resolution), 'isnull'] = True
    meter_data.loc[meter_data['energy']<meter_data['consecutive_null_shift']*contractual_production*1000*(-1)/(60/resolution), 'energy'] = 0.0
    
    return(meter_data)

def fill_missing_periods(meter_data, filling_window: int, fill_night: bool, resolution):
    #creating Total Energy field for the energy(Wh) field so that it can be used for interpolation, to fill in missing data:
    meter_data['Total_Energy'] = meter_data['energy'].cumsum()
    meter_data['Total_Energy'] = pd.to_numeric(meter_data['Total_Energy'])
    meter_data['Total_Energy'] = meter_data['Total_Energy'].interpolate()
    meter_data['energy'] = meter_data['Total_Energy'].diff()

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

def only_full_days(meter_data): 
    meter_data['date'] = meter_data.index.date
    meter_data['time'] = meter_data.index.time
    full_days = meter_data.pivot(index='date', columns ='time', values='isnull')
    full_days = full_days.loc[~full_days[full_days.columns].any(True)]
    meter_data = meter_data[meter_data['date'].isin(full_days.index)]
    return(meter_data)


def data_preprocessing(all_meter_data, dict, meter_description, resolution):    
    resolution = none_checker(resolution)
    resolution = int(resolution)

    with mlflow.start_run(run_name='etl_newmongo', nested=True) as mlrun:
        #for smart meters BBB6017, BBB6052, that had no data during the night, make energy during the night = 0
        for id in ['BBB6017', 'BBB6052']:
            meter = dict[id]
            meter['hour'] = meter.index.hour
            null_in_the_night = meter[meter['isnull']==True][meter['hour'].isin([0,1,2,3,4,5,6,7,18,19,20,21,22,23])]
            null_in_the_night['energy'] = 0.0
            null_in_the_night['isnull'] = False
            meter.update(null_in_the_night)
            dict[id] = meter

        #for smart meter BBB6100, for which we had no data during the day when there was more generation than production
        #make energy during those periods = 0
        meter = dict['BBB6100']
        meter['hour'] = meter.index.hour
        null_generation = meter[meter['isnull']==True][meter['hour'].isin([6,7,8,9,10,11,12,13,14,15,16,17,18,19])]
        null_generation['energy'] = 0.0
        null_generation['isnull'] = False
        meter.update(null_generation)
        dict['BBB6100'] = meter


        IDs = pd.unique(all_meter_data['id'])
        to_export = pd.DataFrame()
        for id in IDs:
            one_meter = dict[id].copy()

            one_meter.loc[one_meter['isnull']==True, 'energy'] = np.nan
            one_meter['consecutive_null'] = one_meter['energy'].isnull().astype(int).groupby(one_meter['energy'].notnull().astype(int).cumsum()).cumsum()

            #dealing with outliers:
            power = meter_description.loc[id]['Contractual power (kW)']  
            production = meter_description.loc[id]['Production (kW)']
            #one_meter = reject_outliers(one_meter, power, production, resolution)

            #filling missing periods:
            one_meter = fill_missing_periods(one_meter, 2, True, resolution)

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
        to_export.to_csv('clean_newmongo.csv')
        mlflow.log_artifact(dir_path+'/clean_newmongo.csv', "clean_newmongo")
        mlflow.set_tag('clean_newmongo_csv', f'{mlrun.info.artifact_uri}/clean_newmongo/clean_newmongo.csv')



if __name__ == '__main__':
    main()
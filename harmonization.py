from configparser import ConfigParser
import pandas as pd
import os
from os.path import abspath
dir_path = os.path.abspath('')
import numpy as np
import math
import click
import mlflow
import sys
from dtaidistance import dtw
from sklearn.preprocessing import Normalizer

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
@click.option("--sql_csv", type=str, default="clean_sql.csv")
@click.option("--oldmongo_csv", type=str, default="clean_oldmongo.csv")
@click.option("--newmongo_csv", type=str, default='clean_newmongo.csv')
@click.option("--out_csv", type=str, default='harmonized.csv')
@click.option("--compute_dtw", type=str, default='False')

def main(sql_csv, oldmongo_csv, newmongo_csv, out_csv, compute_dtw):
    with mlflow.start_run(run_name='harmonization', nested=True) as mlrun:
        df = join_and_export(sql_csv, oldmongo_csv, newmongo_csv, out_csv, mlrun)
        X = df_to_array(df)
        X = scale(X)
        dtw_matrix(compute_dtw, X)

def scale(array):
    # Very important to scale!
    sc = Normalizer(norm='l2')
    array = sc.fit_transform(array)
    return array


def same_resolution(df1, df2, df3):
    one = (len(df1.columns) == len(df2.columns))
    two =  (len(df1.columns) == len(df3.columns))
    return(one and two)

def join_and_export(sql_csv, oldmongo_csv, newmongo_csv, out_csv, mlrun):
    sql_csv = none_checker(sql_csv)
    oldmongo_csv = none_checker(oldmongo_csv)
    newmongo_csv = none_checker(newmongo_csv)
    out_csv = none_checker(out_csv)

    sql_csv = abspath(sql_csv)
    oldmongo_csv = abspath(oldmongo_csv)
    newmongo_csv = abspath(newmongo_csv)


    df1 = pd.read_csv(sql_csv, index_col=0)
    df2 = pd.read_csv(oldmongo_csv, index_col=0)
    df3 = pd.read_csv(newmongo_csv, index_col=0)

    if (not same_resolution(df1, df2, df3)):
        sys.exit("Error: The three csvs don't have the same resolution")

    df1 = df1[df1['id'] != 'BBB6179']
    df1 =df1[df1['id'] != 'BBB6169']

    all_data = pd.concat([df1,df2,df3])
    all_data.dropna(axis=0, inplace=True)
    all_data = all_data.reset_index()

    all_data.to_csv('harmonized.csv')
    mlflow.log_artifact(dir_path+'/harmonized.csv', "harmonized")
    mlflow.set_tag('harmonized_csv', f'{mlrun.info.artifact_uri}/harmonized/harmonized.csv')
    return(all_data)

def df_to_array(df):
    X = df.copy()
    X = X.drop(columns=['id', 'date', 'index'])
    X = X.values.copy()
    return(X)

def dtw_matrix(compute_dtw, X):

    compute_dtw = truth_checker(compute_dtw)
    if(compute_dtw):
        ds = dtw.distance_matrix(X, window=2)
        np.savetxt('dtw.out', ds, delimiter=',')
        return()
    else:
        return()


if __name__ == '__main__':
    main()
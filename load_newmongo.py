"""
Downloads data from the ASM's mongo database, converts it to csv format and saves it as an artifact
"""

from configparser import ConfigParser
import mlflow
from pymongo import MongoClient
import pandas as pd
import click
import os
from os.path import abspath

dir_path = os.path.abspath('')
# get environment variables
#from dotenv import load_dotenv
#load_dotenv()
## explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
##os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().('backend','mlflow_tracking_uri')
#os.environ["MLFLOW_TRACKING_URI"] = 'http://131.154.97.48:5000'
#MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

@click.command(help="Downloads data from the ASM's mongo database, converts it to csv format and saves it as an artifact"
                    "--out_csv:localpath+filename to save the csv file")
@click.option("--out_csv", type=str, default="newmongo.csv")

def main(out_csv):
    with mlflow.start_run(run_name='load_data', nested=True) as mlrun:
        out_csv = out_csv.replace('/', os.path.sep)
        fname = out_csv.split(os.path.sep)[-1]
        out_path = abspath(out_csv)

        db = MongoClient('131.154.97.22', 
                            27017, 
                            username='inergy',
                            password='inergySt0rag32oo22!').get_database('inergy_prod_db')

        current_collection = db.asm_historical_smart_meters_uc7

        current_df = pd.DataFrame(current_collection.find())
        #print(current_df)

        #current_df = current_df.drop(columns={'_id', ''})

        current_df.to_csv(out_path)
        mlflow.log_artifact(out_path, "raw_newmongo")
        mlflow.set_tag('dataset_csv', f'{mlrun.info.artifact_uri}/raw_newmongo/{fname}')
        


if __name__ == '__main__':
    main()
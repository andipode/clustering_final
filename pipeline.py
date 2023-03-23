from configparser import ConfigParser
import os
import click
import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint
from mlflow.tracking.fluent import _get_experiment_id
from tempfile import TemporaryDirectory
from utils import none_checker, truth_checker
from mlflow.artifacts import download_artifacts
from os.path import abspath

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
@click.option("--resolution", type=str, default='60')
@click.option("--smart_meter_description_csv", type=str, default="smart_meter_description.csv")
@click.option("--oldmongo_folder", type=str, default="oldmongo/")
@click.option("--sql_folder", type=str, default="sql/")
@click.option("--compute_dtw", type=str, default='False')
@click.option("--model", type=str, default="kmeans")
@click.option("--distance_metric", type=str, default="euclidian")
@click.option("--number_of_clusters", type=str, default="0")
@click.option("--ignore-previous-runs",
              type=str,
              default="true",
              help="Whether to ignore previous step runs while running the pipeline")

def workflow(load_new, resolution, smart_meter_description_csv, oldmongo_folder, sql_folder, compute_dtw, model, distance_metric,
             number_of_clusters, ignore_previous_runs):
    ignore_previous_runs = truth_checker(ignore_previous_runs)
    resolution = none_checker(resolution)
    with mlflow.start_run(run_name=(model+' '+distance_metric+' '+number_of_clusters+'_pipeline')) as active_run:
        git_commit = active_run.data.tags.get(mlflow_tags.MLFLOW_GIT_COMMIT)
        #1. Load New Mongo Data
        load_newmongo_params = {"out_csv": "newmongo.csv"}
        load_newmongo_run = _get_or_run("load_newmongo", load_newmongo_params, git_commit, ignore_previous_runs)
        newmongo_csv = load_newmongo_run.data.tags['dataset_csv']#.replace("s3:/", S3_ENDPOINT_URL)

        #2a. etl old mongo
        etl_oldmongo_params = {"mongo_folder":oldmongo_folder, "smart_meter_description_csv":smart_meter_description_csv, "resolution":resolution}
        etl_oldmongo_run = _get_or_run("etl_oldmongo", etl_oldmongo_params, git_commit, ignore_previous_runs)
        clean_oldmongo_csv = etl_oldmongo_run.data.tags['clean_oldmongo_csv']#.replace("s3:/", S3_ENDPOINT_URL)

        #2b. etl sql
        etl_sql_params = {"sql_folder":sql_folder, "smart_meter_description_csv":smart_meter_description_csv, "resolution":resolution}
        etl_sql_run = _get_or_run("etl_sql", etl_sql_params, git_commit, ignore_previous_runs)
        clean_sql_csv = etl_sql_run.data.tags['clean_sql_csv']#.replace("s3:/", S3_ENDPOINT_URL)

        #2c. etl new mongo
        with TemporaryDirectory() as temp_dir:
            dst_dir = temp_dir
            #newmongo_csv = download_artifacts(artifact_uri=newmongo_csv, dst_path="/".join([dst_dir, 'newmongo.csv']))
            #newmongo_csv = download_artifacts(artifact_uri=newmongo_csv, dst_path=abspath('results/'))
            newmongo_csv = download_artifacts(artifact_uri=newmongo_csv, dst_path=temp_dir)
            #eprint(newmongo_csv)
            etl_newmongo_params = {"load_new":load_new, "mongo_csv":newmongo_csv, "smart_meter_description_csv":smart_meter_description_csv, "resolution":resolution}
            etl_newmongo_run = _get_or_run("etl_newmongo", etl_newmongo_params, git_commit, ignore_previous_runs)
            clean_newmongo_csv = etl_newmongo_run.data.tags['clean_newmongo_csv']#.replace("s3:/", S3_ENDPOINT_URL)

        #3. Harmonization
        with TemporaryDirectory() as temp_dir:
            clean_sql_csv = download_artifacts(artifact_uri=clean_sql_csv, dst_path=temp_dir)
            clean_oldmongo_csv = download_artifacts(artifact_uri=clean_oldmongo_csv, dst_path=temp_dir)
            clean_newmongo_csv = download_artifacts(artifact_uri=clean_newmongo_csv, dst_path=temp_dir)

            harmonization_params = {"sql_csv": clean_sql_csv, 'oldmongo_csv': clean_oldmongo_csv, 'newmongo_csv':clean_newmongo_csv,
                                    'out_csv': 'harmonized.csv', 'compute_dtw':compute_dtw}
            harmonization_run = _get_or_run("harmonization", harmonization_params, git_commit, ignore_previous_runs)
            harmonized_csv = harmonization_run.data.tags['harmonized_csv']#.replace("s3:/", S3_ENDPOINT_URL)

        #4. Clustering
        with TemporaryDirectory() as temp_dir:
            #harmonized_csv = download_artifacts(artifact_uri=harmonized_csv, dst_path="/".join([dst_dir, 'harmonized.csv']))
            harmonized_csv = download_artifacts(artifact_uri=harmonized_csv, dst_path=temp_dir)
            clustering_params = {"in_csv": harmonized_csv, 'model': model, 'distance_metric':distance_metric,
                                'number_of_clusters': number_of_clusters}
            clustering_run = _get_or_run("clustering", clustering_params, git_commit, ignore_previous_runs)

            # Log eval metrics to father run for consistency and clear results
            #mlflow.log_metrics(clustering_run.data.metrics)

        # Log eval metrics to father run for consistency and clear results
        #mlflow.log_metrics(clustering_run.data.metrics)



def _already_ran(entry_point_name, parameters, git_commit, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        if tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None) != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in parameters.items():
            run_value = full_run.data.params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run_info.to_proto().status != RunStatus.FINISHED:
            eprint(
                ("Run matched, but is not FINISHED, so skipping " "(run_id=%s, status=%s)")
                % (run_info.run_id, run_info.status)
            )
            continue

        previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        if git_commit != previous_version:
            eprint(
                (
                    "Run matched, but has a different source code version, so skipping "
                    "(found=%s, expected=%s)"
                )
                % (previous_version, git_commit)
            )
            continue
        return client.get_run(run_info.run_id)
    eprint("No matching run has been found.")
    return None

def _get_or_run(entrypoint, parameters, git_commit, ignore_previous_run=True, use_cache=True):
    #TODO: this was removed to always run the pipeline from the beginning.
    if not ignore_previous_run:
        existing_run = _already_ran(entrypoint, parameters, git_commit)
        if use_cache and existing_run:
            print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
            return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, env_manager="local")
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


         
if __name__ == "__main__":
    workflow()

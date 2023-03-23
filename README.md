This repository contains a complete MLFlow pipeline for clustering electric load profiles. It performs data collection,
data preprocessing, harmonization, clustering and validation.
New data are collected from a Mongo database (through load_newmongo.py) and there is also historical data that's used for the analysis (collected and processed by etl_oldmongo.py and etl_sql.py). 

The files starting with "etl_" create three csvs, one for each database. Their input should come from [here]
(https://drive.google.com/drive/folders/1xha5zwElF3QRf88BpBhBlWGAvSJ3eOOD).
The current data inside the /oldmongo and /sql folders are incomplete because the files were too large to upload on github. A sample file is included in each folder to demonstrate how the files should be titled and structured.

Then harmonization.py takes the 3 separate csvs and unifies them to one. It also checks that all have the same resolution and calculates the dtw.out file for all datasets (which is necessary for some of the algorithms, those that use DTW). See below technical issue 3 as well. The clustering stage loads the dtw.out file from the local storage. (TODO: Change that to an MLflow parameter). dtw.out is not stored to MLflow due to size issues (4GB). Need some kind of optimization here in production (will it be even used in production? -> does not integrate with MLflow).

The results.ipynb file contains an example analysis of the results of a clustering configuration.

# Pipeline example
```mlflow run --experiment-name uc7_clustering --entry-point pipeline.py . -P model="kmeans" -P number_of_clusters=12 --env-manager=local```

# Technical Issues
1. No way found to log models as MLmodels other than kmeans with euclidean distance.

2. Scaling issues for the oldmongo database were fixed (in the oldmongo.csv file=output of etl_oldmongo.py) by guessing which of the different scales seemed the most accurate based on the contractual value for active power. The selection of the scale doesn't influence the clustering stage since all the profiles are normalized, but in case the old mongo data need to be used for forecasting, more investigation needs to be done on ASM's part to figure out what caused the scaling
issues and which scale is correct for each smart meter.

3. The dtw.out file needs to be calculated every time new data is brought for the clustering functions that need precomputed DTW distances (all algorithms that use DTW except kmeans + the distance metric Silhouette Score DTW) to work. That's done by setting the compute_dtw parameter in the harmonization step to True. Beware that this step is time consuming (can take a few hours). In case the dtw.out file is not up to date with the data, running the clustering stage will produce an error, unless the algorithm of choice is kmeans and the calculation of Silhouette Score DTW is skipped (commented out -> can be set as mlflow parameter to avoid errors).



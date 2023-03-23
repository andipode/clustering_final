This repository contains a complete MLFlow pipeline for clustering electric load profiles. It performs data collection,
data preprocessing, harmonization, clustering and validation.
New data are collected from a Mongo database (through load_newmongo.py) and there is also historical data that's used for the analysis (collected
and processed by etl_oldmongo.py and etl_sql.py).

The data inside the /oldmongo and /sql folders are incomplete because the files were too large to upload on github. A sample file is included in each
folder to demonstrate how the files should be titled and structured.


# Pipeline example
```mlflow run --experiment-name uc7_clustering --entry-point pipeline.py . -P model="kmeans" -P number_of_clusters=12 --env-manager=local```

# Technical Issues
1. No way found to log models other than kmeans with euclidean distance.

2. Scaling issues for the oldmongo database were fixed (in the oldmongo.csv file) by guessing which of the different scales seemed the most accurate
based on the contractual value for active power. The selection of the scale doesn't influence the clustering stage since all the profiles are normalized,
but in case the old mongo data need to be used for forecasting, more investigation needs to be done on ASM's part to figure out what caused the scaling 
issues and which scale is correct for each smart meter.

3. The dtw.out file needs to be calculated every time new data is brought for the clustering functions that need precomputed DTW distances 
(all algorithms that use DTW except kmeans + the distance metric Silhouette Score DTW) to work. That's done by setting the compute_dtw parameter
in the harmonization step to True. Beware that this step is time consuming (can take a few hours). In case the dtw.out file is not up to date witb 
the data, running the clustering stage will produce an error, unless the algorithm of choice is kmeans and the calculation of Silhouette Score DTW is 
skipped (commented out).


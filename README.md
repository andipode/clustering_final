# Pipeline example
```mlflow run --experiment-name uc7_clustering --entry-point pipeline.py . -P model="kmeans" -P number_of_clusters=12 --env-manager=local```

# Technical Issues
1. No way found to log models other than kmeans with euclidean distance.

2. Scaling issues for the oldmongo database were fixed (in the oldmongo.csv file) by guessing which of the different scales seemed the most accurate
based on the contractual value for active power. The selection of the scale doesn't influence the clustering stage since all the profiles are normalized,
but in case the old mongo data need to be used for forecasting, more investigation needs to be done on ASM's part to figure out what caused the scaling 
issues and which scale is correct for each smart meter.

3. The dtw.out file needs to be calculated every time new data is brought foe the clustering functions that need precomputed DTW distances 
(all algorithms that use DTW except kmeans + the distance metric Silhouette Score DTW) to work. That's done by setting the compute_dtw parameter
in the harmonization step to True.


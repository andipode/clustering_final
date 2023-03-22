# Pipeline example
mlflow run --experiment-name uc7_clustering --entry-point pipeline.py . -P model="kmeans" -P number_of_clusters=12 --env-manager=local

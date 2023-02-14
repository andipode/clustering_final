for number in {3..20} 
do
    mlflow run --experiment-name uc7_clustering --entry-point pipeline.py . -P model="kmeans" -P number_of_clusters=$number --env-manager=local
    mlflow run --experiment-name uc7_clustering --entry-point pipeline.py . -P model="kmeans" -P distance_metric="dtw" -P number_of_clusters=$number --env-manager=local
    mlflow run --experiment-name uc7_clustering --entry-point pipeline.py . -P model="kmedioids" -P number_of_clusters=$number --env-manager=local
    mlflow run --experiment-name uc7_clustering --entry-point pipeline.py . -P model="kmedioids" -P distance_metric="dtw" -P number_of_clusters=$number --env-manager=local
    mlflow run --experiment-name uc7_clustering --entry-point pipeline.py . -P model="agglomerative" -P number_of_clusters=$number --env-manager=local
    mlflow run --experiment-name uc7_clustering --entry-point pipeline.py . -P model="agglomerative" -P distance_metric="dtw" -P number_of_clusters=$number --env-manager=local
    mlflow run --experiment-name uc7_clustering --entry-point pipeline.py . -P model="som" -P number_of_clusters=$number --env-manager=local
done
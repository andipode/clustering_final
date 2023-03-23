"""
Clusters the data using a configuration provided by the user (model, distance metric and number of clusters), computes
different validation metrics and also creates graphs for a visual overview of the clusters
----------
Parameters:
in_csv
    Path to input csv file.
model
    The preferred clustering algorithm.
distance_metric
    The preferred metric to compute distances between samples.
number_of_clusters

"""


from configparser import ConfigParser
from operator import index
from re import A
import pandas as pd
import os
from sklearn import cluster

from sklearn.neighbors import DistanceMetric
from os.path import abspath
dir_path = os.path.abspath('')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from scipy import stats
import sklearn
import click
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import mlflow
from sklearn_extra.cluster import KMedoids
import tslearn
from tslearn.clustering import silhouette_score as dtw_silhouette_score
from numpy import reshape as reshape
from sklearn.cluster import AgglomerativeClustering
from dtaidistance import dtw

from utils import none_checker, truth_checker

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
#os.environ["MLFLOW_TRACKING_URI"] = ConfigParser().('backend','mlflow_tracking_uri')
os.environ["MLFLOW_TRACKING_URI"] = 'http://131.154.97.48:5000'
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")


@click.command(help="Clusters the data using a configuration provided by the user (model, distance metric and number of clusters), computes"
                    "different validation metrics and also creates graphs for a visual overview of the clusters.")
@click.option("--in_csv", type=str, default="harmonized.csv")
@click.option("--model", type=click.Choice(['kmeans', 'kmedoids', 'agglomerative']), default="kmeans", multiple=False)
@click.option("--distance_metric", type=click.Choice(['euclidean', 'dtw']), default="euclidean", multiple=False)
@click.option("--number_of_clusters", type=str, default="0")
def main(in_csv, model, distance_metric, number_of_clusters):
    in_csv = none_checker(in_csv)
    in_csv = abspath(in_csv)
    distance_metric = none_checker(distance_metric)
    number_of_clusters = none_checker(number_of_clusters)
    number_of_clusters = int(number_of_clusters)

    all_data = pd.read_csv(in_csv, index_col=0)
    #all_data = all_data.drop(columns={'index'})
    mlflow.set_experiment("uc7_clustering")
    X = df_to_array(all_data)
    X = scale(X)
    if(model=='kmeans'):
        if(distance_metric=='dtw'):
            cluster_kmeans_dtw(distance_metric, number_of_clusters, X, all_data)
        else:
            cluster_kmeans(distance_metric, number_of_clusters, X, all_data)
        
    if(model=='kmedoids'):
        if(distance_metric=='dtw'):
            cluster_kmedioids_dtw(distance_metric, number_of_clusters, X, all_data)
        else:
            cluster_kmedioids(distance_metric, number_of_clusters, X, all_data)
    
    
    if(model=='som'):
        cluster_som(distance_metric, number_of_clusters, X, all_data)
    
    if(model=='agglomerative'):
        if(distance_metric=='dtw'):
            cluster_agglomerative_dtw(distance_metric, number_of_clusters, X, all_data)
        else:
            cluster_agglomerative(distance_metric, number_of_clusters, X, all_data)


def scale(array):
    # This is norm l2 normalization. Also consider minmax scaling
    sc = Normalizer(norm='l2')
    array = sc.fit_transform(array)
    return array

def df_to_array(df):
    X = df.copy()
    X = X.drop(columns=['id', 'date', 'index'])
    X = X.values.copy()
    return(X)


from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
def cluster_kmeans_dtw(distance, k, X, df):
    X1 = to_time_series_dataset(X)
    kmeans = TimeSeriesKMeans(n_clusters=k, init='random', n_init=10, random_state=0, metric="dtw", max_iter=5, metric_params={"global_constraint": "sakoe_chiba", "sakoe_chiba_radius":1})
    #mlflow.set_experiment("Comparison 28/06")
    with mlflow.start_run(run_name= "kmeans " + distance + " k=" + str(k)):
        cluster_found = kmeans.fit_predict(X1)
        cluster_centers = kmeans.cluster_centers_
        mlflow.log_param("k", k)
        cluster_centers = cluster_centers.reshape(cluster_centers.shape[0], (cluster_centers.shape[1]*cluster_centers.shape[2]))

        metrics(X, kmeans.labels_, cluster_centers)
        graphs(X, df, k, cluster_found, cluster_centers)

        #mlflow.sklearn.log_model(kmeans, "kmeans " + distance + " k=" + str(k))
        print("Model run: ", mlflow.active_run().info.run_uuid)
        mlflow.set_tag('results', 'True')
    mlflow.end_run()

from mlflow.models.signature import infer_signature
def cluster_kmeans(distance, k, X, df):
    if(k==0): 
        k = select_k(X)
    kmeans = KMeans(
        n_clusters=k, init='random',
        n_init=10, random_state=0
    )
    #mlflow.set_experiment("Comparison 28/06")
    with mlflow.start_run(run_name= "kmeans " + distance + " k=" + str(k)):
        cluster_found = kmeans.fit_predict(X)
        cluster_centers = kmeans.cluster_centers_
        mlflow.log_param("k", k)
        metrics(X, kmeans.labels_, kmeans.cluster_centers_)
        graphs(X, df, k, cluster_found, cluster_centers)
        
        Xsub = X[:1]
        signature = infer_signature(Xsub, kmeans.predict(Xsub))
        mlflow.sklearn.save_model(sk_model=kmeans, path="models/kmeansEuclidean"+str(k), signature=signature, input_example=Xsub)

        print("Model run: ", mlflow.active_run().info.run_uuid)
    mlflow.end_run()

from sklearn_som.som import SOM
def cluster_som(distance, k, X, df):
    som = SOM(m=k, n=1, dim=24, random_state=0)
    #mlflow.set_experiment("Comparison 28/06")
    with mlflow.start_run(run_name= "som " + distance + " k=" + str(k)):
        labels = som.fit_predict(X)       
        from sklearn.neighbors import NearestCentroid
        clf = NearestCentroid()
        clf.fit(X, labels)

        mlflow.log_param("k", k)
        metrics(X, labels, clf.centroids_)
        graphs(X, df, k, labels, clf.centroids_)

        #mlflow.sklearn.log_model(kmeans, "kmeans " + distance + " k=" + str(k))
        print("Model run: ", mlflow.active_run().info.run_uuid)
    mlflow.end_run()

def cluster_agglomerative(distance, k, X, df):
    agglomerative = AgglomerativeClustering(
        n_clusters=k, affinity='euclidean'
    )
    #mlflow.set_experiment("Comparison 28/06")
    with mlflow.start_run(run_name= "agglomerative " + distance + " k=" + str(k)):
        cluster_found = agglomerative.fit_predict(X)
        
        from sklearn.neighbors import NearestCentroid
        clf = NearestCentroid()
        clf.fit(X, cluster_found)
        cluster_centers = clf.centroids_
    
        mlflow.log_param("k", k)
        metrics(X, agglomerative.labels_, clf.centroids_)
        graphs(X, df, k, cluster_found, clf.centroids_)

        #mlflow.sklearn.log_model(kmeans, "kmeans " + distance + " k=" + str(k))
        print("Model run: ", mlflow.active_run().info.run_uuid)
    mlflow.end_run()

def cluster_agglomerative_dtw(distance, k, X, df):
    agglomerative = AgglomerativeClustering(
        n_clusters=k, affinity='precomputed',
        linkage='average', distance_threshold=None
    )
    #mlflow.set_experiment("Comparison 28/06")
    with mlflow.start_run(run_name= "agglomerative " + distance + " k=" + str(k)):
        ds = np.loadtxt("dtw.out", delimiter=",")
        cluster_found = agglomerative.fit_predict(ds)
            
        from sklearn.neighbors import NearestCentroid
        clf = NearestCentroid()
        clf.fit(X, cluster_found)
        cluster_centers = clf.centroids_
        
        mlflow.log_param("k", k)
        metrics(X, agglomerative.labels_, clf.centroids_)
        graphs(X, df, k, cluster_found, clf.centroids_)
    
        #mlflow.sklearn.log_model(kmeans, "kmeans " + distance + " k=" + str(k))
        print("Model run: ", mlflow.active_run().info.run_uuid)
    mlflow.end_run()
    

def cluster_kmedioids(distance, k, X, df):
    kmedioids = KMedoids(
        n_clusters=k, init='random',
        random_state=0
    )
    #mlflow.set_experiment("Comparison 28/06")
    with mlflow.start_run(run_name= "kmedioids " + distance + " k=" + str(k)):
        cluster_found = kmedioids.fit_predict(X)
        
        mlflow.log_param("k", k)
        metrics(X, kmedioids.labels_, kmedioids.cluster_centers_)
        graphs(X, df, k, cluster_found, kmedioids.cluster_centers_)
        #mlflow.sklearn.log_model(kmedioids, "kmedioids " + distance + " k=" + str(k))
        print("Model run: ", mlflow.active_run().info.run_uuid)
    mlflow.end_run()

def cluster_kmedioids_dtw(distance, k, X, df):
    kmedioids = KMedoids(
        n_clusters=k, init='random',
        random_state=0, metric='precomputed'
    )
    #mlflow.set_experiment("Comparison 28/06")
    with mlflow.start_run(run_name= "kmedioids " + distance + " k=" + str(k)):
        ds = np.loadtxt("dtw.out", delimiter=",")
        cluster_found = kmedioids.fit_predict(ds)
        from sklearn.neighbors import NearestCentroid
        clf = NearestCentroid()
        clf.fit(X, cluster_found)

        mlflow.log_param("k", k)
        metrics(X, kmedioids.labels_, clf.centroids_)
        graphs(X, df, k, cluster_found, clf.centroids_)
        print("Model run: ", mlflow.active_run().info.run_uuid)
    mlflow.end_run()

def metrics(X, labels, centroids):
    ds = np.loadtxt("dtw.out", delimiter=",")
    mlflow.log_metric("DB Index", davies_bouldin_score(X, labels))
    mlflow.log_metric("Silhouette Score", silhouette_score(X, labels))
    # Comment out next line to avoid dtw.out
    mlflow.log_metric("Silhouette DTW", silhouette_score(X=ds, labels=labels, metric='precomputed'))
    mlflow.log_metric("Peak Match Score", vanilla_peak_match_score(X, labels, centroids))
    mlflow.log_metric("Pr Peak Detection", prominent_peak_detection_score(X, labels, centroids))
    mlflow.log_metric("Pr Peak Overdetection", peak_overprediction(X, labels, centroids))
    mlflow.log_metric("Pr Peak Performance", peak_performance(X, labels, centroids))
    mlflow.log_metric("Rel Pr P Performance", relaxed_prominent_peak_performance(X, labels, centroids))
    mlflow.log_metric("Rel Pr P Detection", relaxed_prominent_peak_detection_score(X, labels, centroids))
    mlflow.log_metric("Rel Pr P Overdetection", relaxed_peak_overprediction(X, labels, centroids))


def graphs(X, all_data, k, cluster_found, cluster_centers):
    scaled_data = pd.DataFrame(X, all_data.index, all_data.columns.tolist().remove('id'))
    scaled_data['cluster'] = pd.Series(cluster_found, name='cluster')

    #plot the cluster centers and the samples that belong to each cluster:
    fig, ax= plt.subplots(8, 4, figsize=(15,26))
    clusters = {}
    i = 0 
    j = 0
    for cluster in range(0,k):
        clusters[cluster] = scaled_data[scaled_data['cluster'] == cluster].drop(columns=['cluster'])
        clusters[cluster].T.plot(ax=ax[i][j], legend=False, alpha=0.01, color='orange', label= f'Cluster {cluster}')

        cluster_centers = pd.DataFrame(cluster_centers)
        cluster_centers.iloc[[cluster]].T.plot(legend=True, color='black', ls='--', ax=ax[i][j])

        ax[i][j].set_title("Cluster " + str(cluster))
        ax[i][j].set_xlabel("Time")
        ax[i][j].set_ylabel("Normalized Energy")
        j = j+1
        if(j==4):
            j = 0
            i = i + 1
    fig.tight_layout()
    fig.savefig("clusterCenters.png", dpi=80)
    fig.savefig("clusterCentersHighRes.png", dpi=300)
    mlflow.log_artifact('clusterCenters.png')
    mlflow.log_artifact('clusterCentersHighRes.png')
    #clear figure:
    fig.clf()

    colors = sns.color_palette('pastel')[0:k]
    pie = scaled_data.groupby('cluster').size().plot(kind='pie', autopct='%.0f%%', textprops={'fontsize': 20},
                                  colors=colors)
    fig = pie.get_figure()
    fig.savefig("clusterPie.png", dpi=100)
    mlflow.log_artifact('clusterPie.png')

def select_k(X):
    # autocompute k based on silhoutte score (not used)
    silhouette = [-1,-1]
    for i in range(3, 30):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, random_state=0
        )
        km.fit(X)
        silhouette.append(silhouette_score(X, km.labels_))
    return (silhouette.index(max(silhouette))+1)

from scipy.signal import find_peaks
def vanilla_peak_match_score(X, labels, centers):
    input_peaks = np.apply_along_axis(to_binary_peaks, axis=1, arr=X)
    center_peaks = np.apply_along_axis(to_binary_peaks, axis=1, arr=centers)
    
    score = 0
    ones = np.ones(shape=[24])
    for i in list(zip(input_peaks, labels)):
        if (np.all(i[0]==0) and np.all(center_peaks[i[1]]==0)):
            line_score = 1
        elif(np.all(i[0]==0)):
            line_score = 0
        else:
            line_score = (i[0] @ center_peaks[i[1]]) / (i[0]@ones) #dot product center_peaks[labels]
        score = score + line_score 
    score = score/X.shape[0]
    return(score)

def prominent_peak_match_score(X, labels, centers):
    input_peaks = np.apply_along_axis(im_to_binary_peaks, axis=1, arr=X)
    improved_center_peaks = np.apply_along_axis(im_to_binary_peaks, axis=1, arr=centers)
    
    score = 0
    ones = np.ones(shape=[24])
    for i in list(zip(input_peaks, labels)):
        if (np.all(i[0]==0) and np.all(improved_center_peaks[i[1]]==0)):
            line_score = 1
        elif(np.all(i[0]==0)):
            line_score = 0
        else:
            line_score = (i[0] @ improved_center_peaks[i[1]]) / (i[0]@ones) #dot product center_peaks[labels]
        score = score + line_score 
    score = score/X.shape[0]
    return(score)

def peak_performance(X, labels, centers):
    input_peaks = np.apply_along_axis(im_to_binary_peaks, axis=1, arr=X)
    improved_center_peaks = np.apply_along_axis(im_to_binary_peaks, axis=1, arr=centers)
    
    score = 0
    ones = np.ones(shape=[24])
    for i in list(zip(input_peaks, labels)):
        if (np.all(i[0]==0) and np.all(improved_center_peaks[i[1]]==0)):
            line_score = 1
        else:
            line_score = (i[0] @ improved_center_peaks[i[1]]) / (max(i[0]@ones, improved_center_peaks[i[1]]@ones)) #dot product center_peaks[labels]
        score = score + line_score 
    score = score/X.shape[0]
    return(score)

def relaxed_prominent_peak_performance(X, labels, centers):
    input_peaks = np.apply_along_axis(im_to_binary_peaks, axis=1, arr=X)
    improved_center_peaks = np.apply_along_axis(im_to_binary_peaks, axis=1, arr=centers)
    relaxed_center_peaks = np.apply_along_axis(max_filter, axis=1, arr=improved_center_peaks)
    
    score = 0
    ones = np.ones(shape=[24])
    for i in list(zip(input_peaks, labels)):
        if (np.all(i[0]==0) and np.all(improved_center_peaks[i[1]]==0)):
            line_score = 1
        else:
            line_score = (i[0] @ relaxed_center_peaks[i[1]]) / (max(i[0]@ones, improved_center_peaks[i[1]]@ones)) #dot product center_peaks[labels]
        score = score + line_score 
    score = score/X.shape[0]
    return(score)


def prominent_peak_detection_score(X, labels, centers):
    input_peaks = np.apply_along_axis(im_to_binary_peaks, axis=1, arr=X)
    improved_center_peaks = np.apply_along_axis(im_to_binary_peaks, axis=1, arr=centers)
    
    score = 0
    ones = np.ones(shape=[24])
    profiles_no_peaks = 0
    for i in list(zip(input_peaks, labels)):
        if(np.all(i[0]==0)):
            profiles_no_peaks = profiles_no_peaks + 1
        else:
            line_score = (i[0]@improved_center_peaks[i[1]]) / (i[0]@ones) #dot product center_peaks[labels]
            score = score + line_score 
    score = score/(X.shape[0]-profiles_no_peaks)
    return(score)

def relaxed_prominent_peak_detection_score(X, labels, centers):
    input_peaks = np.apply_along_axis(im_to_binary_peaks, axis=1, arr=X)
    improved_center_peaks = np.apply_along_axis(im_to_binary_peaks, axis=1, arr=centers)
    relaxed_center_peaks = np.apply_along_axis(max_filter, axis=1, arr=improved_center_peaks)
    
    score = 0
    ones = np.ones(shape=[24])
    profiles_no_peaks = 0
    for i in list(zip(input_peaks, labels)):
        if(np.all(i[0]==0)):
            profiles_no_peaks = profiles_no_peaks + 1
        else:
            line_score = (i[0]@relaxed_center_peaks[i[1]]) / (i[0]@ones) #dot product center_peaks[labels]
            score = score + line_score 
    score = score/(X.shape[0]-profiles_no_peaks)
    return(score)


def peak_overprediction(X, labels, centers): # not used
    input_peaks = np.apply_along_axis(im_to_binary_peaks, axis=1, arr=X)
    improved_center_peaks = np.apply_along_axis(im_to_binary_peaks, axis=1, arr=centers)
    
    score = 0
    ones = np.ones(shape=[24])
    profiles_no_peaks = 0
    for i in list(zip(input_peaks, labels)):
        if (np.all(improved_center_peaks[i[1]]==0)):
            profiles_no_peaks = profiles_no_peaks + 1
        else:
            line_score = ((improved_center_peaks[i[1]]@ones) - (i[0] @ improved_center_peaks[i[1]]))/(improved_center_peaks[i[1]]@ones) #dot product center_peaks[labels]
            score = score + line_score 
    if((X.shape[0]-profiles_no_peaks) != 0):
        score = score/(X.shape[0]-profiles_no_peaks)
    else:
        score = 0
    return(score)

def relaxed_peak_overprediction(X, labels, centers): # not used
    input_peaks = np.apply_along_axis(im_to_binary_peaks, axis=1, arr=X)
    improved_center_peaks = np.apply_along_axis(im_to_binary_peaks, axis=1, arr=centers)
    relaxed_center_peaks = np.apply_along_axis(max_filter, axis=1, arr=improved_center_peaks)
    
    score = 0
    ones = np.ones(shape=[24])
    profiles_no_peaks = 0
    for i in list(zip(input_peaks, labels)):
        if (np.all(improved_center_peaks[i[1]]==0)):
            profiles_no_peaks = profiles_no_peaks + 1
        else:
            line_score = ((improved_center_peaks[i[1]]@ones) - (i[0] @ relaxed_center_peaks[i[1]]))/(improved_center_peaks[i[1]]@ones) #dot product center_peaks[labels]
            score = score + line_score
    if((X.shape[0]-profiles_no_peaks) != 0):
        score = score/(X.shape[0]-profiles_no_peaks)
    else:
        score = 0
    return(score)


from itertools import combinations
from scipy.spatial.distance import hamming
def distinctiveness_score(X, labels, centers): # not used, it is from a paper but didnt make sense
    improved_center_peaks = np.apply_along_axis(im_to_binary_peaks, axis=1, arr=centers)
    relaxed_center_peaks = improved_center_peaks
    l = [*range(0,relaxed_center_peaks.shape[0])]
    total_distance = 0

    for i,j in combinations(l, r=2):
        hamming_distance = hamming(relaxed_center_peaks[i], relaxed_center_peaks[j])
        hamming_distance_1 = hamming(np.roll(relaxed_center_peaks[i],1), relaxed_center_peaks[j])
        hamming_distance_2 = hamming(np.roll(relaxed_center_peaks[i],-1), relaxed_center_peaks[j])
        min_hamming_distance = min(hamming_distance, hamming_distance_1, hamming_distance_2)

        total_distance = total_distance+min_hamming_distance

    return(total_distance/len(set(combinations(l, r=2))))


from scipy.ndimage import maximum_filter1d
def max_filter(x):
    return(maximum_filter1d(x, size=3))


def to_binary_peaks(x):
    index_of_peaks, _ =  find_peaks(x)
    array = np.zeros(shape=24)
    array[index_of_peaks] = 1
    return(array)

def im_to_binary_peaks(x):
    index_of_peaks, _ =  find_peaks(x=x, prominence=0.2)
    array = np.zeros(shape=24)
    array[index_of_peaks] = 1
    return(array)


if __name__ == '__main__':
    main()

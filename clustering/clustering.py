
import numpy as np 
import pandas as pd
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn_extra.cluster import KMedoids

from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.metrics import silhouette_score as sc
from sklearn.metrics import davies_bouldin_score as db


# the max the better for silhouette (belong to (-1,1)
# the lower the better for db(min = 0)


def cv_silhouette_scorer(estimator, X):
    estimator.fit_predict(X)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = len(X.index)
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return sc(X, cluster_labels)

def cv_db_scorer(estimator, X):
    estimator.fit_predict(X)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = len(X.index)
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        return db(X, cluster_labels)





# Clustering test functions 
def Kmeans_clustering(df, tweets_column, embeddings_df, n_clusters, n_init = 10, max_iter= 1000, algorithm = 'full') : 
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_init, max_iter=max_iter, tol=1e-05, random_state=3, algorithm = algorithm)
    cluster_ids = kmeans.fit_predict(embeddings_df)
    result_df = pd.DataFrame({'tweets' : df[tweets_column], 'topic_cluster' : cluster_ids })
    return result_df


def Agglomerative_clustering(df, tweets_column, embeddings_df, n_clusters): 
    agglo_clust = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    y_agg = agglo_clust.fit_predict(embeddings_df)
    result_df = pd.DataFrame({'tweets' : df[tweets_column], 'topic_cluster' : y_agg })
    return result_df


def Spectral_clustering(df, tweets_column, embeddings_df, n_clusters): 
    agglo_clust = SpectralClustering(n_clusters=8, affinity = 'nearest_neighbors', n_jobs = -1 , eigen_solver = 'arpack',n_components= 50, n_neighbors = 100, assign_labels="kmeans")
    y_spectral = spectral_clustering.fit_predict(embeddings_df)
    result_df = pd.DataFrame({'tweets' : df[tweets_column], 'topic_cluster' : y_spectral })
    return result_df


def Kmedoids_clustering(df, tweets_column, embeddings_df, n_clusters, max_iter= 1000) : 
  kmeans = KMedoids(n_clusters=n_clusters, init='k-medoids++', metric = 'cosine', max_iter=max_iter)
  cluster_ids = kmeans.fit_predict(embeddings_df)
  result_df = pd.DataFrame({'tweets' : df[tweets_column], 'topic_cluster' : cluster_ids })
  return result_df






# Grid Seach

# grid search parameters
num_clusters = [6,8,10,12]
tolerances= [5*1e-04,1e-05,1e-06]
inits = [10,25, 50, 100]
iterations = [700,1000,1500]
algorithms = ['auto', 'full']




def KMeans_grid_search(num_clusters, embeddings_df, tolerances, inits, algorithms,iterations, return_results = True): 

    scoring = {'Silhouette': cv_silhouette_scorer, 'Bouldin': cv_db_scorer}
    param_grid = {'n_clusters' : num_clusters, 'tol': tolerances, 'n_init' : inits, 'max_iter' : iterations, 'algorithm' : algorithms}
    cv = [(slice(None), slice(None))]

    clustering_model = KMeans(init = 'k-means++')
    grid_search = GridSearchCV(estimator = clustering_model, param_grid = param_grid, scoring = scoring, cv = cv, refit = 'Silhouette', verbose = 1 , n_jobs = -1)
    grid_search.fit(embeddings_df)


    results = pd.DataFrame(grid_search.cv_results_)
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_

    if return_results == True: 
        filename = 'Gridsearch' + str(datetime.now().hour) + str(datetime.now().minute) + '.csv'
        results.to_csv(filename)

    return results, best_params, best_estimator


def KMedoids_grid_search(num_clusters, embeddings_df,iterations, return_results = True): 

    scoring = {'Silhouette': cv_silhouette_scorer, 'Bouldin': cv_db_scorer}
    param_grid = {'n_clusters' : num_clusters,  'max_iter' : iterations}
    cv = [(slice(None), slice(None))]

    clustering_model = KMedoids(init = 'k-means++', metric = 'cosine')
    grid_search = GridSearchCV(estimator = clustering_model, param_grid = param_grid, scoring = scoring, cv = cv, refit = 'Silhouette', verbose = 1 , n_jobs = -1)
    grid_search.fit(embeddings_df)


    results = pd.DataFrame(grid_search.cv_results_)
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_

    if return_results == True: 
        filename = 'Gridsearch' + str(datetime.now().hour) + str(datetime.now().minute) + '.csv'
        results.to_csv(filename)

    return results, best_params, best_estimator



def Best_KMeans(num_clusters, embeddings_df, tolerances, inits, algorithms,iterations, print_params = True): 
    results, best_params, best_estimator = KMeans_grid_search(num_clusters, embeddings_df, tolerances, inits, algorithms,iterations, return_results = True)
    cluster_ids = best_estimator.fit_predict(embeddings_df)
    result_df = pd.DataFrame({'tweets' : df[tweets_column], 'topic_cluster' : cluster_ids })
    if print_params == True : 
        print(best_params)
    return result_df


def Best_KMedoids(num_clusters, embeddings_df,iterations, print_params = True): 
    results, best_params, best_estimator = KMedoids_grid_search(num_clusters, embeddings_df,iterations, return_results = True)
    cluster_ids = best_estimator.fit_predict(embeddings_df)
    result_df = pd.DataFrame({'tweets' : df[tweets_column], 'topic_cluster' : cluster_ids })
    if print_params == True : 
        print(best_params)
    return result_df













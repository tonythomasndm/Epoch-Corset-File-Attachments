"""*****************************************************************************************
IIIT Delhi License
Copyright (c) 2023 Supratim Shit
*****************************************************************************************"""

from sklearn.datasets import fetch_kddcup99
from sklearn.cluster import kmeans_plusplus
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import wkpp as wkpp 
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")

dataset = fetch_kddcup99()								# Fetch kddcup99 
data = dataset.data										# Load data
data = np.delete(data,[0,1,2,3],1) 						# Preprocess
data = data.astype(float)								# Preprocess
data = StandardScaler().fit_transform(data)				# Preprocess

n = np.size(data,0)										# Number of points in the dataset
d = np.size(data,1)										# Number of dimension/features in the dataset.
k = 17													# Number of clusters (say k = 17)
Sample_size = 100										# Desired coreset size (say m = 100)


def D2(data,k):											# D2-Sampling function.


	#----------Your code for Algo-1----------#


	return center 										# Returns B in Algo-1.

# centers = D2(data,k)									# Call D2-Sampling (D2())

def Sampling(data,k,centers,Sample_size):				# Coreset construction function.


	#----------Your code for Algo-2----------#


	return coreset, weight 								# Return coreset points and its weights.

# coreset, weight = Sampling(data,k,centers,Sample_size)	# Call coreset construction algorithm (Sampling())

#---Running KMean Clustering---#
fkmeans = KMeans(n_clusters=k,init='k-means++')
fkmeans.fit_predict(data)

#----Practical Coresets performance----# 	
# Coreset_centers, _ = wkpp.kmeans_plusplus_w(coreset, k, w=weight, n_local_trials=100)						# Run weighted kMeans++ on coreset points
# wt_kmeansclus = KMeans(n_clusters=k, init=Coreset_centers, max_iter=10).fit(coreset,sample_weight = weight)	# Run weighted KMeans on the coreset, using the inital centers from the above line.
# Coreset_centers = wt_kmeansclus.cluster_centers_															# Compute cluster centers
# coreset_cost = np.sum(np.min(cdist(data,Coreset_centers)**2,axis=1))										# Compute clustering cost from the above centers
# reative_error_practicalCoreset = abs(coreset_cost - fkmeans.inertia_)/fkmeans.inertia_						# Computing relative error from practical coreset, here fkmeans.inertia_ is the optimal cost on the complete data.

#-----Uniform Sampling based Coreset-----#
tmp = np.random.choice(range(n),size=Sample_size,replace=False)		
sample = data[tmp][:]																						# Uniform sampling
sweight = n*np.ones(Sample_size)/Sample_size 																# Maintain appropriate weight
sweight = sweight/np.sum(sweight)																			# Normalize weight to define a distribution

#-----Uniform Samling based Coreset performance-----# 	
wt_kmeansclus = KMeans(n_clusters=k, init='k-means++', max_iter=10).fit(sample,sample_weight = sweight)		# Run KMeans on the random coreset
Uniform_centers = wt_kmeansclus.cluster_centers_															# Compute cluster centers
uniform_cost = np.sum(np.min(cdist(data,Uniform_centers)**2,axis=1))										# Compute clustering cost from the above centers
reative_error_unifromCoreset = abs(uniform_cost - fkmeans.inertia_)/fkmeans.inertia_						# Computing relative error from random coreset, here fkmeans.inertia_ is the optimal cost on the full data.
	

# print("Relative error from Practical Coreset is",reative_error_practicalCoreset)
print("Relative error from Uniformly random Coreset is",reative_error_unifromCoreset)
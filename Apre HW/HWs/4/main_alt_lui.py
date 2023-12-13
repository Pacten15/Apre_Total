import pandas as pd
import numpy as np

#*####################################################################################
#* 1)
#*####################################################################################

#* import data
from scipy.io.arff import loadarff
data = loadarff("pd_speech.arff")
df = pd.DataFrame(data[0])
df['class'] = df['class'].str.decode('utf-8')
df['class'] = pd.to_numeric(df["class"])

#* aux variable
num_columns = df.shape[1]

#* pre-process data
from sklearn.preprocessing import MinMaxScaler
df_scaled = df.copy()
df_scaled.iloc[:, 0:num_columns-1] = MinMaxScaler().fit_transform(df.iloc[:, 0:num_columns-1])

#* partition data
X, y = df_scaled.iloc[:, 0:num_columns-1], df_scaled["class"]

#* parameterize clustering
from sklearn import cluster
kmeans_algo_0 = cluster.KMeans(n_clusters=3, random_state=0)
kmeans_algo_1 = cluster.KMeans(n_clusters=3, random_state=1)
kmeans_algo_2 = cluster.KMeans(n_clusters=3, random_state=2)

#* learn the model
kmeans_model_0 = kmeans_algo_0.fit(X)
kmeans_model_1 = kmeans_algo_1.fit(X)
kmeans_model_2 = kmeans_algo_2.fit(X)

#* produced clusters
y_pred_0 = kmeans_model_0.labels_
y_pred_1 = kmeans_model_1.labels_
y_pred_2 = kmeans_model_2.labels_

#* compute Silhouette
from sklearn import metrics
print("[0] Silhouette (euclidian):", metrics.silhouette_score(X, y_pred_0, metric='euclidean'))
print("[1] Silhouette (euclidian):", metrics.silhouette_score(X, y_pred_1, metric='euclidean'))
print("[2] Silhouette (euclidian):", metrics.silhouette_score(X, y_pred_2, metric='euclidean'))

#/ compute Purity
import numpy as np
def purity_score(y_true, y_pred):
#/compute contingency/confusion matrix  
    confusion_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)
y_true = y
print("[0] Purity:", purity_score(y_true, y_pred_0))
print("[1] Purity:", purity_score(y_true, y_pred_1))
print("[2] Purity:", purity_score(y_true, y_pred_2),"\n")

#*####################################################################################
#* 2)
#*####################################################################################

# O que está a causar o não determinismo é facto de estarmos inicialmente a considerar que o centroide das três 
# clusters de uma forma completamente aleatória e usando a distancia da euclidiana dos pontos 
# a esses centroides associamos os pontos aos clusteres dos respetivos clusteres e caso o centroide 
# altere quando os pontos estão associados aos seus respetivos clusters,  então voltamos a recalcular 
# os centroides até que o calculo do novo centroide resulte nos centroides usados no calculo do mesmo. 

#*####################################################################################
#* 3)
#*####################################################################################

#* compute features' variances
from sklearn.feature_selection import VarianceThreshold
selection = VarianceThreshold().fit(X)

#* get second max variance
import heapq
max_three_variances = heapq.nlargest(3, selection.variances_)
third_max_variance = max_three_variances[2]

print(max_three_variances)

#* feature selection
X_new = VarianceThreshold(threshold=third_max_variance).fit_transform(X)


#* plot
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.title(label="Original Parkinsons Diagnoses")
plt.scatter(X_new[:,0], X_new[:,1], c=y)
plt.subplot(122)
plt.title(label="Learned k=3 Clusters")
plt.scatter(X_new[:,0], X_new[:,1], c=y_pred_0)
plt.savefig("figures/plot.png")
plt.show()

#*####################################################################################
#* 4)
#*####################################################################################
#? a variability é a soma das fraction of variance das principal components?

#* learn the transformation (components as linear combination of features)
from sklearn.decomposition import PCA
pca = PCA(n_components=X.shape[1])
pca.fit(X)
variability = 0
number_of_principal_components = 1
for f in pca.explained_variance_ratio_:
    variability += f
    if variability > 0.80: 
        break
    number_of_principal_components += 1

print(number_of_principal_components)

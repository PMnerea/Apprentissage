# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import kmedoids
import scipy . cluster . hierarchy as shc

from sklearn import cluster
from sklearn.metrics import davies_bouldin_score
from scipy.io import arff
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import mutual_info_score
from sklearn.metrics import silhouette_score

"""
 =================== Jeux de données =================
"""

path ='./artificial/'
databrut = arff.loadarff(open(path+"twenty.arff",'r'))
data = [[x[0],x[1]] for x in databrut[0]]

f0 =[f[0] for f in data]
f1 =[f[1] for f in data]

plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales")
plt.show()

"""
 =================== Clustering K-M =================
"""

datanp = data

print ( " Appel KMeans pour une valeur fixee de k " )
tps1 = time.time()

vectMet=[]

for i in range(2, 30) : 
    model = cluster.KMeans(n_clusters=i, init ='k-means++')
    model.fit(datanp)
    met = davies_bouldin_score(datanp, model.labels_)
    vectMet.append(met) 

plt.scatter(list(range(2,30)), vectMet)
plt.title("Metrics de Davies-Bouldin" )
plt.show()

# +2 car les indices commencent à 0 et nos inices commencent à 2
k=np.argmin(vectMet)+2
model = cluster.KMeans(n_clusters=k, init ='k-means++')
model.fit(datanp)

tps2 = time.time()
labels = model.labels_
iteration = model.n_iter_

plt.scatter(f0,f1, c=labels, s=8)
plt.title(" Donnees apres clustering Kmeans " )
plt.show()
print( "nb clusters = " ,k , " , nb iter =" , iteration, " , . . . . . . runtime = " , round((tps2 - tps1 ) * 1000 , 2 ) , " ms " )


"""
 =================== Kmedoids =================
"""

tps1 = time.time()
distmatrix=euclidean_distances(datanp)

vectMet=[]

for i in range(2, 30) :
    fp= kmedoids.fasterpam(distmatrix,i)
    met = davies_bouldin_score(datanp, fp.labels)
    vectMet.append(met)

plt.scatter(list(range(2,30)), vectMet)
plt.title("Metrics de Davies-Bouldin" )
plt.show()
    

# +2 car les indices commencent à 0 et nos inices commencent à 2
k= int(np.argmin(vectMet)+2)
fp= kmedoids.fasterpam(distmatrix,k)
    
tps2= time.time()
iter_kmed = fp.n_iter
labels_kmed = fp.labels
print( " Loss with FasterPAM : " , fp.loss)
plt.scatter( f0 , f1 , c=labels_kmed , s =8)
plt.title( " Donnees apres clustering KMedoids " )
plt.show()
print( " nb clusters =" ,k , " , nb iter =" , iter_kmed , " , . . . . . . runtime = " , round ( ( tps2 - tps1 ) * 1000 , 2 ) , " ms " )


"""
 ================ Compare k-means & kmedoids ==========
"""

print("rand_score of kmeans compared to kmed : ", rand_score(labels, labels_kmed))

print("mutual_info_score of kmeans compared to kmed : ", mutual_info_score(labels, labels_kmed))


"""
 ================== Test of different metrics ==========
"""

tps1 = time.time()
euclidean=euclidean_distances(datanp)
manhattan=manhattan_distances(datanp)

# on reprend le k trouvé précédemment
fp_euclidean= kmedoids.fasterpam(euclidean,k)
fp_manhattan= kmedoids.fasterpam(manhattan,k)
    
tps2= time.time()
iter_kmed_euclidean = fp_euclidean.n_iter
iter_kmed_manhattan = fp_manhattan.n_iter
labels_kmed_euclidean = fp_euclidean.labels
labels_kmed_manhattan = fp_manhattan.labels

plt.scatter( f0 , f1 , c=labels_kmed_euclidean , s =8)
plt.title( " Donnees apres clustering KMedoids (distance euclidienne" )
plt.show()

plt.scatter( f0 , f1 , c=labels_kmed_manhattan, s =8)
plt.title( " Donnees apres clustering KMedoids (distance de manhattan" )
plt.show()

print("rand_score of kmedoids with euclidean and manhattan distances : ", rand_score(labels_kmed_euclidean, labels_kmed_manhattan))



"""
 =================== Cluestering agglomératif ================
"""

# Donnees dans datanp
print( " Dendrogramme ’single’ donnees initiales " )
linked_mat = shc.linkage( datanp , 'single')
plt.figure( figsize = ( 12 , 12 ) )
shc.dendrogram ( linked_mat , orientation = 'top', distance_sort = 'descending', show_leaf_counts = False )
plt.show()

# set di stance_threshold ( 0 ensures we compute the full tree )
tps1 = time.time()
model = cluster.AgglomerativeClustering(distance_threshold = 10 , linkage = 'single' , n_clusters = None )
model = model.fit( datanp )
tps2 = time.time()
labels = model.labels_
k = model.n_clusters_
leaves = model.n_leaves_

# Affichage clustering
plt.scatter( f0 , f1 , c = labels , s = 8 )
plt.title( " Resultat du clustering " )
plt.show()
print( " nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )

# set the number of clusters
k = 4
tps1 = time.time()
model = cluster.AgglomerativeClustering( linkage = 'single' , n_clusters = k )
model = model.fit( datanp )
tps2 = time.time()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_

# Affichage clustering
plt.scatter( f0 , f1 , c = labels , s = 8 )
plt.title( " Resultat du clustering " )
plt.show()
print( " nb clusters = " ,k , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )


"""
 ====================== Optimization seuil de distance =============
"""

tps1 = time.time()

vectMet=[]

for i in range(1, 10) : 
    model = cluster.AgglomerativeClustering(distance_threshold = i , linkage = 'single' , n_clusters = None )
    model.fit(datanp)
    labels = model.labels_
    met = silhouette_score(datanp, labels, metric='euclidean')
    vectMet.append(met) 

plt.scatter(list(range(1,10)), vectMet)
plt.title("Silhouette score" )
plt.show()

# +2 car les indices commencent à 0 et nos inices commencent à 2
nb_leaves=np.argmin(vectMet)+3
model = cluster.AgglomerativeClustering(distance_threshold = nb_leaves , linkage = 'single' , n_clusters = None )
model.fit(datanp)
tps2 = time.time()
labels = model.labels_
kres = model.n_clusters_
leaves = model.n_leaves_

# Affichage clustering
plt.scatter( f0 , f1 , c = labels , s = 8 )
plt.title( " Resultat du clustering " )
plt.show()
print( " nb clusters = " ,kres , " , nb feuilles = " , leaves , " runtime = " , round (( tps2 - tps1 ) * 1000 , 2 ) ," ms " )


"""
 ================ Optimization type de combination de clusters =============
"""

tps1 = time.time()

# On prend un nombre fixe k de clusters

# set the number of clusters
k = 20
tps1 = time.time()

model_single = cluster.AgglomerativeClustering( linkage = 'single' , n_clusters = k )
model_single = model_single.fit( datanp )

model_average = cluster.AgglomerativeClustering( linkage = 'average' , n_clusters = k )
model_average = model_average.fit( datanp )

model_ward = cluster.AgglomerativeClustering( linkage = 'ward' , n_clusters = k )
model_ward = model_ward.fit( datanp )

model_complete = cluster.AgglomerativeClustering( linkage = 'complete' , n_clusters = k )
model_complete = model_complete.fit( datanp )

tps2 = time.time()
labels_single = model_single.labels_
labels_average = model_average.labels_
labels_ward = model_ward.labels_
labels_complete = model_complete.labels_

kres_single = model_single.n_clusters_
kres_average = model_average.n_clusters_
kres_ward = model_ward.n_clusters_
kres_complete = model_complete.n_clusters_

# Affichage clustering single
plt.scatter( f0 , f1 , c = labels_single , s = 8 )
plt.title( " Resultat du clustering avec linkage single " )
plt.show()

# Affichage clustering average
plt.scatter( f0 , f1 , c = labels_average , s = 8 )
plt.title( " Resultat du clustering avec linkage average " )
plt.show()

# Affichage clustering ward
plt.scatter( f0 , f1 , c = labels_ward , s = 8 )
plt.title( " Resultat du clustering avec linkage ward " )
plt.show()

# Affichage clustering complete
plt.scatter( f0 , f1 , c = labels_complete , s = 8 )
plt.title( " Resultat du clustering avec linkage complete " )
plt.show()
















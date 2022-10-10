# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn import metrics
from sklearn import cluster
from sklearn.metrics import davies_bouldin_score
from scipy.io import arff

"""
 =================== Jeux de données =================
"""

path ='./artificial/'
databrut = arff.loadarff(open(path+"spiral.arff",'r'))
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
    print("Pour nb cluster =",i," on a une metrics de ",met)
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
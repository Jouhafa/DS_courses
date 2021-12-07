
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Classification hierarchique de coordonnées géographiques
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time

#read data
dataframe=pandas.read_csv("./J18_E2_coordinates_morroco.csv",sep=',')
print(dataframe.head)
listColNames=list(dataframe.columns)


#get usefull information
X=dataframe[['lat','lng']].values

X_with_pop=dataframe[['lat','lng','population']].values

city_names=list(dataframe['city'])

plt.scatter(X[:,1],X[:,0],c=X_with_pop[:,2],cmap='rainbow',alpha=0.5)
plt.colorbar()
plt.show()


#QUESTION 1:
# -> Allez a la page https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
# -> Trouvez comment effectuer une classification hierarchique pour clusteriser les villes du Maroc
#    en 4 groupes a l'aide de leurs coordonnees GPS (lattitude et longitude)
# -> Que signifient les differentes options de 'linkage'?
# -> Effectuez les clusterings avec les differentes options de 'linkage'. Comparez les temps de calculs
#    et les clusters trouves
# -> Comment expliquez-vous les differences de temps et de clusterings?

from sklearn.cluster import AgglomerativeClustering

#...
previous_time=time.time()
AggClu_ward=AgglomerativeClustering(n_clusters=4, linkage='ward')
AggClu_ward.fit(X)
print('time AggClu_ward:',time.time()-previous_time)

plt.scatter(X[:,1],X[:,0],c=AggClu_ward.labels_,cmap='rainbow',alpha=0.5)
plt.colorbar()
plt.title('ward')
plt.show()

#...

previous_time=time.time()
AggClu_complete=AgglomerativeClustering(n_clusters=4, linkage='complete')
AggClu_complete.fit(X)
print('time AggClu_complete:',time.time()-previous_time)

plt.scatter(X[:,1],X[:,0],c=AggClu_complete.labels_,cmap='rainbow',alpha=0.5)
plt.colorbar()
plt.title('complete')
plt.show()


#...



previous_time=time.time()
AggClu_average=AgglomerativeClustering(n_clusters=4, linkage='average')
AggClu_average.fit(X)
print('time AggClu_average:',time.time()-previous_time)

#...

plt.scatter(X[:,1],X[:,0],c=AggClu_average.labels_,cmap='rainbow',alpha=0.5)
plt.colorbar()
plt.title('average')
plt.show()




previous_time=time.time()
AggClu_single=AgglomerativeClustering(n_clusters=4, linkage='single')
AggClu_single.fit(X)
print('time AggClu_single:',time.time()-previous_time)

plt.scatter(X[:,1],X[:,0],c=AggClu_single.labels_,cmap='rainbow',alpha=0.5)
plt.colorbar()
plt.title('single')
plt.show()





#QUESTION 2:
# -> Reproduisez-ce tests avec le lien 'ward' sans definir a l'avance le nombre de
#    clusters voulus, mais plutot une distance maximum de 300km entre les elements
#    de chaque cluster. On notera que l'unite d'une coordonnee GPS correspond
#    environ a 69.47km
# -> Quel est alors le premier avantage de cette methode compare aux k-means?



previous_time=time.time()
AggClu_ward_dt=AgglomerativeClustering(n_clusters=None, distance_threshold=300/69.47,linkage='ward')
AggClu_ward_dt.fit(X)
print('time AggClu_ward:',time.time()-previous_time)

plt.scatter(X[:,1],X[:,0],c=AggClu_ward_dt.labels_,cmap='rainbow',alpha=0.5)
plt.colorbar()
plt.title('ward distance threshold')
plt.show()



#QUESTION 3:
# -> On s'interesse maintenant a "J18_E2_worldcities.csv" qui contient 41001
#    villes au lieu de 162
# -> Essayez de clusteriser ces nouvelles donnees de maniere a avoir 100 clusters
#    Que constatez-vous?
# -> Allez a la page https://scikit-learn.org/stable/modules/clustering.html et
#    trouvez une methode qui passe mieux a l'echelle (scalability) pour resoudre
#    le probleme.
# -> Une fois le clustering effectue, il est quasi impossible de representer
#    toutes les donnees clusterisees. Ne representez alors qu'une ville de chaque
#    classe.

#read data
dataframe=pandas.read_csv("./J18_E2_worldcities.csv",sep=',')
print(dataframe.head)
listColNames=list(dataframe.columns)

#get usefull information
X=dataframe[['lat','lng']].values
city_names=list(dataframe['city'])

#clustering

#previous_time=time.time()
#AggClu_ward_world=AgglomerativeClustering(n_clusters=100,linkage='ward')
#AggClu_ward_world.fit(X)
#print('time AggClu_ward:',time.time()-previous_time)

# -> prend beaucoup trop de temps... on va utiliser les KMeans a la place qui
#    semblent adaptes pour de gos jeux de donnees et un nombre raisonable de
#    clusters

from sklearn.cluster import KMeans

KMclassifier=KMeans(n_clusters=100)
KMclassifier.fit(X)

Id_clusters=list(set(KMclassifier.labels_))

import random

Cluster_reprensentents=[]
for cluster_id in Id_clusters:
    obs_with_cluster_id=np.where(KMclassifier.labels_==cluster_id)[0]
    toto=random.choice(obs_with_cluster_id)
    Cluster_reprensentents.append(toto)

fig, ax = plt.subplots()
ax.scatter(X[Cluster_reprensentents,1],X[Cluster_reprensentents,0],c=Id_clusters,cmap='rainbow')
#fig.colorbar()

for i in range(len(Id_clusters)):
    ax.annotate(city_names[Cluster_reprensentents[i]], (X[Cluster_reprensentents,1][i],X[Cluster_reprensentents,0][i]))

plt.show()

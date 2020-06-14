import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

A = np.array([[2,-1,-1,0,0,0,0,0,0],
              [-1,3,-1,-1,0,0,0,0,0],
              [-1,-1,3,0,0,0,-1,0,0],
              [0,-1,0,3,-1,-1,0,0,0],
              [0,0,0,-1,2,-1,0,0,0],
              [0,0,0,-1,-1,3,0,-1,0],
              [0,0,-1,0,0,0,3,-1,-1],
              [0,0,0,0,0,-1,-1,3,-1],
              [0,0,0,0,0,0,-1,-1,2]
              ])

results = la.eig(A)

#print(results[1])
print(results[0].real)
#print(A@results[1][:,1])
#print(results[0][1].real*results[1][:,1])
y1=results[1][:,1].reshape(9,1) #3
y2=results[1][:,2].reshape(9,1) #0.6
y3=results[1][:,3].reshape(9,1) #0
X=scipy.linalg.orth(results[1][:,3:6])

#print(y1,y2,y3)
#print(results[1][:,3:6])
kmeans = KMeans(n_clusters=3,random_state=0).fit(X)
print(kmeans.labels_)


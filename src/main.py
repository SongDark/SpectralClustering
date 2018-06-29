import sys
sys.path.append("..")

from utils.similarity import calEuclidDistanceMatrix
from utils.knn import myKNN
from utils.laplacian import calLaplacianMatrix
from utils.dataloader import genTwoCircles
from utils.ploter import plot
from sklearn.cluster import KMeans
import numpy as np
np.random.seed(1)

data, label = genTwoCircles(n_samples=500)

Similarity = calEuclidDistanceMatrix(data)

Adjacent = myKNN(Similarity, k=10)

Laplacian = calLaplacianMatrix(Adjacent)

x, V = np.linalg.eig(Laplacian)

x = zip(x, range(len(x)))
x = sorted(x, key=lambda x:x[0])

H = np.vstack([V[:,i] for (v, i) in x[:500]]).T

kmeans = KMeans(n_clusters=2).fit(H)

plot(data, kmeans.labels_)

from matplotlib import pyplot as plt
from itertools import cycle, islice
import numpy as np

def plot(X, y_sp, y_km):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#999999', '#e41a1c', '#dede00']),
                                        int(max(y_km) + 1))))
    plt.subplot(121)
    plt.scatter(X[:,0], X[:,1], s=10, color=colors[y_sp])
    plt.title("Spectral Clustering")
    plt.subplot(122)
    plt.scatter(X[:,0], X[:,1], s=10, color=colors[y_km])
    plt.title("Kmeans Clustering")
    # plt.show()
    plt.savefig("../figures/spectral_clustering.png")
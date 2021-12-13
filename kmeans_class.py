import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from scipy.spatial import distance

class Kmeans:
    centers = None,
    cluster = None,
    K = 0,
    X = None,
    PC = None,

    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.cluster = np.zeros(self.X.shape[0], dtype=int)
        self.PC = PCA(n_components=2).fit(self.X)

    def set_centers(self, centers):
        self.centers = centers
        return self

    def assignment_step(self):
        for idx, x in enumerate(self.X):
            self.cluster[idx] = np.argmin(distance.cdist([x], self.centers, 'euclidean')[0])
        return self

    def updating_step(self):
        centersCounter = [[0, 0.0] for _ in self.centers]
        for idx, c in enumerate(self.cluster):
            centersCounter[c][0] += 1
            centersCounter[c][1] += self.X[idx]
        self.centers = [
            centersCounter[idx][1] / centersCounter[idx][0]
            if centersCounter[idx][0] != 0 else 0
            for idx, c in enumerate(self.centers)
        ]
        return self

    def sequential_step(self, x, alpha):
        minCenter = np.argmin(distance.cdist([x], self.centers, 'euclidean')[0])
        self.centers[minCenter] += alpha * (x - self.centers[minCenter])
        return self

    def plot(self, title=""):
        if self.centers is None:
            print("No centroids defined")

        # Function to plot current state of the algorithm.
        # For visualisation purposes, only the first two PC are shown.
        PC = self.PC.transform(self.X)
        C2 = self.PC.transform(self.centers)

        if self.cluster[0] is None:
            plt.scatter(PC[:, 0], PC[:, 1], alpha=0.5)
        else:
            plt.scatter(PC[:, 0], PC[:, 1], c=self.cluster, alpha=0.5)

        plt.scatter(C2[:, 0], C2[:, 1], s=100, c=np.arange(self.K), edgecolors='black')
        plt.title(title)
        plt.show()
        plt.clf()

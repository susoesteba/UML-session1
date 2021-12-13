import numpy as np
from sklearn import datasets
from kmeans_class import Kmeans

X = datasets.load_iris().data
np.random.seed(1)
K = 3
km = Kmeans(X, K)

S0 = X[(0,1,2),:]

km = km.set_centers(S0)
km = km.assignment_step()
km.plot("1st assignment")
km = km.updating_step().assignment_step()
km.plot("2nd assignment")

mean_ = Kmeans(X, K=1)
mean_.set_centers(np.array([X[0,:]]))
for i in range(X.shape[0]):
  x = X[i,:]
  mean_ = mean_.sequential_step(x, 1.0/(i+1))

mean_.plot()
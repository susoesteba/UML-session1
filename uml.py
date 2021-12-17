import numpy as np
from sklearn import datasets
from kmeans_class import Kmeans

# X = datasets.load_iris().data
# np.random.seed(1)
# K = 3
# km = Kmeans(X, K)
#
# S0 = X[(0,1,2),:]
#
# km = km.set_centers(S0)
# km = km.assignment_step()
# km.plot("1st assignment")
# km = km.updating_step().assignment_step()
# km.plot("2nd assignment")
#
# mean_ = Kmeans(X, K=1)
# mean_.set_centers(np.array([X[0,:]]))
# for i in range(X.shape[0]):
#   x = X[i,:]
#   mean_ = mean_.sequential_step(x, 1.0/(i+1))
#
# mean_.plot()

import numpy as np
X = np.array([[4, 8.5],
[6.5,7.5],
[6, 9],
[1, 1],
[2, 4],
[4, 2],
[7, 4],
[8, 1],
[9, 2]])
K = 3
km = Kmeans(X,K)

km.set_centers(np.array([[5,5], [8,3], [3,8]], dtype = 'float64'))
# km.assignment_step()
# km.updating_step()
# km.assignment_step()
# km.updating_step()
# km.assignment_step()

x = X[0,:]
km.sequential_step(x, 0.8)
x = X[1,:]
km.sequential_step(x, 0.8)
x = X[2,:]
km.sequential_step(x, 0.8)
x = X[3,:]
km.sequential_step(x, 0.8)
x = X[4,:]
km.sequential_step(x, 0.8)
x = X[5,:]
km.sequential_step(x, 0.8)
x = X[6,:]
km.sequential_step(x, 0.8)
x = X[7,:]
km.sequential_step(x, 0.8)
x = X[8,:]
km.sequential_step(x, 0.8)

km.plot()

for I in range(9):
    x = X[I,:]
    km = km.sequential_step(x, 0.6)

km.plot()

for I in range(9):
    x = X[I,:]
    km = km.sequential_step(x, 0.4)

km.plot()

for I in range(9):
    x = X[I,:]
    km = km.sequential_step(x, 0.2)

km.plot()

for I in range(9):
    x = X[I,:]
    km = km.sequential_step(x, 0.1)

km.plot()

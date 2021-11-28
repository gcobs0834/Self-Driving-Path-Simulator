import numpy as np
from numpy.linalg import norm, pinv
import matplotlib as mpl
import matplotlib.pyplot as plt
from Genetic import RBFN
import random
from math import exp

def population(x_dim, hidden_nodes):
        pop = []
        for _ in range(10):
            theta = [random.uniform(-1, 1)]
            w = [random.uniform(-1, 1) for _ in range(hidden_nodes)]
            m = [random.uniform(0, 80) for _ in range(hidden_nodes * x_dim)]
            sigma = [random.uniform(0, 1) for _ in range(hidden_nodes)]
            pop.append(np.array(theta + w + m + sigma))
        with open('../weights/init0.txt', 'w') as f:
            np.savetxt(f, np.array(pop)[0])
        return np.array(pop)


def load_data(file_path):
    result=[]
    with open(file_path) as f:
        data=f.readlines()
    for i in range(len(data)):
        row = list(map(lambda x: float(x), data[i].replace('\n', '').split(' ')))
        result.append(row)
    return np.array(result)

path = "../train4dAll.txt"
weights_path = "../weights/RBFN_params.txt"
# weights_path = "./best1.txt"

rbfn_weights = np.loadtxt(weights_path)

data = load_data(path)
X = data[:,:-1]
y = data[:, -1]
# print(X.shape)

pop = population(X.shape[1], 10)[3]

rbf = RBFN(X, y,10, pop)
z = rbf.output(X)
print(z[:20])



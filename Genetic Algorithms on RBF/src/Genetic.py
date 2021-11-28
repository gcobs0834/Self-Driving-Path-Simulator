import numpy as np
from copy import deepcopy
from numpy.linalg import norm, pinv
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time
from math import exp


class GA():

    def __init__(self,X ,y,J, iteration, POP_SIZE, mutate_rate, cross_rate):
        self.iter = iteration
        self.X = X
        self.y = y
        self.POP_SIZE = POP_SIZE
        self.DNA_SIZE = 1+J+X.shape[1]*J+J
        self.theta_idx = 0
        self.w_idx = 1
        self.m_idx = 1+J
        self.sigma_idx = 1+J+X.shape[1]*J
        self.hidden_nodes = J
        self.pop = self.population(X.shape[1] , self.hidden_nodes)
        self.mutate_rate = mutate_rate
        self.cross_rate = cross_rate
        self.start_time = time()
        self.best_f = 100000000000

    def population(self, x_dim, hidden_nodes):
        pop = []
        for _ in range(self.POP_SIZE):
            theta = [np.random.uniform(-40, 40)]
            w = [np.random.uniform(-1, 1) for _ in range(hidden_nodes)]
            m = [np.random.uniform(0, 80) for _ in range(hidden_nodes * x_dim)]
            sigma = [np.random.uniform(0, 1) for _ in range(hidden_nodes)]
            pop.append(np.array(theta + w + m + sigma))
        with open('../weights/init0.txt', 'w') as f:
            np.savetxt(f, np.array(pop)[0])
        return np.array(pop)

    def error(self, y, predict):
        error = 0
        for idx, data in enumerate(y):
            error += ((predict[idx] - y[idx]) ** 2)
        return (error/2)

    def fitness(self, X, y):
        _pop = self.pop
        fitness = np.zeros(_pop.shape[0], dtype=np.float)
        for idx, DNA in enumerate(_pop):
            # run RBF for every DNA
            rbf = RBF(X, y, self.hidden_nodes, DNA)
            # rbf.train(X,y)
            predict = rbf.predict(X)
            fitness[idx] = (self.error(y, predict))
        return fitness

    def select(self, fitness):
        best_idx = np.argmin(fitness)
        best = deepcopy(self.pop[best_idx])
        best_size = int(self.POP_SIZE/2)
        if min(fitness) < self.best_f:
            self.save(best,'../weights/RBFN_dim{}.txt'.format(self.X.shape[1]))
            self.best_f = min(fitness)
        idx = np.random.choice(np.arange(self.POP_SIZE), size=(self.POP_SIZE), replace=True,
                                p=(1/fitness) / (1/fitness).sum())
        print("before idx = {}".format(idx))
        print('best idx  ={}'.format(best_idx))
        idx[:idx.size//2:-1]= best_idx
        print("idx = {}".format(idx))
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            idx = np.random.randint(0, self.POP_SIZE, size=1)
            mask = np.random.randint(0, 2, size=self.DNA_SIZE).astype(
                np.bool)  # random init mask for crossover
            parent[mask] = pop[idx, mask]
        return parent

    def mutate(self,child):
        #mutate for weight
        for point in range(self.w_idx, self.m_idx):
            if np.random.rand() < self.mutate_rate:
                exchange = np.random.choice(range(self.w_idx, self.m_idx))
                child[point], child[exchange] = child[exchange],child[point]
        #mutate for m
        for point in range(self.m_idx, self.sigma_idx):
            if np.random.rand() < self.mutate_rate:
                exchange = np.random.choice(range(self.m_idx, self.sigma_idx))
                child[point], child[exchange] = child[exchange],child[point]
        #mutate for sigma
        for point in range(self.sigma_idx, self.DNA_SIZE):
            if np.random.rand() < self.mutate_rate:
                exchange = np.random.choice(range(self.sigma_idx, self.DNA_SIZE))
                child[point], child[exchange] = child[exchange],child[point]
        return child

    def evolve(self, X, y):
        for epoch in range(self.iter):
            fitness = self.fitness(X, y)
            print("time(sec):{} epoch-{} Best fitness = {}, averag fitness = {}".format(
                int(time()-self.start_time), epoch, min(fitness) , sum(fitness)/self.POP_SIZE))
            self.pop = self.select(fitness)
            pop_copy = self.pop.copy()
            # crossover and mutate
            for parent in self.pop:
                child = self.crossover(parent, pop_copy)
                child = self.mutate(child)
                parent[:] = child  # parent is replaced by its child
        fitness = self.fitness(X, y)
        sort_idx = sorted(range(len(fitness)), key = lambda x: fitness[x])
        self.pop = self.pop[sort_idx]
    
    def best_DNA(self):
        return [self.pop[0],self.pop[-1]]
    
    def save(self, DNA,path='../weights/RBFN_params6d.txt'):
        with open(path, 'w') as f:
            np.savetxt(f, DNA)


class RBF():

    def __init__(self, X,Y, hidden_nodes, DNA):
        self.input_dim = X.shape[1]
        self.hidden_nodes = hidden_nodes  # J
        self.theta = DNA[0]
        self.W = np.array(DNA[1:1+hidden_nodes])
        self.hidden = np.reshape(DNA[1+hidden_nodes : 1+hidden_nodes+X.shape[1]*hidden_nodes], (-1,X.shape[1]))
        self.sigma = np.array(DNA[1+hidden_nodes+X.shape[1]*hidden_nodes:])

    def _calPhi(self, x_i, m_j, beta):
        return np.exp(beta * norm(x_i - m_j) ** 2)

    def _calAct(self, X):
        Phi = np.zeros((X.shape[0], self.hidden_nodes), dtype=np.float)
        for m_idx, m_num in enumerate(self.hidden):
            for x_idx, x_num in enumerate(X):
                Phi[x_idx, m_idx] = self._calPhi(x_num, m_num, -1/2*(self.sigma[m_idx]**2))
        return Phi

    def normDegree(self, y, deNorm=False):
        if deNorm != True:
            n_y = 2*(y - np.min(y))/np.ptp(y) - 1
            return n_y
        else:
            n_y = (y+1)/2*80-40
            n_y = [max(min(y, 40),-40) for y in n_y]
            return n_y

    def train(self, X, Y):
        """
        :param X: N*input_dim
        :param Y: N*output_dim
        :return :
        """
        # X = self.normDegree(X)
        Y = self.normDegree(Y)
        _idx = np.random.permutation(X.shape[0])[:self.hidden_nodes]
        self.hidden = [X[i, :] for i in _idx]
        Phi = self._calAct(X)
        self.W = np.dot(pinv(Phi), Y)

    def predict(self, X):
        Phi = self._calAct(X)
        Y = np.dot(Phi, self.W) +self.theta
        # print('before rbn :'+str(Y[-15:-10]))
        Y = self.normDegree(Y,True)
        # print('rbn :'+str(Y[-15:-10]))
        return Y


def load_data(file_path):
    result=[]
    with open(file_path) as f:
        data=f.readlines()
    for i in range(len(data)):
        row = list(map(lambda x: float(x), data[i].replace('\n', '').split(' ')))
        result.append(row)
    return np.array(result)


if __name__ == '__main__':
    # n = 100
    # x= np.linspace(-1, 1, n).reshape(n, 1)
    # y = np.sin(2 * (x+0.5)**3 - 1)

    # rbf= RBF(1, 500, 1, 0.5)
    # rbf.train(x, y)
    # z = rbf.predict(x)

    path = "../train6dAll.txt"

    data = load_data(path)
    X = data[:,:-1]
    y = data[:, -1]
    
    # GA(X ,Y,J, iteration, POP_SIZE, mutate_rate, cross_rate)
    # RBF(X,Y, hidden_nodes, out_dim, DNA))
    ga = GA(X, y, 12, 20, 50, 0.2, 0.7)

    ga.evolve(X,y)
    fitness = ga.fitness(X,y)
    sort_idx = sorted(range(len(fitness)), key = lambda x: fitness[x])
    print(fitness)

    print(ga.best_DNA())

    rbf = RBF(X, y, 12, ga.best_DNA()[0])
    z = rbf.predict(X)
    print(z[-10:])

    rbf = RBF(X, y, 12, ga.best_DNA()[1])
    z = rbf.predict(X)
    print(z[-10:])

    a_file = open("best0.txt", "w")
    # for row in ga.best_DNA()[0]:
    np.savetxt(a_file, ga.best_DNA()[0])
    a_file.close()

    b_file = open("best1.txt", "w")
    # for row in ga.best_DNA()[1]:
    np.savetxt(b_file, ga.best_DNA()[1])
    b_file.close()


    # plt.plot(x,y, 'k-', label = 'actual')
    # plt.plot(x,z, 'r-', label = 'predict')


    # plt.xlim(-1.2,1.2)
    # plt.title('RBF Test', fontsize = 20, color = 'r')
    # plt.legend(loc = 'upper left')
    # plt.show()

    # ga = GA(iteration=100, POP_SIZE=100, mutate_rate=0.1, cross_rate=0.5,rbf_sigma=0.5)

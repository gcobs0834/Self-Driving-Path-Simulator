import numpy as np
from copy import deepcopy
from numpy.lib.function_base import average
from numpy.linalg import norm, pinv
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time
from math import exp
import random
import math


class Particle(object):
    def __init__(self, j_dim, i_dim, ranges):
        self.vmax = 3
        self.ranges = ranges
        self.upbound_of_SD = 10
        # individual = > 'thita', 'w', 'm', 'sd', 'adapt_value'
        self.theta = np.array([random.uniform(-1, 1)])
        self.weight = np.zeros(j_dim)
        self.means = np.zeros(j_dim*i_dim)
        self.sd = np.zeros(j_dim)
        self.fitness = None
        # w initialization
        for i in range(j_dim):
            self.weight[i] = random.uniform(-1, 1)
        # m initialization
        for i in range(j_dim*i_dim):
            self.means[i] = random.uniform(ranges[1], ranges[0])
        # sd initialization
        for i in range(j_dim):
            self.sd[i] = random.uniform(0.001, 10)
        
        # p-vector record the best lacation found by particle so far
        self.p_theta = deepcopy(self.theta)
        self.p_weight = deepcopy(self.weight)
        self.p_means = deepcopy(self.means)
        self.p_sd = deepcopy(self.sd)
        self.p_fitness = None

        # v-vector record the best lacation found by particle so far
        self.v_theta = np.array([random.uniform(-1, 1)])
        self.v_weight = np.zeros(j_dim)
        self.v_means = np.zeros(j_dim*i_dim)
        self.v_sd = np.zeros(j_dim)
        # w initialization
        for i in range(j_dim):
            self.v_weight[i] = random.uniform(-1, 1)
        # m initialization
        for i in range(j_dim*i_dim):
            self.v_means[i] = random.uniform(ranges[1], ranges[0])
        # sd initialization
        for i in range(j_dim):
            self.v_sd[i] = random.uniform(1/2 * 10, 10)
        

    def printmyself(self):
        print('theta', self.theta, 'p', self.p_theta)
        print('weight', self.weight, 'p', self.p_weight)
        print('means', self.means, 'p', self.p_means)
        print('sd', self.sd, 'p', self.p_sd)
        print('fitness', self.fitness, 'p', self.p_fitness)

    def update_p(self):
        self.p_theta = deepcopy(self.theta)
        self.p_weight = deepcopy(self.weight)
        self.p_means = deepcopy(self.means)
        self.p_sd = deepcopy(self.sd)
        self.p_fitness = deepcopy(self.fitness)

    def update_location(self):
        self.theta = deepcopy(self.theta + self.v_theta)
        self.weight = deepcopy(self.weight + self.v_weight)
        self.means = deepcopy(self.means + self.v_means)
        self.sd = deepcopy(self.sd + self.v_sd)
        self.fitness = None

    def limit_v(self):
        np.clip(self.v_theta, -1*self.vmax , 1*self.vmax, out=self.v_theta)
        np.clip(self.v_weight, -1*self.vmax, 1*self.vmax, out=self.v_weight)
        np.clip(self.v_means, -1*self.vmax, self.vmax, out=self.v_means)
        np.clip(self.v_sd, -1*self.vmax, self.vmax, out=self.v_sd)
    
    def limit_location_upbound(self):
        np.clip(self.theta, -1, 1, out=self.theta)
        np.clip(self.weight, -1, 1, out=self.weight)
        np.clip(self.means, self.ranges[1], self.ranges[0], out=self.means)
        np.clip(self.sd, 0.00000000000000000000001, self.upbound_of_SD, out=self.sd)

class RBFN():

    def __init__(self, hidden_nodes, theta, weight, hidden, sigma):
        self.input_dim = 3
        self.hidden_nodes = hidden_nodes  # J
        self.theta = theta
        self.W = weight
        self.hidden = np.reshape(hidden, (-1,3))
        self.sigma = sigma

    def _calPhi(self, x_i, m_j, beta):
        temp = (x_i - m_j).dot(x_i - m_j)
        return math.exp(beta * (norm(x_i - m_j) ** 2))

    def _calAct(self, X):
        Phi = np.zeros((X.shape[0], self.hidden_nodes), dtype=np.float)
        for m_idx, m_num in enumerate(self.hidden):
            for x_idx, x_num in enumerate(X):
                Phi[x_idx, m_idx] = self._calPhi(x_num, m_num, -1/2*(self.sigma[m_idx]**2))
        return Phi

    def predict(self, X):
        X = np.reshape(X,(1,-1))
        Phi = self._calAct(X)
        # print('Phi ={}'.format(Phi)) 
        Y = np.dot(Phi, self.W) +self.theta
        return Y

class PSO():

    def __init__(self, X, y, J, iteration, SWARM_SIZE, P1, P2):
        self.iter = iteration
        self.X = X
        self.y = y
        self.J = J
        self.SWARM_SIZE = SWARM_SIZE
        self.self_weight = 0.5
        self.exp_weight = P1
        self.neighbor_weight = P2
        self.particles = []
        for _ in range(self.SWARM_SIZE):
            self.particles.append(Particle(J, 3, [35,0]))

    def evolve(self, X, y):
        for epoch in range(self.iter):
            average_error = 0
            worst_error = math.inf
            better_particle_in_pocket = False
            for idx in range(self.SWARM_SIZE):
                self.particles[idx].fitness = self.adaptation_funct(X,y,idx)
                average_error += self.particles[idx].fitness
                if self.particles[idx].fitness < worst_error:
                    worst_error = self.particles[idx].fitness
                if epoch == 0:
                    self.particles[idx].p_fitness = deepcopy(self.particles[idx].fitness)
                else:
                    if self.particles[idx].p_fitness < self.particles[idx].fitness:
                        self.particles[idx].update_p()
            average_error = average_error/self.SWARM_SIZE
            
            if epoch == 0:
                self.pocket_particle = deepcopy(self.particles[0])
            self.bestneighbor = deepcopy(self.particles[0])
            # find the best neighbor
            for idx in range(self.SWARM_SIZE):
                if self.bestneighbor.fitness < self.particles[idx].fitness:
                    self.bestneighbor = deepcopy(self.particles[idx])
                    print('change best neighbor')
            if self.pocket_particle.fitness < self.bestneighbor.fitness:
                better_particle_in_pocket = True
                self.pocket_particle = deepcopy(self.bestneighbor)

             # update the v & x
            for p in self.particles:
                p.v_theta = (self.self_weight * p.v_theta) + self.exp_weight*random.uniform(0, 1)*(p.p_theta - p.theta) + self.neighbor_weight*random.uniform(0, 1)*(self.bestneighbor.theta - p.theta)
                p.v_weight = (self.self_weight * p.v_weight) + self.exp_weight*random.uniform(0, 1)*(p.p_weight - p.weight) + self.neighbor_weight*random.uniform(0, 1)*(self.bestneighbor.weight - p.weight)
                p.v_means = (self.self_weight * p.v_means) + self.exp_weight*random.uniform(0, 1)*(p.p_means - p.means) + self.neighbor_weight*random.uniform(0, 1)*(self.bestneighbor.means - p.means)
                p.v_sd = (self.self_weight * p.v_sd) + self.exp_weight*random.uniform(0, 1)*(p.p_sd - p.sd) + self.neighbor_weight*random.uniform(0, 1)*(self.bestneighbor.sd - p.sd)
                p.limit_v()
                p.update_location()
                p.limit_location_upbound()
            print("------------------------------------------------------")
            print("Iteration Times: {}".format(epoch+1))
            print("Average Error: {} \n  (Normalize: {})".format(1/average_error, 1/average_error/40))
            print("Worst Error: {} \n  (Normalize: {})".format(1/worst_error, 1/worst_error/40))
            print("Least Error: {} \n  (Normalize: {})".format(1/self.pocket_particle.fitness, 1/self.pocket_particle.fitness/40))
            if better_particle_in_pocket is True:
                print("Detail parameter:")
                print("Theta: {}".format(self.pocket_particle.theta))
                print("Means: {}".format(self.pocket_particle.means))
                print("Weight: {}".format(self.pocket_particle.weight))
                print("SD: {}".format(self.pocket_particle.sd))
        self.save(self.pocket_particle,'../weights/RBFN_dim{}.txt'.format(self.X.shape[1]))
        print("-------------PSO training finished---------------")
    
    def save(self, Particle,path='../weights/RBFN_params4d.txt'):
        with open(path, 'w') as f:
            np.savetxt(f, self.pocket_particle.theta)
            np.savetxt(f, self.pocket_particle.means)
            np.savetxt(f, self.pocket_particle.weight)
            np.savetxt(f, self.pocket_particle.sd)
        

    def adaptation_funct(self,X,y, index):
        e_n = 0
        # [y_n - F(x_n)]^2
        for idx, expected_output in enumerate(y):
            rbf = RBFN(self.J, self.particles[index].theta, self.particles[index].weight,
                        self.particles[index].means, self.particles[index].sd)
            rbfn_value = rbf.predict(X[idx])
            rbfn_value = max(-40,min(rbfn_value*40,40))
            e_n += abs(expected_output - rbfn_value)
        # 1/N*(E_n)
        e_n = e_n/len(y)
        return  1/e_n
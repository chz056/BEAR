import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import gym
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import inv


from typing import Optional
from gym import spaces
from gym.utils import seeding

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class BuildingEnvReal(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, Parameter):

        self.OutTemp = Parameter['OutTemp']
        self.length_of_weather=len(self.OutTemp)
        self.connectmap = Parameter['connectmap']
        self.RCtable = Parameter['RCtable']
        self.roomnum = Parameter['roomnum']
        self.weightCmap = Parameter['weightcmap']
        self.target = Parameter['target']
        self.gamma=Parameter['gamma'] #Discount factor for reward, change this one later
        self.ghi=Parameter['ghi']
        self.GroundTemp=Parameter['GroundTemp']
        self.Occupancy=Parameter['Occupancy']
        self.acmap=Parameter['ACmap']
        self.maxpower=Parameter['max_power']
        self.Occupower=0
 

        self.timestep = Parameter['time_resolution']
  
        self.Qlow = np.ones(self.roomnum, dtype=np.float32) * (-1.0)*self.acmap.astype(np.float32)+1e-12
        # print(self.Qlow)

        self.Qhigh = np.ones(self.roomnum, dtype=np.float32) * (1.0)*self.acmap.astype(np.float32)
        self.action_space = gym.spaces.Box(self.Qlow, self.Qhigh, dtype=np.float32)
        self.min_T = -40
        self.max_T = 40


        self.low = np.ones(self.roomnum+4, dtype=np.float32) * self.min_T
        self.high = np.ones(self.roomnum+4, dtype=np.float32) * self.max_T
        self.observation_space = gym.spaces.Box(self.low, self.high, dtype=np.float32)
        self.q_rate = Parameter['gamma'][0]*24
        self.error_rate = Parameter['gamma'][1]
        connect_indoor = self.connectmap[:, :-1]
        Amatrix = self.RCtable[:, :-1]

        diagvalue = (-self.RCtable) @ self.connectmap.T - np.array([self.weightCmap.T[1]]).T
        np.fill_diagonal(Amatrix, np.diag(diagvalue))
        Bmatrix = self.weightCmap.T

 
        Bmatrix[2] = self.connectmap[:, -1] * (self.RCtable[:, -1])

        Bmatrix = Bmatrix.T

        self.rewardsum = 0
        self.statelist = []
        self.actionlist=[]
        self.epochs = 0


        self.A_d = expm(Amatrix * self.timestep)
        self.B_d = inv(Amatrix) @ (self.A_d - np.eye(self.A_d.shape[0])) @ Bmatrix

        print(np.shape(self.A_d))
        print(np.shape(self.B_d))

    def step(self, action):
        # print(self.epochs)

        done = False 
        X = self.state[:self.roomnum].T
        Y = np.insert(np.append(action,self.ghi[self.epochs]), 0, self.OutTemp[self.epochs]).T
        Y = np.insert(Y, 0, self.GroundTemp[self.epochs]).T
        avg_temp=np.sum(self.state[:self.roomnum])/self.roomnum
        Meta = self.Occupancy[self.epochs]
        self.Occupower=6.461927+.946892*Meta+.0000255737*Meta**2+7.139322*avg_temp-.0627909*avg_temp*Meta+.0000589172*avg_temp*Meta**2-.19855*avg_temp**2+.000940018*avg_temp**2*Meta-.00000149532*avg_temp**2*Meta**2

        Y = np.insert(Y, 0, self.Occupower).T

        X_new = self.A_d @ X + self.B_d @ Y

        reward = 0
 
        error = X_new*self.acmap - self.target*self.acmap
        # if np.sum(abs(error))/len(error)<=0.5:
        #   error = error*0
        # self.error=error
        
        # if np.any( abs((self.error).reshape(1,6)) >=2 ):

          
        #   reward-=100
          

        reward -= LA.norm(action, 2) * self.q_rate + LA.norm(error, 2) * self.error_rate
        self.rewardsum += reward

        info = {}
        # np.array([Meta/1000])
        self.state = np.concatenate((X_new, self.OutTemp[self.epochs].reshape(-1,),self.ghi[self.epochs].reshape(-1),self.GroundTemp[self.epochs].reshape(-1),np.array([self.Occupower/1000])), axis=0)
        self.statelist.append(self.state)
        self.actionlist.append(action*self.maxpower)

        self.epochs += 1
        # if self.counter == 10000:
        #   done = True
   
        if self.epochs>=self.length_of_weather-1:
            done=True
            #self.epochs=0

        return self.state, reward, done, info

    def reset(self):
        # self.epochs = random.randrange(self.length_of_weather)
        self.epochs = 0
        self.statelist = []
        self.actionlist=[]
        self.state = np.ones(self.roomnum+4)*22
        # self.state = np.random.uniform(21,23, self.roomnum+4)
        avg_temp=np.sum(self.state[:self.roomnum])/self.roomnum
        Meta = self.Occupancy[self.epochs]
        self.Occupower=6.461927+.946892*Meta+.0000255737*Meta**2+7.139322*avg_temp-.0627909*avg_temp*Meta+.0000589172*avg_temp*Meta**2-.19855*avg_temp**2+.000940018*avg_temp**2*Meta-.00000149532*avg_temp**2*Meta**2
        self.state=np.concatenate((self.state[:self.roomnum], self.OutTemp[self.epochs].reshape(-1,),self.ghi[self.epochs].reshape(-1),self.GroundTemp[self.epochs].reshape(-1),np.array([self.Occupower/1000])), axis=0)
  
        self.flag=1
        self.rewardsum = 0
        print("Reset", self.state)
        
        return self.state

    def render(self, mode='human'):
        pass

    def close(self):
        pass

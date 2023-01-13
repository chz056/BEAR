# -*- coding: utf-8 -*-
"""build_env.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MXa9YP-VgLiqzoESnHeJIf785XFlP7Al
"""

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
from sklearn import linear_model
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
        self.nonlinear=Parameter['nonlinear']
        self.temp_range=Parameter['temp_range']
        self.spacetype=Parameter['spacetype']
        self.Occupower=0
        self.timestep = Parameter['time_resolution']
        self.datadriven=False

        self.Qlow = np.ones(self.roomnum, dtype=np.float32) * (-1.0)*self.acmap.astype(np.float32)+1e-12


        self.Qhigh = np.ones(self.roomnum, dtype=np.float32) * (1.0)*self.acmap.astype(np.float32)

        if self.spacetype == 'continuous':

          self.action_space = gym.spaces.Box(self.Qlow, self.Qhigh, dtype=np.float32)
        else:
          self.action_space = gym.spaces.Box(self.Qlow*100, self.Qhigh*100, dtype=np.int32)
        self.min_T = self.temp_range[0]
        self.max_T = self.temp_range[1]


        self.low = np.ones(self.roomnum+4, dtype=np.float32) * self.min_T
        self.high = np.ones(self.roomnum+4, dtype=np.float32) * self.max_T
        self.observation_space = gym.spaces.Box(self.low, self.high, dtype=np.float32)
        self.q_rate = Parameter['gamma'][0]*24
        self.error_rate = Parameter['gamma'][1]
        connect_indoor = self.connectmap[:, :-1]
        Amatrix = self.RCtable[:, :-1]

        diagvalue = (-self.RCtable) @ self.connectmap.T - np.array([self.weightCmap.T[1]]).T
        np.fill_diagonal(Amatrix, np.diag(diagvalue))


        Amatrix+=self.nonlinear*7.139322/self.roomnum


        Bmatrix = self.weightCmap.T


        Bmatrix[2] = self.connectmap[:, -1] * (self.RCtable[:, -1])

        Bmatrix = (Bmatrix.T)

        self.rewardsum = 0
        self.statelist = []
        self.actionlist=[]
        self.epochs = 0


        self.A_d = expm(Amatrix * self.timestep)
        self.B_d = inv(Amatrix) @ (self.A_d - np.eye(self.A_d.shape[0])) @ Bmatrix



    def step(self, action):

        if self.spacetype != 'continuous':
          action=action/100
        self.statelist.append(self.state)
        done = False
        X = self.state[:self.roomnum].T        
        Y = np.insert(np.append(action,self.ghi[self.epochs]), 0, self.OutTemp[self.epochs]).T
        Y = np.insert(Y, 0, self.GroundTemp[self.epochs]).T
        avg_temp=np.sum(self.state[:self.roomnum])/self.roomnum
        Meta = self.Occupancy[self.epochs]
        if self.datadriven==True:
          Y = np.insert(Y, 0, Meta).T
          Y = np.insert(Y, 0, Meta**2).T
          Y = np.insert(Y, 0, avg_temp).T
          Y = np.insert(Y, 0, avg_temp**2).T
        else:

          self.Occupower=6.461927+0.946892*Meta+0.0000255737*Meta**2 - 0.0627909*avg_temp*Meta+0.0000589172*avg_temp*Meta**2 - 0.19855*avg_temp**2+0.000940018*avg_temp**2*Meta - 0.00000149532*avg_temp**2*Meta**2

          Y = np.insert(Y, 0, self.Occupower).T

        X_new = self.A_d @ X + self.B_d @ Y


        reward = 0

        error = X_new*self.acmap - self.target*self.acmap


        reward -= LA.norm(action, 2) * self.q_rate + LA.norm(error, 2) * self.error_rate
        self.rewardsum += reward

        info = {}#[X_new,X,Y,self.A_d,self.B_d]

        self.state = np.concatenate((X_new, self.OutTemp[self.epochs].reshape(-1,),self.ghi[self.epochs].reshape(-1),self.GroundTemp[self.epochs].reshape(-1),np.array([self.Occupower/1000])), axis=0)
        # self.statelist.append(self.state)
        self.actionlist.append(action*self.maxpower)

        self.epochs += 1

        if self.epochs>=self.length_of_weather-1:
            done=True
            self.epochs=0
        

        return self.state, reward, done, info

    def reset(self,T_initial=0,Auto=True):

        self.epochs = 0
        self.statelist = []
        self.actionlist=[]
        if Auto==True:
          T_initial = self.target
          # T_initial =np.array([18.24489859, 18.58710076, 18.47719682, 19.11476084, 19.59438163,15.39221207])
        # T_initial = np.random.uniform(21,23, self.roomnum+4)
        avg_temp=np.sum(T_initial)/self.roomnum
        Meta = self.Occupancy[self.epochs]
        self.Occupower=6.461927+0.946892*Meta+0.0000255737*Meta**2- 0.0627909*avg_temp*Meta+0.0000589172*avg_temp*Meta**2 - 0.19855*avg_temp**2+0.000940018*avg_temp**2*Meta - 0.00000149532*avg_temp**2*Meta**2
        self.state=np.concatenate((T_initial, self.OutTemp[self.epochs].reshape(-1,),self.ghi[self.epochs].reshape(-1),self.GroundTemp[self.epochs].reshape(-1),np.array([self.Occupower/1000])), axis=0)

        self.flag=1
        self.rewardsum = 0
        print("Reset", self.state)

        return self.state
    def train(self,states,actions):
        current_state=[]
        next_state=[]
        for i in range(len(states)-1):
          X=states[i]
          Y = np.insert(np.append(actions[i]/self.maxpower,self.ghi[i]), 0, self.OutTemp[i]).T
          Y = np.insert(Y, 0, self.GroundTemp[i]).T
          avg_temp=np.sum(X)/self.roomnum
          Meta = self.Occupancy[i]
          self.Occupower=6.461927+0.946892*Meta+0.0000255737*Meta**2 - 0.0627909*avg_temp*Meta+0.0000589172*avg_temp*Meta**2 - 0.19855*avg_temp**2+0.000940018*avg_temp**2*Meta - 0.00000149532*avg_temp**2*Meta**2
          Y = np.insert(Y, 0, Meta).T
          Y = np.insert(Y, 0, Meta**2).T
          Y = np.insert(Y, 0, avg_temp).T
          Y = np.insert(Y, 0, avg_temp**2).T
          stackxy=np.concatenate((X,Y), axis=0)
          current_state.append(stackxy)
          next_state.append(states[i+1])
        model = linear_model.LinearRegression(fit_intercept=False,positive=True) 
        modelfit = model.fit(np.array(current_state),np.array(next_state))
        beta = modelfit.coef_
        self.A_d=beta[:,:self.roomnum]
        self.B_d=beta[:,self.roomnum:]
        self.datadriven=True
        # return current_state,next_state

      
    # def step_d(self, action):

    #     if self.spacetype != 'continuous':
    #       action=action/100
    #     self.statelist.append(self.state)
    #     done = False
    #     X = self.state[:self.roomnum].T
    #     Y = np.insert(np.append(action,self.ghi[self.epochs]), 0, self.OutTemp[self.epochs]).T
    #     Y = np.insert(Y, 0, self.GroundTemp[self.epochs]).T
    #     avg_temp=np.sum(self.state[:self.roomnum])/self.roomnum
    #     Meta = self.Occupancy[self.epochs]
    #     Y = np.insert(Y, 0, Meta).T
    #     Y = np.insert(Y, 0, Meta**2).T
    #     Y = np.insert(Y, 0, avg_temp).T
    #     Y = np.insert(Y, 0, avg_temp**2).T

    #     X_new = self.A_d @ X + self.B_d @ Y


    #     reward = 0

    #     error = X_new*self.acmap - self.target*self.acmap


    #     reward -= LA.norm(action, 2) * self.q_rate + LA.norm(error, 2) * self.error_rate
    #     self.rewardsum += reward

    #     info = {}#[X_new,X,Y,self.A_d,self.B_d]

    #     self.state = np.concatenate((X_new, self.OutTemp[self.epochs].reshape(-1,),self.ghi[self.epochs].reshape(-1),self.GroundTemp[self.epochs].reshape(-1),np.array([self.Occupower/1000])), axis=0)
    #     # self.statelist.append(self.state)
    #     self.actionlist.append(action*self.maxpower)

    #     self.epochs += 1

    #     if self.epochs>=self.length_of_weather-1:
    #         done=True
    #         self.epochs=0
        

    #     return self.state, reward, done, info


    def render(self, mode='human'):
        pass

    def close(self):
        pass

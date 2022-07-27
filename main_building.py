from bldg_utils import Getroominfor, ParameterGenerator, Nfind_neighbor
import numpy as np
from building_HVAC import BuildingEnvReal
from MPC_Controller import MPCAgent

import pvlib
import pandas as pd

from scipy import interpolate

import os
import time
from collections import deque

import gym
import torch
import torch.optim as optim
from stable_baselines3 import PPO, SAC, A2C, DDPG
from stable_baselines3.common.logger import configure

# from stable_baselines.bench import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
#In the table, search 'Construction CTF' ThermalConductance or 'HeatTransfer Surface' U nofilm value

#Building=['__________filename__________', [intwall,floor,outwall,roof,ceiling,groundfloor]].
Buildinglist=[['ASHRAE901_ApartmentHighRise_STD2019_Tucson.table.htm',[6.299,3.285,0.384,0.228,3.839,0.287]]
,['ASHRAE901_ApartmentMidRise_STD2019_Tucson.table.htm',[6.299,3.285,0.384,0.228,3.839,0.287]]
,['ASHRAE901_Hospital_STD2019_Tucson.table.htm',[6.299,3.839,0.984,0.228,3.839,3.285]]
,['ASHRAE901_HotelLarge_STD2019_Tucson.table.htm',[6.299,0.228,0.984,0.228,0.228,2.705]]
,['ASHRAE901_HotelSmall_STD2019_Tucson.table.htm',[6.299,3.839,0.514,0.228,3.839,0.1573]]
,['ASHRAE901_OfficeLarge_STD2019_Tucson.table.htm',[6.299,3.839,0.984,0.228,4.488,3.839]]
,['ASHRAE901_OfficeMedium_STD2019_Tucson.table.htm',[6.299,3.839,0.514,0.228,4.488,0.319]]
,['ASHRAE901_OfficeSmall_STD2019_Tucson.table.htm',[6.299,3.839,0.514,0.228,4.488,0.319]]
,['ASHRAE901_OutPatientHealthCare_STD2019_Tucson.table.htm',[6.299,3.839,0.514,0.228,3.839,0.5650E-02]]
,['ASHRAE901_RestaurantFastFood_STD2019_Tucson.table.htm',[6.299,0.158,0.547,4.706,0.158,0.350]]
,['ASHRAE901_RestaurantSitDown_STD2019_Tucson.table.htm',[6.299,0.158,0.514,4.706,0.158,0.194]]
,['ASHRAE901_RetailStandalone_STD2019_Tucson.table.htm',[6.299,0.047,0.984,0.228,0.228,0.047]]
,['ASHRAE901_RetailStripmall_STD2019_Tucson.table.htm',[6.299,0.1125,0.514,0.228,0.228,0.1125]]
,['ASHRAE901_SchoolPrimary_STD2019_Tucson.table.htm',[6.299,0.144,0.514,0.228,0.228,0.144]]
,['ASHRAE901_SchoolSecondary_STD2019_Tucson.table.htm',[6.299,3.839,0.514,0.228,3.839,0.144]]]

#12 month GroundTemperature dictionary.
GroundTemp_dic={'Albuquerque':[13.7,7.0,2.1,2.6,4.3,8.8,13.9,17.8,23.2,25.6,24.1,20.5],
        'Atlanta':[16.0,11.9,7.7,4.0,7.9,13.8,17.2,20.8,24.8,26.1,26.5,22.5],
        'Buffalo':[9.7,6.0,-2.2,-3.4,-4.2,2.7,7.5,13.7,18.6,22.0,20.7,16.5],
        'Denver':[7.1,3.0,-1.0,0.8,-0.2,4.8,6.1,13.7,22.2,22.7,21.7,18.5],
        'Dubai':[29.5,25.5,21.1,19.2,20.8,23.1,26.5,31.4,33.0,35.1,35.3,32.5],
        'ElPaso':[18.3,11.2,6.8,8.1,10.3,12.5,19.2,23.8,27.9,27.5,26.3,23.4],
        'Fairbanks':[-3.1,-17.7,-19.3,-17.6,-15.4,-10.3,0.7,10.6,16.0,16.9,14.2,6.7],
        'GreatFalls':[8.6,2.8,-4.1,-8.8,-2.2,0.3,6.7,10.1,16.5,20.6,19.2,14.7],
        'HoChiMinh':[26.9,26.7,26.0,26.4,27.5,28.3,29.2,29.0,28.9,27.2,27.5,27.6],
        'Honolulu':[26.2,24.8,23.7,22.5,22.8,23.2,23.8,25.2,25.9,26.9,27.1,26.9],
        'InternationalFalls':[5.4,-2.0,-14.6,-16.9,-11.5,-6.2,4.0,13.4,18.0,19.7,17.9,12.3],
        'NewDelhi':[25.1,19.6,14.5,13.4,17.0,22.4,29.1,33.0,33.6,31.7,30.0,28.7],
        'NewYork':[14.0,7.3,3.3,1.2,-0.2,5.6,10.9,16.1,21.7,25.0,24.8,19.9],
        'PortAngeles':[9.3,6.7,4.1,4.2,4.2,5.9,9.0,10.0,13.3,15.0,15.7,13.4],
        'Rochester':[7.4,-0.0,-7.6,-12.6,-7.7,0.3,7.0,14.2,19.2,20.9,20.0,15.4],
        'SanDiego':[18.8,14.3,13.6,13.2,13.3,12.6,15.3,15.6,17.7,19.4,19.7,18.5],
        'Seattle':[11.4,8.1,5.4,4.5,5.8,8.3,10.9,13.0,15.6,17.7,18.8,15.1],
        'Tampa':[24.2,18.9,15.7,13.6,15.5,17.1,21.2,26.9,27.6,27.9,27.4,26.2],
        'Tucson':[20.9,15.4,11.9,14.8,12.7,15.4,23.3,26.3,31.2,30.4,29.8,27.8]}
###############Example:OfficeSmall at Tucson#####################
time_reso=3600 #1 hour
city=GroundTemp_dic['Tucson']
groundtemp=np.concatenate([np.ones(31*24*3600//time_reso)*city[0],np.ones(28*24*3600//time_reso)*city[1],
              np.ones(31*24*3600//time_reso)*city[2],np.ones(30*24*3600//time_reso)*city[3],
              np.ones(31*24*3600//time_reso)*city[4],np.ones(30*24*3600//time_reso)*city[5]
              ,np.ones(31*24*3600//time_reso)*city[6],np.ones(31*24*3600//time_reso)*city[7],
              np.ones(30*24*3600//time_reso)*city[8],np.ones(31*24*3600//time_reso)*city[9],
              np.ones(30*24*3600//time_reso)*city[10],np.ones(31*24*3600//time_reso)*city[11]])
filename=Buildinglist[7][0]
weatherfile=['USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw',groundtemp] 
U_Wall=Buildinglist[7][1]
Target_setpoints=np.array([20,18,23,21,22,24])

Parameter=ParameterGenerator(filename,weatherfile,U_Wall,target=Target_setpoints,time_reso=time_reso) #Description of ParameterGenerator in bldg_utils.py
env = BuildingEnvReal(Parameter)
obs_dim = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(obs_dim))
action_dim = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(action_dim))
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]
print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

"""
MPC：
"""
model = MPCAgent(env,
                 gamma=env.gamma,
                 safety_margin=0.96, planning_steps=10)

"""
RL：
"""
# time_steps=1000000

# seed=25
# tmp_path = "log/"
# callback = SaveOnBestTrainingRewardCallback(check_freq=10, log_dir=tmp_path)
# obs = env.reset()
# #model = SAC("MlpPolicy", env, learning_rate=0.00001, verbose=1)
# #model.learn(total_timesteps=time_steps, callback=callback, eval_freq=10)

# set_random_seed(seed=seed)
# model = PPO("MlpPolicy", env, verbose=1)
# #model.set_logger(new_logger)
# rewardlist=[]
# for i in range(time_steps//1000):
  
#   model.learn(total_timesteps=1000, eval_freq=1000)
#   action, _states = model.predict(obs)
#   obs, rewards, dones, info = env.step(action)
#   print(rewards)
#   rewardlist.append(rewards)
# plt.plot(rewardlist)
# plt.title('Reward during traning(PPO)')
# plt.xlabel('iteration')
# plt.ylabel('lost')
# plt.show()

# print("################TRAINING is Done############")
# #model.save("Models/ANM_39_PPO_2")
# #model = PPO.load("Models/ANM_6.zip")
obs = env.resetest()
print("Initial observation", obs)
for i in range(24*(3)):
    action, _states = model.predict(env)#MPC
    #action, _states = model.predict(obs)#RL
    obs, rewards, dones, info = env.step(action)
    if i%24==0:
      print("Rewards", rewards)

"""
Plot
"""
plt.plot(np.array(env.statelist)[:,:-3])
plt.title('Tucson Office_s Jan')
plt.xlabel('hours')
plt.ylabel('Temperature(degree)')
# plt.legend(['South','East','North','West','Core','Plenum','Out'],loc='lower right',fontsize='x-small')
plt.show()
plt.plot(np.sum(np.abs(np.array(env.actionlist))*400*20,1))
# plt.plot(env.actionlist)
plt.xlabel('hours')
plt.ylabel('Power(Watts)')
plt.show()

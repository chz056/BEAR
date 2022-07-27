from building_HVAC import BuildingEnvReal
from MPC_Controller import MPCAgent
from bldg_utils import Getroominfor,checkconnect,checkconnect_layer,Nfind_neighbor,ParameterGenerator

import pvlib
import pandas as pd

from scipy import interpolate
import datetime
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

numofhours=24*(4)
time_reso=3600 #1 hour
city=np.concatenate([np.ones(24*(31+28+31+30))*20.4,np.ones(24*(31))*21.5,np.ones(24*(30))*22.7,
                                        np.ones(24*(31))*22.9,np.ones(61*24)*23,np.ones(31*24)*21.9,
                                        np.ones(30*24)*20.7,np.ones(31*24+1)*20.5]) #np.ones(100000000)*22.7
groundtemp=np.concatenate([np.ones(31*24*3600//time_reso)*city[0],np.ones(28*24*3600//time_reso)*city[1],
              np.ones(31*24*3600//time_reso)*city[2],np.ones(30*24*3600//time_reso)*city[3],
              np.ones(31*24*3600//time_reso)*city[4],np.ones(30*24*3600//time_reso)*city[5]
              ,np.ones(31*24*3600//time_reso)*city[6],np.ones(31*24*3600//time_reso)*city[7],
              np.ones(30*24*3600//time_reso)*city[8],np.ones(31*24*3600//time_reso)*city[9],
              np.ones(30*24*3600//time_reso)*city[10],np.ones(31*24*3600//time_reso)*city[11]])
filename='Exercise2A-mytestTable.html'
weatherfile=['USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',groundtemp] 
U_Wall=[2.811,12.894,0.408,0.282,1.533,12.894,1.493]
Target_setpoints=np.array([22,22,22,22,22,22])
Parameter=ParameterGenerator(filename,weatherfile,U_Wall,target=Target_setpoints,time_reso=time_reso,shgc=0.568,AC_map=np.array([1,1,1,1,1,0])
                            ,shgc_weight=0.05,ground_weight=0.5,full_occ=np.array([1,2,3,4,5,0]),
                            reward_gamma=[0.5,0.5],activity_sch=np.ones(100000000)*1*117.24) #Description of ParameterGenerator in bldg_utils.py
env = BuildingEnvReal(Parameter)
obs_dim = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(obs_dim))
action_dim = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(action_dim))
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]
print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))
agent = MPCAgent(env,
                gamma=env.gamma,
                safety_margin=0.96, planning_steps=10)
obs = env.reset()
# env.OutTemp=outtempdatanew[:1000]
sum=0
schedule=np.concatenate([np.ones(7)*0,np.ones(7)*1,np.ones(10)*0])
for i in range(numofhours):
    a,s = agent.predict(env)
    if schedule[i%24]==0:
      a=a*0
    # print('lala',np.sum(a)*400)
    obs, r, done, _ = env.step(a)
    sum+=r

MPCstate=env.statelist
MPCaction=env.actionlist

all=pd.read_csv('Exercise2A-mytest.csv') 
pltemp=all['PLENUM:Zone Air Temperature [C](Hourly)']
southheat = all['SOUTH PERIMETER:Zone Air System Sensible Heating Rate [W](Hourly)']
southcool=all['SOUTH PERIMETER:Zone Air System Sensible Cooling Rate [W](Hourly)']
southtemp=all['SOUTH PERIMETER:Zone Air Temperature [C](Hourly)']
eastheat = all['EAST PERIMETER:Zone Air System Sensible Heating Rate [W](Hourly)']
eastcool=all['EAST PERIMETER:Zone Air System Sensible Cooling Rate [W](Hourly)']
easttemp=all['EAST PERIMETER:Zone Air Temperature [C](Hourly)']
northheat = all['NORTH PERIMETER:Zone Air System Sensible Heating Rate [W](Hourly)']
northcool=all['NORTH PERIMETER:Zone Air System Sensible Cooling Rate [W](Hourly)']
northtemp=all['NORTH PERIMETER:Zone Air Temperature [C](Hourly)']
westheat = all['WEST PERIMETER:Zone Air System Sensible Heating Rate [W](Hourly)']
westcool=all['WEST PERIMETER:Zone Air System Sensible Cooling Rate [W](Hourly)']
westtemp=all['WEST PERIMETER:Zone Air Temperature [C](Hourly)']
coreheat = all['CORE:Zone Air System Sensible Heating Rate [W](Hourly)']
corecool=all['CORE:Zone Air System Sensible Cooling Rate [W](Hourly)']
coretemp=all['CORE:Zone Air Temperature [C](Hourly)']
whole=all['Whole Building:Facility Total HVAC Electricity Demand Rate [W](Hourly)']
outtemp=all['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)']

EP_totalpower=southheat[:numofhours]+eastheat[:numofhours]+northheat[:numofhours]+westheat[:numofhours]+coreheat[:numofhours]+southcool[:numofhours]+eastcool[:numofhours]+northcool[:numofhours]+westcool[:numofhours]+corecool[:numofhours]
EP_state=[southtemp[:numofhours],easttemp[:numofhours],northtemp[:numofhours],westtemp[:numofhours],coretemp[:numofhours],pltemp[:numofhours]]

dates = [datetime.datetime(2018, 1, 1) + datetime.timedelta(hours=k * 1)
        for k in range(24*4)]
fig, ax1 = plt.subplots() 

ax1.plot(dates,EP_state[1])
ax1.plot(dates,np.array(MPCstate)[:,1],color = 'blue')

ax1.hlines(y=22, xmin=dates[0], xmax=dates[-1], colors='red', linestyles='--', lw=0.5, label='Multiple Lines')
ax1.set_title('4 days Temperature')
ax1.set_xlabel('hours')
ax1.set_ylabel('Temperature(($^\circ$C))')
ax1.legend(['EastZone-Energyplus','EastZone-MPC','Setpoint'],loc='lower right')
ax1.set_xlim([dates[0], dates[-1]])
plt.xticks(rotation=70)
data2=np.concatenate([schedule,schedule,schedule,schedule])
ax2 = ax1.twinx() 
ax2.plot(dates, data2,'--', color = 'black')  
ax2.set_ylim([-0.01, 1.01])
ax2.legend(['HVAC on/off'])
plt.ylabel('HVAC on/off') 
plt.show()

EP_totalpower=southheat[:numofhours]+eastheat[:numofhours]+northheat[:numofhours]+westheat[:numofhours]+coreheat[:numofhours]+southcool[:numofhours]+eastcool[:numofhours]+northcool[:numofhours]+westcool[:numofhours]+corecool[:numofhours]
MPC_totalpower=np.sum(np.abs(np.array(MPCaction))*400*20,1)
plt.plot(dates,EP_totalpower)
plt.plot(dates,MPC_totalpower,color = 'blue')


plt.xticks(rotation=70)
plt.title('Total Power Demand')
plt.xlabel('Time(hours)')
plt.ylabel('Power(Watts)')
plt.legend(['Energyplus','MPC'],loc='lower right',fontsize='x-small')
plt.xlim([dates[0], dates[-1]])
plt.show()

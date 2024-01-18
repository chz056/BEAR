from BEAR.Env.env_building  import BuildingEnvReal
from BEAR.Controller.MPC_Controller import MPCAgent
from BEAR.Utils.utils_building import ParameterGenerator, get_user_input
import numpy as np
import functools
import datetime
import os
import time
from collections import deque
import matplotlib.pyplot as plt

def main():

    numofhours=24
    # Create environment
    Parameter = ParameterGenerator('OfficeSmall', 'Hot_Dry', 'Tucson')
    env = BuildingEnvReal(Parameter)

    #Initialize
    env.reset()
    for i in range(numofhours):
        a = env.action_space.sample() # Randomly select an action
        obs, r, terminated, truncated, _ = env.step(a) # Return observation and reward
    RandomController_state = env.statelist # Collect the state list
    RandomController_action = env.actionlist # Collect the action list

    obs_dim = env.observation_space.shape[0]
    print("Size of State Space ->  {}".format(obs_dim))
    action_dim = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(action_dim))
    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]
    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))
    print('Sample State :', RandomController_state[0])
    print('Sample Action :', RandomController_action[0])

if __name__ == "__main__":
    main()

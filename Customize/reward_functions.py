import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import numpy as np
from numpy import linalg as LA
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
from scipy.linalg import expm, sinm, cosm
from numpy.linalg import inv


def my_custom_reward_function(self, state, action, error, state_new):
    # This is your default reward function
    # Initialize the reward
    reward = 0
    self.co2_rate=0.01
    self.temp_rate=0.01

    # Desired temperature range
    lower_temp = 18
    upper_temp = 22

    # Calculate the contribution of action to the reward
    action_contribution = LA.norm(action, 2) * self.q_rate
    reward -= action_contribution

    # Calculate the contribution of error to the reward
    error_contribution = LA.norm(error, 2) * self.error_rate
    reward -= error_contribution

    # Calculate the contribution of temperature deviation to the reward
    temp_deviation = np.sum(np.maximum(0, state_new - upper_temp) + np.maximum(0, lower_temp - state_new)) * self.temp_rate
    reward -= temp_deviation

    # Calculate the contribution of CO2 emissions to the reward
    co2_emission = LA.norm(action, 2) * self.co2_rate
    reward -= co2_emission

    self._reward_breakdown['action_contribution'] -= action_contribution
    self._reward_breakdown['error_contribution'] -= error_contribution
    self._reward_breakdown['temp_deviation'] -= temp_deviation
    self._reward_breakdown['co2_emission'] -= co2_emission
    return reward
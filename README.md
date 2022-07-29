# BEAR
**Physics-Principled Building Environment for Control and Reinforcement Learning**

Thank you for choosing BEAR.

BEAR is a physics based Building Environment for Control and Reinforcement Learning. The platform allows researchers to benchmark both model-based and model-free controllers using a broad collection of standard building models in Python without co-simulation with other building simulators. For implementation convenience, only three elements (building type, weather type, and city) need to be selected by user to create an environment.

The main functionalities of **BEAR** are the following :

BEAR enables an automated pipeline to process building geometry, weather and occupancy information to a composable RL environment. Compared to the actual building model, our model makes several simplifications regarding the zone shape, the window/door open schedules, and the shadowing function. Detailed model assumptions are listed below:

  - **Rectangular thermal mass**. The model is built using  zone data of maximum and minimum length, width, and height values, meaning an assumption of all thermal mass being rectangular shape.
  - **Sensible heat gained from activities**. The sensible heat gained from human activity schedule $Q_p$ is calculated through an approximated function in EnergyPlus:  
\begin{equation}
    \begin{aligned}
    Q_p= & c_1+c_2m+c_3m^2+c_4\bar{T}-c_5\bar{T}m+c_6\bar{T}m^2\\
    &-c_7\bar{T}^2+c_8\bar{T}^2m-c_9\bar{T}^2m^2,
    \end{aligned}
\end{equation}
where $m$ is the metabolic rate, $\bar{T} = \frac{1}{M} (T_1 + T_2 + ... + T_M)$ is the average zone temperature, and $c_1, ..., c_9$ are constants generated by fitting sensible heat data under varying conditions.
   Our model use mean temperature instead of individual zone temperature for occupancy heat gain calculation. 
  - **Light and electric equipment**. Heat generated by lighting and electric equipment are ignored in our model. Also, shadowing function is not implemented in our model, which could have a noticeable impact on temperature for certain locations during the summer. 
  -  **Windows and doors**: Windows and doors opening schedules could result in sudden temperature changes between zones. Specific design of the building geometry could also have influence on the thermal environment. These factors are not included in our model.


## Installation

### Docker container

### Manual installation

## Check Installation



## Usage example





## Citing BEAR


```

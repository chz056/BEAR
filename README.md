# BEAR
*Physics-Principled Building Environment for Control and Reinforcement Learning*

Authors: Chi Zhang, Yuanyuan Shi and Yize Chen

University of California San Diego, Hong Kong University of Science and Technology

Thank you for choosing BEAR.

BEAR is a physics based Building Environment for Control and Reinforcement Learning. The platform allows researchers to benchmark both model-based and model-free controllers using a broad collection of standard building models in Python without co-simulation with other building simulators. For implementation convenience, only three elements (building type, weather type, and city) need to be selected by user to create an environment.
## Usage example
Two examples "QuickStart" and "CustomizeModel_example" are provided for demonstration in Google Colab. Try it out [**HERE**.](https://drive.google.com/drive/folders/1-pFR1-RfhM8UiN2fmBra883NlP1RF1Qj?usp=sharing)

*Quick reminder: Download the folder "BEAR" and upload it to your Drive in "My Drive", then run the cells.*
## Functionalities
The main functionalities of **BEAR** are the following :
  - **Create environment**
  - **Building model**
  - **RL testbed**
### Environment Variables
BEAR supports a set of user-defined variables as inputs and provides an OpenAI Gym interface. Here is a table of all settings that users could modify:
<div align="center">
  <img src="Images/variable.PNG" width=80%><br><br>
</div>

### Buildings
For creating a large variety of building models, BEAR includes 16 different types of building at 19 locations. Here are lists of available buildings, weathers, and locations:
<div align="center">
  <img src="Images/Inputs.PNG" width=80%><br><br>
</div>

### RL algorithms
Researchers from the machine learning and reinforcement learning community can design new environments and algorithms with minimal knowledge of the underlying dynamics and models and thus can focus more on algorithm development and evaluation. BEAR provides an OpenAI Gym interface. Users can perform simulations in the customized environment with any classic model-based control or learning-based controllers. Examples are shown in the google colab notebook.
 
### Simplifications
BEAR enables an automated pipeline to process building geometry, weather and occupancy information to a composable RL environment. Compared to the actual building model, our model makes several simplifications regarding the zone shape, the window/door open schedules, and the shadowing function. Detailed model assumptions are listed below:

  - **Rectangular thermal mass**. The model is built using  zone data of maximum and minimum length, width, and height values, meaning an assumption of all thermal mass being rectangular shape.
  - **Sensible heat gained from activities**. The sensible heat gained from human activity schedule $Q_p$ is calculated through an approximated function in EnergyPlus. Our model uses mean temperature instead of individual zone temperature for occupancy heat gain calculation:  
<div align="center">
  <img src="Images/eqn.PNG" width=80%><br><br>
</div>

<div align="center">
  <img src="Images/constants.PNG" width=30%><br><br>
</div>
       
  - **Light and electric equipment**. Heat generated by lighting and electric equipment are ignored in our model. Also, shadowing function is not implemented in our model, which could have a noticeable impact on temperature for certain locations during the summer. 
  -  **Windows and doors**: Windows and doors opening schedules could result in sudden temperature changes between zones. Specific design of the building geometry could also have influence on the thermal environment. These factors are not included in our model.












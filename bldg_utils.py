import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results


def Getroominfor(filename):
    '''
    This function get information from the html file and sort each zone by layer.
    zoneinfor:[Zone_name,Zaxis,Xmin,Xmax,Ymin,Ymax,Zmin,Zmax,ExteriorGrossArea,ExteriorWindowArea]
    input: html file
    output:
    layerAll--
    nxm zoneinfor list. n:zones number in this layer, m:layers number.
    roonum--int
    cordall-- n zoneinfor list. n:total zones number.
    '''
    htmllines = open(filename).readlines()
    count = 0
    printflag = False
    cordall = []
    cord = []
    for line in htmllines:
        count += 1

        if 'Zone Internal Gains Nominal' in str(line):
            printflag = False # turn off the printflag after the 'Zone info' chart
        if printflag:
            #Zone_name
            if (count - 35) % 32 == 0 and count != 3:
                linestr = str(line)
                cord.append(linestr[22:-6])
            #Zaxis
            if (count - 42) % 32 == 0 and count != 10:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))
            #Xmin
            if (count - 46) % 32 == 0 and count != 14:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))
            #Xmax
            if (count - 47) % 32 == 0 and count != 15:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))
            #Ymin
            if (count - 48) % 32 == 0 and count != 16:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))
            #Ymax
            if (count - 49) % 32 == 0 and count != 17:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))
            #Zmin
            if (count - 50) % 32 == 0 and count != 18:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))
            #Zmax
            if (count - 51) % 32 == 0 and count != 19:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))  
            #FloorArea
            if (count - 56) % 32 == 0 and count != 24:
                linestr = str(line)
                cord.append(float(linestr[22:-6])) 
            #ExteriorNetArea
            if (count - 58) % 32 == 0 and count != 26:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))
            #ExteriorWindowArea
            if (count - 59) % 32 == 0 and count != 27:
                linestr = str(line)
                cord.append(float(linestr[22:-6]))

                cordall.append(cord)
                cord = []
        if 'Zone Information' in str(line):
            # print(line)
            printflag = True
            count = 0

    roomnum = len(cordall)
    cordall.sort(key=lambda x: x[1])
    samelayer = cordall[0][1]
    roomlist = [cordall[0]]
    layerAll = []
    cordall[0].append(0)
    # Sort each zone by layer
    for i in range(1, len(cordall)):

        cordall[i].append(i)
        if cordall[i][1] != samelayer:
            layerAll.append(roomlist)
            roomlist = [cordall[i]]
            samelayer = cordall[i][1]
            if i == len(cordall) - 1:
                layerAll.append(roomlist)
        else:
            roomlist.append(cordall[i])
            if i == len(cordall) - 1:
                layerAll.append(roomlist)
    return layerAll, roomnum, cordall



def checkconnect(room1min,room1max,room2min,room2max):
  '''
  This function check whether zones in the same layer are connected.
  '''
  if (room1min[0]>=room2min[0] and room1min[0]<=room2max[0] \
    and room1min[1]>=room2min[1] and room1min[1]<=room2max[1])\
    or(room1max[0]>=room2min[0] and room1max[0]<=room2max[0] \
    and room1max[1]>=room2min[1] and room1max[1]<=room2max[1]):
    return True
  return False


def checkconnect_layer(room1min,room1max,room2min,room2max):
  '''
  This function check whether zones in different layers are connected.
  '''
  if (room1min[0]>=room2min[0] and room1min[0]<room2max[0] \
    and room1min[1]>=room2min[1] and room1min[1]<room2max[1])\
    or(room1max[0]>room2min[0] and room1max[0]<=room2max[0] \
    and room1max[1]>room2min[1] and room1max[1]<=room2max[1]):
    return True
  return False

def Nfind_neighbor(roomnum,Layerall,U_Wall,SpecificHeat_avg):
  '''
  This function is for the building model.
  Input: roomnumber,sorted layer list, U-factor for all walls, Specific heat.
  Output: map dictionary for neighbour,n+1 by n R table(n:roomnumber),
      n by 1 C table(n:roomnumber),n by 1 Window table(n:roomnumber).
  '''
  Rtable=np.zeros((roomnum,roomnum+1))
  Ctable=np.zeros(roomnum)
  Windowtable = np.zeros(roomnum)
  Walltype=U_Wall[0]
  Floor =U_Wall[1]
  OutWall = U_Wall[2]
  OutRoof = U_Wall[3]
  Ceiling = U_Wall[4]
  Window = U_Wall[6]
  Air = 1.225 #kg/m^3

  dicRoom={}
  outind=roomnum
  for k in range(len(Layerall)):
    Layer_num=len(Layerall)
    cordall = Layerall[k]
    FloorRoom_num = len(cordall)
    if k+1<Layer_num:
      nextcord=Layerall[k+1]
      for i in range(len(cordall)):
        for j in range(len(nextcord)):
          x1min=[float(cordall[i][2]),float(cordall[i][4])]
          x1max=[float(cordall[i][3]),float(cordall[i][5])]
          x2min=[float(nextcord[j][2]),float(nextcord[j][4])]
          x2max=[float(nextcord[j][3]),float(nextcord[j][5])]
          if checkconnect_layer(x2min,x2max,x1min,x1max)or checkconnect_layer(x1min,x1max,x2min,x2max):
            crossarea=(min(x1max[1],x2max[1])-max(x1min[1],x2min[1]))*(min(x1max[0],x2max[0])-max(x1min[0],x2min[0]))
            
            U = crossarea*(Floor*Ceiling/(Floor+Ceiling))
            # U = crossarea*((Ceiling))
            
            Rtable[nextcord[j][11],cordall[i][11]]=U
            Rtable[cordall[i][11],nextcord[j][11]]=U
            if cordall[i][0] in dicRoom:
              dicRoom[cordall[i][0]].append(nextcord[j][11])
            else:
              dicRoom[cordall[i][0]]=[nextcord[j][11]]
            if nextcord[j][0] in dicRoom:
              dicRoom[nextcord[j][0]].append(cordall[i][11])
            else:
              dicRoom[nextcord[j][0]]=[cordall[i][11]]


    for i in range(len(cordall)) :
      height = cordall[i][7]-cordall[i][6]
      xleng= (cordall[i][3]-cordall[i][2])
      yleng= cordall[i][5]-cordall[i][4]
      C_room=SpecificHeat_avg*height*xleng*yleng*Air
      Ctable[cordall[i][11]]= C_room
      Windowtable[cordall[i][11]]=cordall[i][10]
      if cordall[i][9]>0 or (i==len(cordall)-1):
        if i==len(cordall)-1:
          Rtable[cordall[i][11],-1]= cordall[i][9]*OutWall+xleng*yleng*OutRoof+cordall[i][10]*Window
   
          
        else:
          Rtable[cordall[i][11],-1]= cordall[i][9]*OutWall+cordall[i][10]*Window
        

        if cordall[i][0] in dicRoom:
          dicRoom[cordall[i][0]].append(outind)
        else:
          dicRoom[cordall[i][0]]=[outind]    
      for j in range(i+1,FloorRoom_num):
        x1min=[float(cordall[i][2]),float(cordall[i][4])]
        x1max=[float(cordall[i][3]),float(cordall[i][5])]
        x2min=[float(cordall[j][2]),float(cordall[j][4])]
        x2max=[float(cordall[j][3]),float(cordall[j][5])]
        if checkconnect(x2min,x2max,x1min,x1max)or checkconnect(x1min,x1max,x2min,x2max):
          length=np.sqrt((min(x1max[1],x2max[1])-max(x1min[1],x2min[1]))**2+(min(x1max[0],x2max[0])-max(x1min[0],x2min[0]))**2)
          U = height*length*Walltype
          Rtable[cordall[j][11],cordall[i][11]]=U
          Rtable[cordall[i][11],cordall[j][11]]=U
          if cordall[i][0] in dicRoom:
            dicRoom[cordall[i][0]].append(cordall[j][11])
          else:
            dicRoom[cordall[i][0]]=[cordall[j][11]]
          if cordall[j][0] in dicRoom:
            dicRoom[cordall[j][0]].append(cordall[i][11])
          else:
            dicRoom[cordall[j][0]]=[cordall[i][11]]  
  return dicRoom,Rtable,Ctable,Windowtable


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True
def ParameterGenerator(filename,weatherfile,U_Wall,
                       shgc=0.252,shgc_weight=0.1,ground_weight=0.5,full_occ=0,
                       max_power=8000,AC_map=1,time_reso=3600,reward_gamma=[0.001,0.999],
                       target=22,activity_sch=np.ones(100000000)*1*120):
    """
    This function could generate parameters from the selected building and temperature file for the env.
    Input:
      filename:str, htm file for building idf
      weatherfile:str, epw file
      U_Wall:list, U value of [intwall,floor,outwall,roof,ceiling]
      shgc:int, shgc value for window
      shgc_weight:int, extra loss of ghi addressed using this weight
      ground_weight:int, extra lost of heat from ground addressed using this weight
      full_occ:nparray with shape (roomnum,1), max number of people occupy a room
      max_power:int,maximum power of a single hvac output,unit watts
      AC_map:nparray∈[0,1] with shape(roomnum,),boolean of whether a zone has AC
      time_reso:int, determine the length of 1 timestep, unit second 
      reward_gamma:list of two, [energy penalty,temperature error penalty]
      target:nparray with shape (roomnum,), target temperature setpoints for each zone
      activity_sch:nparray with shape(length of the simulation,), the activity schedule of people in the building,unit watts/person

    """  
    Layerall, roomnum, buildall  = Getroominfor(filename)
    print("###############All Zones from Ground############")
    for building in buildall:
      print (building)
    print("###################################################")
    #intwall,floor,outwall,roof,ceiling,groundfloor,window.
    data=pvlib.iotools.read_epw(weatherfile[0])
    oneyear=data[0]['temp_air']
    num_datapoint=len(oneyear)
    x = np.arange(0, num_datapoint)
    y = np.array(oneyear)

    f = interpolate.interp1d(x, y)
    xnew = np.arange(0, num_datapoint-1, 1/3600*time_reso)
    outtempdatanew = f(xnew)   # use interpolation function returned by `interp1d`]

    oneyearrad=data[0]['ghi'] #[5088:]8-12yue#[2882:]5yue#[4344:]7yue
    x = np.arange(0, num_datapoint)
    y = np.array(oneyearrad)

    f = interpolate.interp1d(x, y)
    xnew = np.arange(0, num_datapoint-1, 1/3600*time_reso)
    solardatanew = f(xnew)

    Air=1.225 #kg/m^3
    SpecificHeat_avg = 1000
    SHGC=shgc*shgc_weight*(max(data[0]['ghi'])/(abs(data[1]['TZ'])/60))
    dicRoom,Rtable,Ctable,Windowtable=Nfind_neighbor(roomnum,Layerall,U_Wall,SpecificHeat_avg)
    connectmap=np.zeros((roomnum,roomnum+1))
    RCtable=Rtable/np.array([Ctable]).T
    ground_connectlist=np.zeros((roomnum,1)) #list to see which room connects to the ground
    groundrooms=Layerall[0] #the first layer connects to the ground
    for room in groundrooms:
      ground_connectlist[room[11]]=room[8]*U_Wall[5]*ground_weight #for those rooms, assign 1/R table by floor area and u factor

    for i in range(len(buildall)):
        connect_list = dicRoom[buildall[i][0]]

        for number in connect_list:
            connectmap[i][number] = 1

    people_full= np.zeros((roomnum,1))+np.array([full_occ]).T 
    ACweight=np.diag(np.zeros(roomnum)+AC_map)* max_power
    # ACweight[-1,-1]=0
    weightcmap = np.concatenate((people_full,ground_connectlist,np.zeros((roomnum, 1)), ACweight, np.array([Windowtable*SHGC]).T), axis=-1)/np.array([Ctable]).T



    Parameter = {}
    Parameter['OutTemp'] = outtempdatanew
    Parameter['connectmap'] = connectmap
    Parameter['RCtable'] = RCtable
    Parameter['roomnum'] = roomnum
    Parameter['weightcmap'] = weightcmap
    Parameter['target'] = np.zeros(roomnum) + target
    Parameter['gamma'] = reward_gamma
    Parameter['time_resolution']=time_reso 
    Parameter['ghi'] = solardatanew/(abs(data[1]['TZ'])/60)/ (max(data[0]['ghi'])/(abs(data[1]['TZ'])/60))
    Parameter['GroundTemp'] = weatherfile[1]
    Parameter['Occupancy'] = activity_sch/1000 #schedule(maxocc percent)*metobolic power
    Parameter['ACmap']= AC_map
    
    return Parameter

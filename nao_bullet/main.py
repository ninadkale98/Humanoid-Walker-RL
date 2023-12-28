from qibullet import SimulationManager
from qibullet import NaoVirtual , NaoFsr
import matplotlib.pyplot as plt
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import pickle

if torch.cuda.is_available():
  device = "cuda" 
else:
  device = "cpu"
print(device, " in use")

# ---------------------------------------------------------------------------------------------------------------
obs_space = 10

class ActorNet(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
      super(ActorNet, self).__init__()
      self.rnn = nn.RNN(input_dim, hidden_dim)
      #self.fc = nn.Linear(hidden_dim, output_dim)
      self.fc1 = nn.Linear(hidden_dim, hidden_dim)
      self.fc2 = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
      out, _ = self.rnn(x)
      #out = out[:, -1, :]
      #out = self.fc(out)
      out = self.fc1(out)
      out = self.fc2(out)
      return out
    
Actor = ActorNet(obs_space, 10, obs_space)

# load model weights
Actor.load_state_dict(torch.load('model_weights_day3_1.pth', map_location=torch.device('cpu')))


# ---------------------------------------------------------------------------------------------------------------

# Launch Simulation Environment
vis = True
simulation_manager = SimulationManager()
nao_sim = simulation_manager.launchSimulation(gui=True, auto_step=True)
nao_sim = simulation_manager.launchSimulation(gui=False, auto_step=True)
simulation_manager.setGravity(nao_sim, [0.0, 0.0, -9.81])
nao = None

# Utility Functions
# Joints to monitor
legJoints = [ "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
              "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll"]

min_values = torch.tensor([-0.37, -1.53, -0.09, -1.18, -0.39 , -0.79, -1.53, -0.10, -1.18, -0.76], dtype=torch.float32)
max_values = torch.tensor([ 0.79, 0.48, 2.11, 0.92, 0.76,  0.37, 0.48, 2.12, 0.93, 0.39], dtype=torch.float32)

nao_prev_position_X = 0
nao_prev_position_Y = 0

prev_weight_cntr = 0
def env_Reward():
    global prev_weight_cntr
    x, y, z = nao.getPosition()
    terminated = False
    delX = nao_prev_position_X - x 
    delY = nao_prev_position_Y - x 
    weight = -(nao.getTotalFsrValues(NaoFsr.LFOOT) + nao.getTotalFsrValues(NaoFsr.RFOOT) )

    # if robot fallen down
    if weight == 0: 
        reward = -1
        prev_weight_cntr += 1
    else:
        reward = delX
        prev_weight_cntr = 0

    if prev_weight_cntr == 5:
        terminated = True
        prev_weight_cntr = 0
    return reward, terminated

def env_state():
    return nao.getAnglesPosition(legJoints)
    
def env_action(angles, speed):
    nao.setAngles(legJoints, angles ,speed)

def env_itrm_step(itr):
    for _ in range(itr):
        simulation_manager.stepSimulation(nao_sim)

def env_reset():
    global nao_prev_position_X , nao_prev_position_Y
    global nao
    simulation_manager.resetSimulation(nao_sim)
    simulation_manager.setGravity(nao_sim, [0.0, 0.0, -9.81])
    nao = simulation_manager.spawnNao( 
    nao_sim,
    translation=[0, 0, 0],quaternion=[0, 0, 0, 1],
    spawn_ground_plane=True)
    env_itrm_step(100)
    nao_prev_position_X, nao_prev_position_Y, _ = nao.getPosition()


def env_stop():
    simulation_manager.stopSimulation(nao_sim)
    # wait for limited iterations 

def env_step(angles):
    env_action(angles, 0.1)
    #env_itrm_step(50)
    reward, terminated = env_Reward()
    next_state = env_state()
    return next_state, reward, terminated

env_reset()

# ---------------------------------------------------------------------------------------------------------------

for _ in range(10):
    state =  nao.getAnglesPosition(legJoints)
    total_reward = 0
    itr = 0
    while True:
        itr += 1
        x = torch.tensor(state).float()
        x = x.view(1,10)
        next_state = Actor(x)
        next_state = torch.clamp(next_state[0], min_values, max_values)   
        angles = next_state.tolist()
        #state[0] = (itr%3)/10  
        next_state, reward, terminated = env_step(angles)
        total_reward += reward
        state = next_state
        
        time.sleep(0.5)
        if terminated:
            env_reset()
            break


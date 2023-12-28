import numpy as np
import matplotlib.pyplot as plt
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data as data_utils
import pickle

if torch.cuda.is_available():
  device = "cuda" 
else:
  device = "cpu"
device_ = torch.device(device)
print(device, " in use")

torch.autograd.set_detect_anomaly(True)


obs_space = 10

class ActorNet(nn.Module):
    def __init__(self, state_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, state_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = 2*torch.tanh(self.fc3(x))
        return x
    
Actor = ActorNet(obs_space)

# load model weights
Actor.load_state_dict(torch.load('model_weights.pth'))

def give_nextstate(state):
   next_state = Actor(torch.tensor(state).float())
   return next_state.tolist()

# Sockets
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('127.0.0.1', 1112))
server_socket.listen()
print('Waiting for Connection ......')

conn, addr = server_socket.accept()
print('Connection Established ......')

while True:
  request = conn.recv(1024)
  if request:    
    e = pickle.loads(request)
    out = give_nextstate(pickle.loads(request))
    conn.sendall(pickle.dumps([out] , protocol = 2))

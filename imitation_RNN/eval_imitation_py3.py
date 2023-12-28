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
Actor.load_state_dict(torch.load('model_weights_day3_1.pth'))

def give_nextstate(state):
  x = torch.tensor(state).float()
  x = x.view(1,10)
  next_state = Actor(x)
  print(next_state[0])
  return next_state[0].tolist()

# Sockets
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('127.0.0.1', 1117))
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

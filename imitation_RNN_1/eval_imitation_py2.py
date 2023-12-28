from naoqi import ALProxy
import almath
import motion
import time
import signal
import matplotlib.pyplot as plt
import numpy as np
import socket
import pickle

# Nao configurations
#ip = "10.0.255.22"  # Physical
ip = "laptop-ti0kt4lp.local." # Simulations
port = 9559
fractionMaxSpeed = 0.2
run = True

# Proxies
motion_proxy = ALProxy("ALMotion" , ip, port)
init_proxy = ALProxy("ALRobotPosture" , ip, port)
memory_Proxy = ALProxy("ALMemory", ip, port)

# Enable Balance Constraints
# Set the balance constraint
motion_proxy.wbEnableBalanceConstraint(True, "Legs")

# Cntrl + C ISR
def handler(signum, frame):
    global run
    run = False
    print("Ending experiments")
signal.signal(signal.SIGINT, handler)

# Init Pose
def init_pose():
    print(" Nao Reset")
    init_proxy.goToPosture("StandInit", 0.5)
    

# Joints to monitor
legJoints = [ "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
              "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll"]

# socket connection with python3
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 1116))

init_pose()

print(" Start Walking using model after 5 sec")

time.sleep(5)

motion_proxy.setStiffnesses(["LLeg", "RLeg"], 1.0)


def nao_step(next_state):
    motion_proxy.angleInterpolationWithSpeed(legJoints, next_state, fractionMaxSpeed)  #setAngles

def nao_reset():
    init_pose()

def nao_get_action():
    joint_states = motion_proxy.getAngles(legJoints, True)
    
    client_socket.sendall(pickle.dumps(joint_states)) # send current state to model to get next state
    t = client_socket.recv(1024) #blocking call, wait till u recive next state
    next_state = pickle.loads(t) # convert recived buffer in list
    
    next_state= list(next_state[0])
    
    #next_state[4] = 0
    #next_state[9] = 0
    #next_state[0] = 0
    #next_state[5] = 0
    # next_state[2] *= 1.5
    # next_state[7] *= 1.5
    # next_state[1] *= 1.5
    # next_state[6] *= 1.5

    return next_state

while run:

    next_state = nao_get_action()
    nao_step(next_state)
    time.sleep(0.25)
    
# End sequence 
motion_proxy.stopMove()
motion_proxy.killAll()
nao_reset()

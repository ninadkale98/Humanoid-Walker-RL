from util import *
from naoqi import ALProxy
import almath
import motion
import time
import signal

# Experiment Configurations
bodyParts = ["Head"]

joints = ["LKneePitch"]
legJoints = ["LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll",
             "RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll"]

# Init
init( bodyParts)

update_time = 1

angles = 2
incr = -0.1
walk()

while run:
    angles += incr
    if angles < 0:
        incr = 0.0
        angles = 0.2
    elif angles > 2:
        incr = -0.1
        angles = 0.2
    state = get_state(legJoints)
    #for i in range(len(state)):
        #print(legJoints[i] , " :\t", state[i])
    fallen , position = get_reward()
    print( "Fallen \t" , fallen , " Position \t", position)
    
    #set_angle(joints, angles)
    
    time.sleep(update_time)

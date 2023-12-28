from naoqi import ALProxy
import almath
import motion
import time
import signal

# Nao configurations
ip = "10.0.255.22"  # laptop-ti0kt4lp.local. for simulations
port = 9559
fractionMaxSpeed = 0.1
run = True

# Proxies
motion_proxy = ALProxy("ALMotion" , ip, port)
init_proxy = ALProxy("ALRobotPosture" , ip, port)
memory_Proxy = ALProxy("ALMemory", ip, port)



# Cntrl + C ISR
def handler(signum, frame):
    global run
    run = False
    print("Ending experiments")
signal.signal(signal.SIGINT, handler)

# Init Pose
print(" Getting in init position, wait 5sec")
init_proxy.goToPosture("StandInit", 1.0)
motion_proxy.moveInit()
time.sleep(5)

# Start Walking
motion_proxy.move(0.3, 0.0, 0.0)

# Joints to monitor
legJoints = ["LHipYawPitch", "LHipRoll", "LHipPitch", "LKneePitch", "LAnklePitch", "LAnkleRoll"]
            # "RHipYawPitch", "RHipRoll", "RHipPitch", "RKneePitch", "RAnklePitch", "RAnkleRoll"]

while run:
    joint_states = motion_proxy.getAngles(legJoints, True)
    print(joint_states)
    time.sleep(0.2)

motion_proxy.stopMove()
motion_proxy.killAll()

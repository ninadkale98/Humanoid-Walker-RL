import signal
from naoqi import ALProxy

# Nao configurations
ip = "10.0.255.22"#"laptop-ti0kt4lp.local."
port = 9559
fractionMaxSpeed = 0.1

motion_proxy = ALProxy("ALMotion" , ip, port)
init_proxy = ALProxy("ALRobotPosture" , ip, port)
memory_Proxy = ALProxy("ALMemory", ip, port)

bodyParts_util = None
run = True

def setStiffness(joints , sig):
    if sig == 0:
        motion_proxy.setStiffnesses(bodyParts_util, 0.0)
    else:
        motion_proxy.setStiffnesses(bodyParts_util, 1)

def handler(signum, frame):
    global run
    run = False
    init_pose()
    #setStiffness(bodyParts_util, 0)
    print('You pressed Ctrl+C!')

def init_pose():
    init_proxy.goToPosture("StandInit", 1.0)

def init(bodyParts):
    global bodyParts_util
    signal.signal(signal.SIGINT, handler)
    bodyParts_util = bodyParts
    init_pose()
    motion_proxy.stopMove()
    motion_proxy.killAll()
    


def set_angle(joints, angles ):
    motion_proxy.setAngles(joints,angles,fractionMaxSpeed)


def get_state(legJoints):
    joint_states = motion_proxy.getAngles(legJoints, True)
    return joint_states

def get_reward():
    fallStatus = memory_Proxy.getData("robotHasFallen")
    position = motion_proxy.getRobotPosition(True)
    return fallStatus, position

def walk():

    x = 0.2  # forward speed in m/s
    y = 0.0  # lateral speed in m/s
    theta = 0.0  # rotational speed in rad/s
    motion_proxy.move(x, y, theta)

    # Wait for the robot to finish walking
    #motion_proxy.waitUntilMoveIsFinished()
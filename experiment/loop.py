from naoqi import ALProxy
import almath
import motion
import time
import signal

ip = "laptop-ti0kt4lp.local."   # "laptop-ti0kt4lp.local." #laptop-ti0kt4lp.local.:9559
port = 9559
fractionMaxSpeed = 0.1

motion_service = ALProxy("ALMotion" , ip, port)
joints = [""]

def setStiffness(joints , sig):
    if sig == 0:
        motion_service.setStiffnesses(joints, 0.0)
    else:
        motion_service.setStiffnesses(joints, 1)



def handler(signum, frame):
    print('You pressed Ctrl+C!')
    setStiffness(joints, )

motion_service.setStiffnesses("Head", 1.0)
names = "HeadYaw"

motion_service.setAngles(names,2,fractionMaxSpeed)
print("\t Getting in Intial Position")
time.sleep(15)
print("\t Starting loop")


angles = 2
incr = -0.3
while True:

    angles += incr
    if angles < -2:
        incr = 0.3
        angles = 0.2
    elif angles > 2:
        incr = -0.3
        angles = 0.2
    print(angles)
    motion_service.setAngles(names,angles,fractionMaxSpeed)

    time.sleep(1)


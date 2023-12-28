from naoqi import ALProxy
import almath
import motion
import time

#ip = "laptop-ti0kt4lp.local."
ip = "10.0.255.22"
port = 9559
fractionMaxSpeed = 0.1

motion_service = ALProxy("ALMotion" , ip, port)

motion_service.setStiffnesses("Head", 1.0)

names = "HeadPitch"

angles = 0.3
motion_service.setAngles(names,angles,fractionMaxSpeed)
# wait half a second
time.sleep(0.5)

# change target
angles = 0.0
motion_service.setAngles(names,angles,fractionMaxSpeed)
# wait half a second
time.sleep(0.5)

# change target
angles = 0.3
motion_service.setAngles(names,angles,fractionMaxSpeed)
time.sleep(3.0)

motion_service.setStiffnesses("Head", 0.0)


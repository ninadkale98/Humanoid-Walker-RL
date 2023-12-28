from naoqi import ALProxy
import almath
import motion

ip = "laptop-ti0kt4lp.local."
port = 9559


motion_service = ALProxy("ALMotion" , ip, port)
posture_service = ALProxy("ALRobotPosture", ip, port)

# Wake up robot
motion_service.wakeUp()

# Send robot to Stand Init
posture_service.goToPosture("StandInit", 0.5)

effector   = "LLeg"
frame      = motion.FRAME_TORSO
axisMask   = almath.AXIS_MASK_VEL # just control position
useSensorValues = False

path = []
currentTf = motion_service.getTransform(effector, frame, useSensorValues)
targetTf  = almath.Transform(currentTf)
targetTf.r1_c4 += 0.3 # x
targetTf.r2_c4 += 0.3 # y

path.append(list(targetTf.toVector()))
path.append(currentTf)

# Go to the target and back again
times      = [2.0, 4.0] # seconds

motion_service.transformInterpolations(effector, frame, path, axisMask, times)

# Go to rest position
motion_service.rest()

import time
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import cv2

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
#restposes for null space
jointPositions=[-0.02281602, -1.59401434, -0.13742989, -3.05013807, -0.28466714,
          3.10330309,  0.80803389, -0.00446232,  0.01322508]
rp = jointPositions

class PandaSim():
    def __init__(self, bullet_client, panda, timeStep):
        self.MAX_FORCES = 150
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.panda = panda
        self.timeStep = timeStep
        self.digit_links = [11,14]

        self.initial_joints = [-0.02281602, -1.59401434, -0.13742989, -3.05013807, -0.28466714,
                3.10330309,  0.80803389, -0.00446232,  0.01322508]
        # self.in_cabinet_mug = [-0.03786923956561822, 0.8463535653326023, 0.02608225909109106, -0.4700517116015577, -0.026823348393911886, 2.934614315072545, 0.6611812056642706, -0.00446232,  0.01322508]
        self.in_cabinet_mug = [-0.03786923956561822, 0.8463535653326023, 0.02608225909109106, -0.4700517116015577, -0.026823348393911886, 2.934614315072545, 0.8, -0.00446232,  0.01322508]
        # self.in_cabinet_bottle = [0.4099618646286729, 0.9361554354796208, -0.5709239438980049, -0.45152278146769625, -0.21685657955199333, 2.8152052777325705, 0.8, -0.00446232,  0.01322508]
        self.in_cabinet_bottle = [0.4099618646286729, 0.9361554354796208, -0.5709239438980049, -0.45152278146769625, -0.21685657955199333, 2.8152052777325705, 1.311145578262842, -0.00446232,  0.01322508]
        self.in_cabinet_bowl = self.in_cabinet_mug
        self.real = [-0.0807112151808015, 0.12689950624857624, 0.20026483563505107, -2.1055931307302225, 0.20446346430420115, 3.6963584305710264, 0.6289109912481573 -0.00446232,  0.01322508]
        
        # self.initial_joints = [-0.02281602, -1.59401434, -0.13742989, -3.05013807, -0.28466714,
        #         3.10330309,  0.80803389, 0.02,  0.01322508]

        # self.initial_joints = [0.98, 0.458, 0.31, -2.24, -0.15, 2.66, 2.32, 0.02, 0.02]

        # create a constraint to keep the fingers centered
        # c = self.bullet_client.createConstraint(self.panda,
        #                     9,
        #                     self.panda,
        #                     12,
        #                     jointType=self.bullet_client.JOINT_GEAR,
        #                     jointAxis=[0, 1, 0],
        #                     parentFramePosition=[0, 0, 0],
        #                     childFramePosition=[0, 0, 0])
        # self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.5, maxForce=50)
        self.reset()
    

    def reset(self):
        index = 0
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            #print("info=",info)
            # jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.panda, j, self.initial_joints[index]) 
                index=index+1
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.panda, j, self.initial_joints[index]) 

    def set_in_cabinet(self, obj):
        if "mug" in obj:
            in_cabinet = self.in_cabinet_mug
        elif "bottle" in obj:
            in_cabinet = self.in_cabinet_bottle
        elif "bowl" in obj:
            in_cabinet = self.in_cabinet_bowl
        elif "real" in obj:
            in_cabinet = self.real
        index = 0
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            #print("info=",info)
            # jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.panda, j, in_cabinet[index]) 
                index=index+1
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.panda, j, in_cabinet[index]) 
    
    def get_joints_position(self):
        index = 0
        joint_position  = []
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                r = self.bullet_client.getJointState(self.panda, j) 
                joint_position.append(r[0])
                index=index+1
        return joint_position



    def _close_gripper_command(self):
        # self.bullet_client.submitProfileTiming("Close Gripper")
        finger_ids = [9,12]
        self.bullet_client.setJointMotorControl2(self.panda, finger_ids[0], self.bullet_client.POSITION_CONTROL, -0.0001, force=self.MAX_FORCES)
        self.bullet_client.setJointMotorControl2(self.panda, finger_ids[1], self.bullet_client.POSITION_CONTROL, 0.0001, force=self.MAX_FORCES)

    def _open_gripper_command(self):
        # self.bullet_client.submitProfileTiming("Open Gripper")
        finger_ids = [9,12]
        self.bullet_client.setJointMotorControl2(self.panda, finger_ids[0], self.bullet_client.POSITION_CONTROL, -0.06, force=self.MAX_FORCES)
        self.bullet_client.setJointMotorControl2(self.panda, finger_ids[1], self.bullet_client.POSITION_CONTROL, 0.06, force=self.MAX_FORCES)

    def open_gripper(self):
        for i in range(50):
            self._open_gripper_command()
            self.bullet_client.stepSimulation()
            time.sleep(self.timeStep)

    def close_gripper(self):
        for i in range(50):
            self._close_gripper_command()
            self.bullet_client.stepSimulation()
            time.sleep(self.timeStep)

    def get_pose(self, euler=False):
        r = self.bullet_client.getLinkState(self.panda, 15, computeLinkVelocity=1) 
        r_pos = np.array(r[0])
        if euler:
            xyz = self.bullet_client.getEulerFromQuaternion(r[1])
            r_orn = np.array(xyz)
        else:
            r_orn = np.array(r[1])
        return r_pos, r_orn

    def set_joints(self, jointPoses, velocity=None):
    # self.bullet_client.submitProfileTiming("set joints")
        if velocity: 
            for i in range(len(jointPoses)):
                self.bullet_client.setJointMotorControl2(self.panda, i, \
                self.bullet_client.POSITION_CONTROL, jointPoses[i],targetVelocity = velocity, force=5 * 240., positionGain=0.1,
                                    velocityGain=1.0)
        else:
            for i in range(len(jointPoses)):
                self.bullet_client.setJointMotorControl2(self.panda, i, \
                self.bullet_client.POSITION_CONTROL, jointPoses[i], force=5 * 240.)
                self.bullet_client.submitProfileTiming()

    def get_IK(self, pos, orn=[0,1,0,0]):
        self.bullet_client.submitProfileTiming("IK")
        jointPoses = self.bullet_client.calculateInverseKinematics(self.panda, 15, pos, orn, ll, ul, jr, rp, maxNumIterations=100)  
        return jointPoses
    
    def go_relative(self, pos):
        curr_pos, curr_orn = self.get_pose()
        new_pos = curr_pos + np.array(pos)
        for i in range(50):
            curr_pos, curr_orn = self.get_pose()
            joint_position = self.get_IK(new_pos, curr_orn)
            self.set_joints(joint_position)
            self.bullet_client.stepSimulation()
            time.sleep(self.timeStep)
    
    def go_relative_orn(self, orn):
        curr_pos, curr_orn = self.get_pose()
        new_orn = R.from_quat(curr_orn).as_euler('zyx', degrees=True)
        new_orn = new_orn + np.array(orn)
        new_orn = R.from_euler('zyx', new_orn, degrees=True).as_quat()
        joint_position = self.get_IK(curr_pos, new_orn)
        for i in range(50):
            # curr_pos, curr_orn = self.get_pose()
            self.set_joints(joint_position)
            self.bullet_client.stepSimulation()
            time.sleep(self.timeStep)

    def set_point_position(self, joint_position):
        # for i in range(3):
        self.set_joints(joint_position)
        self.bullet_client.stepSimulation()
        time.sleep(self.timeStep)


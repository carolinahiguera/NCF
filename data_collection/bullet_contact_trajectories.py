import numpy as np
import cv2
import pybullet as p
import tacto
import time
from collections import deque
from pyb_utils.collision import NamedCollisionObject, CollisionDetector
from panda_sim import PandaSim
from obj_3d_scene import Object_3D_scene


class SimContactTrajectories:
    def __init__(self, bullet_client, sim_id, cfg, path_meshes, grasp_conf, len_buffer, show_digits=True):
        # bullet client
        self.bullet_client = bullet_client
        self.sim_id = sim_id
        self.timeStep = 1./240.
        # scene config yaml file
        self.cfg = cfg
        self.path_meshes = path_meshes
        self.num_pts_meshes = 10000 # number of points in point cloud
        # background for the DIGIT sensors
        self.bg_left = np.load('./conf/bg_D20479.npy')
        self.bg_right = np.load('./conf/bg_D20510.npy')
        self.show_digits = show_digits

        self.objects_scene = {}
        self.collision_bodies = {}
        self.collision_detector = None
        self.robot = None
        self.franka_panda = None
        self.scene_3d_model = None

        # log data
        self.len_buffer = len_buffer
        self.digit_left_buffer = deque(maxlen=self.len_buffer)
        self.digit_right_buffer = deque(maxlen=self.len_buffer)
        self.ee_pose_buffer = deque(maxlen=self.len_buffer)
        self.obj_pose_buffer = deque(maxlen=self.len_buffer)
        self.all_objs_pose_buffer = deque(maxlen=self.len_buffer)
        self.sim_frame_buffer = deque(maxlen=self.len_buffer)
        self.idx_contact_buffer = deque(maxlen=self.len_buffer)
        self.d_collision_buffer = deque(maxlen=self.len_buffer)

        # grasping pose for each test object 
        # for new objects: add grasping pose
        self.grasp_conf = grasp_conf
        # load pybullet scene simulation
        self.load_bullet()

    def load_bullet(self):
        self.bullet_client.setGravity(0, 0, -10)
        self.bullet_client.setTimeStep(self.timeStep)

        planeId = self.bullet_client.loadURDF("plane.urdf", useMaximalCoordinates=False)
        # load kitchen cabinet
        self.cabinet = self.bullet_client.loadURDF(fileName = self.cfg["cabinet"]["urdf_path"],
                            basePosition = self.cfg["cabinet"]["base_position"],
                            baseOrientation = self.cfg["cabinet"]["base_orientation"],
                            useMaximalCoordinates=False,
                            globalScaling = self.cfg["cabinet"]["global_scaling"]
                            )

    def load_scene(self, obj_name=None, scene_id=None):
        self.obj_name = obj_name
        if not scene_id:
            scene_id = np.random.choice([1,2,3]) # 3 available scenes for testing in conf_scenes_test.yaml
        self.scene_id = scene_id

        num_obs = len(self.cfg[f'scene{scene_id}'])

        # add robot
        self.robot = self.bullet_client.loadURDF(  fileName = self.cfg["franka_gripper"]["urdf_path"],
                            basePosition = self.cfg["franka_gripper"]["base_position"],
                            useMaximalCoordinates = False,
                            useFixedBase = self.cfg["franka_gripper"]["use_fixed_base"],
                            )
        self.franka_panda = PandaSim(self.bullet_client, self.robot, self.timeStep)
  
        # add object to scene
        self.objects_scene = {}
        # self.bullet_client.setGravity(0, 0, -10)
        self.objects_scene[obj_name] = self.bullet_client.loadURDF(fileName = self.cfg[obj_name]["urdf_path"],
                                                basePosition = self.cfg[obj_name]["base_position"],
                                                baseOrientation = self.cfg[obj_name]["base_orientation"],
                                                useMaximalCoordinates=False,
                                                globalScaling = self.cfg[obj_name]["global_scaling"]
                                                )
        # Initialize DIGIT sensors
        if self.robot:
            # left
            self.digits_left = tacto.Sensor(**self.cfg.tacto, background=self.bg_left)
            self.digits_left.add_camera(self.robot, self.franka_panda.digit_links[0])
            self.digits_left.add_object(urdf_fn=self.cfg[self.obj_name]["urdf_path"], obj_id=self.objects_scene[self.obj_name], globalScaling=self.cfg[self.obj_name]["global_scaling"])
            # right
            self.digits_right = tacto.Sensor(**self.cfg.tacto, background=self.bg_right)
            self.digits_right.add_camera(self.robot, self.franka_panda.digit_links[1])
            self.digits_right.add_object(urdf_fn=self.cfg[self.obj_name]["urdf_path"], obj_id=self.objects_scene[self.obj_name], globalScaling=self.cfg[self.obj_name]["global_scaling"])
        
        # grasp object
        self.franka_panda.set_in_cabinet(obj_name)
        self.franka_panda.open_gripper()
        
        r_pos, r_orn = self.franka_panda.get_pose(euler=False)
        o_pos, o_orn = self.get_grasped_obj_pose()

        o_pos = r_pos + np.array(self.grasp_conf[self.obj_name][0])
        tmp_orn = np.array(self.grasp_conf[self.obj_name][1]) 
        o_orn = tmp_orn if len(tmp_orn)!=0 else o_orn
        self.bullet_client.resetBasePositionAndOrientation(self.objects_scene[obj_name], o_pos, o_orn)

        self.franka_panda.close_gripper()
        
        # self.bullet_client.setGravity(0, 0, -10)

        # add obstacles to kitchen cabinet
        for i in range(1,num_obs+1):
            self.objects_scene[f"obs{i}"] = self.bullet_client.loadURDF(fileName = self.cfg[f"scene{scene_id}"][f"obs{i}"]["urdf_path"],
                                                basePosition = self.cfg[f"scene{scene_id}"][f"obs{i}"]["base_position"],
                                                baseOrientation = self.cfg[f"scene{scene_id}"][f"obs{i}"]["base_orientation"],
                                                useMaximalCoordinates=False,
                                                globalScaling = self.cfg[f"scene{scene_id}"][f"obs{i}"]["global_scaling"]
                                                )
            self.path_meshes[f'obs{i}'] = self.cfg[f'scene{self.scene_id}'][f'obs{i}']['collision_path']
        self.objects_scene["cabinet"] = self.cabinet

        # collision detector data
        all_objs = list(self.objects_scene.keys())
        for k in all_objs[1:]:
            self.collision_bodies[k] = self.objects_scene[k]
        self.collision_bodies["robot"]  =self.objects_scene[self.obj_name]

        obj = NamedCollisionObject("robot")
        obs1 = NamedCollisionObject("obs1")
        obs2 = NamedCollisionObject("obs2")
        obs3 = NamedCollisionObject("obs3")
        obs4 = NamedCollisionObject("obs4")
        obs5 = NamedCollisionObject("obs5")
        obs6 = NamedCollisionObject("obs6")
        obs7 = NamedCollisionObject("obs7")
        cabinet = NamedCollisionObject("cabinet", link_name="cabinet_0004") 


        self.collision_detector = CollisionDetector(
                                                    self.sim_id,
                                                    self.collision_bodies,
                                                    [(obj, obs1), (obj, obs2), (obj, obs3), (obj, obs4), (obj, obs5), (obj, obs6), (obj, obs7), (obj, cabinet)],
                                                )
        
        # create scene 3d in trimesh
        self.scene_3d_model = Object_3D_scene(   cfg=self.cfg, 
                                                 path_meshes=self.path_meshes, 
                                                 rot_euler_deg=[90,0,0],
                                                 obj_name=self.obj_name,
                                                 scene_id=self.scene_id,
                                                 num_pts_pc=self.num_pts_meshes,
                                                 test=True)   
        
        self.wait()

    def clean_scene(self, path_meshes):
        if self.robot:
            self.bullet_client.removeBody(self.robot)
        if len(self.objects_scene.keys()) != 0:
            for obj in self.objects_scene.keys():
                if obj != "cabinet":
                    self.bullet_client.removeBody(self.objects_scene[obj])
        self.objects_scene = {}
        self.collision_bodies = {}
        self.collision_detector = None
        self.robot = None
        self.franka_panda = None
        self.path_meshes = path_meshes

    def wait(self):
        for i in range(50):
            self.bullet_client.stepSimulation()
            time.sleep(self.timeStep)

    def get_grasped_obj_pose(self):
        obj_pos, obj_orn = self.bullet_client.getBasePositionAndOrientation(self.objects_scene[self.obj_name])
        obj_pos = np.array(obj_pos)
        obj_orn = np.array(obj_orn)
        return obj_pos, obj_orn

    def get_objs_poses(self):
        objs_pos = []
        objs_orn = []
        # offset = np.array([0, 0.002, 0.007]) 
        offset = np.array([0.00, 0.00, 0.009]) 
        for k in self.objects_scene.keys():
            pos, orn = self.bullet_client.getBasePositionAndOrientation(self.objects_scene[k])
            if k!=self.obj_name and k!="cabinet":
                pos = tuple(np.array(pos) + offset)
            objs_pos.append(pos)
            objs_orn.append(orn)
        return objs_pos, objs_orn

    def reset_buffers(self):
        self.digit_left_buffer = deque(maxlen=self.len_buffer)
        self.digit_right_buffer = deque(maxlen=self.len_buffer)
        self.ee_pose_buffer = deque(maxlen=self.len_buffer)
        self.obj_pose_buffer = deque(maxlen=self.len_buffer)
        self.all_objs_pose_buffer = deque(maxlen=self.len_buffer)
        self.sim_frame_buffer = deque(maxlen=self.len_buffer)
        self.idx_contact_buffer = deque(maxlen=self.len_buffer)
        self.d_collision_buffer = deque(maxlen=self.len_buffer)

    def collect_data(self, sim_frame, idx_contact, d_collision):
        digit_left, digit_right, _, _ = self.update_digits(with_gui=self.show_digits)
        self.digit_left_buffer.append(digit_left)
        self.digit_right_buffer.append(digit_right)
        r_pos, r_orn = self.franka_panda.get_pose()
        self.ee_pose_buffer.append(np.concatenate((r_pos, r_orn)))
        objs_pos, objs_orn = self.get_objs_poses()
        self.obj_pose_buffer.append( np.concatenate((np.array(objs_pos[0]), np.array(objs_orn[0]))) )
        poses = []
        for i in range(1, len(objs_pos)):
            p = np.concatenate((np.array(objs_pos[i]), np.array(objs_orn[i])))
            poses.append(p)
        self.all_objs_pose_buffer.append(poses)
        self.sim_frame_buffer.append(sim_frame)
        self.idx_contact_buffer.append(idx_contact)
        self.d_collision_buffer.append(d_collision)

    def update_digits(self, with_gui=True):
        color_left, depth_left = self.digits_left.render()
        color_right, depth_right = self.digits_right.render()

        image_size = color_left[0].shape
        dim = (image_size[1]//2, image_size[0]//2)
        color_left = cv2.resize(color_left[0], dim, interpolation = cv2.INTER_AREA)
        color_right = cv2.resize(color_right[0], dim, interpolation = cv2.INTER_AREA)
        depth_left = cv2.resize(depth_left[0], dim, interpolation = cv2.INTER_AREA)
        depth_right = cv2.resize(depth_right[0], dim, interpolation = cv2.INTER_AREA)

        if with_gui:
            a=np.concatenate((color_left, color_right), axis=1)
            b=np.concatenate((depth_left, depth_right), axis=1)
            self.digits_right.updateGUI([a], [b])
        return color_left, color_right, depth_left, depth_right

    def check_digit_images(self):
        color_left, color_right, depth_left, depth_right = self.update_digits(with_gui=self.show_digits)
        if (depth_left.mean() < depth_left.max()*0.1) or (depth_right.mean() < depth_right.max()*0.1):
            return -1
        elif (color_left.mean()<80) or (color_right.mean()<80):
            return -1
        else:
            return 1
    
    def save_data(self, filename):
        np.savez(f'{filename}.npz',
                    name_obj = self.obj_name,
                    scene=self.scene_id,
                    d_collision = self.d_collision_buffer,
                    idx_point_contact = self.idx_contact_buffer,
                    digit_left = self.digit_left_buffer,
                    digit_right = self.digit_right_buffer,
                    ee_pose = self.ee_pose_buffer,
                    obj_pose = self.obj_pose_buffer,
                    all_poses = self.all_objs_pose_buffer,
                    sim_frame = self.sim_frame_buffer
                    )
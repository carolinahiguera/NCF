import numpy as np
import hydra
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
from bullet_contact_trajectories import SimContactTrajectories

NUM_TEST_TRAJECTORIES = 1
OBJECT_NAME = "mug1" # demo available options ["mug1", "mug2", "bottle1", "bottle2", "bowl1", "bowl2"]
PATH_TEST_DATA = './test_data/'
LEN_BUFFER = 50 # duration of each trajectory

codes_meshes = {'mug1':"0_1",    'mug2':"0_2",
                'bottle1':"1_1", 'bottle2':"1_2",
                'bowl1':"2_1",   'bowl2':"2_2"}


# For a new object:
# 1. place in ./objects/test/ the obj and urdf files for the new object
# 2. add path to .obj file for new objects
path_meshes = {'cabinet': "objects/cabinet/none_motion_vhacd.obj"}
for obj in codes_meshes.keys():
	path_meshes[obj] = f"objects/test/{obj}/model_normalized_vhacd.obj"

# grasping pose for each test object 
# for new objects: add grasping pose
grasp_conf = {  'mug1' : [ [0.035, 0, -0.01], [ 1, 0, -1, 0 ] ],
                'mug2' : [ [0.03, 0, -0.01], [ 1, 0, -1, 0 ] ],
                'bottle1' : [ [0.01, 0, -0.01],  [] ],
                'bottle2' : [ [0.01, 0, 0.05],  [] ],
                'bowl1' : [ [0.045, 0.05, 0.00], [ 0, 0, 0, -1 ] ],
                'bowl2' : [ [0.040, 0.05, -0.01], [ 0, 0, 0, -1 ] ]
                }
# ==============================================================================


@hydra.main(config_path="conf", config_name="conf_scenes_test")
def main(cfg):  
    sim_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    viewMatrix = p.computeViewMatrixFromYawPitchRoll(                          
                          cameraTargetPosition=[1.62, -0.21, 0.29],
                          distance=1.80,
                          yaw=-118.0,
                          pitch=-17.40,
                          roll=0.0,
                          upAxisIndex=2)

    projectionMatrix = p.computeProjectionMatrixFOV(
                          fov=45.0,
                          aspect=1.0,
                          nearVal=0.1,
                          farVal=2.1)



    sim = SimContactTrajectories(   bullet_client = p,
                                    sim_id = sim_id,
                                    cfg = cfg,
                                    path_meshes=path_meshes,
                                    grasp_conf=grasp_conf,
                                    len_buffer=LEN_BUFFER,
                                    show_digits = True
                                )
    sim.load_bullet()

    k = 0
    while k < NUM_TEST_TRAJECTORIES:
        print(f"\nCollecting trajectory {k} ...")  
        sim.clean_scene(path_meshes)
        sim.load_scene(obj_name=OBJECT_NAME)

        sim.franka_panda.go_relative(pos=[0,0,0.03])
        n_steps = np.random.randint(low=0, high=20)
        for i in range(n_steps):
            sim.franka_panda.go_relative(pos=[0.0, -0.01, 0.0])

        r_pose, _ = sim.franka_panda.get_pose()
        o_pose, o_orn = sim.get_grasped_obj_pose()
        d_grasp = np.linalg.norm(o_pose-r_pose)

        if d_grasp < 0.1:
            sim.reset_buffers()
            _, _, sim_frame, _, _  = p.getCameraImage(640, 640, viewMatrix,projectionMatrix)

            for i in range(sim.len_buffer):
                q = np.concatenate((o_pose, o_orn))
                d = sim.collision_detector.compute_distances(q)
                sim.collect_data(sim_frame, np.array([]), d)
            
            contact_detected = False

            for t in range(sim.len_buffer):
                _, _, sim_frame, _, _  = p.getCameraImage(640, 640, viewMatrix,projectionMatrix)

                if np.random.rand() < 0.6:
                    delta_pose = np.random.uniform(low=-0.03, high=0.03, size=(3,))
                    curr_pos, curr_orn = sim.franka_panda.get_pose()
                    new_pos = curr_pos + delta_pose
                    while new_pos[0] < 0.88:
                        delta_pose = np.random.uniform(low=-0.05, high=0.05, size=(3,))
                        curr_pos, curr_orn = sim.franka_panda.get_pose()
                        new_pos = curr_pos + delta_pose
                    joint_position = sim.franka_panda.get_IK(new_pos, curr_orn)
                else:
                    delta_orn = np.random.uniform(low=-7.0, high=7.0, size=(3,))
                    curr_pos, curr_orn = sim.franka_panda.get_pose()
                    new_orn = R.from_quat(curr_orn).as_euler('zyx', degrees=True) + delta_orn
                    new_orn = R.from_euler('zyx', new_orn, degrees=True).as_quat()
                    joint_position = sim.franka_panda.get_IK(curr_pos, new_orn)
                # robot execute motion
                sim.franka_panda.set_point_position(joint_position)
                
                # check digits
                r = sim.check_digit_images()
                if r==-1:
                    print("[ERROR] No reading from DIGIT sensor")
                    break

                # check if the object is still grasped
                r_pose, r_orn = sim.franka_panda.get_pose()
                o_pose, o_orn = sim.get_grasped_obj_pose()
                d_grasp = np.linalg.norm(o_pose-r_pose)

                if d_grasp > 0.1:
                    print('[ERROR] Object is not longer grasped. Retaking trajectory.')
                    break
                else:                   
                    q = np.concatenate((o_pose, o_orn))
                    d = sim.collision_detector.compute_distances(q)
                    if sim.collision_detector.in_collision(q, margin=0.006):
                        objs_pos, objs_orn = sim.get_objs_poses()
                        sim.scene_3d_model.apply_transformation(objs_pos, objs_orn)  
                        idx_pts_contact = sim.scene_3d_model.get_pts_external_contact()
                        # collect data in buffers
                        sim.collect_data(sim_frame, idx_pts_contact, d)
                        if len(idx_pts_contact)>20:
                            print(f'[INFO] External contact detected, t={t}')
                            # sim.scene_3d_model.show_object_contact(idx_pts_contact) # uncomment if you want to see the external contact ground truth
                            # sim.scene_3d_model.show_scene_mesh_contact(idx_pts_contact, d) # uncomment if you want to see the external contact ground truth + external object in collision with
                            contact_detected = True                            
                    else:
                        sim.collect_data(sim_frame, np.array([]), d)

            if  t>=sim.len_buffer-1 and contact_detected:
                filename = f"{PATH_TEST_DATA}/{codes_meshes[sim.obj_name]}_{k}"
                sim.save_data(filename)
                contact_detected = False
                k += 1 
    print("*** END ***")   


if __name__ == "__main__":
    main()
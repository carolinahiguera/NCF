import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import cv2
from skimage import exposure
from utils import torch_util
from scipy.spatial.transform import Rotation as R


class TrajectoryData(Dataset):
    def __init__(self, root_path, data_path, num_pc_pts, num_query_pts, debug=False, transform=None):
        self.root_path = root_path
        self.path = data_path
        self.digits_bg_path = f"{root_path}/ncf/conf/"
        self.image_size = None
        self.seed_is_set = False  
        self.channels = 3
        self.digit_images = []
        self.debug = debug
        self.transform = transform

        self.num_pc_pts = num_pc_pts
        self.num_query_pts = num_query_pts

        files_total = []       
        files = list(sorted(glob.glob(self.path+"/*.npz")))
        n = len(files)
        if self.debug:
            idx = int(n*0.1)
            files = files[0:idx]
        files_total.extend(files)
        self.files = files_total

        self.bg_left =  np.load(f"{self.digits_bg_path}/bg_D20479.npy")
        self.bg_right = np.load(f"{self.digits_bg_path}/bg_D20510.npy")
        self.bg_left = cv2.cvtColor(self.bg_left, cv2.COLOR_BGR2RGB)
        self.bg_right = cv2.cvtColor(self.bg_right, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.files)

    def _process_images(self, data, t):
        try:
            digit_left =  data["digit_left"]
            digit_right =  data["digit_right"]
            ee_pose = data["ee_pose"]
            frames_sq = np.array([9,7,4,2,0])

            # get history of digit images and ee pose
            img_sz = digit_left[0].shape
            digit_seq = np.zeros((len(frames_sq), img_sz[2], img_sz[0], img_sz[1]*2))
            ee_seq = np.zeros((len(frames_sq), 7))
            for i, idx in enumerate(np.flipud(frames_sq)):
                if t-idx < 0:
                    digit_seq[len(frames_sq)-i-1] = digit_seq[len(frames_sq)-i]
                    ee_seq[len(frames_sq)-i-1] = ee_seq[len(frames_sq)-i]
                else:
                    img1 = digit_left[t-idx]
                    img2 =  digit_right[t-idx]

                    img1 = cv2.GaussianBlur(img1, (7, 7), 0)
                    img2 = cv2.GaussianBlur(img2, (7, 7), 0)

                    img1 = exposure.match_histograms(img1, self.bg_left, multichannel=True)
                    img2 = exposure.match_histograms(img2, self.bg_right, multichannel=True)

                    img = cv2.hconcat([img1, img2])
                    img = img / 255.

                    if self.transform:
                        img = self.transform(img)
                    digit_seq[len(frames_sq)-i-1] = img

                    ee_seq[len(frames_sq)-i-1] = ee_pose[t-idx]
            
            return digit_seq, ee_seq
        
        except Exception as e:
           print(e)
           digit_seq, ee_seq = self._process_images(data, t)
           return digit_seq, ee_seq

    def _get_sample_item(self, data, t):
        # get digit sequence
        len_buffer = len(data["digit_left"])        
        digit_seq, ee_pose = self._process_images(data, t)
        digit_seq = torch.from_numpy(digit_seq).float()
        ee_pose = torch.from_numpy(ee_pose).float()
        
        # ee transformation matrix
        ee_quat = torch.from_numpy(data["ee_pose"][t][3:]).float()
        rot_mat = torch_util.quaternion_to_angle_axis(ee_quat).unsqueeze(dim=0)
        rot_mat = torch_util.angle_axis_to_rotation_matrix(rot_mat)
        rot_mat = rot_mat.squeeze().float()

        # set init obj orientation when grasping
        obj_name = str(data["name_obj"])
        if "mug" in obj_name:
            quat = np.array([ 1, 0, -1, 0 ])
        elif "bottle" in obj_name:
            quat = np.array([ 0, 0, 0, 1 ])
        elif obj_name == "bowl2":
            quat = np.array([ -0.3826834, 0, 0, -0.9238795 ])
        else:
            quat = np.array([ 0, 0, 0, -1 ])
        init_quat = torch.from_numpy(quat).float()
        rot_obj = torch_util.quaternion_to_angle_axis(init_quat).unsqueeze(dim=0)
        rot_obj = torch_util.angle_axis_to_rotation_matrix(rot_obj)
        rot_obj = rot_obj.squeeze().float()

        # samples from point cloud
        ref_point_cloud = np.load(f"{self.root_path}/data_collection/objects/test/point_clouds/{obj_name}_pc.npy")
        n = len(ref_point_cloud)
        point_cloud = ref_point_cloud - np.mean(ref_point_cloud, axis=0)
        idx = np.random.choice(n, size=self.num_pc_pts)
        mi_point_cloud = torch.from_numpy(point_cloud[idx]).float()
        mi_point_cloud = torch_util.transform_pcd_torch(mi_point_cloud, rot_obj)
        mi_point_cloud = torch_util.transform_pcd_torch(mi_point_cloud, rot_mat)
        pc = torch.from_numpy(point_cloud).float()
        rot_ref_pc = torch_util.transform_pcd_torch(pc, rot_obj)
        rot_ref_pc = torch_util.transform_pcd_torch(rot_ref_pc, rot_mat)

        # query points
        idx_gt_contact = data["idx_point_contact"][t]
        n_contact = len(idx_gt_contact)
        if n_contact < self.num_query_pts:
            a= np.arange(n)
            b  = np.setdiff1d(a, idx_gt_contact)
            idx_no_contact = np.random.choice(b, size=self.num_query_pts-n_contact)
            idx_samples = np.concatenate((idx_gt_contact, idx_no_contact))
            p_contact = np.concatenate((np.ones(n_contact), np.zeros(self.num_query_pts-n_contact)))
        else:
            idx_samples = idx_gt_contact[0:self.num_query_pts]
            p_contact = np.ones(self.num_query_pts)
        idx_random = np.random.choice(self.num_query_pts, size=self.num_query_pts, replace=False)
        idx_samples = idx_samples[idx_random].astype(np.int32)
        p_contact = torch.from_numpy(p_contact[idx_random]).float()
        query_points = torch.from_numpy(point_cloud[idx_samples]).float()
        query_points = torch_util.transform_pcd_torch(query_points, rot_obj)
        X = torch_util.transform_pcd_torch(query_points, rot_mat)

        # get previous prob of contact 
        t_seq =  np.array([2,1])
        p_contact_seq = []
        for i in range(len(t_seq)):
            if t < t_seq[i]:
                p_t_i = np.zeros(len(idx_samples))
            else:
                p_t_i = np.array([1.0 if idx_samples[k] in data["idx_point_contact"][t-t_seq[i]] else 0.0 for k in range(len(idx_samples))])
            p_contact_seq.append(torch.from_numpy(p_t_i).unsqueeze(dim=0).float())

        # get ee_pose difference wrt current pose
        r0 = R.from_quat(data["ee_pose"][t][3:]).as_matrix()
        if t==0:
            p_t_1 = np.zeros(len(idx_samples))
            p_t_2 = np.zeros(len(idx_samples))
            q_t_1 = R.from_matrix(np.eye(3)).as_quat()
            q_t_2 = R.from_matrix(np.eye(3)).as_quat()
            t_t_1 = np.zeros(3)
            t_t_2 = np.zeros(3)
        elif t==1:
            p_t_1 = np.array([1.0 if idx_samples[i] in data["idx_point_contact"][t-1] else 0.0 for i in range(len(idx_samples))])
            p_t_2 = np.zeros(len(idx_samples))
            r1 = R.from_quat(data["ee_pose"][t-1][3:]).as_matrix()
            q_t_1 = R.from_matrix(np.matmul(r0, r1.T)).as_quat()
            q_t_2 = R.from_matrix(np.eye(3)).as_quat()
            t_t_1 = data["ee_pose"][-1][0:3] - data["ee_pose"][-2][0:3]
            t_t_2 = np.zeros(3)
        else:
            p_t_1 = np.array([1.0 if idx_samples[i] in data["idx_point_contact"][t-1] else 0.0 for i in range(len(idx_samples))])
            p_t_2 = np.array([1.0 if idx_samples[i] in data["idx_point_contact"][t-2] else 0.0 for i in range(len(idx_samples))])
            r1 = R.from_quat(data["ee_pose"][t-1][3:]).as_matrix()
            r2 = R.from_quat(data["ee_pose"][t-2][3:]).as_matrix()
            q_t_1 = R.from_matrix(np.matmul(r0, r1.T)).as_quat()
            q_t_2 = R.from_matrix(np.matmul(r0, r2.T)).as_quat()
            t_t_1 = data["ee_pose"][t][0:3] - data["ee_pose"][t-1][0:3]
            t_t_2 = data["ee_pose"][t][0:3] - data["ee_pose"][t-2][0:3]
        p_contact_t1 = torch.from_numpy(p_t_1).float()
        p_contact_t2 = torch.from_numpy(p_t_2).float()
        ee_t1 = torch.from_numpy( np.concatenate((t_t_1, q_t_1)) ).float()
        ee_t2 = torch.from_numpy( np.concatenate((t_t_2, q_t_2)) ).float()
     
        sample = {'info': (obj_name, t),
                  'digit_seq': digit_seq,
                  'ee_seq': ee_pose,
                  'query_pts': X,
                  'pts_pc': mi_point_cloud,
                  'p_occ': p_contact,
                  'idx_points': idx_samples,
                  'ref_pc': rot_ref_pc,
                  'p_contact_t1': p_contact_t1,
                  'p_contact_t2': p_contact_t2,
                  'p_contact_seq': p_contact_seq,
                  'ee_t1': ee_t1,
                  'ee_t2': ee_t2,
                  }

        scene_id = int(data["scene"])
        d_collision = data["d_collision"]
        sim_frame = data["sim_frame"]
        obj_pose = data["obj_pose"]
        all_poses = data["all_poses"]
        objs_pos = [obj_pose[t][0:3]]
        objs_orn = [obj_pose[t][3:]]
        for i in range(len(all_poses[t])):
            objs_pos.append(all_poses[t][i][0:3])
            objs_orn.append(all_poses[t][i][3:])
        
        sample["scene"] = scene_id
        sample["objs_pos"] = objs_pos
        sample["objs_orn"] = objs_orn
        sample["d_collision"] = d_collision[t]
        sample["sim_frame"] = sim_frame[t-1] if t != 0  else  sim_frame[0]

        return sample

    def get_item(self, tj_file, t):
        try:
            data = np.load(self.path+tj_file+".npz", allow_pickle=True)
            return self._get_sample_item(data, t)
        except Exception as e:
           print(e)

    def __getitem__(self, tj_file, t):
        return self.get_item(tj_file, t)

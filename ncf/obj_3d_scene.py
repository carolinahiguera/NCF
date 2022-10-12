import os
import numpy as np
import matplotlib as mpl
from matplotlib import cm
import trimesh
from vedo import Plotter, Mesh, Points
from vedo import trimesh2vedo
from scipy.spatial.transform import Rotation as R



base_colors = { 'mug': [111,23,29,255], 'bowl': [111,23,29,255], 'sugar': [174,165,22,255], 
                'crackers':[133,10,11,255], 'pitcher':[62,87,146,255], "cabinet":[97,79,46,255],
                'tomato_soup':[133,10,11,255]}

base_textures = { "mug": "./objects/ycb/025_mug/texture.png",
                  "bowl": "./objects/ycb/024_bowl/texture.png",
                  "sugar": "./objects/ycb/004_sugar_box/texture.png",
                  "crackers": "./objects/ycb/003_cracker_box/texture.png",  
                  "pitcher": "./objects/ycb/019_pitcher_base/texture.png",
                  "tomato_soup": "./objects/ycb/005_tomato_soup_can/texture.png",
                  "cabinet": "./objects/cabinet/WoodFloor13_col.jpg"
                }

class Object_3D_scene:
    def __init__(self, cfg, base_path_mesh, path_meshes, rot_euler_deg, obj_name, num_pts_pc=5000, test=False):
        self.cfg = cfg
        self.base_path_mesh = base_path_mesh
        self.path_meshes = path_meshes        
        self.rot_euler_deg = rot_euler_deg
        self.obj_name = obj_name
        self.base_num_pts_pc = num_pts_pc
        self.test = test
    
    def set_scene(self, scene_id):
        self.scene_id = scene_id
        self.num_obs = len(self.cfg[f'scene{self.scene_id}'])
        self.num_pts_pc = np.ones(self.num_obs+2, dtype=np.int32) * self.base_num_pts_pc
        self.num_pts_pc[0] = self.num_pts_pc[0]*3
        self.num_pts_pc[-1] = self.num_pts_pc[-1]*6
        self.on_shelf = {}
        self.obj_color = {}
        self.obj_texture = {}

        rot = np.eye(4)
        rot[:-1, :-1] = R.from_euler('zyx',np.array(self.rot_euler_deg),degrees=True).as_matrix()

        self.meshes = {}
        self.meshes[self.obj_name] = self.load_mesh(path=self.base_path_mesh+self.path_meshes[self.obj_name], 
                                        scale=self.cfg[self.obj_name]["global_scaling"], rot=rot,
                                        with_offset=True, convex_hull=False)
        self.meshes_t = {}
        self.obj_color[self.obj_name] = [46,171,135,255]

        for i in range(self.num_obs):
            pth = self.cfg[f'scene{self.scene_id}'][f"obs{i+1}"]["collision_path"]
            self.path_meshes[f"obs{i+1}"] = f"{self.base_path_mesh}/{pth}"
            self.meshes[f"obs{i+1}"] = self.load_mesh(path=self.path_meshes[f"obs{i+1}"], rot=rot, with_offset=True)
            self.obj_color[f"obs{i+1}"] = base_colors[ self.cfg[f'scene{self.scene_id}'][f"obs{i+1}"]["name"] ]
            self.obj_texture[f"obs{i+1}"] = self.base_path_mesh + base_textures[ self.cfg[f'scene{self.scene_id}'][f"obs{i+1}"]["name"] ]
        
        self.meshes["cabinet"] = self.load_mesh(path=self.base_path_mesh+self.path_meshes[f"cabinet"])
        self.obj_color["cabinet"] = base_colors["cabinet"]
        self.obj_texture["cabinet"] = self.base_path_mesh + base_textures["cabinet"]
        
            
        self.point_clouds_array = {}
        self.points_clouds = {}
        path_root = os.path.dirname(os.path.abspath("."))
        for i, k in enumerate(self.meshes.keys()):
            color = self.obj_color[k] if k != self.obj_name else [46,171,135,255]
            if k==self.obj_name:                
                root_path = f"{path_root}/data_collection/objects/train/point_clouds/" if not self.test else f"{path_root}/data_collection/objects/test/point_clouds/"  
                pts = np.load(f"{root_path}/{self.obj_name}_pc.npy")                 
                non_convex_pts = trimesh.sample.sample_surface(self.meshes[k], self.num_pts_pc[i])[0]
            else:
                pts = trimesh.sample.sample_surface(self.meshes[k], self.num_pts_pc[i])[0]
            color = np.ones((self.num_pts_pc[i], 4))*color
            self.point_clouds_array[k] = pts
            self.points_clouds[k] = Points(pts, r=3, c=tuple(color))
            self.non_convex_pc = Points(non_convex_pts, r=3)
            self.on_shelf[k] = True
           

    def load_mesh(self, path, scale=None, rot=[], with_offset=False, convex_hull=False):
        m = trimesh.load_mesh(path)
        if convex_hull:
            m = trimesh.convex.convex_hull(m)
        m.visual = trimesh.visual.ColorVisuals()
        if scale != None:
            scale = scale+0.03 if with_offset else scale
            m.apply_scale(scale)
        if len(rot) != 0:
            m.apply_transform(rot)
        return m



    def apply_transformation(self, objs_pos, objs_orn):
        self.meshes_t = {}
        for i, k in enumerate(self.meshes.keys()):
            rot = np.eye(4)
            rot[:-1, :-1] = R.from_quat(objs_orn[i]).as_matrix()
            rot[0:3, -1] = objs_pos[i]
            self.points_clouds[k].applyTransform(rot)
            if i==0:
                self.non_convex_pc.applyTransform(rot)
            if objs_pos[i][-1] < 0.4:
                self.on_shelf[k] = False
            self.meshes_t[k] = self.meshes[k].copy()
            self.meshes_t[k].apply_transform(rot)
            self.point_clouds_array[k] = rot[:-1, :-1].dot(self.point_clouds_array[k].T).T + rot[0:3, -1]

        
    def get_scene_3d(self, idx_points_contact, gt_contact, pred_contact, d_collision, objs_pos):
            rot = np.eye(4)
            rot[0:3, -1] = objs_pos[0]*-1

            # mesh gt
            mesh_gt_sc = trimesh2vedo(self.meshes_t[self.obj_name]).clone()
            mesh_gt_sc.subdivide(N=3,method=2)

            mesh_gt = trimesh2vedo(self.meshes_t[self.obj_name]).clone()
            mesh_gt.subdivide(N=3,method=2)
            pts_obj = self.points_clouds[self.obj_name].points()

            if ("mug" in self.obj_name) or ("bowl" in self.obj_name):
                pocc = np.zeros(self.num_pts_pc[0])
                if len(idx_points_contact) != 0:
                    pocc[idx_points_contact] = pred_contact
                pts_external_contact_pred =  Points(pts_obj, r=3).cmap('viridis', pocc, vmin=0.0, vmax=1.0)

                non_convex_pc = self.non_convex_pc.points()
                idx = np.array([np.argmin(np.linalg.norm(non_convex_pc - pts_obj[ idx_points_contact[j] ],axis=1)) for j in range(len(idx_points_contact))])
                idx_d = np.array([np.min(np.linalg.norm(non_convex_pc - pts_obj[ idx_points_contact[j] ],axis=1)) for j in range(len(idx_points_contact))])

                idx_points_contact = idx[np.where(idx_d < 1e-3)]
                gt = gt_contact[np.where(idx_d < 1e-3)]
                pred = pred_contact[np.where(idx_d < 1e-3)]

                idx2 = list([np.where(np.linalg.norm(non_convex_pc - non_convex_pc[ idx_points_contact[j] ],axis=1)<2e-3)[0] for j in range(len(idx_points_contact))])
                c=list([np.ones(len(idx2[j]))*gt[j] for j in range(len(idx_points_contact))])
                d=list([np.ones(len(idx2[j]))*pred[j] for j in range(len(idx_points_contact))])
                idx2 = np.concatenate(idx2)
                c = np.concatenate(c)
                d = np.concatenate(d)
                idx_points_contact = np.concatenate((idx_points_contact, np.array(idx2)))
                gt = np.concatenate((gt,c))
                pred = np.concatenate((pred,d))
                idx_points_contact, idx_unique = np.unique(idx_points_contact, return_index=True)
                gt = gt[idx_unique]
                pred = pred[idx_unique]
                pocc = np.zeros(self.num_pts_pc[0])
                if len(idx_points_contact) != 0:
                    pocc[idx_points_contact] = gt
                pts_external_contact =  Points(non_convex_pc, r=3).cmap('viridis', pocc, vmin=0.0, vmax=1.0)
            else:
                pocc = np.zeros(self.num_pts_pc[0])
                if len(idx_points_contact) != 0:
                    pocc[idx_points_contact] = gt_contact
                pts_external_contact =  Points(pts_obj, r=3).cmap('viridis', pocc, vmin=0.0, vmax=1.0)
                pocc = np.zeros(self.num_pts_pc[0])
                if len(idx_points_contact) != 0:
                    pocc[idx_points_contact] = pred_contact
                pts_external_contact_pred =  Points(pts_obj, r=3).cmap('viridis', pocc, vmin=0.0, vmax=1.0)
            
            mesh_gt.interpolateDataFrom(pts_external_contact, N=3, on="points", kernel='linear').cmap('viridis',  vmin=0.0, vmax=1.0)
            mesh_gt_sc.interpolateDataFrom(pts_external_contact, N=3, on="points", kernel='linear').cmap('viridis',  vmin=0.0, vmax=1.0)

            all_meshes_scene_3d = [mesh_gt_sc]
            objs = list(self.points_clouds.keys())
            objs = objs[1:]
            for i, k in enumerate(objs):
                if self.on_shelf[k]:
                    mesh = trimesh2vedo(self.meshes_t[k]).clone()
                    mesh.color(c=self.obj_color[k][:3])
                    mesh.alpha(1.0)
                    all_meshes_scene_3d.append(mesh)

            all_meshes_zoom = [mesh_gt_sc]
            objs = list(self.points_clouds.keys())
            objs = objs[1:]
            for i, k in enumerate(objs):
                if self.on_shelf[k]:
                    mesh = trimesh2vedo(self.meshes_t[k]).clone()
                    mesh.color(c=self.obj_color[k][:3])
                    mesh.alpha(1.0)
                    all_meshes_zoom.append(mesh)
       
            mesh_pred = trimesh2vedo(self.meshes_t[self.obj_name]).clone()
            mesh_pred.subdivide(N=3,method=2)
            mesh_pred.interpolateDataFrom(pts_external_contact_pred, N=3, on="points", kernel='gaussian').cmap('viridis',  vmin=0.0, vmax=1.0)#.addScalarBar()
            
            mesh_gt.applyTransform(rot)
            mesh_pred.applyTransform(rot)

            meshes_gt_pred = [mesh_gt, mesh_pred]
           
            return all_meshes_scene_3d, all_meshes_zoom, meshes_gt_pred

    def scene_image(self, all_meshes_scene_3d, scene_3d_cam, tj_file, t):
        plt = Plotter(size=(1800, 1000), interactive=False)
        plt.show(all_meshes_scene_3d, resetcam=True, camera=scene_3d_cam)
        img = plt.screenshot(filename=f"screenshot.png", asarray=True)
        plt.close()
        return img


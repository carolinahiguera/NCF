import numpy as np
import trimesh
from vedo import Plotter, Mesh, Points, show
from vedo import trimesh2vedo
from scipy.spatial.transform import Rotation as R
from pysdf import SDF
from os.path import exists


idx_pts_contact = []
base_colors = { 'mug': [111,23,29,255], 'bowl': [111,23,29,255], 'sugar': [174,165,22,255], 
                'crackers':[133,10,11,255], 'pitcher':[62,87,146,255], "cabinet":[81,56,41,255],
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
    def __init__(self, cfg, path_meshes, rot_euler_deg, obj_name, scene_id, num_pts_pc=10000, test=False):
        self.cfg = cfg
        self.path_meshes = path_meshes
        self.path_meshes["cabinet"] = "./objects/cabinet/bookcase_half.stl"
        self.rot_euler_deg = rot_euler_deg
        self.obj_name = obj_name
        self.scene_id = scene_id
        self.test = test
        self.num_obs = len(self.cfg[f'scene{self.scene_id}'])
        self.num_pts_pc = np.ones(self.num_obs+2, dtype=np.int32) * num_pts_pc
        self.num_pts_pc[0] = self.num_pts_pc[0]*3
        self.num_pts_pc[-1] = self.num_pts_pc[-1]*6
        self.on_shelf = {}
        self.obj_color = {}
        self.obj_texture = {}

        rot = np.eye(4)
        rot[:-1, :-1] = R.from_euler('zyx',np.array(rot_euler_deg),degrees=True).as_matrix()
        

        self.meshes = {}
        pth = self.path_meshes[self.obj_name]
        pth = pth.replace("_vhacd","")
        self.meshes[self.obj_name] = self.load_mesh(path=pth, 
                                        scale=self.cfg[self.obj_name]["global_scaling"], rot=rot,
                                        with_offset=True, convex_hull=False)
        self.meshes_t = {}

        
        for i in range(self.num_obs):
            pth = self.path_meshes[f"obs{i+1}"]
            pth = pth.replace("_vhacd","")
            self.meshes[f"obs{i+1}"] = self.load_mesh(path=pth, rot=rot, with_offset=True)
            self.obj_color[f"obs{i+1}"] = base_colors[ self.cfg[f'scene{self.scene_id}'][f"obs{i+1}"]["name"] ]
            self.obj_texture[f"obs{i+1}"] = base_textures[ self.cfg[f'scene{self.scene_id}'][f"obs{i+1}"]["name"] ]
        
        self.meshes["cabinet"] = self.load_mesh(path=self.path_meshes[f"cabinet"])
        self.obj_color["cabinet"] = base_colors["cabinet"]
        self.obj_texture["cabinet"] = base_textures["cabinet"]
        
            
        self.point_clouds_array = {}
        self.points_clouds = {}
        for i, k in enumerate(self.meshes.keys()):
            color = self.obj_color[k] if k != self.obj_name else [46,171,135,255]
            if k==self.obj_name:
                root_path = "./objects/train/point_clouds/" if not self.test else "./objects/test/point_clouds/"                
                if exists(f"{root_path}/{self.obj_name}_pc.npy"):
                    pts = np.load(f"{root_path}/{self.obj_name}_pc.npy")
                else:
                    pts = trimesh.sample.sample_surface(self.meshes[k], self.num_pts_pc[i])[0]
                    np.save(f"{root_path}/{self.obj_name}_pc.npy", pts)                
                non_convex_pts = trimesh.sample.sample_surface(self.meshes[k], self.num_pts_pc[i])[0]
            else:
                pts = trimesh.sample.sample_surface(self.meshes[k], self.num_pts_pc[i])[0]
            color = np.ones((self.num_pts_pc[i], 4))*color
            self.point_clouds_array[k] = pts
            self.points_clouds[k] = Points(pts, r=3, c=tuple(color))
            self.non_convex_pc = Points(non_convex_pts, r=3)
            self.on_shelf[k] = True

        self.ref_point_cloud = self.point_clouds_array[self.obj_name].copy()
            

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
            if i == 0:
                rot[0:3, -1] = objs_pos[i] #+ np.array([0.005,-0.005,0.0])
                self.non_convex_pc.applyTransform(rot)
            else:
                rot[0:3, -1] = objs_pos[i] 
            self.points_clouds[k].applyTransform(rot)
            if objs_pos[i][-1] < 0.4:
                self.on_shelf[k] = False
            self.meshes_t[k] = self.meshes[k].copy()
            self.meshes_t[k].apply_transform(rot)
            self.point_clouds_array[k] = rot[:-1, :-1].dot(self.point_clouds_array[k].T).T + rot[0:3, -1]


    def get_pts_external_contact(self):
        idx_points_contact = np.array([])
        objs = list(self.points_clouds.keys())
        objs = objs[1:]
        pts_obj = self.points_clouds[self.obj_name].points()
        for i, k in enumerate(objs):
            f = SDF(self.meshes_t[k].vertices, self.meshes_t[k].faces)
            d = f(pts_obj)
            idx =np.where(d >= -0.002)[0]
            if len(idx) != 0:
                idx_points_contact = np.concatenate((idx_points_contact, idx))
        return idx_points_contact.astype(np.int32)

    def show_object_contact(self, idx_points_contact):
        mesh = trimesh2vedo(self.meshes_t[self.obj_name]).clone()
        mesh.subdivide(N=3,method=2)
        pts_obj = self.points_clouds[self.obj_name].points()
        pocc = np.zeros(self.num_pts_pc[0])
        pocc[idx_points_contact] = 1.0
        pts_external_contact =  Points(pts_obj, r=10).cmap('viridis', pocc, vmin=0.0, vmax=1.0)
        mesh.interpolateDataFrom(pts_external_contact, N=2, on="points", kernel='gaussian').cmap('viridis',  vmin=0.0, vmax=1.0)
        show(mesh, axes=1, viewup="z").close()

    def show_scene_mesh_contact(self, idx_points_contact, d_collision):
        all_meshes = []
        mesh = trimesh2vedo(self.meshes_t[self.obj_name]).clone()
        mesh.subdivide(N=3,method=2)
        pts_obj = self.points_clouds[self.obj_name].points()
        non_convex_pc = self.non_convex_pc.points()

        idx = np.array([np.argmin(np.linalg.norm(non_convex_pc - pts_obj[ idx_points_contact[j] ],axis=1)) for j in range(len(idx_points_contact))])
        idx_d = np.array([np.min(np.linalg.norm(non_convex_pc - pts_obj[ idx_points_contact[j] ],axis=1)) for j in range(len(idx_points_contact))])
        
        pocc = np.zeros(self.num_pts_pc[0])
        idx_points_contact = idx[np.where(idx_d < 1e-3)]
        idx = list([np.where(np.linalg.norm(non_convex_pc - non_convex_pc[ idx_points_contact[j] ],axis=1)<2e-3)[0] for j in range(len(idx_points_contact))])
        idx = np.concatenate(idx)
        idx_points_contact = np.unique(np.concatenate((idx_points_contact, idx)))
        if len(idx_points_contact) != 0:
            pocc[idx_points_contact] = 1.0
        pts_external_contact =  Points(non_convex_pc, r=3).cmap('viridis', pocc, vmin=0.0, vmax=1.0)
        mesh.interpolateDataFrom(pts_external_contact, N=3, on="points", kernel='gaussian').cmap('viridis',  vmin=0.0, vmax=1.0)
        all_meshes.append(mesh)
        all_meshes.append(pts_external_contact)

        objs = list(self.points_clouds.keys())
        objs = objs[1:]
        for i, k in enumerate(objs):
            if self.on_shelf[k] and d_collision[i]<=0.008:
                mesh = trimesh2vedo(self.meshes_t[k]).clone()
                mesh.texture(self.obj_texture[k])
                mesh.alpha(0.5)
                all_meshes.append(mesh)
        
        show(all_meshes, axes=1, viewup="z").close()



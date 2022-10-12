import os
import shutil
import hydra
import numpy as np
import matplotlib.pyplot as plt


# torch
import torch
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl

# vedo
from vedo import Plotter, show

# custom
from data.dataio_trajectories import TrajectoryData
from digit_seq2seq.autoencoder import DIGIT_Autoencoder
from digit_seq2seq.rnn_model import Encoder, Decoder, TactilePoseEncoder
import ndf_model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ncf_model.ncf_model import ExternalContactModel
from obj_3d_scene import Object_3D_scene
from params_cam import get_cam_parameters


scene_3d_cam = dict(
        pos=(-0.1943591, 0.5042670, 0.8488891),
        focalPoint=(1.009964, 0.01104544, 0.6306321),
        viewup=(0.1562545, -0.05486857, 0.9861917),
        distance=1.319582,
        clippingRange=(0.5392920, 2.306808),
    )

scene_3d_zoom_camera = dict(
                            pos=(0.04416, 0.4545, 0.8476),
                            focalPoint=(1.023, 0.04872, 0.6408),
                            viewup=(0.1796, -0.06692, 0.9815),
                            distance=1.080,
                            clippingRange=(0.2935, 2.082),
                        )
font_size = 14

@hydra.main(config_path=f"conf", config_name="conf_scenes_test")
def main(cfg):
    # NCF parameters
    n_query_pts = 3500
    n_pc_pts =  5000
    num_hidden_lstm = 6
    c_dim = 128
    debug_dataset = False
    trajectory_file_name = "0_1_0"

    objects = ["mug1", "mug2",
               "bottle1", "bottle2",
               "bowl1", "bowl2"]

    # Setting the seed
    seed = 42
    pl.seed_everything(seed)

    # check if gpu available
    dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # testing trajectories dataset
    path_root = os.path.dirname(os.path.abspath("."))
    path_data = f"{path_root}/data_collection/test_data/"    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,))])

    dataset = TrajectoryData(root_path=path_root,
                             data_path=path_data, 
                             num_pc_pts=n_pc_pts, 
                             num_query_pts=n_query_pts, 
                             debug=debug_dataset, 
                             transform=transform)

    # load digit encoder    
    base_path_digit = f"{path_root}/ncf/digit_seq2seq/models_checkpoints/"
   
    # torch.save(digit_embeddings_model.state_dict(), f"{base_path_digit}/digit_embs_weights.pth")
    digit_embeddings_path = f"{base_path_digit}/digit_embs_weights.pth"
    digit_embeddings_model = DIGIT_Autoencoder(base_channel_size= 64, num_hidden = num_hidden_lstm)
    digit_embeddings_model.load_state_dict(torch.load(digit_embeddings_path, map_location=dev))
    for param in digit_embeddings_model.parameters():
        param.requires_grad = False
    digit_embeddings_model.eval()

    # load seq2seq model
    checkpoint_digit_seq2seq = f"{base_path_digit}/digit_seq2seq_model.ckpt"
    emb_sz = (20*30) + 7
    encoder = Encoder(emb_dim=emb_sz, hid_dim=emb_sz, num_layers=1, dropout=0.8)
    decoder = Decoder(emb_dim=emb_sz, hid_dim=emb_sz, num_layers=1, dropout=0.8)

    checkpoint = torch.load(checkpoint_digit_seq2seq, map_location=dev)
    state_dict_encoder = {}
    state_dict_decoder = {}
    for key in checkpoint['state_dict'].keys():
        if 'encoder.rnn' in key:        
            new_key = key[len('encoder.'):]
            state_dict_encoder[new_key] = checkpoint['state_dict'][key]
        if 'decoder.rnn' in key:    
            new_key = key[len('decoder.'):]    
            state_dict_decoder[new_key] = checkpoint['state_dict'][key]
    encoder.load_state_dict(state_dict_encoder)
    decoder.load_state_dict(state_dict_decoder)
    
    digit_seq2seq = TactilePoseEncoder( autoencoder = digit_embeddings_model, 
                                        encoder = encoder, 
                                        decoder = decoder )
    for param in digit_seq2seq.parameters():
        param.requires_grad = False
    digit_seq2seq.eval()
    print("[OK] Loaded digit seq2seq autoencoder")

    # load NDF network
    ndf_model_path =  f"{path_root}/ncf/ndf_model/ndf_weights/multi_category_weights.pth"
    ndf_model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True)
    ndf_model.load_state_dict(torch.load(ndf_model_path, map_location=dev))
    for param in ndf_model.parameters():
        param.requires_grad = False
    ndf_model.eval()
    print("[OK] Loaded NDF model")

    # load NCF model
    ncf_model_path = f"{path_root}/ncf/ncf_model/ncf_weights/ncf_weights.pth"
    contact_model = ExternalContactModel(   num_hidden_lstm=num_hidden_lstm,
                                            c_dim = c_dim,
                                            seq_len= 1,
                                            digit_feature_extractor=digit_seq2seq,
                                            ndf_model = ndf_model
                                            )
    contact_model.load_state_dict(torch.load(ncf_model_path, map_location=dev))
    contact_model.to(dev)
    for param in contact_model.parameters():
        param.requires_grad = False
    for m in contact_model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.eval()
    print("[OK] Loaded NCF model")


    # load 3d meshes of possible test objects
    base_path_meshes = f"{path_root}/data_collection/"
    path_meshes = {'cabinet': "objects/cabinet/none_motion_vhacd.obj"}
    for obj in objects:
        path_meshes[obj] = f"objects/test/{obj}/model_normalized_vhacd.obj"

    object_3Dmodels = {}
    for k in objects:
        object_3Dmodels[k] = Object_3D_scene(   cfg=cfg, 
                                                base_path_mesh = base_path_meshes,
                                                path_meshes=path_meshes, 
                                                rot_euler_deg=[90,0,0],
                                                obj_name = k,
                                                num_pts_pc=10000,
                                                test=True)  

    # metric
    mse = torch.nn.MSELoss()

    # start evaluation of  NCF
    path_results = f"{path_root}/ncf/results/{trajectory_file_name}/"
    try:
        os.mkdir(path_results)
        print(f"[OK] Will save results in {path_results}")
    except: 
        if os.path.exists(path_results):
            print(f"[OK] Will save results in {path_results}")
        else:
            print(f"[ERROR] Couldn't create {path_results}")
    cameras, t_switch_cameras = get_cam_parameters(trajectory_file_name)


    data = np.load(f"{path_data}/{trajectory_file_name}.npz", allow_pickle=True)
    frames = data["sim_frame"]
    len_t = len(data["digit_left"])-1

    print("[INFO] Starting evaluation of  NCF")
    for t in range(0, len_t):        
        sample = dataset.get_item(trajectory_file_name, t)
        obj_name = sample['info'][0]
        scene_id = sample["scene"]
        ee_seq = sample["ee_seq"].unsqueeze(dim=0).to(dev)
        digit_img_seq  = sample['digit_seq'].unsqueeze(dim=0).to(dev)
        query_pts = sample['query_pts'].unsqueeze(dim=0).to(dev)
        points_pc = sample['pts_pc'].unsqueeze(dim=0).to(dev)
        p_occ = sample['p_occ'].unsqueeze(dim=0).to(dev)
        idx_pts = sample['idx_points']
        d_collision = sample["d_collision"]
        p_contact_t1 = sample['p_contact_t1'].unsqueeze(dim=0).to(dev)
        p_contact_t2 = sample['p_contact_t2'].unsqueeze(dim=0).to(dev)
        ee_t1 = sample['ee_t1'].unsqueeze(dim=0).to(dev)
        ee_t2 = sample['ee_t2'].unsqueeze(dim=0).to(dev)
        frame = frames[t+1]

        ndf_input = {}
        ndf_input['coords'] = query_pts
        ndf_input['point_cloud'] = points_pc

        pocc_hat, _ = contact_model(digit_seq=digit_img_seq, ee_seq=ee_seq, ndf_input=ndf_input, 
                                    p_contact_t1=p_contact_t1, p_contact_t2=p_contact_t2, 
                                    ee_t1=ee_t1, ee_t2=ee_t2)

        gt_contact = p_occ.squeeze().cpu().numpy()
        pred_contact = pocc_hat.squeeze().cpu().numpy()
        res_mse = mse(p_occ+1e-3, pocc_hat.detach()+1e-3).cpu().numpy()

        object_3Dmodels[obj_name].set_scene(scene_id)
        objs_pos = sample["objs_pos"]
        objs_orn = sample["objs_orn"]

        object_3Dmodels[obj_name].apply_transformation(objs_pos, objs_orn)  
        all_meshes_scene_3d, all_meshes_zoom, meshes_gt_pred = object_3Dmodels[obj_name].get_scene_3d(idx_pts, gt_contact, pred_contact, d_collision, objs_pos)
        scene = object_3Dmodels[obj_name].scene_image(all_meshes_scene_3d, scene_3d_cam, trajectory_file_name, t)

        obj_pos = objs_pos[0]
        scene_zoom = show(all_meshes_zoom, resetcam=True, camera=scene_3d_zoom_camera, interactive=False).flyTo(point=obj_pos).show().screenshot(filename=f"screenshot.png", asarray=True)
        show().close()
        cam_id = int(t_switch_cameras[t])
        cam = cameras[cam_id]
        mesh_gt_contact = show(meshes_gt_pred[0], resetcam=True, camera=cam, interactive=False, offscreen=True).screenshot(filename=f"screenshot.png", asarray=True)
        show().close()
        mesh_pred_contact = show(meshes_gt_pred[1], resetcam=True, camera=cam, interactive=False, offscreen=True).screenshot(filename=f"screenshot.png", asarray=True)
        show().close()

        fig, axs = plt.subplots(2,3,squeeze=True, figsize=(14, 8))
        gs = axs[0, 1].get_gridspec()
        axs[0,1].remove()
        axs[0,2].remove()
        axbig = fig.add_subplot(gs[0, 1:])
        axs[0,0].imshow(frame)
        axs[0,0].set_title("PyBullet Simulation", fontsize=font_size)
        axbig.imshow(scene)
        axbig.set_title("3D view of the scene", fontsize=font_size)
        axs[1,0].imshow(scene_zoom)
        axs[1,0].set_title("Zoom-in View", fontsize=font_size)
        axs[1,1].imshow(mesh_gt_contact)
        axs[1,1].set_title("External Contact - Ground Truth", fontsize=font_size)
        im = axs[1,2].imshow(mesh_pred_contact/255.)
        cbar = fig.colorbar(im, ax=axs[1,2], orientation='vertical')
        plt.gcf().text(0.98, 0.15, f"Contact Probability", fontsize=font_size-2, rotation=270) 
        axs[1,2].set_title(f"External Contact - Neural Contact Fields", color="red", fontsize=font_size)
        plt.gcf().text(0.68, 0.035, f"MSE between GT and NCF = {res_mse:.{3}f}", fontsize=font_size-2) # 0.030 0.44
        a = [axi.set_axis_off() for axi in axs.ravel()]
        axbig.set_axis_off()
        plt.gcf().text(0.85, 0.955, f"Timestep {t} / {len_t}", fontsize=font_size)
        fig.tight_layout()
        plt.savefig(f"{path_results}/{t:02}.png", dpi=150)
        plt.close()
        print(f"[t={t} / {len_t}] mse={res_mse:.{3}f}")

    print('*** END ***')


if __name__ == '__main__':
    path_root = os.path.dirname(os.path.abspath("."))
    src = path_root + "/data_collection/conf"
    dest = path_root + "/ncf/conf"
    try:
        destination = shutil.copytree(src, dest) 
    except:
        print("[INFO] Config files are already in the ncf directory. Good to go.")
    main()
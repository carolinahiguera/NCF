import numpy as np
import matplotlib.pyplot as plt
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl

# Torchvision
import torchvision

import time

from ncf_model.layers_contact_model import CResnetBlockConv1d, CBatchNorm1d

class ExternalContactModel(pl.LightningModule):

    def __init__(self,
                 num_hidden_lstm: int,
                 c_dim: int,
                 seq_len: int,
                 digit_feature_extractor : object,
                 ndf_model: object
                 ):
        super().__init__()
        self.digit_feature_extractor = digit_feature_extractor
        self.ndf_model = ndf_model
        self.num_hidden_lstm = num_hidden_lstm
        self.c_dim = c_dim
        self.seq_len = seq_len
        self.digit_ch_conv = [1, 3, 3] 
        # self.img_emb_sz = [3, 8, 13] # i got this running the forward step
        self.latent_var_digit = 607
        self.latent_var_coord = 2049 + 16 # i knew this running the ndf

        # self.cnn_digit_0 = torch.nn.Conv2d(in_channels=self.digit_ch_conv[0], out_channels=self.digit_ch_conv[1], kernel_size=3, stride=2)
        # self.cnn_digit_1 = torch.nn.Conv2d(in_channels=self.digit_ch_conv[1], out_channels=self.digit_ch_conv[2], kernel_size=3, stride=2)
        # self.im_sz = (self.im_sz//2)-1

        # d_dim = self.img_emb_sz[0] * self.img_emb_sz[1] * self.img_emb_sz[2] # dim when flattening the embedding
        self.transform = transforms.Resize(size=(14,21))

        self.block0 = CResnetBlockConv1d(c_dim=self.latent_var_digit, size_in=self.latent_var_coord, size_h=c_dim, size_out=c_dim)
        self.block1 = CResnetBlockConv1d(c_dim=self.latent_var_digit, size_in=c_dim)
        self.block2 = CResnetBlockConv1d(c_dim=self.latent_var_digit, size_in=c_dim)
        self.block3 = CResnetBlockConv1d(c_dim=self.latent_var_digit, size_in=c_dim)

        self.bn = CBatchNorm1d(self.latent_var_digit, c_dim)
        self.dropout = nn.Dropout1d(p=0.2)
        self.fc_out = nn.Conv1d(c_dim, 1, 1)
        self.actvn = nn.Sigmoid()

    def forward(self, digit_seq, ee_seq, ndf_input, p_contact_t1, p_contact_t2, ee_t1, ee_t2):
        self.digit_feature_extractor.eval()
        _, _, d = self.digit_feature_extractor(digit_seq, ee_seq)
        d = d.permute(1,0,2)
        # emb_t = emb_t.unsqueeze(dim=1)
        # d_z = torch.relu(self.cnn_digit_0(emb_t))
        # d_z = torch.relu(self.cnn_digit_1(d_z))
        # d = torch.flatten(emb_t, start_dim=1)

        # x = (torch.rand((5, 1024, self.latent_var_dim))).float().to(p.device) # this simulates z
        self.ndf_model.eval()
        z = self.ndf_model.forward(ndf_input)
        x = z['features'].transpose(1, 2)

        p_contact_t1 = p_contact_t1.unsqueeze(dim=1)
        p_contact_t2 = p_contact_t2.unsqueeze(dim=1)
        ee_t1 = ee_t1.unsqueeze(dim=2).repeat(1,1,x.shape[-1])
        ee_t2 = ee_t2.unsqueeze(dim=2).repeat(1,1,x.shape[-1])

        x_in = torch.cat((x, p_contact_t1, p_contact_t2, ee_t1, ee_t2),1)

        # x = self.fc_p(x)
        x1 = self.block0(x_in,  d)
        x2 = self.block1(x1, d)
        x3 = self.block2(x2, d)
        x4 = self.block2(x3, d)
        x4 = torch.relu(self.bn(x4, d))
        x5 = self.dropout(x4)
        out = self.fc_out(x5)
        out = self.actvn(out)
        out = out.squeeze(1)
        return out, d

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        digit_img_seq  = batch['digit_seq']
        ee_seq = batch["ee_seq"]
        query_pts = batch['query_pts']
        points_pc = batch['pts_pc']
        p_occ = batch['p_occ']
        p_contact_t1 = batch['p_contact_t1']
        p_contact_t2 = batch['p_contact_t2']
        ee_t1 = batch['ee_t1']
        ee_t2 = batch['ee_t2']

        ndf_input = {}
        ndf_input['coords'] = query_pts
        ndf_input['point_cloud'] = points_pc

        # ini_t = time.time()
        pocc_hat, _ = self.forward( digit_seq=digit_img_seq, 
                                    ee_seq=ee_seq, 
                                    ndf_input=ndf_input,
                                    p_contact_t1=p_contact_t1,
                                    p_contact_t2=p_contact_t2,
                                    ee_t1=ee_t1,
                                    ee_t2=ee_t2,
                                  )
        # print(f'[INFERENCE TRAIN] {time.time()-ini_t} seconds')
        # loss = F.mse_loss(p_occ, pocc_hat, reduction="none")
        # loss = loss.sum(dim=[1]).mean(dim=[0])
        loss = self.loss_external_contact(model_outputs=pocc_hat, ground_truth=p_occ)
        return loss

    def loss_external_contact(self, model_outputs, ground_truth, val=False):
        label = ground_truth.squeeze()
        # label = (label + 1) / 2.
        loss = -1 * (label * torch.log(model_outputs + 1e-5) + (1 - label) * torch.log(1 - model_outputs + 1e-5)).mean()
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        #                                                  mode='min',
        #                                                  factor=0.2,
        #                                                  patience=20,
        #                                                  min_lr=5e-5)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return {"optimizer": optimizer,  "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

# class GenerateCallback(pl.Callback):

#     def __init__(self, dataset, obj_3Dmodels, path_save, every_n_epochs=1):
#         super().__init__()
#         self.dataset = dataset
#         self.obj_3dmodels = obj_3Dmodels
#         self.every_n_epochs = every_n_epochs
#         self.path_save = path_save

#     def on_epoch_end(self, trainer, pl_module):
#         if trainer.current_epoch % self.every_n_epochs == 0:
#             # Reconstruct images
#             sample = self.dataset.sample_test[10]            
#             with torch.no_grad():
#                 pl_module.eval()
                
#                 info_seq = sample['info'][2]
#                 digit_img_seq  = sample['digit_seq']
#                 query_pts = sample['query_pts']
#                 points_pc = sample['pts_pc']
#                 p_occ = sample['p_occ']

#                 ndf_input = {}
#                 ndf_input['coords'] = query_pts
#                 ndf_input['point_cloud'] = points_pc

#                 pocc_hat, _ = pl_module(info_seq, digit_img_seq, ndf_input)

                
                
#                 for i in range(num_data):
#                     info_seq = trajectory['info_frames'][i]
#                     digit_img_seq  = trajectory['digit_seq'][i]
#                     pc = trajectory['point_cloud']
#                     gt_contact = trajectory['gt_contact'][i]

#                     ndf_input = {}
#                     ndf_input['coords'] = pc
#                     ndf_input['point_cloud'] = pc

#                     pocc_hat, _ = pl_module(info_seq, digit_img_seq, ndf_input)

#                     filename = f'{self.path_save}/t{i}'

#                     gt_contact_array = gt_contact.squeeze().cpu().numpy()
#                     pc_array = pc.squeeze().cpu().numpy()
#                     pocc_hat_array = pocc_hat.squeeze().cpu().numpy()
#                     self.dataset.obj_model[obj_name].plot_external_contact(gt_contact_array, pc_array, pocc_hat_array, filename)
#                     # self.obj_model.plot_external_contact(gt, pts, pocc_hat, filename)
#                 # self.obj_model.close_plotter()
#                 pl_module.train()
  


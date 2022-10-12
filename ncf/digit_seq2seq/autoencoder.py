import numpy as np
import math
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# PyTorch Lightning
import pytorch_lightning as pl

# Torchvision
import torchvision
from torchvision import transforms

from digit_seq2seq.ConvLSTMCell import ConvLSTMCell



class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. 
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            # new
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2),
            act_fn()
        )

    def forward(self, x):
        x = x.float()
        x = self.net(x)
        return x


class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 act_fn : object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. 
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), 
            act_fn(),
            #new
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), 
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = x.float()
        x = self.net(x)
        return x



class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 num_hidden: int,
                 random_seq: bool = False):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.encoder = Encoder(num_input_channels, base_channel_size)

        self.encoder_1_convlstm = ConvLSTMCell(input_dim=base_channel_size*2,
                                               hidden_dim=num_hidden,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=num_hidden,
                                               hidden_dim=num_hidden,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=num_hidden,  # nf + 1
                                               hidden_dim=num_hidden,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=num_hidden,
                                               hidden_dim=num_hidden,
                                               kernel_size=(3, 3),
                                               bias=True)


        self.decoder_CNN = nn.Conv3d(in_channels=num_hidden,
                                     out_channels=base_channel_size*2,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))

        self.decoder = Decoder(num_input_channels, base_channel_size)

        self.random_seq = random_seq

        if random_seq:
            self. pos_encoding = self.positionalencoding1d(dim=128*40*60, length=128) #s5
            # self. pos_encoding = self.positionalencoding1d(dim=128*80*120, length=128) #s1

    def positionalencoding1d(self, dim, length):
        """
        :param d_seq: dimension of the window
        :param length: length of positions
        :return: length*d_seq position matrix
        """
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                            -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe


    def autoencoder(self, x, info_in_seq, seq_len, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4):

        outputs = []

        # encoder
        for t in range(seq_len):

            x_reduced = self.encoder(x[:, t, :, :])

            # if self.random_seq:
            #     # add positional encoding
            #     i = info_in_seq[:,-1] - info_in_seq[:,t]
            #     pos_enc = self.pos_encoding[i, :]
            #     pos_enc = torch.reshape(pos_enc, x_reduced.shape)
            #     x_reduced += pos_enc.to(x_reduced.device)



            h_t, c_t = self.encoder_1_convlstm(input_tensor=x_reduced,
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(seq_len):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4] 

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = nn.GELU()(outputs)
        outputs = outputs.permute(0, 2, 1, 3, 4)

        out_seq = []
        for t in range(seq_len):
            x_hat_red = outputs[:, t, :, :]
            out_seq += [self.decoder(x_hat_red)]

        out_seq = torch.stack(out_seq, 1)
        return encoder_vector, out_seq

    def forward(self, x, info_in_seq, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()
        # h = h//4
        # w = w//4
        h = h//8
        w = w//8

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))


        # autoencoder forward
        embedding, outputs = self.autoencoder(x, info_in_seq, seq_len, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return embedding, outputs



class DIGIT_Autoencoder(pl.LightningModule):

    def __init__(self,
                 base_channel_size: int,
                 num_hidden: int,
                 autoencoder_class : object = EncoderDecoderConvLSTM,
                 num_input_channels: int = 3,
                 random_seq = False,
                 width: int = 64,
                 height: int = 64):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()

        self.random_seq = random_seq

        # Create autoencoder
        self.autoencoder_anchor = autoencoder_class(num_input_channels, base_channel_size, num_hidden, random_seq)
        # self.autoencoder_key = autoencoder_class(num_input_channels, base_channel_size, num_hidden)
        # self.autoencoder_key.load_state_dict(self.autoencoder_anchor.state_dict())

        if random_seq:
            self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
            self.encoder_tau = 0.005
        self.coef = [500.0, 1.0]

        # Example input array needed for visualizing the graph of the network
        # self.example_input_array = torch.zeros(2, 6, width, height, num_input_channels)

    def forward(self, x, info_in_seq):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        # x = x.permute(0,1,4,2,3)
        z, x_hat = self.autoencoder_anchor(x, info_in_seq)
        # x_hat = x_hat.permute(0,1,3,4,2)
        return z, x_hat

    def encode(self, x, info_x, ema=False):
        x = x.permute(0,1,4,2,3)
        if ema:
            with torch.no_grad():
                # z_out, x_hat = self.autoencoder_key(x, info_x)
                z_out, x_hat = self.autoencoder_anchor(x, info_x)
        else:
            z_out, x_hat = self.autoencoder_anchor(x, info_x)
        x_hat = x_hat.permute(0,1,3,4,2)
        return z_out, x_hat

    def _get_reconstruction_loss(self, batch, mode='train'):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        # self.update_key_encoder()
        # if mode=='train':
        #     batch = self.dataset_train.get_batch()
        # else:
        #     batch = self.dataset_val.get_batch()

        anchor_info = batch['info'][1].float()
        anchor_seq  = batch['anchor_seq'].float()

        if self.random_seq:
            pos_info = batch['info'][2][0].float()
            pos_seq  = batch['pos_seq'][0].float()

            neg_info = batch['info'][3][0].float()
            neg_seq  = batch['neg_seq'][0].float()

        # y  = batch['out_seq']
        # y = y.type(torch.cuda.FloatTensor)
        _, x_hat = self.forward(anchor_seq, anchor_info)
        loss = F.mse_loss(anchor_seq, x_hat, reduction="none")
        loss_reconstruction = self.coef[1] * loss.sum(dim=[1,2,3,4]).mean(dim=[0]) # check dim t dim=[1,2,3,4]??

        if self.random_seq:
            z_anchor, _ = self.encode(anchor_seq, anchor_info)
            z_pos, _ = self.encode(pos_seq, pos_info, ema=True)
            z_neg, _ = self.encode(neg_seq, neg_info, ema=True)

            z_anchor = z_anchor[:,-1,:,:]
            z_pos = z_pos[:,-1,:,:]
            z_neg = z_neg[:,-1,:,:]

            loss_triplet = self.coef[0] * self.triplet_loss(z_anchor, z_pos, z_neg)

            loss = loss_triplet + loss_reconstruction
        else:
            loss_triplet = torch.tensor(0.0).to(loss_reconstruction.device)
            loss = loss_reconstruction

        return loss_reconstruction, loss_triplet, loss

    def update_key_encoder(self):
        for param, target_param in zip(self.autoencoder_key.parameters(), self.autoencoder_anchor.parameters()):
            target_param.data.copy_(
                self.encoder_tau * param.data + (1 - self.encoder_tau) * target_param.data
            )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=5,
                                                         min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss_reconstruction, loss_triplet, loss = self._get_reconstruction_loss(batch, mode='train')
        self.log('train_loss', loss)
        self.log('train_loss_reconstruction', loss_reconstruction)
        self.log('train_loss_triplet', loss_triplet)
        return loss

    def validation_step(self, batch, batch_idx):
        loss_reconstruction, loss_triplet, loss = self._get_reconstruction_loss(batch, mode='val')
        self.log('val_loss', loss)
        self.log('val_loss_reconstruction', loss_reconstruction)
        self.log('val_loss_triplet', loss_triplet)


class GenerateCallback(pl.Callback):

    def __init__(self, input, every_n_epochs=1):
        super().__init__()
        self.input_info = input[0]
        self.input_imgs = input[1] # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            i = np.random.randint(0, self.input_imgs.shape[0])
            # input_imgs = self.input_imgs[i,:,:,:,:].float()
            # input_info = self.input_info[i,:].float()
            input_imgs = self.input_imgs.float().to(pl_module.device)
            input_info = self.input_info.float().to(pl_module.device)
            # input_info = input_info.type(torch.cuda.LongTensor)
            with torch.no_grad():
                pl_module.eval()
                _, reconst_imgs = pl_module(input_imgs, input_info)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs.squeeze(), reconst_imgs.squeeze()], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)



            # i = np.random.randint(0, reconst_imgs.shape[0])
            # a = input_imgs[i].permute(0,3,1,2)
            # b = reconst_imgs[i].permute(0,3,1,2)
            # d = torch.zeros([1,a.shape[1], a.shape[2], a.shape[3]]).to(a.device)
            # c = torch.cat([a,d,b], dim=0)

            # # unnormalize = transforms.Normalize((-0.3 / 0.05), (1.0 / 0.05))
            # # c = unnormalize(c)
            # # grid = torchvision.utils.make_grid(c, nrow=c.shape[0])
            # grid = torchvision.utils.make_grid(c, nrow=c.shape[0], normalize=True, range=(-1,1))
            # # imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
            # # grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
            # trainer.logger.experiment.add_image(f"Reconstructions_ep{trainer.current_epoch}", grid, global_step=trainer.global_step)























# class DIGITFeatureExtractor(pl.LightningModule):

#     def __init__(self,
#                  checkpoint_path: str,
#                  base_channel_size: int,
#                  num_hidden: int,
#                  autoencoder_class : object = EncoderDecoderConvLSTM,
#                  num_input_channels: int = 3,
#                  random_seq: bool = True):
#         super().__init__()
#         self.random_seq = random_seq
#         # Saving hyperparameters of autoencoder
#         self.save_hyperparameters()
#         # load autoencoder model
#         self.model = self.load_model_from_checkpoint(checkpoint_path, base_channel_size, num_hidden, autoencoder_class, num_input_channels, random_seq)
#         self.model.encoder_2_convlstm.register_forward_hook(self.get_embedding_hook())
#         self.feature_vector = torch.empty(0)
#         # Example input array needed for visualizing the graph of the network
#         # self.example_input_array = torch.zeros(2, 6, 64, 64, num_input_channels)

#     def get_embedding_hook(self):
#         def fn(_, __, output):
#             h_t2, _ = output
#             # self.feature_vector = torch.flatten(h_t2.detach(), start_dim=1)
#             self.feature_vector = h_t2.detach()
#         return fn

#     def forward(self, x, info_in_seq):
#         self.model.eval()
#         with torch.no_grad():
#             x = x.permute(0,1,4,2,3)
#             _, x_hat = self.model(x, info_in_seq)
#             x_hat = x_hat.permute(0,1,3,4,2)
#         return self.feature_vector, x_hat

#     def load_model_from_checkpoint(self, checkpoint_path, base_channel_size, num_hidden, autoencoder_class, num_input_channels, random_seq):
#         checkpoint = torch.load(checkpoint_path, map_location="cpu")
#         autoencoder_anchor = autoencoder_class(num_input_channels, base_channel_size, num_hidden, random_seq)
#         state_dict = {}
#         for old_key in checkpoint['state_dict'].keys():
#             if 'autoencoder_anchor' in old_key:
#                 new_key = old_key[len('autoencoder_anchor.'):]
#                 state_dict[new_key] = checkpoint['state_dict'][old_key]
#         autoencoder_anchor.load_state_dict(state_dict)

#         for param in autoencoder_anchor.parameters():
#             param.requires_grad = False

#         return autoencoder_anchor
    
#     def _get_reconstruction_loss(self, batch):
#         """
#         Given a batch of images, this function returns the reconstruction loss (MSE in our case)
#         """
#         anchor_info = batch['info'][1]
#         anchor_seq  = batch['anchor_seq'].type(torch.cuda.FloatTensor)

#         embedding, y_hat = self.forward(anchor_seq, anchor_info)
#         loss = F.mse_loss(anchor_seq, y_hat, reduction="none")
#         loss = loss.sum(dim=[1,2,3,4]).mean(dim=[0]) # check dim t dim=[1,2,3,4]??
#         return loss, embedding, y_hat

#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer

#     def training_step(self, batch, batch_idx):
#         loss, _, _ = self._get_reconstruction_loss(batch)
#         self.log('train_loss', loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss, _, _ = self._get_reconstruction_loss(batch)
#         self.log('val_loss', loss)

#     def test_step(self, batch, batch_idx):
#         loss, emb, x_hat = self._get_reconstruction_loss(batch)
#         self.log('test_loss', loss)
#         for i in range(len(batch['seq'])):
#             a = batch['seq'][i].permute(0,3,1,2)
#             b = x_hat[i].permute(0,3,1,2)
#             d = torch.zeros([1,a.shape[1], a.shape[2], a.shape[3]]).to(a.device)
#             c = torch.cat([a,d,b], dim=0)
#             grid = torchvision.utils.make_grid(c, nrow=c.shape[0], normalize=True, range=(-1,1))
#             # imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
#             # grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
#             self.logger.experiment.add_image(f"Reconstructions_{i}", grid, global_step=i)





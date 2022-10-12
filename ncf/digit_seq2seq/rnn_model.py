import numpy as np
import random
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


class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        
       #  self.embedding = nn.Embedding(input_dim, emb_dim) #no dropout as only one layer!
        
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(src)
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, hidden = self.rnn(embedded) #no cell state!
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden


class Decoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, num_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        
       #  self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, num_layers)
        
       #  self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, context):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #context = [n layers * n directions, batch size, hid dim]
        
        #n layers and n directions in the decoder will both always be 1, therefore:
        #hidden = [1, batch size, hid dim]
        #context = [1, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(input)
        
        #embedded = [1, batch size, emb dim]
                
        emb_con = torch.cat((embedded, context[-1].unsqueeze(dim=0)), dim = 2)
            
        #emb_con = [1, batch size, emb dim + hid dim]
            
        output, hidden = self.rnn(emb_con, hidden)
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        
        #seq len, n layers and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [1, batch size, hid dim]
        
       #  output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), 
       #                     dim = 1) # yo lo quite
        
        #output = [batch size, emb dim + hid dim * 2]
        
       #  prediction = self.fc_out(output) # yo lo quite
        
        #prediction = [batch size, output dim]
        
        return output, hidden



class TactilePoseEncoder(pl.LightningModule):
    def __init__(self, autoencoder, encoder, decoder):
        super().__init__()

        self.autoencoder = autoencoder
        self.encoder = encoder
        self.decoder = decoder
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src, ee, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        src = src.permute(1,0,2,3,4)
        batch_size = src.shape[1]
        src_len = src.shape[0]
        trg_len = src.shape[0]
        emb_dim = self.decoder.emb_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, emb_dim).to(src.device)

        # get embeddings
        src_emb = torch.zeros(src_len, batch_size, emb_dim).to(src.device)
        emb_sz = None

        for i in range(src_len):
            s = src[i,:,:,:,:].unsqueeze(dim=0)
            self.autoencoder.eval()
            emb,_ = self.autoencoder(s.permute(1,0,2,3,4), [])
            emb_sz = emb.shape
            emb = torch.flatten(emb[:,-1,:,:], start_dim=1)
            seq = torch.cat((emb, ee[:,i,:]), dim=1)
            src_emb[i,:,:] = seq

        
        #last hidden state of the encoder is the context
        context = self.encoder(src_emb)
        
        #context also used as the initial hidden state of the decoder
        hidden = context
        
        #first input to the decoder is the <sos> tokens
        input = torch.zeros(batch_size, emb_dim).to(src.device)

        
        for t in range(1, src_len):
            
            #insert input token embedding, previous hidden state and the context state
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output[0]
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            # top1 = output
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = src_emb[t] if teacher_force else output[0]

        # outputs = outputs.permute(1,0,2)
        y_hat = torch.zeros(src_len-1, batch_size, emb_sz[2], emb_sz[3]).to(src.device)
        o = outputs[:,:, 0:emb_dim-7]
        for i in range(src_len-1):
            y_hat[i] = o[i+1].view(batch_size,emb_sz[2], emb_sz[3])
        y_hat = y_hat.permute(1,0,2,3)
        y_true = src_emb[0:src_len-1,:,0:600].permute(1,0,2)
        y_true = y_true.view(batch_size, src_len-1, emb_sz[2], emb_sz[3])
        return y_true, y_hat, context

    def _get_reconstruction_loss(self, batch):
        in_seq = batch["in_seq"]
        # out_seq = batch["out_seq"]
        ee = batch["ee_pose"]
        y_true, y_hat, context = self.forward(in_seq, ee)
        loss = F.mse_loss(y_true, y_hat, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        return loss

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
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss)

    
class GenerateCallback(pl.Callback):

    def __init__(self, input, every_n_epochs=1):
        super().__init__()
        self.in_seq = input[0]
        self.ee_pose = input[1] # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            # i = np.random.randint(0, self.in_seq.shape[0])
            in_s = self.in_seq.float().to(pl_module.device)
            in_pose = self.ee_pose.float().to(pl_module.device)
            # input_info = input_info.type(torch.cuda.LongTensor)
            with torch.no_grad():
                pl_module.eval()
                y_true, y_hat, context = pl_module(in_s, in_pose)
                pl_module.train()
            
            y_true = y_true.permute(1,0,2,3)
            y_hat = y_hat.permute(1,0,2,3)
            img_in = torch.cat([y_true[:,0], y_true[:,1], y_true[:,2], y_true[:,3], y_true[:,4]], dim=2)
            img_in = torch.cat([img_in[0], img_in[1], img_in[2], img_in[3]], dim=0)
            img_out = torch.cat([y_hat[:,0], y_hat[:,1], y_hat[:,2], y_hat[:,3], y_hat[:,4]], dim=2)
            img_out = torch.cat([img_out[0], img_out[1], img_out[2], img_out[3]], dim=0)
            # Plot and add to tensorboard
            # imgs = torch.cat([y_true.squeeze(), y_hat.squeeze()], dim=2)
            # grid = torchvision.utils.make_grid(imgs, nrow=5, normalize=False, range=(-1,1))
            trainer.logger.experiment.add_image("y_true", img_in.unsqueeze(dim=0), global_step=trainer.global_step)
            trainer.logger.experiment.add_image("y_hat", img_out.unsqueeze(dim=0), global_step=trainer.global_step)
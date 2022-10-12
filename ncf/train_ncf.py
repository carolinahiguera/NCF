import os
import shutil
import hydra

# torch
import torch
import torch.utils.data as data
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# custom
from data.dataio_train import ExternalContactDataset
from digit_seq2seq.autoencoder import DIGIT_Autoencoder
from digit_seq2seq.rnn_model import Encoder, Decoder, TactilePoseEncoder
import ndf_model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ncf_model.ncf_model import ExternalContactModel


@hydra.main(config_path=f"conf", config_name="conf_scenes_test")
def main(cfg):
    # NCF parameters
    n_query_pts = 3500
    n_pc_pts =  5000
    num_hidden_lstm = 6
    c_dim = 128
    batch_sz = 5
    debug_dataset = False

    # Setting the seed
    seed = 42
    pl.seed_everything(seed)

    # check if gpu available
    dev = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # train and validation dataset
    path_root = os.path.dirname(os.path.abspath("."))
    path_data = f"{path_root}/data_collection/train_data/" 
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,))])

    dataset = ExternalContactDataset(root_path=path_root,
                                     data_path=path_data, 
                                     num_pc_pts=n_pc_pts, 
                                     num_query_pts=n_query_pts, 
                                     debug=debug_dataset, 
                                     transform=transform)
    n = dataset.__len__()
    n_train = int(n*0.7)
    n_val = n-n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)) 

    train_loader = data.DataLoader(train_set, batch_size=batch_sz, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)
    val_loader =   data.DataLoader(val_set,   batch_size=batch_sz, shuffle=False, drop_last=False, num_workers=4)
    print("[OK] Train and validation dataloaders ready")

    # load digit encoder    
    base_path_digit = f"{path_root}/ncf/digit_seq2seq/models_checkpoints/"
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
    contact_model = ExternalContactModel(   num_hidden_lstm=num_hidden_lstm,
                                            c_dim = c_dim,
                                            seq_len= 1,
                                            digit_feature_extractor=digit_seq2seq,
                                            ndf_model = ndf_model
                                            )
    contact_model.to(dev)

    # create train data logger
    logger = TensorBoardLogger('train_log', name=f'ncf_model')
    print(f"[INFO] NCF checkpoint will be saved in {path_root}/ncf/train_log/ncf_model/")
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(logger=logger,
                         gpus=1 if str(dev).startswith("cuda") else 0,
                         strategy="ddp",
                         max_epochs=201,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = False         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # train external contact model
    print("[INFO] Starting NCF training ...")
    trainer.fit(contact_model, train_loader, val_loader)
    print("*** END ***")


if __name__ == '__main__':
    path_root = os.path.dirname(os.path.abspath("."))
    src = path_root + "/data_collection/conf"
    dest = path_root + "/ncf/conf"
    try:
        destination = shutil.copytree(src, dest) 
    except:
        print("[INFO] Config files are already in the ncf directory. Good to go.")
    main()
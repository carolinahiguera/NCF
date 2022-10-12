import os
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.dataio_seq2seq import DIGIT_Dataset
from digit_seq2seq.autoencoder import DIGIT_Autoencoder
from digit_seq2seq.rnn_model import Encoder, Decoder, TactilePoseEncoder, GenerateCallback


def get_val_images(num):
    idx = np.random.randint(0, val_set.__len__()-1, num)
    in_seq = torch.stack([val_set[idx[i]]['in_seq'] for i in range(num)], dim=0)
    ee_pose = torch.stack([val_set[idx[i]]['ee_pose'] for i in range(num)], dim=0)
    return (in_seq, ee_pose)

if __name__ == '__main__':
    batch_sz = 8

    # Setting the seed
    seed = 42
    pl.seed_everything(seed)

    # check if gpu available
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # train and validation dataset
    path_root = os.path.dirname(os.path.abspath("."))
    path_data = f"{path_root}/data_collection/train_data/" 
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,))])

    train_dataset = DIGIT_Dataset(root_path=path_root, data_path=path_data, transform=transform)
    n = train_dataset.__len__()
    n_train = int(n*0.7)
    n_val = n-n_train
    train_set, val_set = torch.utils.data.random_split(train_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    train_loader = data.DataLoader(train_set, batch_size=batch_sz, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)
    val_loader =   data.DataLoader(val_set,   batch_size=batch_sz, shuffle=False, drop_last=False, num_workers=4)
    print("[OK] Train and validation dataloaders ready")

    # load digit encoder    
    base_path_digit = f"{path_root}/ncf/digit_seq2seq/models_checkpoints/"
    digit_embeddings_path = f"{base_path_digit}/digit_embs_weights.pth"
    digit_embeddings_model = DIGIT_Autoencoder(base_channel_size=64, num_hidden=6)
    digit_embeddings_model.load_state_dict(torch.load(digit_embeddings_path, map_location=device))
    for param in digit_embeddings_model.parameters():
        param.requires_grad = False
    digit_embeddings_model.eval()
    print("[OK] Loaded DIGIT embeddings model")

    # define digit seq2seq model
    emb_sz = (20*30) + 7 # this is the dimension of the embeddings
    encoder = Encoder(emb_dim=emb_sz, hid_dim=emb_sz, num_layers=2, dropout=0.8).to(device)
    decoder = Decoder(emb_dim=emb_sz, hid_dim=emb_sz, num_layers=2, dropout=0.8).to(device)
    digit_seq2seq = TactilePoseEncoder(digit_embeddings_model, encoder, decoder).to(device)
    print("[OK] DIGIT seq2seq autoencoder defined")

    # create train data logger
    logger = TensorBoardLogger('train_log', name=f'digit_seq2seq')
    # create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(   logger=logger,
                            gpus=1 if str(device).startswith("cuda") else 0,
                            max_epochs=51,
                            callbacks=[ ModelCheckpoint(save_weights_only=True),
                                        GenerateCallback(get_val_images(5), every_n_epochs=5),
                                        LearningRateMonitor("epoch")]
                        )
    trainer.logger._log_graph = True         
    trainer.logger._default_hp_metric = None 

    # train external contact model
    print("[INFO] Starting DIGIT seq2seq autoencoder training ...")
    trainer.fit(digit_seq2seq, train_loader, val_loader)
    

    print("*** END ***")
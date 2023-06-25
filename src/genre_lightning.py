import os, sys, time, pickle
from pathlib import Path
from collections import Counter
from IPython.display import Audio


PROJECT_DIR = Path(sys.path[0])/".."
DATA_DIR = PROJECT_DIR/"data"
SRC_DIR = PROJECT_DIR/"src"
DEPS_DIR = PROJECT_DIR/"deps"

sys.path.append(str(SRC_DIR))
sys.path.append(str(DEPS_DIR))

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from fairseq_wav2vec import Wav2Vec2Config, Wav2Vec2Model
import pandas as pd

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from mugen_train import (
    musicDataset, IDX_to_GENRE, GENRE_TO_IDX, 
    get_music_data_loaders, GenreClassifier, 
#     train_genre_predict, validation_genre_predict,
    GenreEncoder,
    get_num_params,
    get_lr)

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    feature_size = 128
    n_genres = 50
    top50_genre_samples_df = pd.read_csv("/n1Tb/sc_mp3_top50_genre_samples.tsv_gz", compression='gzip', sep='\t')

    train_loader, valid_loader = get_music_data_loaders(top50_genre_samples_df, cut=8e4)

    # precision = 16

    encoder = Wav2Vec2Model(Wav2Vec2Config)

    t_specs = {
    #     "d_model": 128, 
        "dim_feedforward": 64, 
        "num_decoder_layers": 0,
        "num_encoder_layers": 0,
        "nhead": 1,
    }
    decoder = GenreClassifier(n_genres, feature_size, t_specs=t_specs, use_transformer=False)

    print(f"wav2vec-model n_params: {get_num_params(encoder):,d}")
    print(f"genre_classifier n_params: {get_num_params(decoder):,d}")

    genre_predictor = GenreEncoder(encoder, decoder)

    # torch.multiprocessing.set_start_method('spawn')

    checkpoint_callback = ModelCheckpoint(
        dirpath="/home/keld/projects/music_gen/mugen_ml/train_checkpoints", 
        every_n_epochs=1, 
        monitor="train_loss")

    trainer = pl.Trainer(
        # limit_train_batches=1000, 
        max_epochs=3, 
        accelerator="gpu", 
        devices=2,
        strategy='ddp',     
        default_root_dir="/home/keld/projects/music_gen/mugen_ml/train_checkpoints",
        callbacks=[checkpoint_callback]
        # strategy='ddp_notebook',     
    #     precision='16-mixed'
    #     precision=16, 
    #     devices=2, 
    )
    
    optimum_lr = get_lr(trainer, genre_predictor, train_loader, plot=True)
    genre_predictor.lr = optimum_lr

    trainer.fit(model=genre_predictor, train_dataloaders=train_loader)
import os

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import matplotlib.pylab as plt
from audio_lib import Mp3Stream

N_CPU_CORES = os.cpu_count()

def get_lr(trainer, model, train_loader, plot=False):
    tuner = pl.tuner.tuning.Tuner(trainer)
    out_lr_finder = tuner.lr_find(model=model, train_dataloaders=train_loader)

    lrf_lr = out_lr_finder.results["lr"]
    lrf_loss = out_lr_finder.results["loss"]
    opt_idx = out_lr_finder._optimal_idx
    if plot:
        plt.plot(lrf_lr, lrf_loss);
        plt.plot(lrf_lr[opt_idx], lrf_loss[opt_idx], "x", label="optimum");
        plt.legend()

        plt.xscale("log");
    print(f"Optimal lr: {lrf_lr[opt_idx]}")        
    return lrf_lr[opt_idx]

class GenreEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, lr=1e-3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.BCELoss = torch.nn.BCELoss()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        lpcms, genres = batch
        outputs = self.encoder(lpcms)
        z = outputs["x"]
        z, genres = remove_inf_rows(z, genres)
        
        genre_props = self.decoder(z)

        # genre prediction loss
        loss = self.BCELoss(genre_props, genres)

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)
        return optimizer

class GenreClassifier(nn.Module):
    def __init__(self, n_classes, feature_size, t_specs={}, use_transformer=True):
        super().__init__()
        self.feature_size = feature_size
        self.use_transformer = use_transformer
        self.encoder_out_size = 101

        if self.use_transformer:
            self.genre_transformer = nn.Transformer(
                **{**t_specs, **{"d_model": feature_size}}
            )

            self.fc = nn.Linear(feature_size, n_classes)
        else:
            self.fc = nn.Linear(self.encoder_out_size, n_classes)

    def forward(self, x):
        # shape is now batch, timesteps, layer features
        x = x.permute(1, 2, 0)
        
        if self.use_transformer:
            # Add padding so that it works with transformer 
            # adding padding to get to feature size 
            x = torch.pad(x, (0, self.feature_size-101))
            x = self.genre_transformer(x, x)
        x = x.mean(dim=1)
        prob = torch.sigmoid(self.fc(x))
        return prob

def remove_inf_rows(x, genres, debug=False):
    # discard where there is inf values 
    num_inf = torch.sum(torch.isinf(x))
    if num_inf:
        ignore_samples = torch.where(x.isinf())[1].unique()
        filter_samples = [i not in ignore_samples for i in range(x.shape[1])]
        x = x[:, filter_samples, :]
        genres = genres[filter_samples, :]
        # [genre for i, genre in enumerate(genres) if i not in ignore_samples]
        if debug:
            print(f"num_inf in output of model {num_inf}")        
            print(f"Num of ignored samples: {len(ignore_samples)}") 
    return x, genres

def get_music_data_loaders(
        music_df, batch_size=12, n_size=1e5, cut=100000, random_mix_stereo=True):
    train_ds = musicDataset(
        music_df, n_size=n_size, cut=cut, random_mix_stereo=random_mix_stereo)
    valid_ds = musicDataset(
        music_df[:100], n_size=n_size, cut=cut, offset=2e6, random_mix_stereo=random_mix_stereo)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=music_collate_fn_wrap(cut),
        num_workers=N_CPU_CORES,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=music_collate_fn_wrap(cut),
        num_workers=N_CPU_CORES,
    )
    return train_loader, valid_loader


def music_collate_fn_wrap(cut):
    def music_collate_fn(x):
        features = []
        genres = []
        for xi, genre in x:
            if  xi is not None:
                if len(xi.shape) == 1 and xi.shape[0] >= cut:
                    # if we are pulling mono channels
                    features.append(xi[None]) # add shape dimension to support concat
                    genres.append(genre)
                elif len(xi.shape) == 2 and xi.shape[1] >= cut:
                    # if we are using two channels (stereo)
                    features.append(xi)
                    # need to make two entries, one for each channel
                    genres.append(genre)
                    genres.append(genre)

        return torch.concat(features), torch.concat(genres)

    return music_collate_fn

class musicDataset(Dataset):
    def __init__(
        self,
        data_df,
        n_size=1e5,
        cut=100000,
        offset=None,
        start_cut=1000,
        random_mix_stereo=False,
        n_genres=50,
    ):
        self.data_df = data_df
        self.n_size = n_size
        self.cut = int(cut)
        self.start_cut = int(start_cut)
        self.offset = offset
        self.random_mix_stereo = random_mix_stereo
        self.n_genres = n_genres

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        song_info = self.data_df.iloc[idx]
        genre_idx = GENRE_TO_IDX[song_info["genre"]]

        genre_one_hot = (
            torch.nn.functional.one_hot(torch.tensor([genre_idx]), self.n_genres)
            .requires_grad_(False)
            .type(torch.float)
        )
        
        song = Mp3Stream(song_info["nvme_fp"])
        if self.offset is None:
            d = song.np_decode_rand_offset(self.n_size)
        else:
            d = song.np_decode(self.n_size, offset=int(self.offset))

        if d is not None and d.shape[1] not in {1, 2}:
            print(d.shape)
            raise ValueError("More than 2 channels! Need a way to handle.")            

        if d is not None and d.shape[1] == 2:
            # Remove a bit of the beginning as the first 250 samples are always around zero
            # start_cut should be >250
            d = torch.tensor(d).movedim(0, -1)[
                :, self.start_cut : self.cut + self.start_cut
            ]
            
            if self.random_mix_stereo and d.shape[0] == 2:
                LR_mix_ratio = np.random.random()
                d = LR_mix_ratio * d[0] + (1 - LR_mix_ratio) * d[1]

        return d, genre_one_hot 

def get_num_params(model):
    return np.sum(
        np.fromiter(
            (
                torch.prod(torch.tensor(param.size())).item() 
                for name, param in model.named_parameters()), 
            dtype=int))

GENRE_TO_IDX = {
    "edm": 0,
    "hiphop": 1,
    "soundtrack": 2,
    "house": 3,
    "classical": 4,
    "rock": 5,
    "drum & bass": 6,
    "ambient": 7,
    "dubstep": 8,
    "trap": 9,
    "alternative": 10,
    "r&b": 11,
    "trance": 12,
    "piano": 13,
    "reggae": 14,
    "jazz": 15,
    "alternative rock": 16,
    "new age": 17,
    "psytrance": 18,
    "indie": 19,
    "folk": 20,
    "funk": 21,
    "world": 22,
    "country": 23,
    "metal": 24,
    "jazz & blues": 25,
    "instrumental": 26,
    "disco": 27,
    "latin": 28,
    "video game music": 29,
    "bass house": 30,
    "bass": 31,
    "contemporary classical": 32,
    "future bass": 33,
    "orchestral": 34,
    "beat & instrumental": 35,
    "religion & spirituality": 36,
    "melodic techno": 37,
    "trailer": 38,
    "experimental": 39,
    "chill": 40,
    "soul": 41,
    "epic": 42,
    "reggaeton": 43,
    "lofi": 44,
    "future garage": 45,
    "drumstep": 46,
    "cinematic": 47,
    "ethereal techno": 48,
    "nu disco": 49,
}

IDX_to_GENRE = dict(zip(GENRE_TO_IDX.values(), GENRE_TO_IDX.keys()))

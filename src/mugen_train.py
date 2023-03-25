import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from audio_lib import Mp3Stream


def train(model, optimizer, data_loader):
    model.train()
    running_loss = 0.0
    for i, (batch, genres) in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
        optimizer.zero_grad()
        outputs = model(batch)
        loss = outputs["features_pen"]
        loss.backward()
        optimizer.step()
        running_loss += loss.to("cpu").item()
    return running_loss / len(data_loader)


def validation(model, data_loader):
    model.eval()
    running_loss = 0.0
    for i, (batch, genres) in tqdm(enumerate(data_loader, 0), total=len(data_loader)):
        outputs = model(batch)
        loss = outputs["features_pen"]
        running_loss += loss.to("cpu").item()
    return running_loss / len(data_loader)


def get_music_data_loaders(music_df, batch_size=6, n_size=1e5, cut=100000):
    train_ds = musicDataset(music_df, n_size=n_size, cut=cut)
    valid_ds = musicDataset(music_df[:100], n_size=n_size, cut=cut, offset=2e6)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=music_collate_fn_wrap(cut),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=music_collate_fn_wrap(cut),
    )
    return train_loader, valid_loader


def music_collate_fn_wrap(cut):
    def music_collate_fn(x):
        features = []
        genres = []
        for xi, genre in x:
            if xi is not None and xi.shape[1] >= cut:
                features.append(xi)
                # need to make two entries, one for each channel
                genres.append(genre)
                genres.append(genre)
        return torch.concat(features), genres

    return music_collate_fn


class musicDataset(Dataset):
    def __init__(
        self,
        data_df,
        n_size=1e5,
        cut=100000,
        device="cuda",
        offset=None,
        start_cut=1000,
    ):
        self.device = device
        self.data_df = data_df
        self.n_size = n_size
        self.cut = cut
        self.start_cut = start_cut
        self.offset = offset

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        song_info = self.data_df.iloc[idx]
        genre_idx = GENRE_TO_IDX[song_info["genre"]]
        song = Mp3Stream(song_info["nvme_fp"])
        if self.offset is None:
            d = song.np_decode_rand_offset(self.n_size)
        else:
            d = song.np_decode(self.n_size, offset=int(self.offset))
        if d is not None:
            # Remove a bit of the beginning as the first 250 samples are always around zero
            # start_cut should be >250
            d = torch.tensor(d).movedim(0, -1)[
                :, self.start_cut : self.cut + self.start_cut
            ]
            d = d.to(self.device)
        return d, genre_idx


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

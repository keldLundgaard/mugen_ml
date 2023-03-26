import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from audio_lib import Mp3Stream

def get_batch_genre_one_hot(genres, n_genres, device):
        return (
            torch.nn.functional.one_hot(torch.tensor(genres), n_genres)
            .requires_grad_(False)
            .type(torch.float)
            .to(device)
        )


class MusicCNN(nn.Module):
    def __init__(self, n_classes):
        super(MusicCNN, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Define pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(in_features=128*4*4, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=n_classes)
        
    def forward(self, x):
        # Perform convolutional layers and pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        # Flatten output for fully connected layers
        x = x.view(-1, 128*4*4)
        
        # Perform fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

class GenreClassifier(nn.Module):
    def __init__(self, n_classes, feature_size, t_specs={}, use_transformer=True):
        super().__init__()
        self.feature_size = feature_size
        self.use_transformer = use_transformer

        if self.use_transformer:
            self.genre_transformer = nn.Transformer(
                **{**t_specs, **{"d_model": feature_size}}
            )

        self.fc = nn.Linear(feature_size, n_classes)

    def forward(self, x):
        # Add padding so that it works with transformer 
        zero_padding = torch.zeros((self.feature_size - x.shape[0], *x.shape[1:])).to("cuda")
        x = torch.cat((x, zero_padding), dim=0)

        # shape is now batch, timesteps, layer features
        x = x.permute(1, 2, 0)

        if self.use_transformer:
            x = self.genre_transformer(x, x)
        x = x.mean(dim=1)
        prob = torch.sigmoid(self.fc(x))
        return prob


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


def get_music_data_loaders(music_df, batch_size=12, n_size=1e5, cut=100000, random_mix_stereo=True):
    train_ds = musicDataset(
        music_df, n_size=n_size, cut=cut, random_mix_stereo=random_mix_stereo)
    valid_ds = musicDataset(
        music_df[:100], n_size=n_size, cut=cut, offset=2e6, random_mix_stereo=random_mix_stereo)

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
        random_mix_stereo=False,
    ):
        self.device = device
        self.data_df = data_df
        self.n_size = n_size
        self.cut = int(cut)
        self.start_cut = int(start_cut)
        self.offset = offset
        self.random_mix_stereo = random_mix_stereo

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
            if self.random_mix_stereo:
                LR_mix_ratio = np.random.random()
                d = LR_mix_ratio * d[0] + (1 - LR_mix_ratio) * d[1]
            d = d.to(self.device)
        return d, genre_idx


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

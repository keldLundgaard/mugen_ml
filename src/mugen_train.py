import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from audio_lib import Mp3Stream 


# Data
class musicDataset(Dataset):
    def __init__(self, data_df, n_size=1e5, cut=100000, device="cuda"):
        self.device=device
        self.data_df = data_df
        self.n_size = n_size
        self.cut = cut

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        song = Mp3Stream(self.data_df.iloc[idx]["nvme_fp"])
        d = song.np_decode_rand_offset(self.n_size)
        if d is not None:
            d = torch.tensor(d).movedim(0,-1)[:, :self.cut]
            d = d.to(self.device)
        return d




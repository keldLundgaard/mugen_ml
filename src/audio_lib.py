import os

import numpy as np
from IPython.display import Audio

from miniaudio import mp3_read_f32, DecodeError

class Mp3Stream:
    """
    Simple and fast python class to decode and play mp3 files.
    """
    
    def __init__(self, fp):
        self.fp = fp
        self.decoded = None
        self.rate = None
        self.fsize = os.path.getsize(self.fp)
    
    def np_decode_rand_offset(self, n_size): 
        offset = int(np.random.random() * (self.fsize-(n_size + n_size/2)))
        return self.np_decode(n_size, offset)

    def np_decode(self, n_size=None, offset=0):
        decoded = self.decode(n_size, offset)
        if decoded is not None:
            return np.array(decoded.samples, dtype=np.float32).reshape((-1, decoded.nchannels))
        
    def decode(self, n_size=None, offset=0):
        n_size = int(n_size or self.fsize)  # defaults to fsize 
        assert n_size > 0
        try:
            with open(self.fp, "rb") as f:
                f.seek(offset, 0)
                decoded = mp3_read_f32(f.read(n_size))
            self.decoded = decoded
            self.sample_rate = decoded.sample_rate
            return decoded
        except DecodeError:
            pass 

    def play(self, autoplay=False):
        return Audio(self.fp, autoplay=autoplay)
    

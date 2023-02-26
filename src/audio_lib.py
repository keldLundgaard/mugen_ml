from miniaudio import mp3_read_f32
from IPython.display import Audio
import numpy as np

class Mp3Stream:
    """
    Simple and fast python class to decode and play mp3 files.
    """
    
    def __init__(self, fp):
        self.fp = fp
        self.decoded = None
        self.rate = None
        
    def np_decode(self, n_size=None, offset=0):
        decoded = self.decode(n_size, offset)
        return np.array(decoded.samples, dtype=np.float32).reshape((-1, decoded.nchannels))
        
    def decode(self, n_size=None, offset=0):
        if n_size is None: 
            n_size = os.path.getsize(self.fp)
        assert n_size > 0
        n_size = int(n_size)
        with open(self.fp, "rb") as f:
            f.seek(offset, 0)
            decoded = mp3_read_f32(f.read(n_size))
        self.decoded = decoded
        self.sample_rate = decoded.sample_rate
        return decoded

    def play(self, autoplay=False):
        return Audio(self.fp, autoplay=autoplay)
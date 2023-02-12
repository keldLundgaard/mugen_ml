from miniaudio import mp3_read_s16
import numpy

class Mp3Stream:
	"""
	Small simple python class to open, load, and play mp3 files.
	"""
    
    def __init__(self, fp):
        self.fp = fp
        self.decoded = None
        self.rate = None
        
    def np_decode(self, n_size, offset):
        decoded = self.decode(n_size, offset)
        return np.array(decoded.samples, dtype=numpy.int16).reshape((-1, decoded.nchannels))
        
    def decode(self, n_size, offset):
        with open(self.fp, "rb") as f:
            f.seek(offset, 0)
            decoded = mp3_read_s16(f.read(n_size))
        self.decoded = decoded
        self.sample_rate = decoded.sample_rate
        return decoded

    def play(self, n_size=-1, offset=0, autoplay=False):
        npa = self.np_decode(n_size, offset)
        return Audio([npa[:, 0], npa[:, 1]],rate=self.sample_rate, autoplay=autoplay)
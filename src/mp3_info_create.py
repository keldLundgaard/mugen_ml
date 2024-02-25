import time 
import random
import pickle
import os
import sys

from tqdm import tqdm, trange
import pandas as pd
import eyed3

DATA_DIR = "/n4Ta/sc_webapp_data/"

fname = "mp3_r32_info_dct.pcl"
path = DATA_DIR + fname
if os.path.exists(path):
    mp3_r32_info_dct = pickle.load(open(path, "rb"))
    mp3_files = mp3_r32_info_dct["filepaths"]
else:
    user_path = "/r32Ta/soundcloud_music/files/"
    mp3_files = []
    for root, dirs, files in os.walk(user_path):
        for file in files:
            if file.endswith(".mp3"):
                mp3_files.append(os.path.join(root, file))
    mp3_r32_info_dct = {"filepaths": mp3_files}
    pickle.dump(mp3_r32_info_dct, open(path, "wb"))

eyed3.log.setLevel("ERROR")
def extract_mp3_info(path):
    try:
        audiofile = eyed3.load(path)
        
        if audiofile is not None and audiofile.tag is not None:
            genre_dct = {}
            if audiofile.tag.genre is None:
                genre_dct = {
                    'genre': None,
                    'genre_id': None,                
                }
                
            else:
                genre_dct = {
                    'genre': audiofile.tag.genre._name,
                    'genre_id': audiofile.tag.genre._id,                
                }
                
            
            return pd.Series(
                {
                    **{
                'title': audiofile.tag.title,
                'artist': audiofile.tag.artist,
                'album': audiofile.tag.album,
                'album artist': audiofile.tag.album_artist,
                'track number': audiofile.tag.track_num[0] if audiofile.tag.track_num else None,
                'release year': str(audiofile.tag.getBestDate())[:4] if audiofile.tag.getBestDate() else None,
                'duration (seconds)': audiofile.info.time_secs if audiofile.info else None,
                'bitrate (kbps)': audiofile.info.bit_rate[1] if audiofile.info and audiofile.info.bit_rate else None
                },
                    **genre_dct
                
                })
    except Exception as e:
        print(f"Error processing {path}: {e}")
    return pd.Series({
        'title': None, 'artist': None, 'album': None, 'album artist': None,
        'track number': None, 'genre': None, 'genre_id': None, 'release year': None, 'duration (seconds)': None,
        'bitrate (kbps)': None
    })

df = pd.DataFrame.from_dict({'paths': mp3_files}).astype({"paths": "string[pyarrow]"})
df["filename"]=df["paths"].apply(lambda x: x.split("/")[-1].split(".mp3")[0])

proc_df = df

out = []
for i, path in tqdm(enumerate(proc_df["paths"].values), total=len(proc_df)):
    out.append(extract_mp3_info(path))
mp3_info_df =  pd.DataFrame(out)    
w_info_df = pd.concat([proc_df, mp3_info_df], axis=1)
w_info_df.to_csv(DATA_DIR+"df_with_mp3_info.tsv.gz", sep="\t", compression="gzip")

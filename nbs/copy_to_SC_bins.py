import shutil
import pickle
import os 
from tqdm import tqdm

bins_content = pickle.load(open("bin_packing.pckl", "rb"))

bin_range = list(range(1, 11))
out_folder = "/usb-20Ta/soundcloud_bins/"

if not os.path.exists(out_folder):
    os.mkdir(out_folder)
    
for k, user_folders in bins_content.items():
    bin_num = int(k.split("_")[1])
    if bin_num in bin_range:
        print(f"operating on {k}")
        bin_path = out_folder + k
        if not os.path.exists(bin_path):
            os.mkdir(bin_path)
        for user_folder in tqdm(user_folders, total=len(user_folders)):
            # user_folder will have the / when split in this way
            dest = bin_path + user_folder.split("files")[1]
            shutil.copytree(user_folder, dest)
    else:
        pass
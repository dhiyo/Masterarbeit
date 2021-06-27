import os
import pickle
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
# from utils.mypath import MyPath
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


data = []
targets = []
file_path = r'C:\Users\shang\Downloads\cifar-10-batches-py\data_batch_1'
checksum = 'c99cafc152244af753f735de768cd75f'
with open(file_path, 'rb') as f:
    if sys.version_info[0] == 2:
        entry = pickle.load(f)
    else:
        entry = pickle.load(f, encoding='latin1')
    data.append(entry['data'])
    if 'labels' in entry:
        targets.extend(entry['labels'])
    else:
        targets.extend(entry['fine_labels'])

data = np.vstack(data).reshape(-1, 3, 32, 32)
data = data.transpose((0, 2, 3, 1))  # convert to HWC
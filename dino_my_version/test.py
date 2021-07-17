import os
import argparse
import numpy as np
import torch
import math
import os
import sys
import tempfile
from PIL import Image, ImageFilter
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torchvision import transforms as pth_transforms
from sklearn.cluster import KMeans
import shutil
import utils
import cv2
import vision_transformer as vits
from PIL import Image
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

from matplotlib import pyplot as plt
import utils
import main_dino
import vision_transformer as vits
from vision_transformer import DINOHead

state_dict = torch.load(r'C:\Users\shang\Downloads\checkpoint_uiqa_resnet50_200out_200epochs.pth',map_location=torch.device('cpu'))

# data_path = r'C:\Users\shang\Documents\yueqi_ws\Masterarbeit\Datasets\uiqa_serie\uiqa\uiqa_Ausschnitte'
# areas = []
# img_list = os.listdir(data_path)
# for img in img_list:
#     img_path = data_path + '/' + img
# # img1_path = r'C:\Users\shang\Documents\yueqi_ws\Masterarbeit\Datasets\uiqa_serie\uiqa\uiqa_Ausschnitte\label5576.png'
#     img = Image.open(img_path)
#     area = img.size[0] * img.size[1]
#     areas.append(area)
#
#
# print(len(areas))
# print(max(areas))
# plt.hist(areas, bins=100)
# plt.show()


# global_crops_scale = (0.4, 1.)
# local_crops_scale = (0.05, 0.4)
# local_crops_number = 8
#
#
# class ReturnIndexDataset(datasets.ImageFolder):
#     def __getitem__(self, idx):
#         img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
#         return img, idx
#
#
# class DataAugmentationDINO(object):
#     def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
#         flip_and_color_jitter = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomApply(
#                 [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
#                 p=0.8
#             ),
#             transforms.RandomGrayscale(p=0.2),
#         ])
#         normalize = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ])
#
#         # first global crop
#         self.global_transfo1 = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             utils.GaussianBlur(1.0),
#             normalize,
#         ])
#         # second global crop
#         self.global_transfo2 = transforms.Compose([
#             transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             utils.GaussianBlur(0.1),
#             utils.Solarization(0.2),
#             normalize,
#         ])
#         # transformation for the local small crops
#         self.local_crops_number = local_crops_number
#         self.local_transfo = transforms.Compose([
#             transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
#             flip_and_color_jitter,
#             utils.GaussianBlur(p=0.5),
#             normalize,
#         ])
#
#     def __call__(self, image):
#         crops = []
#         crops.append(self.global_transfo1(image))
#         crops.append(self.global_transfo2(image))
#         for _ in range(self.local_crops_number):
#             crops.append(self.local_transfo(image))
#         return crops
#
#
#
# # ============ preparing data ... ============
#
# dist.init_process_group(backend="nccl")
# # utils.fix_random_seeds(args.seed)
# transform = DataAugmentationDINO(
#     global_crops_scale,
#     local_crops_scale,
#     local_crops_number,
# )
# dataset = datasets.ImageFolder(data_path, transform=transform)
# sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
# data_loader = torch.utils.data.DataLoader(
#     dataset,
#     sampler=sampler,
#     batch_size=64,
#     num_workers=10,
#     pin_memory=True,
#     drop_last=True,
# )
# print(f"Data loaded: there are {len(dataset)} images.")
#


print()

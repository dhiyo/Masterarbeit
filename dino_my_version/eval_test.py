import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import shutil
import matplotlib.pyplot as plt
import numpy


data_path = r'C:\Users\shang\Documents\yueqi_ws\Masterarbeit\Datasets\ssd100_original_mini\train\0'
imgs_size = {}

img_list = os.listdir(data_path)
# img = numpy.empty(len(img_list), dtype=float, order='F')

for img_name in img_list:
    img = Image.open(os.path.join(data_path, img_name))
    area = img.size[0] * img.size[1]
    aspect_ratio = round(img.size[0] / img.size[1] , 1)
    imgs_size[img_name] = [ aspect_ratio, area]

print(imgs_size)




# for i in tqdm(range(len(imgs))):
#     for j in range(i+1, len(imgs)):
#         if j not in invalid_index:
#             if imgs[i].shape[0] == imgs[j].shape[0] and imgs[i].shape[1] == imgs[j].shape[1]:
#                 diff = imgs[i]-imgs[j]
#                 if diff.all() == 0:
#                     invalid_index.append(j)
#                     print(img_list[i],img_list[j])
#                     shutil.move(dir_path+img_list[j], 'duplicate/')
#         else:
#             continue

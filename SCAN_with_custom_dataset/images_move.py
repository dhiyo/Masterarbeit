import os, shutil
import  re
import pickle
import sys
import numpy as np
import torch
from PIL import Image

classes = ['label', 'hint_text', 'textfield', 'range', 'switch', 'textarea', 'validation_text', 'button', 'radio_button', 'checkbox']
image_name = os.listdir(r'C:\Users\shang\Documents\yueqi_ws\Masterarbeit\SCAN_with_custom_dataset\uiqa_Ausschnitte\train')
for i in range(len(image_name)):
    num_index = re.search(r'\d', image_name[i]).start()
    label = image_name[i][:num_index]
    label_index = classes.index(label)
    shutil.copyfile(r'C:\Users\shang\Documents\yueqi_ws\Masterarbeit\SCAN_with_custom_dataset\uiqa_Ausschnitte\train/' + image_name[i], r'C:\Users\shang\Documents\yueqi_ws\Masterarbeit\SCAN_with_custom_dataset\PNG-2-CIFAR10-master\classes/' + str(label_index) +'/' + image_name[i])


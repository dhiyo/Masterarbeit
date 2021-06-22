
import os, shutil
import  re
import pickle
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.mypath import MyPath


class UIQA(Dataset):

    def __init__(self, root=MyPath.db_root_dir('uiqa'), train=True, transform=None):

        super(UIQA, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.classes = ['label', 'hint_text', 'textfield', 'range', 'switch', 'textarea', 'validation_text', 'button', 'radio_button', 'checkbox']

        self.targets = []
        self.data = []

        if self.train:
            self.image_name = os.listdir('/content/uiqa_Ausschnitte/train')
        else:
            self.image_name = os.listdir('/content/uiqa_Ausschnitte/val')

        for i in range(len(self.image_name)):
            num_index = re.search(r'\d', self.image_name[i]).start()
            label = self.image_name[i][:num_index]
            self.targets.append(label)
            if train:
              img_dir = '/content/uiqa_Ausschnitte/train/' + self.image_name[i]
            else:
              img_dir = '/content/uiqa_Ausschnitte/val/' + self.image_name[i]
            img = Image.open(img_dir).convert('RGB')
            x_s = 32  # define standard size
            y_s = 32
            out = img.resize((x_s, y_s), Image.ANTIALIAS)
            self.data.append(out)

        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        x, y = img.size
        img_size = (x, y)
        img = Image.fromarray(img)
        class_name = target

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}}

        return out

    def __len__(self):
        return len(self.data)



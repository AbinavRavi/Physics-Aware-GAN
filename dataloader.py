import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

#with open('./data/velocity64.pickle','rb') as handle:
 #   velocities = pickle.load(handle)

class SmokeDataset(Dataset):
    """Contains the dataloader for Smoke Simulation images generated from Mantaflow"""
    def __init__(self, pickle,transform=None):
        """pickle - The pickle file containing the simulated dataset of images"""
        with open('./data/velocity64.pickle', 'rb') as handle:
            velocities = pickle.load(handle)
        for i in range(velocities.shape[0]):
            velocities[i,:,:,:]=2*(velocities[i,:,:,:]-velocities[i,:,:,:].min())/(velocities[i,:,:,:].max()-velocities[i,:,:,:].min())-1
        self.pickle = velocities
        self.transform = transform

    def __len__(self):
        return len(self.pickle)

    def __getitem__(self,index):
        img_name = os.path.join(self.root.dir,self.pickle[idx])
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)

        return image


    
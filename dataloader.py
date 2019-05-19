import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose

import warnings
warnings.filterwarnings("ignore")

#with open('./data/velocity64.pickle','rb') as handle:
 #   velocities = pickle.load(handle)

class SmokeDataset(Dataset):
    """Contains the dataloader for Smoke Simulation images generated from Mantaflow"""
    def __init__(self, data,transform=None):
        """pickle - The pickle file containing the simulated dataset of images"""
                # for i in range(velocities.shape[0]):
            # velocities[i,:,:,:]=(velocities[i,:,:,:]-velocities[i,:,:,:].min())/(velocities[i,:,:,:].max()-velocities[i,:,:,:].min())
        self.velocities = data #np.load(data, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return self.velocities.shape[0]

    def __getitem__(self,index):
        # img_name = os.path.join(self.root.dir,self.pickle[idx])
        image = self.velocities[index,:,:,:]
        if self.transform:
            resized_image = self.transform(image)

        return image,resized_image

class Normalize(object):
    """Normalise the images"""

    def __call__(self, image):
        image = (image - np.min(image))/(np.max(image)-np.min(image))
        # image = (image[:,:,:,0] - np.min(image[:,:,:,0]))/(np.max(image[:,:,:,0])-np.min(image[:,:,:,0]))
        # print(image)
        # image = (image[:,:,:,1] - np.min(image[:,:,:,1]))/(np.max(image[:,:,:,1])-np.min(image[:,:,:,1]))
        return image
    
    def __repr__(self):
        return self.__class__.__name__+'()'

class Resize(object):
    """Resize the image for the generator"""

    def __call__(self,image):
        image = resize(image,(8,8),anti_aliasing=False)  
        return image

    def __repr__(self):
        return self.__class__.__name__+'()'
    


    
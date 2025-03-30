import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, imgpath, maskpath, classes, size):
        self.imgpath = imgpath
        self.maskpath = maskpath
        self.classes = classes
        self.size = size
        
    def __len__(self):
        return (len(self.imgpath))
    
    def __getitem__(self, idx):
        img = self.imgpath[idx]
        mask = self.maskpath[idx]
        img = Image.open(img).convert('RGB')
        mask = Image.open(mask).convert('L')
        img = img.resize(self.size, Image.BICUBIC)
        img = transforms.ToTensor()(img)
        mask = mask.resize(self.size, Image.NEAREST)
        mask = np.array(mask, dtype = np.int64)
        mask = torch.from_numpy(mask).long()
        mask = F.one_hot(mask, num_classes = self.classes).permute(2, 0, 1).float()
        return img, mask

def load_dataset(path, subfolders, h, w, classes):
    data = {}
    
    for sub in subfolders:
        img_fol = os.path.join(path, sub, f"{sub}-org-img")
        mask_fol = os.path.join(path, sub, f"{sub}-label-img")
        images = sorted([os.path.join(img_fol, f) for f in os.listdir(img_fol)])
        masks = sorted([os.path.join(mask_fol, f) for f in os.listdir(mask_fol)])

        if sub == 'train':
            dataset = CustomDataset(images, masks, classes, size = (h,w))
            data[sub] = DataLoader(dataset, batch_size = 8, shuffle = True)
        else:
            dataset = CustomDataset(images, masks, classes, size = (h,w))
            data[sub] = DataLoader(dataset, batch_size = 16, shuffle = False)
    return data

# transformations such as rotation, flipping and cropping only to be applied for training set

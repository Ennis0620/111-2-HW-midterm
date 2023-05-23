import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
import PIL.Image as Image
import re
from torch.utils.data import DataLoader
from natsort import natsorted 

class dataset(Dataset):
    def __init__(self, type, transform=None):
        super(dataset,self).__init__()
        self.CURRENT_PATH =  os.path.dirname(__file__)
        self.transform = transform
        self.type = type
        self.train_csv_path = f'{self.CURRENT_PATH}/train.csv' 
        self.test_csv_path = f'{self.CURRENT_PATH}/test.csv'
        self.train_csv = pd.read_csv(self.train_csv_path) 
        self.test_csv = pd.read_csv(self.test_csv_path)
        #抽成數量平均
        tmp = []
        uni_label = self.train_csv['Label'].unique()
        uni_label = set(uni_label)
        print('uni',uni_label)
        for i in uni_label:
            tmp.append(np.where(self.train_csv['Label']==i)[0])
        
        print('0:',len(tmp[0]))
        print('1:',len(tmp[1]))
        print('2:',len(tmp[2]))
        print('3:',len(tmp[3]))
        print('4:',len(tmp[4]))
        print('5:',len(tmp[5]))
 
        train_index = np.array([])
        valid_index = np.array([])
        split_ratio = 0.15
       
        for per_label in tmp:
            tmp_split = int(np.floor(split_ratio * len(per_label)))
            
            train_indices,valid_indices = per_label[tmp_split:], per_label[:tmp_split]
        
            train_index = np.append(train_index,train_indices)
            valid_index = np.append(valid_index,valid_indices)
        
        #打亂index
        train_indices = list(range(len(train_index)))
        valid_indices = list(range(len(valid_index)))
        np.random.shuffle(train_indices)
        np.random.shuffle(valid_indices)

        #重新排index
        train_index =  train_index[train_indices]  
        valid_index = valid_index[valid_indices]
        
        self.train_index = train_index
        self.valid_index = valid_index

        print('train len:',len(self.train_index))
        print('valid len:',len(self.valid_index))

    def __len__(self):
        if self.type == 'train':
            return len(self.train_index)
        elif self.type == 'valid':
            return len(self.valid_index)
        elif self. type == 'test':
            return len(self.test_csv)

    def __getitem__(self, idx):
        
        if self.type == 'train':
            if self.transform:
                img = Image.open(f'{self.CURRENT_PATH}/data/train_images/{self.train_csv["ID"][self.train_index[idx]]}').convert('RGB')
                image = self.transform(img)
                return image, self.train_csv['Label'][self.train_index[idx]]
        elif self.type == 'valid':
            if self.transform:
                img = Image.open(f'{self.CURRENT_PATH}/data/train_images/{self.train_csv["ID"][self.valid_index[idx]]}').convert('RGB')
                image = self.transform(img)
                return image, self.train_csv['Label'][self.valid_index[idx]]
        elif self.type == 'test':
            if self.transform:
                img = Image.open(f'{self.CURRENT_PATH}/data/test_images/{self.test_csv["ID"][idx]}').convert('RGB')
                image = self.transform(img)
                return image

if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomRotation(30),
        transforms.ToTensor(),  
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    train_data = dataset(type='train',
                   transform=train_transform
                   )
    
    valid_data = dataset(type='valid',
                   transform=train_transform
                   )
    
    t_data = DataLoader(train_data,
                      batch_size=1)
    
    v_data = DataLoader(valid_data,
                      batch_size=1)

    img, label =  next(iter(t_data))
    print( label)

    img, label =  next(iter(v_data))
    print( label)
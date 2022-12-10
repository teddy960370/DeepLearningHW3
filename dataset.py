# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 11:37:34 2022

@author: dirty
"""

from typing import List, Dict
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder ,StandardScaler,LabelEncoder

class PM25Dataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        mode: str,
    ):
        self.data = data
        self.mode = mode
        
        #self.collate_fn()
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Dict:
        instance = self.data.iloc[index]
        return np.array(instance)
    

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)


    def collate_fn(self,samples):

        return samples



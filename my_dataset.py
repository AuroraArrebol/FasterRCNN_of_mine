"""
作者：ZWP
日期：2023.06.08
"""
from torch.utils.data import Dataset
import os

#read.readlines()

class VOC2012DataSet(Dataset):
    def __init__(self,voc_root,transforms,train_set=True):
        self.root=os.path.join(voc_root,"VOCdevkit","VOC2012")
        self.img_root=os.path.join(self.root,"JEPGImages")
        self.annotations_root=os.path.join(self.root,"Annotations")
        if train_set:
            txt_list=os.path.join(self.root,
                                  "ImageSets","Main","train.txt")
        else:
            txt_list = os.path.join(self.root,
                                    "ImageSets", "Main", "val.txt")
        with open(txt_list) as read:
            self.xml_list=[os.path.join(self.annotations_root,line.strip()+".xml")
                           for line in read.readlines()]

    def __len__(self):
    def __getitem__(self, item):
    def
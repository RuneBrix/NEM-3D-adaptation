import os
import random
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from torchvision.datasets import ImageNet
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import multiprocessing
import pytorch_lightning as pl


class BaseImageNetDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data.iloc[idx, 0]
        target = self.data.iloc[idx, 1]
        if self.transform:
            img = self.transform(img)
        return img, target
    
class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size, train_transform=None, test_transform=None):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = multiprocessing.cpu_count() - 2
        self.train_transform = train_transform
        self.test_transform = test_transform



    def setup(self, stage=None):
        self.train_dataset = BaseImageNetDataset(self.train_data, transform=self.train_transform)
        self.val_dataset = BaseImageNetDataset(self.val_data, transform=self.test_transform)
        self.test_dataset = BaseImageNetDataset(self.test_data, transform=self.test_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,num_workers= self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)




class ImageNetDataset:
    def __init__(self, EXP_STORAGE_PATH, IMAGENET_DATAPATH,
                  REUSE_BASE_DATA, BATCH_SIZE,TRAIN_TRANSFORM, TEST_TRANSFORM, TEST_PIPELINE, HEAVY_PIPELINE):
        self.DATA_PATH = Path(EXP_STORAGE_PATH) / "imagenet"
        self.IMAGENET_DATAPATH = IMAGENET_DATAPATH
        self.REUSE_BASE_DATA = REUSE_BASE_DATA
        self.BATCH_SIZE = BATCH_SIZE
        self.TRAIN_TRANSFORM = TRAIN_TRANSFORM
        self.TEST_TRANSFORM = TEST_TRANSFORM
        self.TEST_PIPELINE = TEST_PIPELINE
        self.HEAVY_PIPELINE = HEAVY_PIPELINE
        #assert self.HEAVY_PIPELINE != self.TEST_PIPELINE, "CANNOT HAVE BOTH HEAVY AND TEST PIPELINE"
        self.gen_data()
    def load_train_data(self):
        train_data =  pickle.load(open(self.DATA_PATH / "data/train_images.pkl", "rb"))
        datamodule = ImageNetDataModule(train_data[train_data.train_val == "train"],
                                    train_data[train_data.train_val == "val"], 
                                    pd.DataFrame(train_data), batch_size=self.BATCH_SIZE,
                                    train_transform=self.TRAIN_TRANSFORM, test_transform=self.TEST_TRANSFORM)
        datamodule.setup()
        return datamodule
    def load_test_data(self):
        test_data =  pickle.load(open(self.DATA_PATH / "data/test_images.pkl", "rb"))
        if self.TEST_PIPELINE:
            print("Using test pipeline")
            test_data = test_data[:5]
        if self.HEAVY_PIPELINE:
            print("Using heavy pipeline")
            test_data = test_data[:100]
        return  torch.utils.data.DataLoader(
            BaseImageNetDataset(pd.DataFrame(test_data), transform=self.TEST_TRANSFORM), batch_size=1, shuffle=False)
    def get_data(self):
        
        return self.load_train_data(), self.load_test_data()
    def gen_data(self):
        if os.path.exists(self.DATA_PATH / "data/train_images.pkl") & self.REUSE_BASE_DATA:
            print("Loading stored base data")
            return
        assert self.IMAGENET_DATAPATH is not None, "Please provide the path to the ImageNet dataset"
        print("Generating stored base data")
        random.seed(0)
        np.random.seed(0)
        imgs= ImageNet(self.IMAGENET_DATAPATH, split="val" )
        imgs = list(imgs) 
        test_images = random.sample(imgs, 1000)
        train_images = random.sample(imgs, 10000)
        train_images = [img for img in train_images if img not in test_images]
        train_images = pd.DataFrame(train_images)
        train_images.columns = ["input", "target"]
        test_indices = np.random.rand(len(train_images)) < 0.8
        train_images["train_val"] = ["train" if index else "val" for index in test_indices]
        train_images.columns = ["input", "target","train_val"]
        
        os.makedirs(self.DATA_PATH / "data", exist_ok=True)
        pickle.dump(train_images, open(self.DATA_PATH / "data/train_images.pkl", "wb"))
        pickle.dump(test_images, open(self.DATA_PATH / "data/test_images.pkl", "wb"))
        return
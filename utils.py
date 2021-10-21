import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import PIL

class SmilesDataset(Dataset):
    """Smiles dataset"""
    def __init__(self, csv_file, root_dir="images", transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        df = pd.read_csv(csv_file)
        self.img_filename = df["stimulus_filename"].tolist()
        self.encodings = df["avg_encode"].tolist()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_filename)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.img_filename[idx]) + ".png"
        image = Image.open(img_name)
        encode = self.encodings[idx]
        sample = {'image': image, 'encode': encode}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

def image_grid(imgs, rows, cols, height=50, width=50):
    assert len(imgs) == rows*cols

    w, h = height, width
    grid = Image.new('L',size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w+1, i//cols*h+1))
    return grid
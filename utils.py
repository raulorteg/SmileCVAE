import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class SmilesDataset(Dataset):
    def __init__(self, csv_file, root_dir="datasets/images", transform=None):
        """ Smiles dataset. Custom torch.utils.data.Dataset class to deal with the format
        of the dataset UTK-Face Smiles, a subset of the UTK-Face dataset with encodings of the
        smile-degree of a number of subjects.

        :param csv_file: Path to the csv file with annotations (image filename, encoding of smile degree).
        :type csv_file: str
        :param root_dir: Path to the folder where the images are.
        :type root_dir: str
        :param transform: callable object, optional transform to be applied to the data
        """
        df = pd.read_csv(csv_file)
        self.img_filename = df["stimulus_filename"].tolist()
        self.encodings = df["avg_encode"].tolist()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_filename)

    def __getitem__(self, idx):
        """
        Note: This method grabs the next sample in the dataset, for that it looks up the next
        image filename, then loads the picture as a PIL Image, converts it ino an RGB image (it was found some
        picture came in a variety of formats, from RGBA to Grayscale, this allows to treat all of them in a common format)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.img_filename[idx])
        image = Image.open(img_name)
        image = image.convert('RGB')
        encode = self.encodings[idx]
        sample = {'image': image, 'encode': encode}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

def image_grid(imgs, rows, cols, height=50, width=50):
    """ Creates a grid of images in from a list of PIL.Image objects and returns the new grid Image

    :param imgs: list of PIL.Image objects (the Images to be part of the grid)
    :type imgs: list of PIL.Image objects
    :param rows: number of rows to use for the grid
    :type rows: int
    :param cols: number of columns to use for the grid
    :type cols: int
    :param height: height in pixels of each individual image part of the grid
    :type height: int
    :param width: width in pixels of each individual image part of the grid
    :type width: int
    """
    assert len(imgs) == rows*cols

    w, h = height, width
    grid = Image.new('RGB',size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w+1, i//cols*h+1))
    return grid
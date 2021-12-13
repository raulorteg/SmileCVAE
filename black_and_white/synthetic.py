import torch
import torch.nn as nn
import torchvision.transforms as transforms
from utils import SmilesDataset, image_grid
from models import VAE, CVAE
from PIL import Image

def main():

    cvae = CVAE(input_size=[1,1,50,50], latent_size=600)
    cvae.load_state_dict(torch.load("model_savepoints/cvae/model.pt")["model_state_dict"])
    for c in [-3, -2, -1, -0.5, 0.5, 1, 2, 3]:
        xhat, z = cvae.sample_prior(64, c)
        print(xhat.shape)
        
        images = [Image.fromarray(255*img.detach().numpy()[0]).convert("L") for img in xhat]
        grid = image_grid(images, 8, 8)
        grid.save(f"results/encode{c}.png")
        grid.show()

if __name__ == "__main__":
    main()
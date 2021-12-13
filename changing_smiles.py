import torch
import numpy as np
import torchvision.transforms as transforms
from utils import SmilesDataset, image_grid
from plotters import LatentPlotter
from models import CVAE
from PIL import Image

def changing_smiles(degree, RESIZE, LATENT_SIZE):
    """
    Given a degree "smile degree", LATENT_SIZE the size of the latent space used to train the model
    and the RESIZE the size (pixels) of the pictures used to train the model this function performs a forward pass
    on a batch of the training samples while shifting their encoded smlie degree by a factor "degree", that is, increases/decreases
    the level of smileyness of every picture by a "degree" factor.

    :param degree: the smile strenght degree, originally bound between [-1 (really not smiley),1 (really smiley)] bu the model
    can generalize extending a bit over the range.
    :type degree: float
    :param RESIZE: resize integer parameter used when training the model.
    :type RESIZE: int
    :param LATENT_SIZE: latent space dimension integer parameter used when training the model.
    :type LATENT_SIZE: int
    """

    BATCH_SIZE = 32
    
    # prepare the data
    transform = transforms.Compose(
                    [
                    transforms.Resize((RESIZE,RESIZE),interpolation=Image.NEAREST),
                    transforms.ToTensor()
                    ])

    dataset = SmilesDataset(csv_file="datasets/smiles_dataset.txt", root_dir="datasets/images/", transform=transform)
    dataloader_eval = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    for sample in dataloader_eval:
        input_shape = sample["image"].shape
        break
    
    # prepare the network and plotting functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cvae = CVAE(input_size=input_shape, latent_size=LATENT_SIZE).to(device)
    if torch.cuda.is_available():
        cvae.load_state_dict(torch.load("model_savepoints/model.pt")["model_state_dict"])
    else:
        cvae.load_state_dict(torch.load("model_savepoints/model.pt", map_location=torch.device('cpu'))["model_state_dict"])

    cvae.eval()
    for sample in dataloader_eval:
        images = sample["image"].to(device)
        encodings = degree + 1.0*sample["encode"]
        encodings = encodings.to(device)
        outdict = cvae(images, encodings)
        x, xhat = outdict["x"], outdict["xhat"]
        break

    images = []
    for img in xhat:
        img = (255*img).cpu().detach().numpy().astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        pil_image = Image.fromarray(img).convert("RGB")
        images.append(pil_image)

    grid = image_grid(images, 4, 8, RESIZE, RESIZE)
    grid.save(f"results/changed/changed_{degree}.png")
    
if __name__ == "__main__":

    import argparse
    # parsing user input
    # example: python changing_smiles.py --degree=0.6 --resize=50 --latent=50
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", help="Smile degree [-1,1] (Defaults 0.6)", default=0.6, type=float)
    parser.add_argument("--resize", help="Resized dimension (pixels, defaults 50)", default=50, type=int)
    parser.add_argument("--latent", help="Latent dimension (defaults 50)", default=50, type=int)
    args = parser.parse_args()

    changing_smiles(degree=args.degree, RESIZE=args.resize, LATENT_SIZE=args.latent)
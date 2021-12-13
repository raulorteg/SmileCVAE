import torch
import numpy as np
import torchvision.transforms as transforms
from utils import SmilesDataset, image_grid
from plotters import LatentPlotter
from models import CVAE
from PIL import Image

def sample_across_axis(axis, RESIZE, LATENT_SIZE):
    """
    Given the axis to look across, RESIZE parameter and LATENT_SIZE of the trained model this function
    samples the latent space unformly only varying the [axis]th element, leaving the rest of the latent values
    copnstant to 0 and decodes this latent samples into images. This is done to attempt to visualize what each
    dimension of the latent space is "doing". Since the latent spaces for these datasets tend to be large, its 
    good to first look at the the variance across the latent space for all epochs to see what are the components
    with higher variance. Then one can use this function to inspect visualy what those high variance dimensions
    of the latent space are doing

    :param axis: index of the dimension of the latent space to be inspected.
    :type axis: int between [0,LATENT_SIZE]
    :param RESIZE: resize integer parameter used when training the model.
    :type RESIZE: int
    :param LATENT_SIZE: latent space dimension integer parameter used when training the model.
    :type LATENT_SIZE: int
    """
    assert ((axis >= 0) and (axis <= LATENT_SIZE)), f"Index of dimension should be within the dimension of the latent space. It must hold 0<={axis}<={LATENT_SIZE}"
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
        encodings = sample["encode"]
        encodings = encodings.to(device)
        xhat, z_c= cvae.sample_prior(32, 0.0)
        break
    
    xhat = []
    for value in np.linspace(-1.2,1.2,32):
        z = 0.0 * z_c
        z[0][axis] = value
        xhat.append(cvae.observation_model(z)[0])

    images = []
    for img in xhat:
        img = (255*img).cpu().detach().numpy().astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        pil_image = Image.fromarray(img).convert("RGB")
        images.append(pil_image)

    grid = image_grid(images, 4, 8, RESIZE, RESIZE)
    grid.save(f"results/axis/axis_{axis}.png")
    
if __name__ == "__main__":

    import argparse
    # parsing user input
    # example: python sample_across_axis.py --axis=0 --resize=50 --latent=50
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", help="Axis index [0,LATENT_SIZE] (Defaults 0)", default=0, type=int)
    parser.add_argument("--resize", help="Resized dimension (pixels, defaults 50)", default=50, type=int)
    parser.add_argument("--latent", help="Latent dimension (defaults 50)", default=50, type=int)
    args = parser.parse_args()

    sample_across_axis(axis=args.axis, RESIZE=args.resize, LATENT_SIZE=args.latent)
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
from utils import SmilesDataset, image_grid
from plotters import LatentPlotter
from models import CVAE
from PIL import Image

def get_subspace(RESIZE, LATENT_SIZE):
    """
    Given the RESIZE parameter and LATENT_SIZE of the trained model this function
    performs a forward pass of the training data on the last saved trained model and saves the latent
    representation of all the training data. Using then this latent representation of the data we can perform
    PCA to obtain a plot showing how much of the total variance can be explained for every number of compoents, 
    which can in turn be used to argue if the latent representations live in a subspace of much lower dimension or not.

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

    z_buffer = []
    cvae.eval()
    for sample in dataloader_eval:
        images = sample["image"].to(device)
        encodings = sample["encode"]
        encodings = encodings.to(device)
        outdict = cvae(images, encodings)
        x, xhat = outdict["x"], outdict["xhat"]
        z = outdict["z"]
        for z_i in z:
            z_buffer.append(z_i.cpu().detach().numpy())

    z_buffer = np.array(z_buffer)


    pca = PCA()
    pca.fit_transform(z_buffer)

    plt.plot(np.cumsum(pca.explained_variance_ratio_), "*-")
    plt.title("Explained Variance Ratio")
    plt.xlabel("Number of components")
    plt.savefig("results/plots/latent_pca.png")

def plot_latent_variance():

    # parse the latent_variance results and generate the plots
    with open("results/latent_variance.txt", "r") as f:
        lines = f.readlines()
        
        latent_buffer, latent = [], []
        for line in lines:
            if "[" in line:
                if len(latent_buffer) > 1:
                    latent_buffer = [item for sublist in latent_buffer for item in sublist.split()]
                    latent.append(latent_buffer)
                    
                latent_buffer = []
                latent_buffer.append(line.replace("[",""))
            
            else:
                latent_buffer.append(line.replace("]",""))
            
    latent = np.array(latent, dtype=float)
    np.savetxt("results/latent.txt", latent)

    # latent variance on first 50 epochs
    var_epochs = latent[0:50].T
    fig, axs = plt.subplots(1,1,figsize=(12,12))
    img = axs.imshow(var_epochs, cmap='hot', interpolation='nearest')
    axs.set_xlabel("Epochs")
    axs.set_ylabel("Latent")
    plt.colorbar(img, ax=axs)
    plt.savefig("results/plots/latent_var_first_50.png")

    # latent variance on last 50 epochs
    var_epochs = latent[-50:-1].T
    fig, axs = plt.subplots(1,1,figsize=(12,12))
    img = axs.imshow(var_epochs, cmap='hot', interpolation='nearest')
    axs.set_xlabel("Epochs")
    axs.set_ylabel("Latent")
    plt.colorbar(img, ax=axs)
    plt.savefig("results/plots/latent_var_last_50.png")

    # summary latent variance on all training (sampled every 20 epochs)
    var_epochs = latent[::20].T
    fig, axs = plt.subplots(1,1,figsize=(12,12))
    img = axs.imshow(var_epochs, cmap='hot', interpolation='nearest')
    axs.set_xlabel("Epochs")
    axs.set_ylabel("Latent")
    plt.colorbar(img, ax=axs)
    plt.savefig("results/plots/latent_var.png")

def plot_training():

    data_train = np.genfromtxt("results/training.txt")

    # summary plot of training
    fig, axs = plt.subplots(2,2, figsize=(12,12))

    axs[0,0].set_title("Total Loss (ELBO)")
    axs[0,0].set_xlabel("Epoch")
    axs[0,0].set_ylabel("Loss")
    axs[0,0].plot(data_train[:,0], data_train[:,1])

    axs[1,0].set_title("MSE Loss (reconstruction)")
    axs[1,0].set_xlabel("Epoch")
    axs[1,0].set_ylabel("MSE Loss")
    axs[1,0].plot(data_train[:,0], data_train[:,2], "green")

    axs[1,1].set_title("KLD Loss")
    axs[1,1].set_xlabel("Epoch")
    axs[1,1].set_ylabel("KLD Loss")
    axs[1,1].plot(data_train[:,0], data_train[:,3], "orange")

    axs[0,1].set_title("Learning Rate")
    axs[0,1].set_xlabel("Epoch")
    axs[0,1].set_ylabel("Learning Rate")
    axs[0,1].plot(data_train[:,0], data_train[:,4], "red")

    plt.tight_layout()
    plt.savefig("results/plots/summary_training.png")

    # big plot of training
    fig, axs = plt.subplots(1,1, figsize=(12,12))

    axs.set_title("Total Loss (ELBO)")
    axs.set_xlabel("Epoch")
    axs.set_ylabel("Loss")
    axs.plot(data_train[:,0], data_train[:,1])

    plt.savefig("results/plots/training.png")

    
if __name__ == "__main__":
    
    # example: python training_plots.py --resize=50 --latent=50
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resize", help="Resized dimension (pixels, defaults 50)", default=50, type=int)
    parser.add_argument("--latent", help="Latent dimension (defaults 50)", default=50, type=int)
    args = parser.parse_args()

    get_subspace(RESIZE=args.resize, LATENT_SIZE=args.latent)
    plot_latent_variance()
    plot_training()
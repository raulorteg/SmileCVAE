import torch
import torch.nn as nn
import torchvision.transforms as transforms
from utils import SmilesDataset, image_grid
from models import VAE
from PIL import Image

def main():
    transform = transforms.Compose(
                    [transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((50,50),interpolation=Image.NEAREST),
                    transforms.ToTensor()])

    dataset = SmilesDataset(csv_file="smiles_dataset.txt", root_dir="filtered", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
    dataloader_eval = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    for sample in dataloader:
        input_shape = sample["image"].shape
        break
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(input_size=input_shape, latent_size=600).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.00002)
    train_loss_history = []
    num_epochs = 1000
    vae.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for sample in dataloader:
            optimizer.zero_grad()

            images = sample["image"].to(device)

            outdict = vae(images)
            pz, qz, z, x, xhat = outdict["pz"], outdict["qz"], outdict["z"], outdict["x"], outdict["xhat"]
            
            # compute loss
            reconloss = criterion(xhat, x)
            MSE = torch.mean((xhat - x)**2)

            KLD = -0.5 * torch.mean(torch.mean(1 + torch.log(qz.sigma**2) - qz.mu**2 - torch.exp(qz.sigma**2)))
            loss = MSE + KLD

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().item()

        vae.eval()
        for sample in dataloader_eval:
            optimizer.zero_grad()
            images = sample["image"].to(device)
            outdict = vae(images)
            x, xhat = outdict["x"], outdict["xhat"]
            break

        images = [Image.fromarray(255*img.detach().numpy()[0]).convert("L") for img in xhat]
        grid = image_grid(images, 4, 8)
        grid.save(f"results/vae/iter_{epoch}.png")

        if epoch == 0:
            images = [Image.fromarray(255*img.detach().numpy()[0]).convert("L") for img in x]
            grid = image_grid(images, 4, 8)
            grid.save(f"results/vae/original.png")

        train_loss_history.append(train_loss)

        torch.save({
                    'epoch': epoch,
                    'model_state_dict': vae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    }, "model_savepoints/vae/model.pt")

        print(f"epoch:{epoch}, loss:{train_loss_history[-1]}, mse: {BCE}, kld: {KLD}")
    

if __name__ == "__main__":
    main()
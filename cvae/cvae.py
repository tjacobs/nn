# Autoencoder example
# From https://ccrma.stanford.edu/~slegroux/blog/posts/cvae/
# Download dataset from https://www.kaggle.com/datasets/zalando-research/fashionmnist

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

torch.manual_seed(0)

class ConditionalVariationalEncoder(nn.Module):
    def __init__(self, input_dim=28*28, n_conds=10, hid_dim=512, latent_dim=2, device='mps'):
        super().__init__()
        self.layer1 = nn.Linear(input_dim + n_conds, hid_dim)
        self.layer2 = nn.Linear(hid_dim, latent_dim) #mu
        self.layer3 = nn.Linear(hid_dim, latent_dim) #sigma
        self.gaussian = torch.distributions.Normal(0, 1)
        self.gaussian.loc = self.gaussian.loc.to(device) #sampling on 
        self.gaussian.scale = self.gaussian.scale.to(device)
        self.kl = 0
        self.n_conds = n_conds

    def forward(self, x, cond):
        seq1 = nn.Sequential(self.layer1, nn.ReLU())
#        print(f"{x=}")
#        print(f"{cond=}")
        yy = F.one_hot(cond.to(torch.long), self.n_conds)
        x = torch.cat([x, yy], 1)
        xx = seq1(x)
        mu = self.layer2(xx)
        sigma = torch.exp(self.layer3(xx)) # exp added to keep sigma positive
        z = mu + sigma * self.gaussian.sample(mu.shape) # we sample gaussian to get latent
        self.kl = torch.mean(0.5 * torch.sum(sigma**2 + mu**2 - torch.log(sigma**2 + 1e-8) - 1, dim=1))
        return(z)

class ConditionalVariationalDecoder(nn.Module):
    def __init__(self,latent_dim=2, n_conds=10, output_dim=28*28, hid_dim=512):
        super().__init__()
        self.layer1 = nn.Linear(latent_dim + n_conds, hid_dim)
        self.layer2 = nn.Linear(hid_dim, output_dim)
        self.n_conds = n_conds

    def forward(self, z, cond):
        seq = nn.Sequential(self.layer1, nn.ReLU(), self.layer2, nn.Sigmoid())
        yy = F.one_hot(cond.to(torch.long), self.n_conds) 
        zz = torch.cat([z, yy], 1)
        x = seq(zz)
        return(x)

class ConditionalVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=28*28, n_conds=10, hid_dim=512, latent_dim=2):
        super().__init__()
        self.encoder = ConditionalVariationalEncoder(input_dim=input_dim, n_conds=n_conds, hid_dim=hid_dim, latent_dim=latent_dim)
        self.decoder = ConditionalVariationalDecoder(latent_dim=latent_dim, n_conds=n_conds, output_dim=input_dim, hid_dim=hid_dim)
        self.n_conds = n_conds
    
    def forward(self, x, cond):
        z = self.encoder(x, cond)
        return(self.decoder(z, cond))

    def generate(self, z:torch.Tensor, cond):
      return self.decoder(z, cond)

class FashionMnistDataset(Dataset):
    def __init__(self, csv_file="fmnist/fashion-mnist_train.csv"):
        super().__init__()
        self.train = pd.read_csv(csv_file)
        X = torch.tensor(self.train.iloc[:,1:].values / 255)
        self.X = (X-0.5)/0.5
        self.Y = torch.tensor(self.train.iloc[:,0].values).to(torch.int)
        self.X, self.Y = self.X, self.Y
    
    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        return (self.X[idx],  self.Y[idx])

def main():
    # Create instance of the dataset
    ds = FashionMnistDataset()

    # Create labels
    map_labels = { "0": "T-shirt/top", "1": "Trouser", "2": "Pullover", "3": "Dress", "4": "Coat", "5": "Sandal", "6": "Shirt", "7": "Sneaker", "8": "Bag", "9": "Ankle boot"  }

    # Show pants
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)
    X, Y = next(iter(dl))
    if False:
        print("Showing original pants")
        plt.figure(figsize = (1, 1))
        batch_n = 0
        plt.imshow(X[batch_n].numpy().reshape(28, 28), cmap='gray')
        plt.title(map_labels[str(int(Y[batch_n]))])
        plt.pause(2)

    # Device is metal
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_default_dtype(torch.float32) # mps doesn't support float64

    # Data loader
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)

    # Model
    cv_autoencoder = ConditionalVariationalAutoencoder().to(device)

    # Training params
    n_epochs = 10
    lr = 1e-3
    beta = 1  # KL scaling factor

    # Optimizer
    opt = Adam(cv_autoencoder.parameters(), lr=lr)

    # Loss is mean square error + KL loss 
    def loss_func(x_hat, x, autoencoder, beta=1.0):
        mse_loss = nn.MSELoss()(x_hat, x)
        kl_loss = autoencoder.encoder.kl
        return mse_loss + beta * kl_loss, mse_loss, kl_loss

    # Training loop
    train = True
    if train:
        for epoch in range(n_epochs):
          for batch_idx, (x,y) in enumerate(dl):
            x, y = x.to(torch.float).to(device), y.to(device)
            x_hat = cv_autoencoder(x, y)
            loss, mse_loss, kl_loss = loss_func(x_hat, x, cv_autoencoder, beta=beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if batch_idx % 1000:
                print(f"\rEpoch [{epoch+1}/{n_epochs}], Batch [{batch_idx+1}/{len(dl)}], Loss: {loss.item():.4f}, MSE: {mse_loss.item():.4f}, KL: {kl_loss.item():.4f}", end='')

        # Save
        print("\nSaving")
        model_path = "cvae"
        torch.save(cv_autoencoder.state_dict(), model_path)

    # Load saved model
    cv_autoencoder = ConditionalVariationalAutoencoder().to(device)
    cv_autoencoder.load_state_dict(torch.load('cvae', weights_only=True))
    cv_autoencoder.to(device)

    # Eval mode, no training info stored
    cv_autoencoder.eval()

    # Test putting pants in to see if we get pants out
    print("\nShowing encoded decoded pants")
    x = X[0].to(torch.float32).to(device).unsqueeze(0)
    y = Y[0].to(device).unsqueeze(0)
    x_reconstructed = cv_autoencoder(x, y)
    plt.figure(figsize = (1, 1))
    plt.imshow(x_reconstructed.cpu().detach().numpy().reshape(28, 28), cmap='gray')
    plt.pause(2)

    # Define grid range and resolution
    print("\nShowing grid of generated clothes")
    z_min, z_max, steps = -10, 10, 10
    z_values = np.linspace(z_min, z_max, steps)

    # Create grid of latent vectors
    latent_grid = torch.tensor([[x, y] for x in z_values for y in z_values]).to(torch.float32).to(device)

    # Repeat condition
    y = y.repeat(latent_grid.size(0))

    # Generate images
    generated_images = cv_autoencoder.generate(latent_grid, y).cpu().detach().numpy()

    # Reshape generated images to 28x28
    generated_images = generated_images.reshape(-1, 28, 28)

    # Plot the images in a grid
    fig, axes = plt.subplots(steps, steps, figsize=(10, 10))
    for i in range(steps):
        for j in range(steps):
            axes[i, j].imshow(generated_images[i * steps + j], cmap="gray")
            axes[i, j].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

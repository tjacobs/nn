# Autoencoder example
# From https://ccrma.stanford.edu/~slegroux/blog/posts/cvae/
# Download dataset from https://www.kaggle.com/datasets/zalando-research/fashionmnist

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

torch.manual_seed(0)

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim=2, device='mps'):
        super().__init__()
        self.layer1 = nn.Linear(784, 512)
        self.layer2 = nn.Linear(512, latent_dim) # mu = mean
        self.layer3 = nn.Linear(512, latent_dim) # sigma = deviation
        self.gaussian = torch.distributions.Normal(0, 1)
        self.gaussian.loc = self.gaussian.loc.to(device) # sampling on 
        self.gaussian.scale = self.gaussian.scale.to(device)
        self.kl = 0

    def forward(self, x):
        seq1 = nn.Sequential(self.layer1, nn.ReLU())
        xx = seq1(x)
        mu = self.layer2(xx)
        sigma = torch.exp(self.layer3(xx)) # exp added to keep sigma positive
        z = mu + sigma * self.gaussian.sample(mu.shape) # we sample gaussian to get latent
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return(z)

class Decoder(nn.Module):
  def __init__(self, latent_dim:int=2):
    super().__init__()
    self.layer1 = nn.Linear(latent_dim, 512)
    self.layer2 = nn.Linear(512, 784)
  
  def forward(self, z:torch.Tensor):
    seq = nn.Sequential(self.layer1, nn.ReLU(), self.layer2, nn.Sigmoid())
    x = seq(z)
    return(x)

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim=2):
      super().__init__()
      self.encoder = VariationalEncoder(latent_dim)
      self.decoder = Decoder(latent_dim)
    
    def forward(self, x):
        z = self.encoder(x)
        return(self.decoder(z))

    def generate(self, z:torch.Tensor):
      return self.decoder(z)

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
    X, y = next(iter(dl))
    if False:
        print("Showing original pants")
        plt.figure(figsize = (1, 1))
        batch_n = 0
        plt.imshow(X[batch_n].numpy().reshape(28,28),cmap='gray')
        plt.title(map_labels[str(int(y[batch_n]))])
        plt.pause(2)

    # Device is metal
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Data loader
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)

    # Model
    autoencoder = VariationalAutoencoder().to(device)

    # Training params
    n_epochs = 20

    # Optimizer
    opt = Adam(autoencoder.parameters(), lr=1e-3)

    # Loss is mean square error + KL loss 
    def loss_func(x_hat, x, autoencoder):
        mse_loss = nn.MSELoss()(x_hat, x)
        kl_loss = autoencoder.encoder.kl
        return mse_loss + kl_loss

    # Training loop
    train = True
    if train:
        for epoch in range(n_epochs):
            for batch_idx, (x, y) in enumerate(dl):
                x = x.to(torch.float32).to(device)
                opt.zero_grad()
                x_hat = autoencoder(x)
                loss = loss_func(x_hat, x, autoencoder)
                loss.backward()
                opt.step()
                if batch_idx % 1000:
                  print('\rTrain Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch + 1, n_epochs, batch_idx * len(x), len(dl.dataset), 100.0 * batch_idx / len(dl), loss.cpu().data.item()), end='')
        # Save
        print("Saving")
        model_path = "varautoencoder"
        torch.save(autoencoder.state_dict(), model_path)

    # Load saved model
    autoencoder = VariationalAutoencoder()
    autoencoder.load_state_dict(torch.load('varautoencoder', weights_only=True))
    autoencoder.to(device)

    # Eval mode, no training info stored
    autoencoder.eval()

    # Test putting pants in to see if we get pants out
    print("\nShowing encoded decoded pants")
    x = X[0]
    x = x.to(torch.float).to(device)
    x_reconstructed = autoencoder(x)
    plt.figure(figsize = (1, 1))
    plt.imshow(x_reconstructed.cpu().detach().numpy().reshape(28, 28), cmap='gray')
    plt.pause(2)

    # Define grid range and resolution
    print("\nShowing grid of generated clothes")
    z_min, z_max, steps = -3, 30, 10
    z_values = np.linspace(z_min, z_max, steps)

    # Create grid of latent vectors
    latent_grid = torch.tensor([[x, y] for x in z_values for y in z_values]).to(torch.float32).to(device)

    # Generate images
    generated_images = autoencoder.generate(latent_grid).cpu().detach().numpy()

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

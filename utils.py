from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image
import os

class Celeba(Dataset):
    def __init__(self, path, df, tfms=None):
        self.tfms = tfms
        self.path = path
        self.df = df
        self.names = [os.path.join(self.path, name) for name in self.df["image_id"].values.tolist()]
    
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        image = Image.open(self.names[idx])
        if self.tfms is not None:
            image = self.tfms(image)
        y = torch.Tensor((self.df.iloc[idx, 1:].values > 0).astype(int).tolist())
        return (image, y)

def show_images(image_tensor, num_images=16, size=(3, 64, 64), nrow=3):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)

def get_disc_loss(disc, batch, fake, criterion):
    # training for real images
    real_labels = torch.ones(batch.size(0))
    yhat = disc(batch)
    real_loss = criterion(yhat.squeeze(), real_labels.squeeze())

    # training for fake images
    fake_labels = torch.zeros(fake.size(0))
    # we detach because we don't want the generator params to update
    fake_loss = criterion(disc(fake.detach()).squeeze(), fake_labels)

    disc_loss = real_loss + fake_loss

    return disc_loss

def get_gen_loss(disc, fake, criterion):
    # training the generator
    labels = torch.ones(fake.size(0))
    loss = criterion(disc(fake).squeeze(), labels)

    return loss

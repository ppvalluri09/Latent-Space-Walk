from models import *
from utils import *
from trainer import *
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchsummary import summary
import pandas as pd

z_dim = 64
batch_size=64
hidden_dim=64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv("./celeba/list_attr_celeba.csv")

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
dataset = Celeba(path="./celeba/img_align_celeba/img_align_celeba", df=df, tfms=transform)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


gen = Generator(z_dim, im_channels=3, hidden_dim=hidden_dim).to(device)
disc = Discriminator(3, hidden_dim, n_classes=1).to(device)
classifier = Classifier(num_classes=40).to(device)
noise = get_noise(batch_size, z_dim, device=device)

print("Generator\n", gen)
print("\nDiscriminator\n", disc)
print("\nClassifier\n", classifier)

# train_gan(gen, disc, EPOCHS=50, loader=loader, batch_size=64, z_dim=z_dim)
train_classifier(classifier, "./models/classifier.pth", device=device, loader=loader)


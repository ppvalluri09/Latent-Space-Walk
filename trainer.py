import torch
from torchvision.datasets import CelebA
from torchvision import transforms
from tqdm.auto import tqdm
from utils import *

def train_gan(gen, disc, EPOCHS, loader=None, lr=0.001, batch_size=64, beta1=0.5, beta2=0.99, z_dim=64):
    image_size=64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if loader is None:
        loader = torch.utils.data.DataLoader(
            CelebA(".", split="train", download=True, transform=transform),
            batch_size=batch_size,
            shuffle=True
        )

    optim_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
    optim_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta1, beta2))
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(1, EPOCHS+1)):
        disc_loss = 0.0
        gen_loss = 0.0
        progress = tqdm(loader, total=len(loader), leave=False)
        for data, _ in progress:
            data = data.to(device)

            optim_g.zero_grad()
            optim_d.zero_grad()

            # training the Discriminator
            z = get_noise(data.size(0), z_dim)
            fake = gen(z)
            disc_loss_batch = get_disc_loss(disc, data, fake, criterion)
            disc_loss_batch.backward()
            disc_loss += disc_loss_batch.item() / len(loader)
            optim_d.step()

            # training the generator
            gen_loss_batch = get_gen_loss(disc, fake, criterion)
            gen_loss += gen_loss_batch.item() / len(loader)
            optim_g.step()

            progress.set_description(f"Epoch [{epoch/EPOCHS}]")
            progress.set_postfix(disc_loss=disc_loss_batch.item(), gen_loss=gen_loss_batch.item())
        torch.save({"gen": gen.state_dict()}, "./models/generator.pth")
        torch.save({"disc": disc.state_dict()}, "./models/discriminator.pth")


def train_classifier(classifier, filename, device, loader=None):
    label_indices = range(40)

    EPOCHS = 3
    lr = 0.001
    beta_1 = 0.5
    beta_2 = 0.999
    image_size = 64

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if loader is None:
        loader = DataLoader(
            CelebA(".", split='train', download=True, transform=transform),
            batch_size=batch_size,
            shuffle=True)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, betas=(beta_1, beta_2))
    criterion = torch.nn.BCEWithLogitsLoss()

    avg_loss = 0.0
    for epoch in tqdm(range(1, EPOCHS+1)):
        loop = tqdm(loader, total=len(loader), leave=False)
        for real, labels in loop:
            real = real.to(device)
            labels = labels[:, label_indices].to(device).float()

            optimizer.zero_grad()
            yhat = classifier(real)
            loss = criterion(yhat, labels)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(loader)

            loop.set_description(f"Epoch [{epoch} / {EPOCHS}]")
            loop.set_postfix(clf_loss=loss.item())
        print(f"Epoch [{epoch} / {EPOCHS}], Loss {avg_loss}")

        torch.save({"classifier": classifier.state_dict()}, filename)

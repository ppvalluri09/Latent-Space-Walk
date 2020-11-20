from utils import *
import torch
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator(z_dim=64, im_channels=3, hidden_dim=64).to(device)
gen_dict = torch.load("models/generator.pth", map_location=torch.device("cpu"))["gen"]
gen.load_state_dict(gen_dict)
gen.eval()

z = get_noise(16, z_dim=64, device=device)
fake = gen(z)
print(fake.shape)

show_tensor_images(fake)

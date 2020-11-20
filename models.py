import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=64, im_channels=3, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.gen = nn.Sequential(
            self.make_gen_block(self.z_dim, hidden_dim * 8),
            self.make_gen_block(hidden_dim*8, hidden_dim*4),
            self.make_gen_block(hidden_dim*4, hidden_dim*2),
            self.make_gen_block(hidden_dim*2, hidden_dim),
            self.make_gen_block(hidden_dim, im_channels, kernel_size=4, final_layer=True)
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, x):
        x = x[:, :, None, None]
        return self.gen(x)



class Discriminator(nn.Module):
    def __init__(self, im_channels, hidden_dim=64, n_classes=1):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # mx3x64x64
            self.make_disc_block(im_channels, hidden_dim),
            # mx64x31x31
            self.make_disc_block(hidden_dim, hidden_dim*2),
            # mx128x14x14
            self.make_disc_block(hidden_dim*2, hidden_dim*4, stride=3),
            # mx256x3x3
            self.make_disc_block(hidden_dim*4, n_classes, final_layer=True)
            # mxnx1x1
        )

    def make_disc_block(self, in_channels, out_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            )

    def forward(self, x):
        disc_pred = self.disc(x)
        return disc_pred.view(len(disc_pred), -1)


class Classifier(nn.Module):
    def __init__(self, im_chan=3, num_classes=2, hidden_dim=64):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            self.make_classifier_block(im_chan, hidden_dim),
            self.make_classifier_block(hidden_dim, hidden_dim * 2),
            self.make_classifier_block(hidden_dim * 2, hidden_dim * 4, stride=3),
            self.make_classifier_block(hidden_dim * 4, num_classes, final_layer=True),
        )

    def make_classifier_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, image):
        class_pred = self.classifier(image)
        return class_pred.view(len(class_pred), -1)

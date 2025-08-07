# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DrumSSMGenerator(nn.Module):
    """
    VAE-GAN SSM generator from Wei et al. (2019) §3.2, §4.4:
      - 8 convolutional layers (downsampling 256→1)
      - 3 fully-connected layers with skip connections, 32-dim latent
      - 8 deconv layers (upsampling 1→256)
      - Input/output: (B,1,256,256)
    """
    def __init__(self, in_channels=1, latent_dim=32):
        super().__init__()
        # Encoder: 8 conv layers, halving spatial dims each time
        channels = [in_channels, 64, 128, 256, 512, 512, 512, 512, 512]
        self.enc_convs = nn.ModuleList()
        for i in range(8):
            self.enc_convs.append(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1)
            )
        # Fully-connected for latent
        # after 8 downsamples: 256/(2^8)=1 → feature map size 1×1
        self.fc1 = nn.Linear(512*1*1, 1024)
        self.fc_mu     = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        # Decoder FC
        self.fc_dec = nn.Linear(latent_dim, 512*1*1)
        # Decoder: 8 deconv layers (upsampling back to 256)
        de_channels = [512, 512, 512, 512, 256, 128, 64, in_channels]
        self.dec_deconvs = nn.ModuleList()
        for i in range(8):
            self.dec_deconvs.append(
                nn.ConvTranspose2d(de_channels[i], de_channels[i+1],
                                   kernel_size=4, stride=2, padding=1)
            )

    def encode(self, x):
        for conv in self.enc_convs:
            x = F.relu(conv(x))
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc_dec(z))
        x = x.view(x.size(0), 512, 1, 1)
        for deconv in self.dec_deconvs[:-1]:
            x = F.relu(deconv(x))
        # last layer: output activation sigmoid to bound [0,1]
        x = torch.sigmoid(self.dec_deconvs[-1](x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class SSMDiscriminator(nn.Module):
    """
    Discriminator for the VAE-GAN SSM generator (Wei et al. §3.2):
      - same conv architecture as encoder, ends with sigmoid.
    """
    def __init__(self, in_channels=1):
        super().__init__()
        channels = [in_channels, 64, 128, 256, 512, 512, 512, 512, 512]
        layers = []
        for i in range(8):
            layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        # final classifier
        self.main = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*1*1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return self.classifier(x)


class DrumPatternVAE(nn.Module):
    """
    Drum-pattern VAE from Wei et al. (2019) §3.4, §4.4:
      - Input: (B,8,84,96)
      - 8 conv layers down to 1x1, 32-dim latent
      - 8 deconv layers back to (8→1) channels with output 46x16
    """
    def __init__(self, in_channels=8, latent_dim=32, out_height=46, out_width=16):
        super().__init__()
        # Encoder convs: seven layers to reduce (84x96)->(1x~1)
        enc_channels = [in_channels, 64, 128, 256, 512, 512, 512, 512]
        self.enc_convs = nn.ModuleList()
        for i in range(len(enc_channels)-1):
            self.enc_convs.append(
                nn.Conv2d(enc_channels[i], enc_channels[i+1],
                          kernel_size=4, stride=2, padding=1)
            )
        # Compute feature map size after convs
        self.enc_final_h = 84 // (2**7)
        self.enc_final_w = 96 // (2**7)
        self.fc1 = nn.Linear(512 * self.enc_final_h * self.enc_final_w, 1024)
        self.fc_mu     = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 512 * self.enc_final_h * self.enc_final_w)
        # Decoder deconv layers
        dec_channels = [512, 512, 512, 512, 256, 128, 64, out_height*out_width]
        self.dec_deconvs = nn.ModuleList()
        for i in range(len(dec_channels)-1):
            self.dec_deconvs.append(
                nn.ConvTranspose2d(dec_channels[i], dec_channels[i+1],
                                   kernel_size=4, stride=2, padding=1)
            )

    def encode(self, x):
        for conv in self.enc_convs:
            x = F.relu(conv(x))
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.fc_dec(z))
        x = x.view(x.size(0), 512, self.enc_final_h, self.enc_final_w)
        for deconv in self.dec_deconvs[:-1]:
            x = F.relu(deconv(x))
        x = torch.sigmoid(self.dec_deconvs[-1](x))
        # reshape to (B,1,H,W)
        B = x.size(0)
        x = x.view(B, 1, out_height, out_width)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# EOF models.py

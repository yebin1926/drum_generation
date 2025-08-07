# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------
# 1) SSM VAE-GAN components (Wei et al. §3.2, §4.4)
# ----------------------------------------

class SSMEncoder(nn.Module):
    """
    Encodes a 256×256 melodic/bar-level SSM into a 32-dim latent vector.
    - 8 Conv2d layers, each stride=2, kernel=4, padding=1
    """
    def __init__(self, in_channels=1, base_channels=64, latent_dim=32):
        super().__init__()
        layers = []
        curr_ch = in_channels
        # 8 downsampling conv blocks
        for i in range(8):
            out_ch = base_channels * min(2**i, 8)  # 64,128,256,512...
            layers.append(nn.Conv2d(curr_ch, out_ch, 4, 2, 1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            curr_ch = out_ch
        self.conv = nn.Sequential(*layers)
        # after 8 downsamples, feature map is 1×1
        self.fc_mu     = nn.Linear(curr_ch * 1 * 1, latent_dim)
        self.fc_logvar = nn.Linear(curr_ch * 1 * 1, latent_dim)

    def forward(self, x):
        x = self.conv(x)                      # (B, C, 1, 1)
        x = x.view(x.size(0), -1)             # (B, C)
        mu     = self.fc_mu(x)                # (B, latent_dim)
        logvar = self.fc_logvar(x)
        return mu, logvar


class SSMDecoder(nn.Module):
    """
    Decodes a 32-dim latent to a reconstructed 256×256 SSM.
    - FC to 512×1×1, then 8 ConvTranspose2d upsampling layers.
    """
    def __init__(self, out_channels=1, base_channels=64, latent_dim=32):
        super().__init__()
        # initial FC
        curr_ch = base_channels * 8  # matches last encoder out_ch
        self.fc = nn.Linear(latent_dim, curr_ch * 1 * 1)
        # 8 upsampling deconv blocks
        layers = []
        for i in reversed(range(8)):
            in_ch = base_channels * min(2**i, 8)
            out_ch = base_channels * min(2**(i-1), 8) if i>0 else out_channels
            layers.append(nn.ConvTranspose2d(curr_ch, out_ch, 4, 2, 1))
            if i>0:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Sigmoid())
            curr_ch = out_ch
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(x.size(0), -1, 1, 1)  # (B, C, 1, 1)
        x = self.deconv(x)              # (B, 1, 256, 256)
        return x


class SSMVAE(nn.Module):
    """Wraps SSMEncoder + reparam + SSMDecoder into a VAE."""
    def __init__(self, encoder: SSMEncoder, decoder: SSMDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


class SSMDiscriminator(nn.Module):
    """
    Discriminator for the SSM VAE-GAN. Mirrors the encoder's conv stack,
    ends in a sigmoid classifier over real/fake.
    """
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        layers = []
        curr_ch = in_channels
        # reuse 8 conv blocks from encoder, with LeakyReLU
        for i in range(8):
            out_ch = base_channels * min(2**i, 8)
            layers.append(nn.Conv2d(curr_ch, out_ch, 4, 2, 1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            curr_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(curr_ch * 1 * 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        return self.classifier(x)


# ----------------------------------------
# 2) Drum-pattern VAE components (Wei et al. §3.4, §4.4)
# ----------------------------------------

class DrumEncoder(nn.Module):
    """
    Encodes an 84×96×8 CQT bar-selection tensor into a 32-dim latent.
    - Similar conv stack as SSMEncoder, but tailored to input size.
    """
    def __init__(self, in_channels=8, base_channels=64, latent_dim=32):
        super().__init__()
        layers = []
        curr_ch = in_channels
        # 7 downsampling conv blocks to reach ~1×1
        for i in range(7):
            out_ch = base_channels * min(2**i, 8)
            layers.append(nn.Conv2d(curr_ch, out_ch, 4, 2, 1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            curr_ch = out_ch
        self.conv = nn.Sequential(*layers)
        # feature map dims ≈ 1×1
        self.fc_mu     = nn.Linear(curr_ch * 1 * 1, latent_dim)
        self.fc_logvar = nn.Linear(curr_ch * 1 * 1, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)


class DrumDecoder(nn.Module):
    """
    Decodes a 32-dim latent back to a 46×16×1 drum pattern.
    - FC → 512×1×1, then 7 ConvTranspose2d to reach 46×16 spatial.
    """
    def __init__(self, out_channels=1, base_channels=64, latent_dim=32):
        super().__init__()
        # initial FC
        curr_ch = base_channels * 4  # choose a mid-level channel size
        self.fc = nn.Linear(latent_dim, curr_ch * 1 * 1)
        # 7 upsampling deconv blocks
        layers = []
        for i in reversed(range(7)):
            in_ch = base_channels * min(2**(i+1), 8)
            out_ch = base_channels * min(2**i, 8) if i>0 else out_channels
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1))
            if i>0:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Sigmoid())
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(x.size(0), -1, 1, 1)
        x = self.deconv(x)  # (B,1,46,16)
        return x


class DrumVAE(nn.Module):
    """Wraps DrumEncoder + reparam + DrumDecoder into the drum-pattern VAE."""
    def __init__(self, encoder: DrumEncoder, decoder: DrumDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# End of models.py
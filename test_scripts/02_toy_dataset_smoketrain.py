# scripts/02_toy_dataset_smoketrain.py
# run by using command: docker exec -it vaegan-dev python3 test_scripts/02_toy_dataset_smoketrain.py

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import SSMEncoder, SSMDecoder, SSMVAE, SSMDiscriminator

class ToySSMDataset(Dataset):
    def __init__(self, n=64):
        super().__init__()
        torch.manual_seed(0)
        self.mel = torch.rand(n, 1, 256, 256)  # input "melodic SSM"
        # target "drum SSM" is a blurred version (easy mapping)
        self.drm = self.mel.clone().mul(0.7) + 0.3 * self.mel.clone().mean(dim=(2,3), keepdim=True)

    def __len__(self): return len(self.mel)
    def __getitem__(self, i): return self.mel[i], self.drm[i]

def kl_div(mu, logvar): 
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ToySSMDataset(n=64)
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    enc = SSMEncoder(1, 64, 32).to(device)
    dec = SSMDecoder(1, 64, 32).to(device)
    vae = SSMVAE(enc, dec).to(device)
    dis = SSMDiscriminator(1, 64).to(device)

    # optG = torch.optim.Adam(vae.parameters(), lr=1e-3)
    # optD = torch.optim.Adam(dis.parameters(), lr=1e-3)
    optG = torch.optim.Adam(vae.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(dis.parameters(), lr=1e-4, betas=(0.5, 0.999))

    bce = nn.BCEWithLogitsLoss()

    for epoch in range(3):
        gtot, dtot = 0.0, 0.0
        for mel, drm in dl:
            mel, drm = mel.to(device), drm.to(device)

            # D step
            optD.zero_grad(set_to_none=True)
            with torch.no_grad():
                recon, _, _ = vae(mel)
            d_real = dis(drm)
            d_fake = dis(recon.detach())
            d_loss = 0.5*(bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake)))
            d_loss.backward(); optD.step()

            # G step
            optG.zero_grad(set_to_none=True)
            recon, mu, logvar = vae(mel)
            rec = F.mse_loss(recon, drm)
            kl  = kl_div(mu, logvar)
            g_fake = dis(recon)
            gan = bce(g_fake, torch.ones_like(g_fake))
            g_loss = rec + 0.1*kl + 0.2*gan
            g_loss.backward(); optG.step()

            gtot += g_loss.item(); dtot += d_loss.item()

        print(f"epoch {epoch}: G {gtot/len(dl):.4f}, D {dtot/len(dl):.4f}")

        # After computing losses
        print(
          f"rec={rec.item():.4f}  kl={kl.item():.4f}  gan={gan.item():.4f}  "
          f"mean(D(real))={d_real.sigmoid().mean().item() if hasattr(d_real,'sigmoid') else d_real.mean().item():.3f}  "
          f"mean(D(fake))={d_fake.sigmoid().mean().item() if hasattr(d_fake,'sigmoid') else d_fake.mean().item():.3f}"
        )


if __name__ == "__main__":
    main()

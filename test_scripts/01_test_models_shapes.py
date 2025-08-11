# scripts/01_test_models_shapes.py
# run by using command: docker exec -it vaegan-dev python3 test_scripts/01_test_models_shapes.py

import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import SSMEncoder, SSMDecoder, SSMVAE, SSMDiscriminator

def main():
    torch.manual_seed(0)
    B = 2
    x = torch.randn(B, 1, 256, 256)  # fake melodic SSM

    enc = SSMEncoder(in_channels=1, base_channels=64, latent_dim=32)
    dec = SSMDecoder(out_channels=1, base_channels=64, latent_dim=32)
    vae = SSMVAE(enc, dec)
    dis = SSMDiscriminator(in_channels=1, base_channels=64)

    # Encoder
    mu, logvar = enc(x)
    assert mu.shape == (B, 32) and logvar.shape == (B, 32)

    # Reparameterize + decode
    recon, mu2, logvar2 = vae(x)
    assert recon.shape == (B, 1, 256, 256)

    # Discriminator
    score = dis(recon)
    assert score.shape == (B, 1)

    # One backward pass on tiny loss
    loss = (recon - x).pow(2).mean() + (mu2**2).mean() + (logvar2**2).mean()
    loss.backward()
    print("✓ shapes OK, backward OK")

if __name__ == "__main__":
    main()

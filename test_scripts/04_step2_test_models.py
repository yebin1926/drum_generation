import importlib
import torch
import pytest

# Try to support both pipelines:
#  - VAE pipeline: SSMEncoder + SSMDecoder + SSMVAE
#  - Arranger pipeline: SSMArranger
models = importlib.import_module("models")

def _has_vae(models_mod):
    return all(hasattr(models_mod, n) for n in ["SSMEncoder", "SSMDecoder", "SSMVAE"])

def _has_arranger(models_mod):
    return hasattr(models_mod, "SSMArranger")

@pytest.mark.parametrize("B", [2])
def test_model_forward_shapes(B):
    H = W = 256
    x_1ch = torch.randn(B, 1, H, W)

    if _has_vae(models):
        enc = models.SSMEncoder(in_channels=1, base_channels=8, latent_dim=8)
        dec = models.SSMDecoder(out_channels=1, base_channels=8, latent_dim=8)
        vae = models.SSMVAE(enc, dec).eval()
        with torch.no_grad():
            recon, mu, logvar = vae(x_1ch)
        assert recon.shape == (B, 1, H, W)
        assert mu.shape[0] == B and logvar.shape[0] == B

    elif _has_arranger(models):
        arr = models.SSMArranger(in_channels=1, base=8, out_channels=1).eval()
        with torch.no_grad():
            y = arr(x_1ch)
        assert y.shape == (B, 1, H, W)

    else:
        pytest.skip("No known model classes found in models.py")

def test_param_count_nonzero():
    if _has_vae(models):
        enc = models.SSMEncoder(in_channels=1, base_channels=8, latent_dim=8)
        dec = models.SSMDecoder(out_channels=1, base_channels=8, latent_dim=8)
        vae = models.SSMVAE(enc, dec)
        n = sum(p.numel() for p in vae.parameters() if p.requires_grad)
        assert n > 0
    elif _has_arranger(models):
        arr = models.SSMArranger(in_channels=1, base=8, out_channels=1)
        n = sum(p.numel() for p in arr.parameters() if p.requires_grad)
        assert n > 0
    else:
        pytest.skip("No known model classes found in models.py")

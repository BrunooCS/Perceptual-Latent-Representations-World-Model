# -------------------- Script Description ----------------------
# This script defines a compact miniVAE model.
# It includes the VAE architecture, reparameterization, encoding/decoding logic,
# and a KL divergence loss with free bits regularization.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import BasicBlock, UpStride, DownStride

# -------------------- miniVAE Model Definition ----------------------

class miniVAE(nn.Module):
    """
    Variational Autoencoder (VAE) model for mask generation.

    Args:
        input_size (Tuple[int, int, int]): Input dimensions (channels, height, width).
        latent_dim (int): Dimensionality of the latent space.
        base_ch (int): Number of base channels for the encoder/decoder.
    """
    def __init__(self, input_size: tuple = (4, 16, 16), latent_dim: int = 32, base_ch: int = 64):
        super(miniVAE, self).__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.base_ch    = base_ch

        # -------------------- Encoder ----------------------
        self.encoder = nn.Sequential(
            BasicBlock(4, base_ch),                     # 16 x 16
            DownStride(base_ch, base_ch * 2),            # 8 x 8
            BasicBlock(base_ch * 2, base_ch * 2),        # 8 x 8
            DownStride(base_ch * 2, base_ch * 4),        # 4 x 4
            BasicBlock(base_ch * 4, base_ch * 4),        # 4 x 4
            nn.Flatten()
        )

        self.flatten_dim = base_ch * 4 * 4 * 4           # Flattened feature size

        # Latent space projections
        self.fc_mu     = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # -------------------- Decoder ----------------------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_dim),
            nn.Unflatten(1, (base_ch * 4, 4, 4)),        # 4 x 4
            UpStride(base_ch * 4, base_ch * 4),          # 8 x 8
            BasicBlock(base_ch * 4, base_ch * 2),        # 8 x 8
            UpStride(base_ch * 2, base_ch * 2),          # 16 x 16
            BasicBlock(base_ch * 2, base_ch),            # 16 x 16
            nn.Conv2d(base_ch, 4, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Samples from the latent space using the reparameterization trick.

        Args:
            mu (torch.Tensor): Mean tensor.
            logvar (torch.Tensor): Log variance tensor.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)                   # Compute standard deviation
        eps = torch.randn_like(std)                     # Sample epsilon
        return mu + eps * std

    def encode(self, x: torch.Tensor) -> tuple:
        """
        Encodes the input tensor into latent space parameters.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log variance tensors.
        """
        h = self.encoder(x)                             # Extract latent features
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes a latent space vector into the reconstructed input.

        Args:
            z (torch.Tensor): Latent vector.

        Returns:
            torch.Tensor: Reconstructed input tensor.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Full forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - x_hat: Reconstructed input.
            - mu: Latent mean.
            - logvar: Latent log variance.
            - z: Sampled latent vector.
        """
        # 1) Encode input
        mu, logvar = self.encode(x)

        # 2) Sample from latent space
        z = self.reparametrize(mu, logvar)

        # 3) Decode latent vector
        x_hat = self.decode(z)

        return x_hat, mu, logvar, z

# -------------------- Free Bits KL Divergence ----------------------

def free_bits_kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 1e-2, alpha: float = 1.) -> torch.Tensor:
    """
    Computes the KL divergence loss using free bits strategy.

    Args:
        mu (torch.Tensor): Mean of the latent distribution [batch_size, latent_dim].
        logvar (torch.Tensor): Log variance of the latent distribution [batch_size, latent_dim].
        free_bits (float): Minimum KL divergence allowed per latent dimension.
        alpha (float): Scaling factor for the KL loss.

    Returns:
        torch.Tensor: KL divergence loss.
    """
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())    # KL per dimension
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)            # Apply free bits threshold
    kl_loss    = kl_per_dim.mean()                                 # Average over dimensions
    return kl_loss * alpha

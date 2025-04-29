"""
ProPos DDS+VAE model for image compression and reconstruction.

This model uses a U-Net, DDS feature selection, Downscaling/ Upscaling modules, and a miniVAE
to compress images into a latent space and reconstruct them.
"""

import torch
import torch.nn as nn
from .UpscaleDownscale.downscaling import DownscalingEncoder
from .UpscaleDownscale.upscaling import UpscalingDecoder
from .minivae.minivae import miniVAE
from .unet.unet import UNet
from .dds.dds import DDS

# -------------------- Vision Module ----------------------
class Vision(nn.Module):
    """
    Vision model for compressing and reconstructing images with mask-based processing.

    :param n_features_to_select: Number of features selected by DDS.
    :param in_ch: Number of input channels.
    :param out_ch: Number of output channels.
    :param base_ch: Base number of channels for UNet and other modules.
    :param alpha: Regularization parameter for DDS.
    :param delta: Threshold parameter for DDS.
    """
    def __init__(self, n_features_to_select: int, in_ch: int, out_ch: int, base_ch: int, 
                 alpha: float = 1.0, delta: float = 0.1, latent_dim: int = 32) -> None:
        super().__init__()

        self.unet1       = UNet(in_ch, out_ch, base_ch)
        self.dds         = DDS(n_features_to_select=n_features_to_select, alpha=alpha, delta=delta)
        self.downscaling = DownscalingEncoder(3, 4, base_ch)
        self.mini_vae    = miniVAE(input_size=(4, 16, 16), latent_dim=latent_dim, base_ch=base_ch * 4)
        self.upscaling   = UpscalingDecoder(4, 3, base_ch)
        self.unet2       = UNet(in_ch, out_ch, base_ch)


    # -------------------- Downscale Input ----------------------
    def downscale(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Downscale the input image through U-Net, DDS, and DownscalingEncoder.

        :param x: Input tensor [batch_size, in_ch, height, width].
        :return: Mini-mask, mask after DDS, binary mask.
        """
        s            = self.unet1(x)
        binary_mask, s = self.dds(s)
        mask         = x * (binary_mask * s)                              # Apply mask to original input
        mini_mask    = self.downscaling(mask)                             # Compress mask to mini-mask
        return mini_mask, mask, binary_mask


    # -------------------- Upscale Mini-Mask ----------------------
    def upscale(self, mini_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Upscale a mini-mask back to image space.

        :param mini_mask: Miniature mask tensor [batch_size, 4, 16, 16].
        :return: Reconstructed image and mask.
        """
        mask_hat = self.upscaling(mini_mask)
        x_hat    = torch.sigmoid(self.unet2(mask_hat))                    # Pass through U-Net and activation
        return x_hat, mask_hat


    # -------------------- Encode ----------------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode an input image into its latent representation.

        :param x: Input tensor [batch_size, in_ch, height, width].
        :return: Mask, mini-mask, sampled latent vector z.
        """
        mini_mask, mask, _ = self.downscale(x)
        mu, logvar         = self.mini_vae.encode(mini_mask)
        z                  = self.mini_vae.reparametrize(mu, logvar)
        return mask, mini_mask, z


    # -------------------- Decode ----------------------
    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode a latent vector back into image space.

        :param z: Latent vector [batch_size, latent_dim].
        :return: Reconstructed image, mask, and mini-mask.
        """
        mini_mask_hat = self.mini_vae.decode(z)
        x_hat, mask_hat = self.upscale(mini_mask_hat)
        return x_hat, mask_hat, mini_mask_hat


    # -------------------- Encode Eval (No Sampling) ----------------------
    def encode_eval(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input deterministically using the mean vector (mu).

        :param x: Input tensor [batch_size, in_ch, height, width].
        :return: Mask, mini-mask, mean latent vector.
        """
        mini_mask, mask, _ = self.downscale(x)
        mu, _              = self.mini_vae.encode(mini_mask)
        z                  = mu                                              # Use mean directly
        return mask, mini_mask, z


    # -------------------- Perceptual Loss ----------------------
    def perceptual_loss_upscaling(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the perceptual loss between predicted and target images after upscaling.

        :param pred: Predicted tensor [batch_size, in_ch, height, width].
        :param target: Target tensor [batch_size, in_ch, height, width].
        :return: Perceptual MSE loss.
        """
        pred_mask_features = self.upscaling(pred, return_sequence=True)

        with torch.no_grad():                                                # No gradients for target
            target_mask_features = self.upscaling(target, return_sequence=True)

        loss = sum(
            nn.functional.mse_loss(pred_feat, target_feat, reduction='mean')
            for pred_feat, target_feat in zip(pred_mask_features, target_mask_features)
        ) / len(pred_mask_features)

        return loss

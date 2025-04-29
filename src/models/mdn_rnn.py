import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional


# -------------------- RNN Module ----------------------
class RNN(nn.Module):
    """
    Recurrent Neural Network (LSTM) for combining latent space and action inputs.

    :param latent_dim: Dimensionality of the latent space.
    :param action_dim: Dimensionality of actions.
    :param hidden_dim: Number of hidden units in the LSTM.
    :param num_layers: Number of LSTM layers.
    """

    def __init__(self,latent_dim: int,action_dim: int,hidden_dim: int,num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm       = nn.LSTM(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self,z: torch.Tensor,a: torch.Tensor,h = None):
        """
        Forward pass through the LSTM using latent space and action inputs.

        :param z: Latent space tensor with shape (batch, seq_len, latent_dim).
        :param a: Action tensor with shape (batch, seq_len, action_dim).
        :param h: Optional (hidden_state, cell_state) tuple for the LSTM.
        :return: Tuple (outs_rnn, (hn, cn)), where outs_rnn is the LSTM outputs
                 and (hn, cn) is the new LSTM hidden state.
        """
        x = torch.cat([z, a], dim=-1)                 # Combine latent space (z) and action (a)
        outs_rnn,h = self.lstm(x, h)                  # Forward pass through LSTM
        return outs_rnn, h

    def init_hidden(self, batch_size: int, device: str = 'cuda'):
        """
        Initialize hidden and cell states for LSTM.

        :param batch_size: Number of sequences per batch.
        :param device: Device for tensor allocation ('cpu' or 'cuda').
        :return: Tuple of zero-initialized hidden and cell states (hn, cn).
        """
        return (
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
            torch.zeros(1, batch_size, self.hidden_dim, device=device)
        )


# -------------------- Mixture Density Network (MDN) ----------------------
class MDN(nn.Module):
    """
    Mixture Density Network that receives hidden states and returns Gaussian mixture parameters.

    :param latent_dim: Dimensionality of the latent space.
    :param hidden_dim: Number of hidden units in the LSTM.
    :param num_gaussians: Number of Gaussian components in the mixture.
    """

    def __init__(self,latent_dim: int,hidden_dim: int,num_gaussians: int):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.latent_dim    = latent_dim

        # Mixture weights (π), Means (μ), and Std Devs (σ)
        self.fc_pi    = nn.Linear(hidden_dim, num_gaussians)
        self.fc_mu    = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, num_gaussians * latent_dim)

    def forward(self,outs_rnn: torch.Tensor,tau: float):
        """
        Forward pass through MDN to obtain mixture weights, means, and standard deviations.

        :param outs_rnn: Hidden outputs from the RNN, shape (batch, seq_len, hidden_dim).
        :param tau: Temperature parameter to control the 'sharpness' of the distributions.
        :return: Tuple (pi, mu, sigma):
                 - pi has shape (batch, seq_len, num_gaussians)
                 - mu has shape (batch, seq_len, num_gaussians, latent_dim)
                 - sigma has shape (batch, seq_len, num_gaussians, latent_dim)
        """
        tau_tensor = torch.tensor(tau, device=outs_rnn.device)

        pi    = self.fc_pi(outs_rnn)                     # Mixture weights
        mu    = self.fc_mu(outs_rnn)                     # Means
        sigma = self.fc_sigma(outs_rnn)                  # Std devs

        batch_size, seq_len, _ = outs_rnn.size()
        mu    = mu.view(batch_size, seq_len, self.num_gaussians, self.latent_dim)
        sigma = sigma.view(batch_size, seq_len, self.num_gaussians, self.latent_dim)

        sigma = torch.exp(sigma) + 1e-15                 # Ensure σ > 0
        sigma = sigma * torch.sqrt(tau_tensor)           # Control sigma with temperature

        pi = F.softmax(pi / tau_tensor, dim=-1) + 1e-15  # Mixture weights with temperature
        return pi, mu, sigma


# -------------------- Combined MDN-RNN ----------------------
class MDNRNN(nn.Module):
    """
    Combined MDN-RNN model that integrates the LSTM (RNN) and the Mixture Density Network (MDN).

    :param latent_dim: Dimensionality of the latent space.
    :param action_dim: Dimensionality of actions.
    :param hidden_dim: Number of hidden units in the LSTM.
    :param num_gaussians: Number of Gaussian components in the mixture.
    :param num_layers: Number of LSTM layers.
    """

    def __init__(self,latent_dim: int,action_dim: int,hidden_dim: int,num_gaussians: int,num_layers: int = 1):
        super().__init__()
        self.rnn = RNN(latent_dim, action_dim, hidden_dim, num_layers)
        self.mdn = MDN(latent_dim, hidden_dim, num_gaussians)

    def forward(self,z: torch.Tensor,a: torch.Tensor,tau: float = 1.0,h = None):
        """
        Forward pass through the combined MDN-RNN, producing MDN parameters from RNN outputs.

        :param z: Latent space tensor with shape (batch, seq_len, latent_dim).
        :param a: Action tensor with shape (batch, seq_len, action_dim).
        :param tau: Temperature parameter for the MDN output.
        :param h: Optional (hidden_state, cell_state) tuple for the LSTM.
        :return: Tuple (pi, mu, sigma, (hn, cn)) for the mixture components and new RNN hidden state.
        """
        outs_rnn, h = self.rnn(z, a, h)
        pi, mu, sigma = self.mdn(outs_rnn, tau)
        return pi, mu, sigma, h


# -------------------- MDN Sampling ----------------------
def sample_mdn(pi: torch.Tensor,mu: torch.Tensor,sigma: torch.Tensor):
    """
    Sample a latent vector z from the MDN parameters, assuming batch_size=1 and seq_len=1.

    :param pi: Mixture weights with shape (1, 1, num_gaussians).
    :param mu: Means with shape (1, 1, num_gaussians, latent_dim).
    :param sigma: Std devs with shape (1, 1, num_gaussians, latent_dim).
    :return: Sampled latent vector z of shape (1, 1, latent_dim).
    """
    component_index = torch.multinomial(pi.view(-1), 1).item()   # Select Gaussian by weights
    selected_mu     = mu[0, 0, component_index]                  # Mean of selected Gaussian
    selected_sigma  = sigma[0, 0, component_index]               # Std dev of selected Gaussian
    epsilon         = torch.randn_like(selected_mu)              # Random noise
    z               = selected_mu + selected_sigma * epsilon     # Sample z
    return z.unsqueeze(0).unsqueeze(0)


# -------------------- Negative Log-Likelihood Loss ----------------------
def gaussian_nll_loss(pi: torch.Tensor,mu: torch.Tensor,sigma: torch.Tensor,z_next: torch.Tensor):
    """
    Compute the Negative Log-Likelihood (NLL) loss for the Mixture Density Network.

    :param pi: Mixture weights with shape (batch, seq_len, num_gaussians).
    :param mu: Means with shape (batch, seq_len, num_gaussians, latent_dim).
    :param sigma: Std devs with shape (batch, seq_len, num_gaussians, latent_dim).
    :param z_next: Target latent vector with shape (batch, seq_len, latent_dim).
    :return: Scalar NLL loss.
    """
    z_next_expanded = z_next.unsqueeze(2)                             # Expand for mixture dimension
    log_pi          = torch.log(pi)
    normal_dist     = Normal(loc=mu, scale=sigma)                     # Gaussian distributions

    log_prob     = normal_dist.log_prob(z_next_expanded).sum(-1)      # Sum log-probs over latents
    log_sum_exp  = torch.logsumexp(log_pi + log_prob, dim=-1)         # Log-sum-exp over mixtures
    nll          = -log_sum_exp.mean()                                # Mean negative log-likelihood
    return nll
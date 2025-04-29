import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

# -------------------- DDS Module ----------------------
class DDS(nn.Module):
    def __init__(self,n_features_to_select: Union[int, float],temperature: float = 2. / 3.,
                 alpha: float = 1.0,delta: float = 0.1,start_dim: int = 1,largest: bool = True,
                 epsilon: float = 1e-6,limit_a: float = 0.0,limit_b: float = 1.0):
        """
        Implementation of L0 regularization for the input units of a fully connected layer.

        :param n_features_to_select: Number or fraction of features to retain.
        :param temperature: Temperature of the concrete distribution for sampling gates.
        :param alpha: Scale factor for gate logits.
        :param delta: Probability of stochastic exploration.
        :param start_dim: Flattening start dimension for multi-dimensional inputs.
        :param largest: Whether to select largest values in top-k.
        :param epsilon: Small constant to avoid log(0).
        :param limit_a: Lower limit for hard-concrete gates.
        :param limit_b: Upper limit for hard-concrete gates.
        """
        super(DDS, self).__init__()
        assert n_features_to_select > 0, "n_features_to_select must be greater than zero."
        self.n_features_to_select = n_features_to_select
        self.temperature           = temperature
        self.alpha                 = alpha
        self.delta                 = delta
        self.start_dim             = start_dim
        self.largest               = largest
        self.epsilon               = epsilon
        self.limit_a               = limit_a
        self.limit_b               = limit_b

    def sample_z(self, u, sample=True):
        """
        Sample hard-concrete gates during training or use deterministic values during evaluation.
        
        Parameters:
        :u: Input logits.
        :sample: If True, adds noise for stochastic gate sampling.
        
        Returns:
        :s: Scaled gate values.
        :mask: Binary mask indicating selected features.
        """
        if sample:
            eps = torch.rand_like(u).clamp(self.epsilon, 1. - self.epsilon)
            z = torch.sigmoid((self.alpha * (torch.log(eps) - torch.log(1 - eps)) + u) / self.temperature)
        else:
            z = torch.sigmoid(u / self.temperature)

        # Determine number of features to select
        if isinstance(self.n_features_to_select, int):
            n_features = self.n_features_to_select
        else:
            n_features = int(u.size(-1) * self.n_features_to_select)

        if self.n_features_to_select is None or self.n_features_to_select == 1.0:
            mask = torch.ones_like(z)
        else:
            # Select top-k features
            _, indices = torch.topk(z, n_features, dim=-1, largest=self.largest)
            mask = torch.zeros_like(z)
            mask.scatter_(-1, indices, 1.0)

            # Add stochastic exploration
            if self.delta > 0 and sample:
                random_mask = (torch.rand_like(mask[:, :1]) < self.delta).float()
                mask = torch.where(random_mask > 0, torch.ones_like(mask), mask)

        # Scale the gates within [limit_a, limit_b]
        s = z * (self.limit_b - self.limit_a) + self.limit_a
        return s, mask

    def forward(self, x):
        """
        Forward pass for applying gates and masks to input.
        
        Parameters:
        :x: Input tensor.
        
        Returns:
        :mask: Binary mask indicating selected features.
        :s: Scaled and gated tensor.
        """
        x_shape = x.size()
        if len(x_shape) > 2:
            x = x.flatten(start_dim=self.start_dim)

        x = x + 1.

        # Sample gates and generate mask
        r, mask = self.sample_z(x, sample=self.training)
        s = F.hardtanh(r, min_val=self.limit_a, max_val=self.limit_b)

        # Reshape mask and gated output if input was multi-dimensional
        if len(x_shape) > 2:
            mask = mask.view(x_shape)
            s = s.view(x_shape)

        return mask, s
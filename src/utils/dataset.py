import os
import h5py
import torch
import numpy as np
import pandas as pd
import pygame
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import plotly.graph_objects as go
import gymnasium as gym

# ---------------------------------
# Global Constants / Configuration
# ---------------------------------
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------
# Dataset Classes
# ---------------------------------
class CarRacingDataset(Dataset):
    """
    A PyTorch Dataset for loading CarRacing data from an HDF5 file.

    Args:
        h5_path (str): Path to the HDF5 file.
        transform (transforms.Compose, optional): Transform to be applied 
            on the input images. Defaults to transforms.ToTensor().
    """
    def __init__(self, h5_path: str, 
                 transform: transforms.Compose = None) -> None:
        # If no transform is provided, default to ToTensor
        self.transform = transform if transform else transforms.ToTensor()
        self.h5_path = h5_path
        
        # Extract dataset dimensions
        with h5py.File(self.h5_path, "r") as h5f:
            self.num_episodes, self.max_steps = h5f["images"].shape[:2]
            self.total_frames = self.num_episodes * self.max_steps
        
        # We'll lazily open the file
        self.h5_file = None

    def __len__(self) -> int:
        return self.total_frames

    def __getitem__(self, idx: int):
        """
        Retrieve a single frame from the dataset.

        Args:
            idx (int): Index to retrieve data from.

        Returns:
            tuple: (image, action, reward, done), each as a torch.Tensor.
        """
        # Lazy open
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        
        episode = idx // self.max_steps
        step = idx % self.max_steps
        
        # Read data from HDF5
        image = self.h5_file["images"][episode, step]
        action = self.h5_file["actions"][episode, step]
        reward = self.h5_file["rewards"][episode, step]
        done = self.h5_file["dones"][episode, step]
        
        # Apply transforms
        image = self.transform(image)
        
        # Convert to torch.Tensor
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        
        return image, action, reward, done

    def __del__(self) -> None:
        """
        Ensure the HDF5 file is closed upon deletion.
        """
        if self.h5_file:
            self.h5_file.close()


class CarRacingDataset_RNN(Dataset):
    """
    A PyTorch Dataset that returns full episodes for RNN-based models.

    Args:
        h5_path (str): Path to the HDF5 file.
        vision (nn.Module): Model (e.g., VAE) to encode images into latent states.
        device (torch.device): Torch device ('cpu' or 'cuda').
        transform (transforms.Compose, optional): Transform to be applied 
            on the input images. Defaults to transforms.ToTensor().
    """
    def __init__(
        self, 
        h5_path: str, 
        vision: nn.Module, 
        device: torch.device, 
        transform: transforms.Compose = None
    ) -> None:
        self.h5_path = h5_path
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.vision = vision
        self.device = device

        # Open HDF5 to extract dimensions
        with h5py.File(self.h5_path, "r") as h5f:
            self.num_episodes = h5f["images"].shape[0]
            self.max_steps = h5f["images"].shape[1]

        self.h5_file = None
        self.vision.eval()

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int):
        """
        Retrieve a single episode from the dataset.

        Args:
            idx (int): Index of the episode to retrieve.

        Returns:
            tuple: z, z_next, actions, rewards, dones 
                   (all torch.Tensor).
        """
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        images = self.h5_file["images"][idx]
        actions = self.h5_file["actions"][idx]
        rewards = self.h5_file["rewards"][idx]
        dones = self.h5_file["dones"][idx]

        # Process each sequence to get the encoded images
        with torch.no_grad():
            x = torch.stack([self.transform(img) for img in images]).to(self.device)
            _, _, z = self.vision.encode(x)

        # Shift latent vectors to get z_next
        z_next = z[1:]
        z = z[:-1]

        actions = torch.tensor(actions[:-1], dtype=torch.float32)
        rewards = torch.tensor(rewards[:-1], dtype=torch.float32)
        dones = torch.tensor(dones[:-1], dtype=torch.float32)

        return z, z_next, actions, rewards, dones

    def __del__(self) -> None:
        """
        Ensure the HDF5 file is closed upon deletion.
        """
        if self.h5_file is not None:
            self.h5_file.close()

# Copyright (c) 2024-2025, Bruno Corcuera
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Find the full license text in the LICENSE file at the root of this repository.

from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import gymnasium as gym
import os
import cv2
from tqdm import tqdm
import pygame

TRANSFORM = transforms.Compose([
    transforms.Lambda(lambda img: img[:-12, :, :]),       # Crop bottom 12 pixels
    transforms.ToPILImage(),                               # Convert to PIL image
    transforms.Resize((96, 96), transforms.InterpolationMode.LANCZOS),  # Resize
    transforms.ToTensor()                                  # Convert to tensor
])



def reset(env: gym.Env) -> np.ndarray:
    """
    Reset the environment.

    Args:
        env (gym.Env): The CarRacing environment.

    Returns:
        np.ndarray: The initial observation after some random steps.
    """
    obs, _ = env.reset()
    # Perform a few no-op steps to clear environment
    for _ in range(1):
        obs, _, _, _, _ = env.step(np.zeros(3))
    return obs

def ensure_dir_exists(directory: str) -> None:
    """
    Ensure the specified directory exists.

    Args:
        directory (str): Directory to check/create.
    """
    os.makedirs(directory, exist_ok=True)

def setup_video_writer(
    filename: str, 
    resolution: tuple, 
    fps: int = 30
) -> cv2.VideoWriter:
    """
    Set up a video writer for saving gameplay.

    Args:
        filename (str): File path for the saved video.
        resolution (tuple): (width, height) of the video.
        fps (int, optional): Frames per second. Defaults to 30.

    Returns:
        cv2.VideoWriter: An OpenCV video writer object.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    return cv2.VideoWriter(filename, fourcc, fps, resolution)

def get_action(keys: pygame.key.ScancodeWrapper) -> np.ndarray:
    """
    Map keyboard input to environment actions.

    Args:
        keys (pygame.key.ScancodeWrapper): Key states from pygame.

    Returns:
        np.ndarray: The action array [steering, gas, brake].
    """
    action = np.zeros(3)
    action[0] = -1.0 if keys[pygame.K_LEFT] else (1.0 if keys[pygame.K_RIGHT] else 0.0)
    action[1] = 1.0 if keys[pygame.K_UP] else 0.0
    action[2] = 1.0 if keys[pygame.K_DOWN] else 0.0
    return action

def preprocess_image(
    tensor: torch.Tensor, 
    resolution: tuple
) -> np.ndarray:
    """
    Preprocess an image tensor for display or saving.

    Args:
        tensor (torch.Tensor): The image tensor with shape [1, H, W, C].
        resolution (tuple): Desired (width, height) for resizing.

    Returns:
        np.ndarray: Resized and converted image (BGR) for OpenCV.
    """
    # Convert from [C, H, W] -> [H, W, C] and invert the color channel order if needed
    img = (tensor.squeeze(0).permute(2, 1, 0).cpu().numpy()[:, :, :3] * 255).astype(np.uint8)
    return cv2.resize(img, resolution, interpolation=cv2.INTER_NEAREST)
# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms
import glob
import sys
from tqdm import tqdm
from IPython.display import display, HTML, clear_output

from src.models.dds_vae import Vision

import gymnasium as gym
import pygame
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
from src.utils.utils import TRANSFORM as transform
from src.models.mdn_rnn import sample_mdn



# Instantiate model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vision = Vision(n_features_to_select=0.03, 
                in_ch=3, 
                out_ch=3, 
                base_ch=16, 
                alpha=1.0, 
                delta=0.1
).to(device)
vision.load_state_dict(torch.load('src/trained_models/vision_03_miniVAE.pth', map_location=device, weights_only=True))
vision.eval()


from src.models.mdn_rnn import MDNRNN

memory = MDNRNN(latent_dim=32, 
                action_dim=3, 
                hidden_dim=256, 
                num_gaussians=5
).to(device)

memory.load_state_dict(torch.load('src/trained_models/memory.pth', map_location=device, weights_only=True))



import gymnasium as gym
import pygame
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from src.utils.utils import setup_video_writer

def run_car_racing_rnn_mda(env_name, vision, mdnrnn, transform, device, scale=1, resolution=(150, 150), tau=1.0, video_filepath='renders/dream.mp4', save_video=False):
    import pygame
    import numpy as np
    import torch
    import cv2

    # Initialize the environment
    env = gym.make(env_name, render_mode='rgb_array')
    obs, _ = env.reset()
    for _ in range(0):
        env.step(np.array([0, 0, 0]))

    # Initialize pygame for rendering
    pygame.init()
    resolution = (resolution[0] * scale, resolution[1] * scale)
    screen = pygame.display.set_mode(resolution)
    clock = pygame.time.Clock()

    action = np.zeros(3)  # Initialize action array
    video_writer = setup_video_writer(video_filepath, resolution[::-1]) if save_video else None

    def get_action(keys):
        """ Map keyboard input to actions """
        action[0] = -1.0 if keys[pygame.K_LEFT] else 1.0 if keys[pygame.K_RIGHT] else 0.0  # Steering
        action[1] = 1.0 if keys[pygame.K_UP] else 0.0  # Accelerate
        action[2] = 1.0 if keys[pygame.K_DOWN] else 0.0  # Brake
        return action

    running = True
    h = mdnrnn.rnn.init_hidden(1)
    h = (h[0].to(device), h[1].to(device))

    cnt = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()  # Get current key states
        action = get_action(keys)  # Update action based on key presses

        # Environment step
        obs, reward, done, info, _ = env.step(action)
        obs_tensor = transform(obs).unsqueeze(0).to(device)  # Transform frame to tensor

        with torch.no_grad():
            if cnt == 0:
                mask, mini_mask, z = vision.encode(obs_tensor)
                z = z.unsqueeze(0)
            else:
                z = z_next

            action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            pi, mu, sigma, h = mdnrnn(z, action_tensor, h=h, tau=tau)
            z_next = sample_mdn(pi, mu, sigma)

            x_hat, mask_hat, mini_mask_hat = vision.decode(z_next.squeeze(0))
            reconstructed = x_hat

        cnt += 1

        # Prepare the reconstructed image for display
        reconstructed = (reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        reconstructed_resized = cv2.resize(reconstructed, (resolution[0], resolution[1]))

        # Convert reconstructed image to pygame surface
        frame_surface = pygame.surfarray.make_surface(reconstructed_resized.swapaxes(0, 1))

        # Blit frame to the screen
        screen.blit(frame_surface, (0, 0))

        # Draw key indicators using pygame shapes
        indicator_color = (255, 0, 0)  # Red color for indicators

        # Left arrow (proportional size)
        if keys[pygame.K_LEFT]:
            pygame.draw.polygon(screen, indicator_color, [(40, resolution[1] - 40), (20, resolution[1] - 30), (40, resolution[1] - 20)])

        # Right arrow (proportional size)
        if keys[pygame.K_RIGHT]:
            pygame.draw.polygon(screen, indicator_color, [(100, resolution[1] - 40), (120, resolution[1] - 30), (100, resolution[1] - 20)])

        # Up arrow (centered and proportional)
        if keys[pygame.K_UP]:
            pygame.draw.polygon(screen, indicator_color, [(70, resolution[1] - 70), (50, resolution[1] - 50), (90, resolution[1] - 50)])

        # Down arrow (proportional rectangle)
        if keys[pygame.K_DOWN]:
            pygame.draw.rect(screen, indicator_color, (140, resolution[1] - 60, 40, 40))

        # Update the display
        clock.tick(30)
        pygame.display.flip()

        if save_video and video_writer is not None:
            video_writer.write(cv2.cvtColor(reconstructed_resized, cv2.COLOR_RGB2BGR))

        if done:
            obs = env.reset()
            h = mdnrnn.rnn.init_hidden(1)
            h = (h[0].to(device), h[1].to(device))

    pygame.quit()
    env.close()

    if save_video and video_writer is not None:
        video_writer.release()


run_car_racing_rnn_mda(env_name="CarRacing-v3", vision=vision, mdnrnn=memory, transform=transform, device=device, scale=4, tau=.001, save_video=False)
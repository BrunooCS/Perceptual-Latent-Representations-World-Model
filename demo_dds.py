
import torch
from src.models.dds_vae import Vision
from src.utils.dataset import CarRacingDataset
from torch.utils.data import DataLoader

import os
import gymnasium as gym
import pygame
import numpy as np
import torch
from torchvision import transforms
import cv2
from src.utils.utils import ensure_dir_exists, setup_video_writer, reset, get_action, preprocess_image
from src.utils.utils import TRANSFORM as transform

# Setup device & load weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vision = Vision(n_features_to_select=0.03, in_ch=3, out_ch=3, base_ch=16, alpha=1.0, delta=0.1).to(device)
vision.load_state_dict(torch.load('src/trained_models/vision_03_miniVAE.pth', map_location=device, weights_only=True))
vision.eval()

# Prepare data
dataset = CarRacingDataset('src/car_racing_data_1024.h5')
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)



def run_car_racing(env_name, transform, device, scale=1, resolution=(96, 96), save_video=False, video_filename="car_racing_vision.mp4"):
    """Run the car racing environment with specified settings."""

    # --- Setup Directories and Video Writer ---
    ensure_dir_exists('renders')
    video_filepath = os.path.join('renders', video_filename)

    pygame.init()
    display_resolution = (resolution[0] * 4 * scale, resolution[1] * scale)
    screen = pygame.display.set_mode(display_resolution)
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    video_writer = setup_video_writer(video_filepath, display_resolution[::-1]) if save_video else None

    # --- Initialize Environment ---
    env = gym.make(env_name, render_mode='rgb_array')
    obs = reset(env)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Handle Actions ---
        keys = pygame.key.get_pressed()
        action = get_action(keys)
        obs, reward, done, info, _ = env.step(action)

        # --- Transform and Process Observation ---
        x = transform(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            mini_mask, mask, binary_mask = vision.downscale(x)
            x_hat, mask_hat = vision.upscale(mini_mask)

        mse = torch.mean((x - x_hat) ** 2).item()

        images = [x, mask, mini_mask, x_hat]
        processed_images = [preprocess_image(img, resolution) for img in images]
        full_image = np.concatenate(processed_images, axis=0)
        full_image_resized = cv2.resize(full_image, (display_resolution[1], display_resolution[0]), interpolation=cv2.INTER_LINEAR)

        # --- Render and Display ---
        pygame.surfarray.blit_array(screen, full_image_resized)
        mse_text = font.render(f"MSE: {mse:.5f}", True, (255, 255, 255))
        screen.blit(mse_text, (10, 10))
        pygame.display.flip()

        if save_video and video_writer is not None:
            video_writer.write(cv2.cvtColor(full_image_resized, cv2.COLOR_RGB2BGR))

        clock.tick(30)
        if done:
            obs = reset(env)

    # --- Cleanup ---
    if save_video and video_writer is not None:
        video_writer.release()

    pygame.quit()
    env.close()

# --- Main Execution ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
run_car_racing(env_name="CarRacing-v3", transform=transform, device=device, scale=5, save_video=False)


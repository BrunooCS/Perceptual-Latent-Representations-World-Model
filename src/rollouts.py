"""
Generate a dataset for the CarRacing environment to train the World Model.

This script collects observations (images), actions, rewards, and done flags from
multiple episodes using a pseudo-random action strategy, processes the images,
and stores the data in an HDF5 file.

It leverages multiprocessing for efficiency and consolidates episode data into a
single dataset.

:example: 
          $ ./script_name.py --num_episodes 100 --max_steps 1000
"""

import argparse
import gc
import os
from multiprocessing import Pool, cpu_count
from typing import Tuple, Optional

import gymnasium as gym
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def resize_obs(image: np.ndarray, target_size: Tuple[int, int]):
    """
    Crop the bottom 12 pixels of the observation image and resize it.

    :param image: Raw observation image from the environment.
    :param target_size: Desired output dimensions as (width, height).
    :return: Processed image of shape target_size.
    """
    image_cropped = image[:-12, :, :]                        # Remove bottom info bar
    img = Image.fromarray(image_cropped)                     # Convert to PIL Image
    img = img.resize(target_size, Image.Resampling.LANCZOS)  # Resize to target size
    return np.array(img)


def random_action(t: int):
    """
    Generate a pseudo-random action based on the current time step.

    For the first 20 steps, applies a fixed action to start the car. Afterward,
    selects actions with the following probabilities:
      - Accelerate : 35%
      - Turn Left  : 30%
      - Turn Right : 30%
      - Brake      : 5%

    :param t: Current time step in the episode.
    :return: Action vector [steering, acceleration, brake].
    """
    if t < 20:
        return np.array([-.1, 1, 0], dtype=np.float32)

    actions = [
        np.array([0, np.random.random(), 0], dtype=np.float32),      # Random accelerate
        np.array([-np.random.random(), 0, 0], dtype=np.float32),     # Random left
        np.array([np.random.random(), 0, 0], dtype=np.float32),      # Random right
        np.array([0, 0, np.random.random()], dtype=np.float32),      # Random brake
    ]
    probabilities = [0.35, 0.3, 0.3, 0.05]
    selected_action = np.random.choice(len(actions), p=probabilities)
    return actions[selected_action]


def collect_episode(env_name: str, max_steps: int, episode_id: int,
                    worker_id: int, output_dir: str, image_size: Tuple[int, int]):
    """
    Collect data for a single episode and save it to an HDF5 file.

    :param env_name: Name of the Gym environment.
    :param max_steps: Number of steps to collect per episode.
    :param episode_id: Unique identifier for the episode within the worker.
    :param worker_id: Unique identifier for the worker process.
    :param output_dir: Directory to store the temporary HDF5 file.
    :param image_size: Desired image size as (width, height).
    """
    episode_images = []
    episode_actions = []
    episode_rewards = []
    episode_dones = []

    # Start environment
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()

    for step in range(max_steps):
        # 1) Process observation: crop and resize image
        resized_image = resize_obs(obs, image_size)
        episode_images.append(resized_image)

        # 2) Generate new action every 20 steps, else reuse previous action
        if step % 20 == 0:
            action = random_action(step)
        episode_actions.append(action)

        # 3) Take a step in the environment and record reward and done flag
        obs, reward, done, truncated, _ = env.step(action)
        episode_rewards.append(reward)
        episode_dones.append(done)

        # 4) If the episode ended, reset the environment
        if done or truncated:
            obs, _ = env.reset()

    episode_images = np.array(episode_images, dtype=np.uint8)
    episode_actions = np.array(episode_actions, dtype=np.float32)
    episode_rewards = np.array(episode_rewards, dtype=np.float32)
    episode_dones = np.array(episode_dones, dtype=bool)

    # Save episode data to an HDF5 file
    episode_filename = f"worker_{worker_id}_episode_{episode_id}.h5"   # Filename for episode
    episode_path = os.path.join(output_dir, episode_filename)           # Full file path
    with h5py.File(episode_path, "w") as h5f:
        h5f.create_dataset("images", data=episode_images, dtype="uint8")
        h5f.create_dataset("actions", data=episode_actions, dtype="float32")
        h5f.create_dataset("rewards", data=episode_rewards, dtype="float32")
        h5f.create_dataset("dones", data=episode_dones, dtype="bool")

    # Clean up: close environment and free memory
    env.close()
    del episode_images, episode_actions, episode_rewards, episode_dones
    del obs, action, reward, done, truncated
    gc.collect()


def collect_data_worker(args: Tuple[str, int, int, str, int, Tuple[int, int]]):
    """
    Worker function for multiprocessing to collect multiple episodes.

    :param args: Tuple containing (env_name, num_episodes, max_steps, output_dir, worker_id, image_size).
    """
    env_name, num_episodes, max_steps, output_dir, worker_id, image_size = args
    # Collect episodes in a loop for this worker
    for ep in range(num_episodes):
        collect_episode(env_name, max_steps, ep, worker_id, output_dir, image_size)
        print(f"Worker {worker_id}: Episode {ep + 1}/{num_episodes} completed")
    print(f"Worker {worker_id}: Finished collecting {num_episodes} episodes.")


def collect_data(env_name: str = "CarRacing-v3", num_episodes: int = 100, max_steps: int = 1000,
                 output_dir: str = "temp_episodes_data", num_workers: Optional[int] = None,
                 image_size: Tuple[int, int] = (96, 96)):
    """
    Collect data across multiple episodes using parallel processing.

    :param env_name: Name of the Gym environment.
    :param num_episodes: Total number of episodes to collect.
    :param max_steps: Number of steps per episode.
    :param output_dir: Directory for temporary episode files.
    :param num_workers: Number of worker processes; if None, defaults to cpu_count() - 1.
    :param image_size: Desired image size as (width, height).
    """
    # Max cpu cores
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting data collection for {num_episodes} episodes with {num_workers} workers...")

    # Distribute episodes evenly across workers
    episodes_per_worker = num_episodes // num_workers
    remaining_episodes = num_episodes % num_workers
    worker_args = [
        (env_name,
         episodes_per_worker + (1 if i < remaining_episodes else 0),
         max_steps, output_dir, i, image_size)
        for i in range(num_workers)
    ]

    # Run worker processes in parallel using a process pool
    with Pool(processes=num_workers) as pool:
        pool.map(collect_data_worker, worker_args)

    print(f"Data collection completed. Temporary files saved in '{output_dir}'.")


def merge_episode_files(episodes_dir: str = "temp_episodes_data", output_file: str = "car_racing_data.h5",
                        num_max_steps: int = 1000, image_width: int = 96,
                        image_height: int = 96):
    """
    Merge individual episode HDF5 files into a single consolidated dataset.

    :param episodes_dir: Directory containing temporary episode files.
    :param output_file: Path for the final consolidated HDF5 file.
    :param num_max_steps: Expected number of steps per episode.
    :param image_width: Width of the images.
    :param image_height: Height of the images.
    """
    # Retrieve list of episode files
    episode_files = [os.path.join(episodes_dir, f) for f in os.listdir(episodes_dir)
                     if f.endswith(".h5")]
    num_episodes = len(episode_files)
    if num_episodes == 0:
        raise ValueError(f"No episode files found in '{episodes_dir}'.")

    print(f"Merging {num_episodes} episode files from '{episodes_dir}'...")

    # Verify that each episode file contains the expected number of steps
    for file in episode_files:
        with h5py.File(file, "r") as h5f:
            if h5f["actions"].shape[0] != num_max_steps:
                raise ValueError(f"File '{file}' has incorrect number of steps.")

    # Define dataset shapes using provided image dimensions
    actions_shape = (num_episodes, num_max_steps, 3)
    dones_shape = (num_episodes, num_max_steps)
    images_shape = (num_episodes, num_max_steps, image_height, image_width, 3)
    rewards_shape = (num_episodes, num_max_steps)

    # Create consolidated HDF5 file and merge data from individual episodes
    with h5py.File(output_file, "w") as merged_h5f:
        merged_h5f.create_dataset("actions", shape=actions_shape, dtype="float16",
                                  chunks=(1, num_max_steps, 3))
        merged_h5f.create_dataset("dones", shape=dones_shape, dtype="bool",
                                  chunks=(1, num_max_steps))
        merged_h5f.create_dataset("images", shape=images_shape, dtype="uint8",
                                  chunks=(1, num_max_steps, image_height, image_width, 3))
        merged_h5f.create_dataset("rewards", shape=rewards_shape, dtype="float32",
                                  chunks=(1, num_max_steps))

        # Merge data from each episode file into the consolidated dataset
        for idx, episode_file in enumerate(tqdm(episode_files, desc="Merging episodes")):
            with h5py.File(episode_file, "r") as ep_h5f:
                merged_h5f["actions"][idx] = ep_h5f["actions"][:]
                merged_h5f["dones"][idx] = ep_h5f["dones"][:]
                merged_h5f["images"][idx] = ep_h5f["images"][:]
                merged_h5f["rewards"][idx] = ep_h5f["rewards"][:]

            # Remove temporary file after merging
            try:
                os.remove(episode_file)
            except OSError as e:
                print(f"Warning: Could not delete '{episode_file}': {e}")

    os.removedirs(episodes_dir)
    print(f"Dataset successfully merged into '{output_file}'.")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a dataset for CarRacing")
    parser.add_argument("--env_name", type=str, default="CarRacing-v3",
                        help="Gym environment name")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Number of episodes to generate")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Steps per episode")
    parser.add_argument("--output_dir", type=str, default="temp_episodes_data",
                        help="Directory for temporary episode files")
    parser.add_argument("--output_file", type=str, default="car_racing_data.h5",
                        help="Path for the final dataset file")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count - 1)")
    parser.add_argument("--image_width", type=int, default=96,
                        help="Width of the resized images")
    parser.add_argument("--image_height", type=int, default=96,
                        help="Height of the resized images")
    args = parser.parse_args()

    collect_data(
        env_name=args.env_name,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        image_size=(args.image_width, args.image_height)
    )

    merge_episode_files(
        episodes_dir=args.output_dir,
        output_file=args.output_file,
        num_max_steps=args.max_steps,
        image_width=args.image_width,
        image_height=args.image_height
    )


if __name__ == "__main__":
    main()
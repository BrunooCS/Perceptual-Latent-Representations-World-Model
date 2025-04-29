# ----------------------------------------------------------------
# Imports
# ----------------------------------------------------------------
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F
import torch.nn as nn
import torch

from gymnasium.vector import AsyncVectorEnv
import gymnasium as gym
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import logging
import time
import cma
import os

# --- Project modules ---
from models.dds_vae import Vision
from models.controller import Controller
from models.mdn_rnn import MDNRNN
from utils.plots import plot_cma_es_results
from torch.nn.utils import parameters_to_vector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------------------------------------
# Environment and VectorEnv
# ----------------------------------------------------------------
def make_env(name='CarRacing-v3'):
    """Create and wrap environment."""
    def _init():
        env = gym.make(
            name, 
            render_mode='rgb_array', 
            lap_complete_percent=1.0, 
            domain_randomize=False, 
            continuous=True
        )
        return env
    return _init

def create_vector_envs(num_envs):
    """Create vectorized environment."""
    return AsyncVectorEnv([make_env() for _ in range(num_envs)], shared_memory=True)

def reset(envs, num_steps=50, action_dim=3):
    """Reset environments and perform a number of no-op steps."""
    obs, _ = envs.reset()
    for step in range(num_steps):
        actions = np.zeros((envs.num_envs, action_dim))
        obs, rewards, dones, truncs, infos = envs.step(actions)
        
        if np.any(dones | truncs):
            reset_obs, _ = envs.reset_done()
            obs = reset_obs
            logging.debug(f"Step {step}: Reset {np.sum(dones | truncs)} environments.")
    return envs, obs

# ----------------------------------------------------------------
# Encoding and Decoding
# ----------------------------------------------------------------
def encode_obs_batch(vision, obs_batch, size=(96, 96), device='cuda'):
    """Preprocess obs and encode to latent z."""
    obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
    obs_tensor = obs_tensor.permute(0, 3, 1, 2) / 255.0
    obs_tensor = obs_tensor[:, :, :-12, :]                  
    obs_tensor = F.interpolate(obs_tensor, size=size, mode='bilinear')
    with torch.no_grad():
        _,_, z = vision.encode(obs_tensor)
    return z

def decode_obs(z, vision):
    """Decode latent z to image."""
    with torch.no_grad():
        x_hat, _, _ = vision.decode(z)
    return x_hat.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()


# ----------------------------------------------------------------
# Controllers and Policy Evaluation
# ----------------------------------------------------------------
def process_actions(controllers, x):
    """Process actions for all controllers at once."""
    return torch.stack([ctrl.get_action(x[i:i+1]) for i, ctrl in enumerate(controllers)], dim=0)

def load_weights(controller_class, solutions):
    """Load CMA-ES solutions into controller weights."""
    controllers = []
    with torch.no_grad():
        for params in solutions:
            ctrl = controller_class(state_dim=LATENT_DIM + HIDDEN_DIM, action_dim=ACTION_DIM).to(device)
            vector_to_parameters(torch.tensor(params, dtype=torch.float32).to(device), ctrl.parameters())
            controllers.append(ctrl)
    return controllers

def evaluate_policies(solutions, controller_class, max_steps, memory, vision):
    """Evaluate multiple policies in parallel via AsyncVectorEnv."""
    num_policies = len(solutions)
    controllers = load_weights(controller_class, solutions)
    envs = create_vector_envs(num_envs=num_policies)
    # obs, _ = envs.reset()
    envs, obs = reset(envs)
    
    hidden = memory.rnn.init_hidden(num_policies, 'cuda')
    cumulative_rewards = np.zeros(num_policies)
    dones = np.full(num_policies, False)

    with torch.no_grad():
        for _ in range(max_steps):
            if np.all(dones):
                break
            z_batch = encode_obs_batch(vision, obs)
            h = hidden[0].squeeze(0)
            x = torch.cat([z_batch, h], dim=-1)
            actions = process_actions(controllers, x)
            obs, rewards, dones_new, _, _ = envs.step(actions.detach().cpu().numpy())
            
            z_batch = z_batch.unsqueeze(1)
            actions = actions.unsqueeze(1)
            _, hidden = memory.rnn(z_batch, actions, hidden)
            
            dones = np.logical_or(dones, dones_new)
            cumulative_rewards += rewards * (~dones) 
    envs.close()
    return cumulative_rewards.tolist()


# ----------------------------------------------------------------
# CMA-ES Training
# ----------------------------------------------------------------
INITIAL_SIGMA = .1
SIGMA_DECAY   = 1.

def train_cma_es(vision, controller_class, memory, 
                 max_generations=100, max_steps=1000, popsize=16, 
                 checkpoint=10, rollouts=7, path='trained_models'):
    """Train controller with CMA-ES."""
    metrics = {'generation': [], 'best_reward': [], 'mean_reward': [], 'worst_reward': []}
    initial_params = parameters_to_vector(
        controller_class(state_dim=LATENT_DIM + HIDDEN_DIM, action_dim=ACTION_DIM).parameters()
    ).detach().cpu().numpy()

    es = cma.CMAEvolutionStrategy(initial_params, INITIAL_SIGMA, {'popsize': popsize})

    for generation in range(1, max_generations+1):
        start_time = time.time()
        solutions = es.ask()

        # Evaluate each solution 'rollouts' times
        all_rewards = []
        for _ in range(rollouts):
            rewards = evaluate_policies(solutions, controller_class, max_steps, memory, vision)
            all_rewards.append(rewards)
        mean_rewards = np.mean(all_rewards, axis=0)

        # Update CMA-ES
        es.tell(solutions, [-r for r in mean_rewards])

        # Logging
        best, mean, worst = np.max(mean_rewards), np.mean(mean_rewards), np.min(mean_rewards)
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)
        print(f'Generation ({generation}/{max_generations}) '
              f'| Best: {round(best)} '
              f'| Avg: {mean:.2f} '
              f'| Worst: {round(worst)} '
              f'| Time: {int(minutes)}:{int(seconds):02d} '
              f'| Sigma: {es.sigma:.4f}')

        metrics['generation'].append(generation)
        metrics['best_reward'].append(best)
        metrics['worst_reward'].append(worst)
        metrics['mean_reward'].append(mean)

        es.sigma *= SIGMA_DECAY

        # Save progress periodically
        if generation % checkpoint == 0:
            best_controller = Controller(state_dim=LATENT_DIM + HIDDEN_DIM, action_dim=ACTION_DIM).to(device)
            vector_to_parameters(
                torch.tensor(es.result.xbest, dtype=torch.float32).to(device),
                best_controller.parameters()
            )
            torch.save(best_controller.state_dict(), f'{path}/controller.pth')
            pd.DataFrame(metrics).to_csv("cma_es_metrics.csv")
            print('---Checkpoint: best controller saved')

    # Final save
    best_controller = Controller(state_dim=LATENT_DIM + HIDDEN_DIM, action_dim=ACTION_DIM).to(device)
    vector_to_parameters(
        torch.tensor(es.result.xbest, dtype=torch.float32).to(device),
        best_controller.parameters()
    )
    torch.save(best_controller.state_dict(), f'{path}/controller.pth')
    pd.DataFrame(metrics).to_csv("cma_es_metrics.csv")
    return es, metrics, es.result.xbest


# ----------------------------------------------------------------
# Rendering and Final Evaluation
# ----------------------------------------------------------------
def render_policy(env_name, vision, controller, mdnrnn, encode_obs_batch):
    """Render a single policy in a gym environment."""
    env = gym.make(env_name, render_mode='human', lap_complete_percent=1.0)
    done = False
    # obs, _ = env.reset()
    env, obs = reset(env)
    h = (torch.zeros(1, HIDDEN_DIM).to(device), torch.zeros(1, HIDDEN_DIM).to(device))
    
    cumulative_reward, cnt = 0, 1
    while True:
        z = encode_obs_batch(vision, obs[np.newaxis, ...])
        x = torch.cat([z, h[0]], dim=-1)
        a = controller.get_action(x)
        obs, reward, done, _, _ = env.step(a.detach().cpu().numpy())
        env.render()

        _, h = mdnrnn.rnn(z, a.unsqueeze(0), h=h)
        cumulative_reward += reward
        cnt += 1
        
        if done or cnt >= 1000:
            break
    env.close()
    print(f'Reward: {cumulative_reward} | Steps: {cnt}')


def final_evaluaion(vision, controller_class, best_solution, 
                    memory, parallel_rollouts=7, max_steps=1000, popsize=16):
    """Evaluate best solution repeatedly for final stats."""
    final_rewards = []
    best_controllers = [best_solution for _ in range(popsize)] 

    for _ in tqdm(range(parallel_rollouts)):
        rewards = evaluate_policies(best_controllers, controller_class, max_steps, memory, vision)
        final_rewards.append(rewards)

    mean_reward = round(np.mean(final_rewards), 2)
    std_reward = round(np.std(final_rewards), 2)
    print(f"Reward over {popsize*parallel_rollouts} rollouts: {mean_reward} +/- {std_reward}")
    return final_rewards

# Constants
LATENT_DIM = 32
HIDDEN_DIM = 256
ACTION_DIM = 3
NUM_GAUSSIANS = 5


# ----------------------------------------------------------------
# Main Script
# ----------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train and evaluate miniVAE model")
    parser.add_argument("--mask", type=float, default=0.03, help="Mask % value")
    parser.add_argument("--popsize", type=float, default=20, help="Population size")
    parser.add_argument("--path", type=str, default='trained_models', help="Path to save trained models")
    parser.add_argument("--generations", type=float, default=200, help="CMA ES Generations")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask = args.mask
    path = args.path
    popsize = args.popsize
    generations = args.generations
    
    # Load models
    vision = Vision(
        n_features_to_select=mask, in_ch=3,
        out_ch=3, base_ch=16, alpha=1.0, delta=0.1
    ).to(device)
    vision_path = os.path.join(path, f'vision_{int(mask * 100):02d}_miniVAE.pth')
    vision.load_state_dict(torch.load(vision_path, map_location=device, weights_only=True))
    vision.eval()

    mdnrnn = MDNRNN(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM, num_gaussians=NUM_GAUSSIANS).to(device)
    mdnrnn_path = os.path.join(path, f'memory.pth')
    mdnrnn.load_state_dict(torch.load(mdnrnn_path, map_location=device, weights_only=True))
    mdnrnn.eval()

    # CMA-ES Training
    es, metrics, best_solution = train_cma_es(
        vision, Controller, mdnrnn,
        max_generations=generations, max_steps=1000, popsize=popsize, 
        checkpoint=10, rollouts=7, path=path
    )
    
    # Plot metrics
    metrics = pd.read_csv('cma_es_metrics.csv')
    plot_cma_es_results(
        metrics, 
        colors=['#86D293', '#FFCF96', '#FF8080'], 
        metric_columns=['best_reward', 'mean_reward', 'worst_reward'], 
        path='plots'
    )

    # Load best controller
    controller = Controller(state_dim=LATENT_DIM + HIDDEN_DIM, action_dim=ACTION_DIM).to(device)
    controller.load_state_dict(torch.load(f'{path}/controller.pth', map_location=device, weights_only=True))
    
    # Render policy
    # print('Rendering policy...')
    # render_policy('CarRacing-v3', vision, controller, mdnrnn, encode_obs_batch)

    # Final evaluation
    controller_best_solution = parameters_to_vector(controller.parameters()).cpu().detach().numpy()
    rewards = final_evaluaion(
        vision, Controller, 
        controller_best_solution, mdnrnn, 
        parallel_rollouts=32, max_steps=1000, popsize=popsize
    )

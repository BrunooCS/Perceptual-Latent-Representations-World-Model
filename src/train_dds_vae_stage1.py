"""
This script loads a dataset, initializes the Vision model, and trains it while saving checkpoints.
"""

# ---------------------------- Imports ----------------------------
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.dataset import CarRacingDataset
from utils.weights import initialize_weights
from models.dds_vae import Vision

torch.autograd.set_detect_anomaly(True)

# ---------------------------- Training Function ----------------------------

def train_model(vision: Vision, 
                dataloader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, 
                scheduler: torch.optim.lr_scheduler._LRScheduler, 
                device: torch.device, 
                epochs: int, 
                name: str, 
                path: str) -> None:
    """
    Train the Vision model.

    :param vision: Vision model to be trained.
    :param dataloader: DataLoader providing the training data.
    :param optimizer: Optimizer used for training.
    :param scheduler: Learning rate scheduler.
    :param device: Device for computation ('cuda' or 'cpu').
    :param epochs: Number of training epochs.
    :param name: Name used to save checkpoints.
    :param path: Directory path to save trained models.
    """
    vision.train()
    loss_history = []
    os.makedirs(path, exist_ok=True)

    print(f'Training: {name}')
    
    for epoch in range(epochs):
        train_loss = 0.0
        cnt        = 0

        with tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}') as pbar:
            for images, _, _, _ in pbar:
                x = images.to(device)
                optimizer.zero_grad()

                # 1) Forward pass
                mini_mask, mask, binary_mask = vision.downscale(x)
                x_hat, mask_hat              = vision.upscale(mini_mask)

                # 2) Loss computation
                loss = F.mse_loss(x_hat, x) + 1e-2 * F.l1_loss(x_hat, x)

                # 3) Backward pass and optimization
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                loss_history.append(loss.item())
                cnt += 1

                # Progress bar update
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{current_lr:.6f}")

                # Save checkpoint every 10% of an epoch
                if cnt % int(len(dataloader) * 0.1) == 0:
                    torch.save(vision.state_dict(), os.path.join(path, f'{name}.pth'))

        print(f'Epoch {epoch + 1}/{epochs} | Avg Loss: {train_loss / len(dataloader):.6f}')

    torch.save(vision.state_dict(), os.path.join(path, f'{name}.pth'))

# ---------------------------- Main Workflow ----------------------------

def main(mask: float, base_ch: int, dataset_name: str, epochs: int, batch_size: int, path: str) -> None:
    """
    Main workflow to initialize the model, dataset, and train.

    :param mask: Masking percentage for feature selection.
    :param base_ch: Base number of channels for UNet and other modules.
    :param dataset_name: Name of the HDF5 dataset file.
    :param epochs: Number of training epochs.
    :param batch_size: Size of mini-batches.
    :param path: Directory path to save models and plots.
    """
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset      = CarRacingDataset(dataset_name)
    dataloader   = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True, 
                    num_workers=8, pin_memory=True)

    model_name = f'vision_{int(mask * 100):02d}'
    vision = Vision( n_features_to_select=mask,
                            in_ch=3,
                            out_ch=3,
                            base_ch=base_ch,
                            alpha=1.0,
                            delta=0.1).to(device)


    optimizer    = torch.optim.AdamW(vision.parameters(), lr=1e-2, weight_decay=1e-5)
    total_steps  = epochs * len(dataloader)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=0.5e-4
    )

    train_model(vision, dataloader, optimizer, scheduler, device, epochs, model_name, path)

# ---------------------------- Argument Parser ----------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate the Vision model.")
    parser.add_argument("--mask", type=float, default=0.03, help="Feature masking value.")
    parser.add_argument("--base_ch", type=int, default=16, help="Base channels for UNet and modules.")
    parser.add_argument("--dataset_name", type=str, default='car_racing_data_10000.h5', help="Dataset file name.")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--path", type=str, default='trained_models', help="Directory to save models.")

    args = parser.parse_args()

    main(
        mask=args.mask,
        base_ch=args.base_ch,
        dataset_name=args.dataset_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        path=args.path
    )

import torch
import argparse
from tqdm import tqdm
from utils.dataset import CarRacingDataset_RNN
from models.mdn_rnn import MDNRNN, gaussian_nll_loss
from torch.utils.data import DataLoader
from models.dds_vae import Vision

def train_model(mdnrnn, dataloader, optimizer, scheduler, num_epochs, device, save=True, tau=1.0, name='memory'):
    """ Train the MDN-RNN model with a given dataloader, optimizer, and scheduler. """
    mdnrnn.train()
    loss_history = []

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(mdnrnn.parameters(), max_norm=1.0)

    for epoch in range(num_epochs):
        total_loss = 0.0

        with tqdm(total=len(dataloader), desc=f"Epoch ({epoch+1}/{num_epochs})", unit="batch", ncols=130) as pbar:
            for z, z_next, actions, rewards, dones in dataloader:
                optimizer.zero_grad()
                z, z_next, a = z.to(device), z_next.to(device), actions.to(device)

                pi, mu, sigma, h = mdnrnn(z, a, tau=tau)

                # Compute losses
                loss = gaussian_nll_loss(pi, mu, sigma, z_next)
                recon_loss = torch.abs(z_next.unsqueeze(2) - mu).mean().item()

                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'NLL Loss': f'{loss.item():.4f}', 'Reconstruction Loss': f'{recon_loss:.4f}'})

        # Average loss for the epoch
        average_loss = total_loss / len(dataloader)
        loss_history.append(average_loss)
        pbar.set_postfix({'NLL Loss': f'{average_loss:.4f}', 'Reconstruction Loss': f'{recon_loss:.4f}'})
        torch.save(mdnrnn.state_dict(), f'{name}_{epoch}.pth')


    if save:
        print(f"Model saved to {name}.pth")

def main(mask, base_ch, latent_dim, dataset_name, epochs, batch_size, path):
    """ Main function to set up the MDN-RNN training pipeline. """


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load DoubleUNet model
    model_name = f'vision_{int(mask * 100):02d}_miniVAE'
    vision = Vision(
        n_features_to_select=mask, in_ch=3,
        out_ch=3, base_ch=base_ch, alpha=1.0, delta=0.1
    ).to(device)
    vision.load_state_dict(torch.load(f'{path}/{model_name}.pth', map_location=device, weights_only=True))
    vision.eval()

    # Prepare dataset and dataloader
    dataset = CarRacingDataset_RNN(h5_path=dataset_name, vision=vision, device=device, transform=None)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    # MDN-RNN setup
    tau = 1.0
    num_gaussians = 5
    hidden_dim = 256

    mdnrnn = MDNRNN(latent_dim=latent_dim, action_dim=3, hidden_dim=hidden_dim, num_gaussians=num_gaussians).to(device)
    optimizer = torch.optim.AdamW(mdnrnn.parameters(), lr=0.5e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs * len(dataloader)), eta_min=1e-4)

    # Train the model
    train_model(mdnrnn, dataloader, optimizer, scheduler, num_epochs=epochs, device=device, save=True, tau=tau, name=f'{path}/memory')

if __name__ == '__main__':
    #  Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train LSTM-MDN model")
    parser.add_argument("--mask", type=float, default=0.03, help="Mask % value")
    parser.add_argument("--base_ch", type=int, default=32, help="Base number of channels")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension size")
    parser.add_argument("--dataset_name", type=str, default='car_racing_data_10000.h5', help="Dataset file name")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--path", type=str, default='trained_models', help="Path to save trained models")
    args = parser.parse_args()

    main(
        mask=args.mask,
        base_ch=args.base_ch,
        latent_dim=args.latent_dim,
        dataset_name=args.dataset_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        path=args.path
    )



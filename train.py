import time
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import VAE, loss_function


def train():
    # 1. Device selection (CPU or CUDA GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # 2. Hyperparameters
    batch_size = 128
    epochs = 15
    learning_rate = 1e-3

    # 3. Load MNIST training dataset (downloads automatically to ./data)
    print("Loading MNIST dataset...", flush=True)
    transform = transforms.ToTensor()  # Converts [0, 255] images to tensors with values in [0.0, 1.0]
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # To ensure training finishes in under 1 minute on standard CPUs,
    # we use a 20,000-sample subset (plenty for a 2D latent space on MNIST).
    # You can set subset_size to len(train_dataset) to train on all 60,000 images!
    subset_size = 20000
    train_subset = Subset(train_dataset, range(subset_size))
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)

    # 4. Instantiate model and optimizer
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. Training loop
    print(f"\nStarting training for {epochs} epochs on {len(train_subset)} samples...", flush=True)
    start_time = time.time()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass: reconstruct batch, get latent distribution params
            recon_batch, mu, logvar = model(data)

            # Compute combined loss (Reconstruction BCE + KL Divergence)
            loss = loss_function(recon_batch, data, mu, logvar)

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Average Loss: {avg_loss:.2f}", flush=True)

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.1f} seconds (Target: < 60s).", flush=True)

    # 6. Save model weights to vae.pt
    save_path = "vae.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to '{save_path}'. You can now run 'streamlit run app.py'!", flush=True)


if __name__ == "__main__":
    train()

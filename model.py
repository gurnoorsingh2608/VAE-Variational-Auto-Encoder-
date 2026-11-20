import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """
    Mini Variational Autoencoder (VAE) for 28x28 MNIST handwritten digits.
    Uses fully connected layers with a 2-dimensional latent space.
    """
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2):
        super(VAE, self).__init__()

        # --- ENCODER ---
        # Compresses a flattened 784-pixel image into hidden features,
        # then predicts the mean (mu) and log-variance (logvar) of the latent distribution.
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)       # Mean of latent space
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)   # Log-variance of latent space

        # --- DECODER ---
        # Takes a 2D latent point z and reconstructs the 784-pixel image.
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """Pass input through encoder to obtain mu and logvar."""
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        The Reparameterization Trick:
        Instead of sampling z directly from N(mu, std^2), which prevents backpropagation,
        sample eps ~ N(0, I) and compute:
            z = mu + std * eps
        where std = exp(0.5 * logvar).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        """Decode a 2D latent vector into a reconstructed 784-pixel image."""
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))  # Sigmoid maps outputs to [0, 1] pixel values

    def forward(self, x):
        """Full pass: Flatten -> Encode -> Reparameterize -> Decode."""
        x_flat = x.view(-1, 784)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    """
    VAE Loss = Reconstruction Loss + KL Divergence.
    
    1. Reconstruction Loss (Binary Cross-Entropy):
       Measures how closely the reconstructed image matches the original image.
    2. KL Divergence:
       Acts as a regularizer, pushing latent distribution q(z|x) toward standard normal N(0, I).
       Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    x_flat = x.view(-1, 784)

    # Sum of binary cross entropy across all pixels and batch items
    bce = F.binary_cross_entropy(recon_x, x_flat, reduction='sum')

    # KL Divergence between N(mu, sigma^2) and N(0, I)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return bce + kld

# Mini VAE Studio 🎨

An educational, interactive PyTorch and Streamlit implementation of a **Variational Autoencoder (VAE)** trained on the MNIST handwritten digits dataset.

The model compresses high-dimensional images ($28 \times 28 = 784$ pixels) into an intuitive, continuous **2D latent coordinate space** $(z_1, z_2)$, making generative deep learning visual, human-navigable, and fun to explore.

---

## 🌐 Live Web Application

> **Experience the model directly in your browser — no installation or coding required!**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

🔗 **[Launch Live Mini VAE Studio on Streamlit]([https://share.streamlit.io/](https://gurnoorsingh2608-vae-variational-auto-encoder--app-yflqud.streamlit.app/))**

With the live web app, anyone can interact with the trained VAE in real time:
- Generate new handwritten digits using interactive coordinate sliders
- Compare original images against model reconstructions
- Explore the entire continuous 2D digit manifold
- Smoothly interpolate (morph) between any two digits

---

## 🧠 Description of the Model

A **Variational Autoencoder (VAE)** is a directed probabilistic generative model. Unlike traditional autoencoders that map inputs to discrete, arbitrary points, a VAE maps inputs to parameters of a probability distribution, ensuring the latent space is continuous, smooth, and easily sampleable.

```
Input Image (784 px)
        │
   [Linear + ReLU]
        ▼
   Hidden Layer (400)
     ┌────┴────┐
     ▼         ▼
  Mean (μ)   Log-Var (log σ²)   ──► Latent Space (2D)
     └────┬────┘
          ▼
   [Reparameterization: z = μ + σ ⊙ ε]
          ▼
  Latent Vector z (2D)
        │
   [Linear + ReLU]
        ▼
   Hidden Layer (400)
        │
   [Linear + Sigmoid]
        ▼
Reconstructed Image (784 px)
```

### 1. Encoder Network
- Takes a flattened $28 \times 28$ image ($784$ pixel intensities between $0.0$ and $1.0$).
- Passes through a dense hidden layer ($784 \to 400$) with **ReLU** activation.
- Predicts two 2D vectors:
  - **Mean ($\mu$)**: The center of the distribution in latent space.
  - **Log-Variance ($\log \sigma^2$)**: The uncertainty / spread of the distribution.

### 2. The Reparameterization Trick
Directly sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ is a non-differentiable operation that halts backpropagation. The **reparameterization trick** bypasses this by generating random noise $\epsilon \sim \mathcal{N}(0, I)$ independently and computing:

$$z = \mu + \sigma \odot \epsilon \quad \text{where } \sigma = \exp\left(\frac{1}{2} \log \sigma^2\right)$$

This allows gradients to flow backwards into the encoder network weights during training.

### 3. Decoder Network
- Takes the sampled 2D latent coordinate $z = (z_1, z_2)$.
- Expands it through a dense hidden layer ($2 \to 400$) with **ReLU** activation.
- Projects back to the original image space ($400 \to 784$) with a **Sigmoid** activation, ensuring pixel output values stay strictly within $[0.0, 1.0]$.

### 4. Loss Function
The model optimizes the Evidence Lower Bound (ELBO), consisting of two complementary loss terms:

$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{Reconstruction}} + \mathcal{L}_{\text{KL Divergence}}$$

1. **Reconstruction Loss (Binary Cross-Entropy):**
   Measures how accurately the decoder reproduces individual pixels of the input image.
   $$\mathcal{L}_{\text{BCE}} = -\sum_{i} \left[ x_i \log(\hat{x}_i) + (1 - x_i)\log(1 - \hat{x}_i) \right]$$
2. **Kullback-Leibler (KL) Divergence:**
   Acts as a regularizer by penalizing deviations between the learned distribution $q(z|x)$ and a standard normal prior $\mathcal{N}(0, I)$.
   $$\mathcal{L}_{\text{KLD}} = -\frac{1}{2} \sum_{j=1}^{2} \left( 1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$
   This regularization prevents cluster gaps, ensuring smooth interpolation and valid generation from any coordinate.

### 5. Why a 2D Latent Space?
Restricting the bottleneck to $d=2$ allows the entire learned latent space to be mapped onto a 2-axis $(z_1, z_2)$ Cartesian plane. Users can manually navigate the manifold with sliders and observe how topological features (loops, strokes, stems) organize naturally.

---

## ✨ Key Features

- **🔍 Reconstruct & Error Heatmap**:
  Select any test image (index 0–999) or click **"Surprise Me!"**. View the original digit side-by-side with its VAE reconstruction and a pixel-wise difference heatmap highlighting reconstruction error.
- **🎨 Live Latent Explorer & Radar Map**:
  Control continuous sliders for $z_1$ and $z_2$ ($-3.0$ to $+3.0$) and watch the decoder synthesize new digits live. Includes a 2D Latent Radar Map with $1\sigma, 2\sigma, 3\sigma$ Gaussian contour rings that tracks your cursor.
- **🗺️ 2D Latent Space Manifold**:
  Renders an $18 \times 18$ grid showing the entire continuous landscape of the latent space across $z_1, z_2 \in [-2.5, 2.5]$, revealing seamless visual transitions between all ten digits.
- **🌀 Morph & Interpolate (Latent Walk)**:
  Pick any two digits (Digit A and Digit B) and drag the morph slider ($\alpha \in [0, 1]$) to perform a linear walk $z(\alpha) = (1 - \alpha)z_A + \alpha z_B$. Displays a 7-step filmstrip preview of the morphing sequence.
- **⚡ Quick Cluster Presets & Sampling**:
  One-click preset buttons to jump straight to characteristic regions (Zero, One, Seven, Eight) or draw random samples $z \sim \mathcal{N}(0, 1)$.

---

## 📁 Project Directory

```
Variational Autoencoder/
├── app.py            # Streamlit web application with interactive tabs & charts
├── model.py          # PyTorch VAE architecture and combined loss function
├── train.py          # Training script (auto-downloads MNIST and saves vae.pt)
├── vae.pt            # Pre-trained model weights (ready for immediate use)
├── requirements.txt  # Python package dependencies
├── .gitignore        # Excludes cache and raw dataset files from version control
└── README.md         # Project documentation and architecture guide
```

---

## 🪟 Windows Installation & Execution Guide

Follow these steps to run the application locally on Windows:

### 1. Prerequisites
- **Python 3.9+** installed on your system.
  *(During Python installation on Windows, ensure you check the box: **"Add python.exe to PATH"**).*

### 2. Open the Project Folder
Open PowerShell or Command Prompt and navigate to your project directory:
```powershell
cd "c:\Users\gurno\Desktop\Variational Autoencoder"
```

### 3. (Optional) Set Up a Virtual Environment
Creating an isolated virtual environment is recommended:
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1

# (If using standard Command Prompt cmd.exe instead, run:)
# .\venv\Scripts\activate.bat
```

> *Tip: If PowerShell blocks script activation, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.*

### 4. Install Dependencies
Install all required packages from `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### 5. Launch the Streamlit Web Application
Launch the interactive studio frontend:
```powershell
streamlit run app.py
```
Your default browser will automatically open to `http://localhost:8501`.

### 6. (Optional) Re-train the Model
The repository already includes pre-trained weights (`vae.pt`), so training is optional. If you wish to retrain the model from scratch:
```powershell
python train.py
```
This automatically downloads the MNIST dataset to `./data` and trains the model in ~30–45 seconds on CPU.

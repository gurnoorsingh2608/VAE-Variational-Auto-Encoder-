import os
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from model import VAE


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Mini VAE Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern card designs, badges, and smooth typography
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .info-card {
        background: rgba(148, 163, 184, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        font-size: 0.9rem;
        line-height: 1.45;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.18);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.35);
        margin-right: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# MODEL & DATA CACHING
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Loads the trained VAE model weights from disk."""
    if not os.path.exists("vae.pt"):
        return None
    model = VAE()
    model.load_state_dict(torch.load("vae.pt", map_location=torch.device("cpu")))
    model.eval()
    return model


@st.cache_data
def load_test_dataset():
    """Loads the MNIST test dataset (auto-downloads if missing)."""
    transform = transforms.ToTensor()
    return datasets.MNIST(root="./data", train=False, download=True, transform=transform)


# Check model existence
model = load_trained_model()
if model is None:
    st.error("⚠️ Model file `vae.pt` was not found in the current directory!")
    st.info("Train the model first by opening your terminal and running:\n\n```bash\npython train.py\n```")
    st.stop()

test_dataset = load_test_dataset()


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧬 Mini VAE Studio")
    st.markdown(
        """
        <div class="info-card">
        <b>Variational Autoencoder (VAE)</b><br>
        A generative model that maps high-dimensional images ($28 \\times 28 = 784$ pixels) 
        into a continuous <b>2D latent coordinate system</b> $(z_1, z_2)$.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 📐 Architecture")
    st.markdown(
        """
        - <span class="badge">Encoder</span> `784` $\\to$ `400` $\\to$ $(\\mu, \\log\\sigma^2)$
        - <span class="badge">Latent</span> `2D` $(z_1, z_2)$
        - <span class="badge">Decoder</span> `2` $\\to$ `400` $\\to$ `784` (Sigmoid)
        - <span class="badge">Loss</span> `BCE` + `KL Divergence`
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### ⚡ Quick Latent Presets")
    st.caption("Jump directly to characteristic regions in latent space:")

    preset_cols = st.columns(2)
    with preset_cols[0]:
        if st.button("🔵 Zero (0)", use_container_width=True):
            st.session_state.z1 = -1.2
            st.session_state.z2 = 1.4
            st.rerun()
        if st.button("🟢 Seven (7)", use_container_width=True):
            st.session_state.z1 = 1.8
            st.session_state.z2 = -1.1
            st.rerun()
    with preset_cols[1]:
        if st.button("🟡 One (1)", use_container_width=True):
            st.session_state.z1 = -1.9
            st.session_state.z2 = -1.5
            st.rerun()
        if st.button("🟣 Eight (8)", use_container_width=True):
            st.session_state.z1 = 0.2
            st.session_state.z2 = 0.5
            st.rerun()

    st.markdown("---")
    st.caption("Mini VAE • PyTorch + Streamlit")


# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🎨 Mini VAE Studio</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Interactive 2D Latent Space Explorer for MNIST Handwritten Digits</div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# NAVIGATION TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 1. Reconstruct & Compare",
    "🎨 2. Live Latent Explorer",
    "🗺️ 3. 2D Latent Manifold",
    "🌀 4. Morph & Interpolate",
])


# =============================================================================
# TAB 1: RECONSTRUCT & COMPARE
# =============================================================================
with tab1:
    st.subheader("Reconstruct Existing Digits")
    st.write(
        "Select an image from the MNIST test set (0 – 999). The encoder compresses its 784 pixels "
        "into a 2D coordinate $(\\mu_1, \\mu_2)$, and the decoder reconstructs it."
    )

    control_col1, control_col2 = st.columns([3, 1])
    with control_col1:
        test_idx = st.slider("Select Test Image Index", min_value=0, max_value=999, value=7, step=1)
    with control_col2:
        st.write("")
        st.write("")
        if st.button("🎲 Surprise Me!", use_container_width=True):
            test_idx = int(np.random.randint(0, 1000))
            st.session_state["recon_idx"] = test_idx

    # Handle random index selection state if active
    if "recon_idx" in st.session_state and st.session_state["recon_idx"] != test_idx:
        test_idx = st.session_state["recon_idx"]

    # Inference on selected digit
    orig_tensor, ground_truth_label = test_dataset[test_idx]
    with torch.no_grad():
        recon_tensor, mu, logvar = model(orig_tensor.unsqueeze(0))

    orig_np = orig_tensor.squeeze().numpy()
    recon_np = recon_tensor.view(28, 28).numpy()
    diff_np = np.abs(orig_np - recon_np)
    mse_error = np.mean((orig_np - recon_np) ** 2)

    # Display in 3 visual columns
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Original Image**")
        st.image(orig_np, width=190, clamp=True)
        st.caption(f"Ground Truth Label: **{ground_truth_label}**")

    with c2:
        st.markdown("**Reconstructed (VAE)**")
        st.image(recon_np, width=190, clamp=True)
        st.caption(f"Latent Mean: $\\mu = [{mu[0, 0]:.2f}, {mu[0, 1]:.2f}]$")

    with c3:
        st.markdown("**Pixel Difference (Error)**")
        # Color difference map using matplotlib colormap
        fig_diff, ax_diff = plt.subplots(figsize=(2.5, 2.5))
        ax_diff.imshow(diff_np, cmap="inferno", vmin=0, vmax=1)
        ax_diff.axis("off")
        fig_diff.tight_layout(pad=0)
        st.pyplot(fig_diff)
        plt.close(fig_diff)
        st.caption(f"Reconstruction MSE: **{mse_error:.4f}**")

    st.info(
        f"💡 **Insight:** The encoder compressed this handwritten '{ground_truth_label}' "
        f"into point **$z = ({mu[0, 0]:.2f}, {mu[0, 1]:.2f})$**. "
        "Notice how the reconstructed digit captures the essential stroke pattern while smoothing out minor pixel noise."
    )


# =============================================================================
# TAB 2: LIVE LATENT EXPLORER (GENERATOR)
# =============================================================================
with tab2:
    st.subheader("Generate Digits From Pure Latent Coordinates")
    st.write(
        "Control $(z_1, z_2)$ live with sliders. The decoder synthesizes brand new handwritten digits "
        "from anywhere in continuous 2D space."
    )

    # Initialize session state for sliders
    if "z1" not in st.session_state:
        st.session_state.z1 = 0.0
    if "z2" not in st.session_state:
        st.session_state.z2 = 0.0

    # Action buttons
    action_cols = st.columns([1, 1, 2])
    with action_cols[0]:
        if st.button("🎲 Sample z ~ N(0, 1)", use_container_width=True):
            st.session_state.z1 = round(float(np.clip(np.random.normal(0, 1), -3.0, 3.0)), 2)
            st.session_state.z2 = round(float(np.clip(np.random.normal(0, 1), -3.0, 3.0)), 2)
            st.rerun()
    with action_cols[1]:
        if st.button("🎯 Center (0.0, 0.0)", use_container_width=True):
            st.session_state.z1 = 0.0
            st.session_state.z2 = 0.0
            st.rerun()

    # Sliders for z1 and z2
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        z1 = st.slider("Latent Dimension 1 ($z_1$)", min_value=-3.0, max_value=3.0, step=0.1, key="z1")
    with s_col2:
        z2 = st.slider("Latent Dimension 2 ($z_2$)", min_value=-3.0, max_value=3.0, step=0.1, key="z2")

    # Decode current z coordinate
    with torch.no_grad():
        z_tensor = torch.tensor([[z1, z2]], dtype=torch.float32)
        generated_tensor = model.decode(z_tensor)
        generated_np = generated_tensor.view(28, 28).numpy()

    # Layout: Decoded Digit on the left, 2D Latent Radar Map on the right
    gen_c1, gen_c2 = st.columns([1, 1])

    with gen_c1:
        st.markdown(f"**Synthesized Digit at $(z_1={z1:.2f}, z_2={z2:.2f})$:**")
        st.image(generated_np, width=220, clamp=True)
        st.caption("Directly outputted by the Decoder via Sigmoid activation.")

    with gen_c2:
        st.markdown("**Latent Map Position:**")
        # Visual 2D coordinate plot with standard normal rings
        fig_radar, ax_radar = plt.subplots(figsize=(3.4, 3.4), facecolor="none")
        ax_radar.set_facecolor("#0f172a")

        # Standard normal distribution contours (1σ, 2σ, 3σ)
        c1_patch = plt.Circle((0, 0), 1.0, color="#6366f1", fill=False, linestyle="--", linewidth=1.2, alpha=0.6, label="1σ")
        c2_patch = plt.Circle((0, 0), 2.0, color="#818cf8", fill=False, linestyle=":", linewidth=1.0, alpha=0.4, label="2σ")
        c3_patch = plt.Circle((0, 0), 3.0, color="#a5b4fc", fill=False, linestyle=":", linewidth=0.8, alpha=0.25, label="3σ")
        ax_radar.add_patch(c1_patch)
        ax_radar.add_patch(c2_patch)
        ax_radar.add_patch(c3_patch)

        # Crosshairs and current location dot
        ax_radar.axhline(0, color="#334155", linestyle="-", linewidth=0.8)
        ax_radar.axvline(0, color="#334155", linestyle="-", linewidth=0.8)
        ax_radar.scatter([z1], [z2], color="#ec4899", s=180, edgecolors="white", linewidth=2, zorder=6)

        ax_radar.set_xlim(-3.2, 3.2)
        ax_radar.set_ylim(-3.2, 3.2)
        ax_radar.set_xlabel("$z_1$", color="#94a3b8", fontsize=10)
        ax_radar.set_ylabel("$z_2$", color="#94a3b8", fontsize=10)
        ax_radar.tick_params(colors="#64748b", labelsize=8)
        for spine in ax_radar.spines.values():
            spine.set_color("#334155")
        ax_radar.legend(loc="upper right", fontsize=7, facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1")
        fig_radar.tight_layout()
        st.pyplot(fig_radar)
        plt.close(fig_radar)


# =============================================================================
# TAB 3: 2D LATENT MANIFOLD GRID
# =============================================================================
with tab3:
    st.subheader("2D Latent Space Manifold")
    st.write(
        "This grid visualizes the entire continuous latent space at once. "
        "Moving horizontally changes $z_1$; moving vertically changes $z_2$. "
        "Notice how numbers naturally transition smoothly (e.g. 1 morphing into 7, 9, 4, 8, 0)!"
    )

    grid_size = st.slider("Grid Resolution (digits per row/col)", min_value=10, max_value=24, value=18, step=2)

    @st.cache_data
    def render_manifold_grid(n, range_lim=2.5):
        """Precomputes an n x n grid of digits across the latent coordinate plane."""
        gx = np.linspace(-range_lim, range_lim, n)
        gy = np.linspace(range_lim, -range_lim, n)  # Top to bottom
        coords = [[x, y] for y in gy for x in gx]
        coords_tensor = torch.tensor(coords, dtype=torch.float32)

        with torch.no_grad():
            decoded = model.decode(coords_tensor).view(n, n, 28, 28).numpy()

        # Stitch all 28x28 patches into a single large canvas
        canvas = np.block([[decoded[i, j] for j in range(n)] for i in range(n)])
        return canvas

    manifold_canvas = render_manifold_grid(grid_size)

    # Plot manifold with axes
    fig_man, ax_man = plt.subplots(figsize=(7, 7), facecolor="none")
    ax_man.imshow(manifold_canvas, cmap="gray", extent=[-2.5, 2.5, -2.5, 2.5])
    ax_man.set_xlabel("Latent Coordinate $z_1$", fontsize=11, color="#cbd5e1")
    ax_man.set_ylabel("Latent Coordinate $z_2$", fontsize=11, color="#cbd5e1")
    ax_man.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax_man.spines.values():
        spine.set_color("#475569")
    fig_man.tight_layout()
    st.pyplot(fig_man)
    plt.close(fig_man)


# =============================================================================
# TAB 4: MORPH & INTERPOLATE
# =============================================================================
with tab4:
    st.subheader("Smooth Digit Morphing (Latent Walk)")
    st.write(
        "Choose any two test digits. The encoder finds their positions $z_A$ and $z_B$. "
        "By taking a linear walk $z(\\alpha) = (1 - \\alpha) z_A + \\alpha z_B$, "
        "we can smoothly morph one digit into another!"
    )

    morph_c1, morph_c2 = st.columns(2)
    with morph_c1:
        idx_a = st.number_input("Start Image (Digit A) Index", min_value=0, max_value=999, value=3)
    with morph_c2:
        idx_b = st.number_input("End Image (Digit B) Index", min_value=0, max_value=999, value=2)

    img_a, label_a = test_dataset[idx_a]
    img_b, label_b = test_dataset[idx_b]

    with torch.no_grad():
        mu_a, _ = model.encode(img_a.view(-1, 784))
        mu_b, _ = model.encode(img_b.view(-1, 784))

    alpha = st.slider("Morph Progress (α: 0% = A, 100% = B)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    with torch.no_grad():
        z_interp = (1.0 - alpha) * mu_a + alpha * mu_b
        morph_digit = model.decode(z_interp).view(28, 28).numpy()

    st.markdown(f"**Current Morph at $\\alpha = {alpha*100:.0f}\\%$:**")
    st.image(morph_digit, width=180, clamp=True)

    st.markdown("#### 🎞️ Continuous Morph Filmstrip (0% → 100%):")
    film_steps = 7
    alphas = np.linspace(0.0, 1.0, film_steps)
    cols = st.columns(film_steps)

    with torch.no_grad():
        for i, a in enumerate(alphas):
            cur_z = (1.0 - a) * mu_a + a * mu_b
            step_img = model.decode(cur_z).view(28, 28).numpy()
            with cols[i]:
                st.caption(f"{int(a * 100)}%")
                st.image(step_img, clamp=True)

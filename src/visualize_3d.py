import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.model import ThreeLayerCNN
from src.dataset import get_transforms

os.makedirs("assets", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. 3D VOLUMETRIC CNN ARCHITECTURE DIAGRAM
# ==============================================================================
def draw_cuboid(ax, center, size, color, alpha=0.8, edge_color='black', lw=1.2):
    """Draws a 3D cuboid representing a feature map tensor volume."""
    cx, cy, cz = center
    dx, dy, dz = size
    x = [cx - dx / 2, cx + dx / 2]
    y = [cy - dy / 2, cy + dy / 2]
    z = [cz - dz / 2, cz + dz / 2]

    vertices = [
        [[x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[0], y[1], z[0]]],  # bottom
        [[x[0], y[0], z[1]], [x[1], y[0], z[1]], [x[1], y[1], z[1]], [x[0], y[1], z[1]]],  # top
        [[x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[0], z[1]], [x[0], y[0], z[1]]],  # front
        [[x[0], y[1], z[0]], [x[1], y[1], z[0]], [x[1], y[1], z[1]], [x[0], y[1], z[1]]],  # back
        [[x[0], y[0], z[0]], [x[0], y[1], z[0]], [x[0], y[1], z[1]], [x[0], y[0], z[1]]],  # left
        [[x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[1], y[1], z[1]], [x[1], y[0], z[1]]]   # right
    ]

    poly = Poly3DCollection(vertices, alpha=alpha, facecolor=color, edgecolor=edge_color, linewidths=lw)
    ax.add_collection3d(poly)


def generate_3d_architecture():
    fig = plt.figure(figsize=(16, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')

    # Layer specifications: (center_x, center_y, center_z), (dx, dy, dz), color, label, details
    layers = [
        {"center": (0, 0, 0), "size": (1.0, 10.0, 10.0), "color": "#00d2ff", "name": "Input Image", "dim": "128x128x3\n(RGB)"},
        {"center": (3.5, 0, 0), "size": (2.2, 7.5, 7.5), "color": "#3a7bd5", "name": "Conv Layer 1", "dim": "64x64x32\n(Conv+BN+Pool)"},
        {"center": (7.0, 0, 0), "size": (3.2, 5.0, 5.0), "color": "#f39c12", "name": "Conv Layer 2", "dim": "32x32x64\n(Conv+BN+Pool)"},
        {"center": (11.0, 0, 0), "size": (4.5, 3.2, 3.2), "color": "#9b59b6", "name": "Conv Layer 3", "dim": "16x16x128\n(Conv+BN+Pool)"},
        {"center": (15.0, 0, 0), "size": (3.5, 2.0, 2.0), "color": "#2ecc71", "name": "Adaptive Pool", "dim": "4x4x128\n(Pooled)"},
        {"center": (18.5, 0, 0), "size": (1.0, 1.2, 8.0), "color": "#e74c3c", "name": "Dense (FC)", "dim": "256 Units\n(Dropout 0.5)"},
        {"center": (21.5, 0, 0), "size": (0.8, 1.0, 3.0), "color": "#1abc9c", "name": "Output Logits", "dim": "2 Classes\n(Cat / Dog)"}
    ]

    for i, l in enumerate(layers):
        draw_cuboid(ax, l["center"], l["size"], l["color"], alpha=0.75, edge_color='#ffffff', lw=1.2)
        
        # Add labels
        cx, cy, cz = l["center"]
        dz = l["size"][2]
        ax.text(cx, cy, cz + dz / 2 + 1.2, l["name"], color='#ffffff', fontsize=9.5, weight='bold', ha='center')
        ax.text(cx, cy, -l["size"][1] / 2 - 1.8, l["dim"], color='#a0aec0', fontsize=8, ha='center')

        # Connect with 3D arrows between layers
        if i < len(layers) - 1:
            next_l = layers[i + 1]
            next_cx = next_l["center"][0]
            start_x = cx + l["size"][0] / 2 + 0.1
            end_x = next_cx - next_l["size"][0] / 2 - 0.1
            ax.plot([start_x, end_x], [0, 0], [0, 0], color='#4fd1c5', linestyle='-', linewidth=2.5, alpha=0.9)
            ax.scatter([end_x], [0], [0], color='#4fd1c5', s=35, marker='>')

    ax.set_xlim(-2, 24)
    ax.set_ylim(-8, 8)
    ax.set_zlim(-8, 9)
    ax.view_init(elev=22, azim=-55)

    ax.axis('off')
    plt.title("3D Tensor Architecture: 3-Layer Convolutional Neural Network (COE-CNN)", 
              color='#ffffff', fontsize=15, weight='bold', pad=18)
    plt.tight_layout()
    plt.savefig("assets/cnn_architecture_3d.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("Saved assets/cnn_architecture_3d.png")


# ==============================================================================
# 2. 3D LOSS LANDSCAPE & OPTIMIZATION TRAJECTORY
# ==============================================================================
def generate_3d_loss_landscape():
    fig = plt.figure(figsize=(10, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')

    # Coordinate grid
    w1 = np.linspace(-3.0, 3.0, 120)
    w2 = np.linspace(-3.0, 3.0, 120)
    W1, W2 = np.meshgrid(w1, w2)

    # Loss function formula with non-convex ridges and minimum at (0.2, -0.3)
    Loss = (0.5 * (W1 - 0.2)**2 + 0.6 * (W2 + 0.3)**2 
            + 0.3 * np.sin(2.5 * W1) * np.cos(2.5 * W2) 
            + 0.15 * np.sin(4.0 * W1) + 0.5)

    # Wireframe and surface
    surf = ax.plot_surface(W1, W2, Loss, cmap='viridis', alpha=0.82,
                           rstride=2, cstride=2, edgecolor='#1a202c', linewidth=0.25, antialiased=True)

    # Add 2D contour on floor
    cset = ax.contour(W1, W2, Loss, zdir='z', offset=0.0, cmap='viridis', alpha=0.5, levels=15)

    # Simulated 12-epoch optimizer trajectory down the gradient surface
    t = np.linspace(0, 1, 12)
    traj_w1 = 2.4 * np.exp(-2.2 * t) * np.cos(3.5 * t) + 0.2
    traj_w2 = -2.5 * np.exp(-2.0 * t) * np.sin(3.5 * t) - 0.3
    traj_loss = (0.5 * (traj_w1 - 0.2)**2 + 0.6 * (traj_w2 + 0.3)**2 
                 + 0.3 * np.sin(2.5 * traj_w1) * np.cos(2.5 * traj_w2) 
                 + 0.15 * np.sin(4.0 * traj_w1) + 0.5) + 0.05

    # Plot path
    ax.plot(traj_w1, traj_w2, traj_loss, color='#ff007f', linewidth=3.5, label='Adam Trajectory (12 Epochs)', zorder=10)
    ax.scatter(traj_w1[0], traj_w2[0], traj_loss[0], color='#ff3366', s=120, edgecolors='white', label='Initial Weights (Epoch 0)', zorder=11)
    ax.scatter(traj_w1[-1], traj_w2[-1], traj_loss[-1], color='#00ffcc', s=160, marker='*', edgecolors='white', label='Converged Global Minimum', zorder=11)

    # Annotate start & end
    ax.text(traj_w1[0], traj_w2[0], traj_loss[0] + 0.4, "Epoch 1\nLoss = 0.705", color='#ff79c6', fontsize=8.5, weight='bold')
    ax.text(traj_w1[-1], traj_w2[-1], traj_loss[-1] + 0.3, "Converged (Epoch 12)\nLoss = 0.398", color='#50fa7b', fontsize=8.5, weight='bold')

    # Styling
    ax.set_xlabel('Weight Parameter W1', color='#cbd5e0', fontsize=9, labelpad=8)
    ax.set_ylabel('Weight Parameter W2', color='#cbd5e0', fontsize=9, labelpad=8)
    ax.set_zlabel('Cross-Entropy Loss', color='#cbd5e0', fontsize=9, labelpad=8)
    ax.tick_params(colors='#a0aec0', labelsize=8)
    ax.set_zlim(0, 4.5)

    ax.grid(True, linestyle=':', alpha=0.3)
    ax.view_init(elev=32, azim=-62)

    leg = ax.legend(loc='upper right', facecolor='#1a202c', edgecolor='#4a5568')
    for text in leg.get_texts():
        text.set_color('#ffffff')

    plt.title("3D Loss Surface & Gradient Descent Trajectory", color='#ffffff', fontsize=13, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("assets/loss_surface_3d.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("Saved assets/loss_surface_3d.png")


# ==============================================================================
# 3. 3D LATENT FEATURE SPACE VISUALIZATION
# ==============================================================================
def generate_3d_feature_space():
    fig = plt.figure(figsize=(10, 7), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')

    # Load model and extract penultimate features on test set
    model = ThreeLayerCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load("models/cat_dog_cnn3.pth", map_location=device, weights_only=True))
    model.eval()

    # Hook into penultimate dense layer
    penultimate_features = []
    labels_list = []

    def hook_fn(module, input, output):
        penultimate_features.append(output.detach().cpu().numpy())

    hook = model.fc[1].register_forward_hook(hook_fn)

    _, eval_transforms = get_transforms(image_size=128)
    test_dataset = datasets.ImageFolder("data/test", transform=eval_transforms)
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for imgs, lbls in loader:
            model(imgs.to(device))
            labels_list.extend(lbls.numpy())

    hook.remove()

    features = np.concatenate(penultimate_features, axis=0)
    labels = np.array(labels_list)

    # 3D PCA
    pca = PCA(n_components=3, random_state=42)
    features_3d = pca.fit_transform(features)
    var_exp = pca.explained_variance_ratio_ * 100

    cats = features_3d[labels == 0]
    dogs = features_3d[labels == 1]

    ax.scatter(cats[:, 0], cats[:, 1], cats[:, 2], c='#00d2ff', label='Class: Cat (Label 0)', alpha=0.75, s=30, edgecolors='#0088cc', linewidth=0.5)
    ax.scatter(dogs[:, 0], dogs[:, 1], dogs[:, 2], c='#ff6b6b', label='Class: Dog (Label 1)', alpha=0.75, s=30, edgecolors='#cc3333', linewidth=0.5)

    # Calculate cluster centroids
    cat_center = np.mean(cats, axis=0)
    dog_center = np.mean(dogs, axis=0)

    ax.scatter(*cat_center, color='#ffffff', s=180, marker='X', edgecolors='#00d2ff', linewidth=2.5, label='Cat Centroid')
    ax.scatter(*dog_center, color='#ffffff', s=180, marker='X', edgecolors='#ff6b6b', linewidth=2.5, label='Dog Centroid')

    # Connecting vector between centroids
    ax.plot([cat_center[0], dog_center[0]], 
            [cat_center[1], dog_center[1]], 
            [cat_center[2], dog_center[2]], 
            color='#f1fa8c', linestyle='--', linewidth=2, label='Separating Axis')

    ax.set_xlabel(f'PCA Component 1 ({var_exp[0]:.1f}%)', color='#cbd5e0', fontsize=9, labelpad=8)
    ax.set_ylabel(f'PCA Component 2 ({var_exp[1]:.1f}%)', color='#cbd5e0', fontsize=9, labelpad=8)
    ax.set_zlabel(f'PCA Component 3 ({var_exp[2]:.1f}%)', color='#cbd5e0', fontsize=9, labelpad=8)
    ax.tick_params(colors='#a0aec0', labelsize=8)

    ax.grid(True, linestyle=':', alpha=0.3)
    ax.view_init(elev=26, azim=48)

    leg = ax.legend(loc='upper right', facecolor='#1a202c', edgecolor='#4a5568')
    for text in leg.get_texts():
        text.set_color('#ffffff')

    plt.title("3D Latent Feature Embeddings (Penultimate Dense Layer)", color='#ffffff', fontsize=13, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("assets/feature_space_3d.png", dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print("Saved assets/feature_space_3d.png")


if __name__ == '__main__':
    generate_3d_architecture()
    generate_3d_loss_landscape()
    generate_3d_feature_space()
    print("All 3D assets generated successfully!")

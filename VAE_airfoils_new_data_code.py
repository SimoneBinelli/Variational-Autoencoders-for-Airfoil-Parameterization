# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 09:39:16 2025

@author: simon
"""

# Data loading

import os
import numpy as np

def load_airfoil_dat(path):

    with open(path, "r") as f:
        # Remove whitespace
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # Remove all the rows with no pair of coordinates
    cleaned = []
    for l in lines:
        parts = l.split()
        if len(parts) != 2:
            continue
        cleaned.append(parts)

    # Conversion in a float numpy array
    coords = np.array(cleaned, dtype=float)
    
    # Remove any duplicated point (first = last)
    if len(coords) >= 2:
        first = coords[0]
        last  = coords[-1]
        if np.allclose(first, last, atol=1e-8):
            coords = coords[:-1]

    # x, y coordinates
    x = coords[:, 0] 
    y = coords[:, 1]

    return x, y

def load_all_airfoils(path_folder):

    airfoil_data = {}

    for folder, subdirs, files in os.walk(path_folder): # path of the current folder, list of the subfolders, list of files

        folder_name = os.path.relpath(folder, path_folder) # converts the absolute path (folder) to a path relative to the principal folder
        if folder_name == ".": # skips the root folder
            continue

        dat_files = [f for f in files if f.lower().endswith(".dat")] # searches the current folder for all files ending with .dat
        
        airfoil_data[folder_name] = {} # creates an empty dictionary that will store all .dat files found in this folder 

        for dat in dat_files:
            full_path = os.path.join(folder, dat) # full path to the .dat file
            x, y = load_airfoil_dat(full_path) # loads the coordinates
            airfoil_data[folder_name][dat] = {"x": x, "y": y} # updates the dictionary

    return airfoil_data

path = "C:/Users/simon/OneDrive/Desktop/UNIFI/Tirocinio - Tesi/data/data/simulations"
data = load_all_airfoils(path)

# Show the points of the first airfoil

print(data["RUN_000015"])

# Total number of loaded airfoils

def print_loaded_airfoils(airfoil_data):
    
    count = 0
    
    for folder in airfoil_data: 
        for file in airfoil_data[folder]:
            count += 1
            print(folder, file, "→ points:", len(airfoil_data[folder][file]["x"]))
    
    print("\nLoaded airfoils:", count)
    
print_loaded_airfoils(data)
        
# Data plotting

import matplotlib.pyplot as plt
import time

def plot_airfoils_sequentially(airfoil_data, delay=0.1, equal_axis=True):

    for folder in airfoil_data:
        
        for file in airfoil_data[folder]:

            x = airfoil_data[folder][file]["x"]
            y = airfoil_data[folder][file]["y"]

            plt.figure(figsize=(6,4))
            plt.scatter(x, y, s=15)
            plt.title(f"{folder}/{file}", fontsize=11)
            plt.grid(True, alpha=0.1)

            if equal_axis:
                plt.gca().set_aspect("equal", adjustable="box")

            plt.tight_layout()
            plt.show()

            time.sleep(delay) # small pause between plots
            plt.close()       # close figure before next one

plot_airfoils_sequentially(data)

# Data preprocessing

from scipy.interpolate import interp1d

def enforce_monotonic(x, y):
    
    xx, idx = np.unique(x, return_index=True) # unique and increasing x values
    
    return xx, y[idx] # y that correspond to the indexes of the original array x 
    # the pair (xx, y[idx]) represents the same points but with monotone x

def cosine_resample(x, y, n):
    
    beta = np.linspace(0, np.pi, n)
    x_cos = 0.5 * (1 - np.cos(beta))
    x_new = x_cos * (x.max() - x.min()) + x.min() 
    
    f = interp1d(x, y, kind="linear", fill_value="extrapolate")
    
    return x_new, f(x_new) # new cosine-spaced x values, new linearly interpolated y values

def preprocess_airfoil(x, y, n_up, n_low, scale=True):

    # Find LE and TE
    idx_LE = np.argmin(x)
    idx_TE = np.argmax(x)

    # Construct the sequences LE -> TE, TE -> LE 
    if idx_LE < idx_TE:
        upper = np.column_stack((x[idx_LE:idx_TE+1], y[idx_LE:idx_TE+1]))
        lower = np.column_stack((x[idx_TE:], y[idx_TE:]))
        lower = np.vstack((lower, np.column_stack((x[:idx_LE+1], y[:idx_LE+1]))))
    else:
        upper = np.column_stack((x[idx_LE:], y[idx_LE:]))
        upper = np.vstack((upper, np.column_stack((x[:idx_TE+1], y[:idx_TE+1]))))
        lower = np.column_stack((x[idx_TE:idx_LE+1], y[idx_TE:idx_LE+1]))

    # Coordinates with monotonic x 
    ux, uy = enforce_monotonic(upper[:,0], upper[:,1]) # upper surface
    lx, ly = enforce_monotonic(lower[:,0], lower[:,1]) # lower surface

    # Cosine resampling to have fixed chordwise x locations
    xu, yu = cosine_resample(ux, uy, n_up)
    xl, yl = cosine_resample(lx, ly, n_low)

    # Assemble the airfoil (LE -> upper -> TE -> lower -> LE)
    x_full = np.concatenate([xu, xl[::-1]]) # invert the lower surface to have TE -> LE (x)
    y_full = np.concatenate([yu, yl[::-1]]) # invert the lower surface to have TE -> LE (y)

    # Normalization (chord = 1)
    chord = x_full.max() - x_full.min()
    x_norm = (x_full - x_full.min()) / chord
    y_norm = y_full / chord

    # Min-Max scaling
    if scale:
        ymin, ymax = y_norm.min(), y_norm.max()
        y_scaled = (y_norm - ymin) / (ymax - ymin)
        return x_norm, y_norm, y_scaled, ymin, ymax
    else:
        return x_norm, y_norm, None, None, None

# Pre-processing application

count_processed = 0
count_failed = 0

for folder in data:
    for file in data[folder]:

        x = data[folder][file]["x"]
        y = data[folder][file]["y"]

        try:
            xr, yr, yr_scaled, y_min, y_max = preprocess_airfoil(x, y, n_up=100, n_low=99, scale=True)

            data[folder][file]["x_norm"] = xr
            data[folder][file]["y_norm"] = yr
            
            data[folder][file]["y_scaled"] = yr_scaled
            data[folder][file]["y_min"] = y_min
            data[folder][file]["y_max"] = y_max

            count_processed += 1

        except Exception as e:
            print(f"Failed preprocessing -> {folder}/{file} | Error: {e}")
            count_failed += 1

print(f"Airfoils successfully preprocessed: {count_processed}")
print(f"Airfoils failed preprocessing:      {count_failed}")

# Pre-processed airfoils plot

def plot_processed_airfoils(airfoil_data, delay=0.1):

    for folder in airfoil_data:
        for file in airfoil_data[folder]:

            item = airfoil_data[folder][file]

            if "x_norm" not in item:
                print(f"Skipping {folder}/{file} -> not preprocessed")
                continue

            x = item["x_norm"]
            y = item["y_scaled"]

            plt.figure(figsize=(6,3))
            plt.scatter(x, y, alpha=0.3, s=10)
            plt.title(f"{folder}/{file}", fontsize=11)
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            time.sleep(delay)
            plt.close()

plot_processed_airfoils(data)

# Dataset preparation

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import random_split, DataLoader, TensorDataset

X = []

for folder in data:
    for file in data[folder]:
        if "y_scaled" in data[folder][file]:
            X.append(data[folder][file]["y_scaled"])

X = np.array(X)
print(X.shape) # (number of airfoils, 199)

X = torch.tensor(X, dtype = torch.float32) # trasform into a tensor for torch
dataset = TensorDataset(X) # tensor dataset for data loader

# Training/Validation/Test split

n_samples = len(dataset) 
n_train = int(0.75 * n_samples)      # 75% of the data as training set
n_val = int(0.15 * n_samples)        # 15% of the data as validation set
n_test = n_samples - n_train - n_val # 10% of the data as test set

g = torch.Generator().manual_seed(20) 
train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=g)

train_loader = DataLoader(train_set, batch_size = 64, shuffle = True) # to read the data as mini-batches of size 64
val_loader = DataLoader(val_set, batch_size = 64)
test_loader = DataLoader(test_set, batch_size = 64)

# PCA

X.shape # [922, 199]
X_centered = X - X.mean(dim=0, keepdim=True) # every column must have mean = 0
U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)  # SVD; if data are centered the PCA and the SVD are equivalent
# Vt principal directions
# S singular values

X_pca = X_centered @ Vt.T # data coordinates in the new principal component space
explained_var_ratio = (S**2) / torch.sum(S**2) # part of the variance explained by each component

x_pca = np.arange(1, 11) # 10 principal components
y_pca = explained_var_ratio[:10] # first 10 elements of the vector explained_var_ratio

# Plot

plt.plot(x_pca, y_pca, marker='o')
plt.xlabel("Principal Component")
plt.ylabel("Variance Ratio")
plt.xlim(0.8, 10.2)   
plt.ylim(min(y_pca) - 0.01, max(y_pca) + 0.02) 
plt.xticks(np.arange(1, 11, 1))
plt.grid(True)
plt.show()

# Hyperparameters (same as in the paper)

x_dim = X.shape[1] # 199, input dimension for each example
z_dim = 8          # number of variables in the latent space
epochs = 250       # number of epochs
lr = 3e-4          # learning rate
beta = 5e-7        # coefficient beta used to balance the reconstruction loss and the KL divergence

# Encoder

# composed of linear layers with LeakyReLU activation functions
# 4 hidden layers composed by 196, 172, 172, 143 neurons respectively
# takes as input an airfoil (199 coordinates) and generates 2 vectors (mu, logvar) of dimension 8 (variables in the latent space)

class Encoder(nn.Module):
    
    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.layers = nn.Sequential( 
            nn.Linear(x_dim, 196),
            nn.LeakyReLU(0.1),
            nn.Linear(196, 172),
            nn.LeakyReLU(0.1),
            nn.Linear(172, 172),
            nn.LeakyReLU(0.1),
            nn.Linear(172, 143),
            nn.LeakyReLU(0.1)
        )
        self.mu = nn.Linear(143, z_dim)
        self.logvar = nn.Linear(143, z_dim)
   
    def forward(self, x):
        h = self.layers(x)
        return self.mu(h), self.logvar(h)
    
# Decoder 

# mirrored architecture 
# takes as input the z vector of dimension 8 (combination of mu and logvar vectors) and reconstructs the original airfoil (199 coordinates)

class Decoder(nn.Module):
    
    def __init__(self, z_dim, x_dim):
        super().__init__()
        self.layers = nn.Sequential( 
            nn.Linear(z_dim, 143),
            nn.LeakyReLU(0.1),
            nn.Linear(143, 172),
            nn.LeakyReLU(0.1),
            nn.Linear(172, 172),
            nn.LeakyReLU(0.1),
            nn.Linear(172, 196),
            nn.LeakyReLU(0.1)
        )
        self.out = nn.Linear(196, x_dim)
        
    def forward(self, z):
        h = self.layers(z)
        return self.out(h)
    
# Model 

enc, dec = Encoder(x_dim, z_dim), Decoder(z_dim, x_dim)

# Optimization with ADAM

opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr = lr)
# the optimizer trains both parts of the VAE simultaneously

# Weights initialization to avoid divergences or NaN in the first steps

def weights_init(l):
    if isinstance(l, nn.Linear): # apply initialization only to layers with parameters
        nn.init.xavier_uniform_(l.weight) # initializes the weights of the matrix with the uniform Xavier distribution (Glorot)
        nn.init.zeros_(l.bias) # resets all biases

enc.apply(weights_init)
dec.apply(weights_init)

# Check before the training phase

print(torch.isnan(X).any())       # False
print(torch.isinf(X).any())       # False
print(torch.min(X), torch.max(X)) # [0,1]
print(torch.mean(X))              

# Training and Validation phase

for epoch in range(epochs): # 250 iterations
    enc.train() # training mode
    dec.train()
    train_loss = 0.0 # to accumulate the average training loss of this epoch

    for (x_batch,) in train_loader: # the train_loader provides mini-batches of shape [64, 199]; the comma is needed because TensorDataset returns a tuple (even if it has only one tensor)
        
        x_batch = x_batch.float() # float32 needed for torch

        # Encoder
        mu, logvar = enc(x_batch)
        sigma = torch.exp(0.5 * logvar)

        # Reparameterization trick
        eps = torch.randn_like(mu) # standard noise (N(0,1))
        z = mu + sigma * eps 
        
        # Decoder
        recon = dec(z)

        # Relative Mean Squared Error as reconstruction loss
        x_shift = x_batch + 1.0 # y ordinates are offset upwards by one to ensure stability and avoid division by zero
        recon_shift = recon + 1.0
        rel_error = ((x_shift - recon_shift) / x_shift) ** 2
        recon_loss = torch.mean(rel_error)

        # KL divergence 
        kl = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - logvar - 1) / x_batch.size(0)

        # Training loss
        loss = recon_loss + beta * kl

        # Gradient update
        opt.zero_grad()
        loss.backward()

        # Parameter update according to ADAM
        opt.step()
        
        # Training loss update
        train_loss += loss.item()

    # Validation
    
    enc.eval() # evaluation mode
    dec.eval()
    val_loss = 0.0 # to accumulate the average validation loss of this epoch
    
    with torch.no_grad():
        
        for (x_batch,) in val_loader:
            
            x_batch = x_batch.float()
            
            # Encoder
            mu, logvar = enc(x_batch)
            sigma = torch.exp(0.5 * logvar)
            
            # Reparameterization trick
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            
            # Decoder
            recon = dec(z)

            # Relative Mean Squared Error as reconstruction loss
            x_shift = x_batch + 1.0
            recon_shift = recon + 1.0
            rel_error = ((x_shift - recon_shift) / x_shift) ** 2
            recon_loss = torch.mean(rel_error)

            # KL divergence
            kl = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - logvar - 1) / x_batch.size(0)
            
            # Total loss
            loss = recon_loss + beta * kl
            
            # Validation loss update
            val_loss += loss.item()

    print(f"Epoch {epoch+1:03d}/{epochs} | "
          f"Training Loss: {train_loss/len(train_loader):.6f} | "
          f"Validation Loss: {val_loss/len(val_loader):.6f}")

# Test phase

enc.eval() # evaluation mode
dec.eval()
test_loss = 0.0 # to accumulate the average test loss of this epoch

with torch.no_grad():
    
    for (x_batch,) in test_loader:
        
        x_batch = x_batch.float()
        
        # Encoder
        mu, logvar = enc(x_batch)
        sigma = torch.exp(0.5 * logvar)
        
        # Reparameterization trick
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        
        # Decoder
        recon = dec(z)

        # Relative Mean Squared Error as reconstruction loss
        x_shift = x_batch + 1.0
        recon_shift = recon + 1.0
        rel_error = ((x_shift - recon_shift) / x_shift) ** 2
        recon_loss = torch.mean(rel_error)

        # KL divergence
        kl = 0.5 * torch.sum(mu.pow(2) + sigma.pow(2) - logvar - 1) / x_batch.size(0)

        # Total loss
        loss = recon_loss + beta * kl

        # Test loss update
        test_loss += loss.item()

print(f"Test Loss: {test_loss / len(test_loader):.6f}")

# Validation

# Rescaling of the results
def inverse_scale(y_scaled, y_min, y_max):
    
    return y_scaled * (y_max - y_min) + y_min

folder = list(data.keys())[0]
file   = list(data[folder].keys())[0] # first airfoil

x_norm   = data[folder][file]["x_norm"]
y_scaled  = data[folder][file]["y_scaled"]
y_min     = data[folder][file]["y_min"]
y_max     = data[folder][file]["y_max"]

x_tensor = torch.tensor(y_scaled).float().unsqueeze(0)
mu, logvar = enc(x_tensor)
z = mu

y_recon_scaled = dec(z).detach().squeeze().numpy()
y_recon_original = inverse_scale(y_recon_scaled, y_min, y_max)
y_original = inverse_scale(y_scaled, y_min, y_max)

# Plot 
plt.figure(figsize=(6, 3))
plt.scatter(x_norm, y_original, s=2, alpha=0.65, label="Original")
plt.scatter(x_norm, y_recon_original, s=8, alpha=0.65, label="Reconstructed")
plt.title(f"Recon vs Original: {folder}/{file}", fontsize=11)
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Now on the whole dataset
def get_reconstruction(y_scaled):
    
    x_tensor = torch.tensor(y_scaled).float().unsqueeze(0)
    mu, logvar = enc(x_tensor)
    z = mu
    
    return dec(z).detach().squeeze().numpy()

loss_list = []

for folder in data:
    
    for file in data[folder]:
        
        item = data[folder][file]
        if "y_scaled" not in item:
            continue
        
        y_scaled = item["y_scaled"]
        y_recon_scaled = get_reconstruction(y_scaled)

        eps = 1e-8
        x_shift = y_scaled + 1.0 + eps
        recon_shift = y_recon_scaled + 1.0
        
        rel_err = ((x_shift - recon_shift) / x_shift) ** 2
        rel_mean = np.mean(rel_err)

        loss_list.append((rel_mean, folder, file, y_scaled, y_recon_scaled, item["y_min"], item["y_max"], item["x_norm"]))

loss_list.sort(key=lambda x: x[0])

best12  = loss_list[:12]         
worst12 = loss_list[-12:]        

# Plot worst 12 (highest reconstruction loss)
fig, axes = plt.subplots(4, 3, figsize=(14, 10))
axes = axes.flatten()

for ax, (rel_mean, folder, file, y_s, y_r_s, y_min, y_max, x_norm) in zip(axes, worst12):
    
    y_original = inverse_scale(y_s, y_min, y_max)
    y_recon = inverse_scale(y_r_s, y_min, y_max)

    ax.scatter(x_norm, y_original, s=2, alpha=0.65, label="Original")
    ax.scatter(x_norm, y_recon, s=2, alpha=0.65, label="Reconstructed")
    ax.set_title(f"{file}\nReconstruction Loss = {rel_mean:.2e}", fontsize=9)
    ax.axis('equal')
    ax.grid(True, alpha=0.35)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.tight_layout()
plt.show()

# Plot best 12 (lowest reconstruction loss)
fig, axes = plt.subplots(4, 3, figsize=(14, 10))
axes = axes.flatten()

for ax, (rel_mean, folder, file, y_s, y_r_s, y_min, y_max, x_norm) in zip(axes, best12):
    
    y_original = inverse_scale(y_s, y_min, y_max)
    y_recon = inverse_scale(y_r_s, y_min, y_max)

    ax.scatter(x_norm, y_original, s=2, alpha=0.65, label="Original")
    ax.scatter(x_norm, y_recon, s=2, alpha=0.65, label="Reconstructed")
    ax.set_title(f"{file}\nReconstruction Loss = {rel_mean:.2e}", fontsize=9)
    ax.axis('equal')
    ax.grid(True, alpha=0.35)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
plt.tight_layout()
plt.show()

# Output processing

# Inferred z latent distributions 

# Extract all latent vectors z_i associated with the airfoils 

enc.eval() # evaluation mode 

Z = [] # list for the latent vectors z_i

with torch.no_grad(): 
    
    for (x_batch,) in train_loader:
        
        x_batch = x_batch.float()
    
        mu, _ = enc(x_batch) 
        z = mu 
        
        Z.append(z.numpy()) # z from tensor to array

# all batches are merged into a single matrix
Z = np.concatenate(Z, axis=0) # [n_samples in the training set, latent_dim]
latent_dim = Z.shape[1]

# Plot

import seaborn as sns
import matplotlib.lines as mlines

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(latent_dim):
    
    z_i = Z[:, i] # extract all values ​​of the i_th latent component z_i for all samples
    
    mean = np.mean(z_i)
    std = np.std(z_i)
    var = np.var(z_i)

    # Histograms and KDE
    h = sns.histplot(z_i, kde=True, stat='probability', bins=30, color='skyblue', ax=axes[i], alpha=0.4)
    
    for line in h.lines:
    
        line.set_color('blue')
        line.set_linewidth(2)
    
    # Mean and standard deviation lines
    axes[i].axvline(mean, color='red', linestyle='--', linewidth=1.5)
    axes[i].axvline(mean + std, color='green', linestyle='--', linewidth=1.0)
    axes[i].axvline(mean - std, color='green', linestyle='--', linewidth=1.0)

    # Title
    axes[i].set_title(f"z{i}", fontsize=12)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Probability")

    # Box with statistics (μ, σ, σ²)
    stats_text = f"μ = {mean:.3f}\nσ = {std:.3f}\nσ² = {var:.3f}"
    axes[i].text(0.97, 0.97, stats_text, transform=axes[i].transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round, pad=0.4', facecolor='white', alpha=0.8, edgecolor='black'))

# General legenda
kde_line = mlines.Line2D([], [], color='blue', label='KDE')
mean_line = mlines.Line2D([], [], color='red', linestyle='--', label='Mean')
std_line = mlines.Line2D([], [], color='green', linestyle='--', label='Standard Deviation')
fig.legend(handles=[kde_line, mean_line, std_line], loc='lower center', ncol=3, fontsize=10, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1]) 
plt.show()

# Generation

# Bezier curves -> Composite B-spline

from scipy.special import comb
from scipy.optimize import minimize
from scipy.spatial import KDTree
from scipy.interpolate import BSpline

# Function to compute a Bezier curve

def bezier_curve(t, ctrl): # t parameter in [0, 1]; ctrl array of the control points
    
    n = len(ctrl) - 1 # degree of the curve
    curve = np.zeros((len(t), 2)) # array for the points on the curve (x, y)
    
    for i in range(n + 1):
        
        B = comb(n, i) * ((1 - t) ** (n - i)) * (t ** i) # Bernstein formula
        curve += np.outer(B, ctrl[i]) # external product between B (len(t),) and ctr (2,)

    return curve # matrix 2D (len(t), 2) that represents the curve in the plane

# Objective function for the fitting
# to minimize the sum of the squared perpendicular distances between the points of the airfoil and the Bezier curve

def perpendicular_distance_sum(ctrl_flat, x_data, y_data):
    
    ctrl = ctrl_flat.reshape(-1, 2) # ctrl.flat -> array 1D of the coordinates (x, y) of the control points
    # reshape (-1, 2) trasforms the 1D array in a matrix 2D -> ctrl (n_ctrl, 2)
    curve = bezier_curve(np.linspace(0, 1, 200), ctrl) # computes the x(t), y(t) coordinates of the 200 points
    tree = KDTree(curve) # for any point of the airfoil (x, y) we can quickly find the closest point on the curve
    total = 0.0 # to accumulate the sum of the squares of the perpendicular distances between the curve and the points of the airfoil
    
    for (x, y) in zip(x_data, y_data):
        
        _, idx = tree.query([x, y], k=2) # find the 2 closest points of the curve to the point (x, y)
        # I want not only the closest point but also the adjacent one to construct the segment of the curve on which the perpendicular projection lies
        p1, p2 = curve[idx[0]], curve[idx[1]] # extract the consecutive points of the curve (2,)
        
        v = p2 - p1 # directional vector of the considered Bezier segment
        w = np.array([x, y]) - p1 # vector from p1 to point (x, y)
        
        t = np.dot(v, w) / np.dot(v, v) # formula for the projection coefficient of one vector onto another
        # coefficient t of the orthogonal projection of the point (x, y) on the segment p1–p2

        proj = p1 + t * v # coordinates of the point projected onto the Bezier curve
        # proj is the point on the curve closest to the point (x, y) in a perpendicular direction
        
        total += np.sum((proj - np.array([x, y])) ** 2) # squared distances
        
    return total

# Function to find the optimal coordinates of the control points

def fit_bezier_surface(x, y, n_ctrl=11):
    
    # Initial values of the coordinates of the control points
    x_init = np.linspace(0, 1, n_ctrl) # divides the chord into n_ctrl-1 equal segments (as in the paper)
    y_init = np.interp(x_init, (x - x.min()) / (x.max() - x.min()), y) # linear interpolation
    ctrl_init = np.column_stack((x_init, y_init)) # initial estimate of the control points

    # Geometric constraints -> bounds for each coordinate (x, y)
    bounds = [] 
    
    for i in range(n_ctrl):
        
        if i == 0:
            # LE fixed: x=0, y free
            bounds += [(0, 0), (None, None)]
        elif i == n_ctrl - 1:
            # TE fixed: x=1, y free
            bounds += [(1, 1), (None, None)]
        elif i == 1:
            # second point: x fixed to 0, y free
            bounds += [(0, 0), (None, None)]
        else:
            # all other control points: x in [0, 1], y free
            bounds += [(0, 1), (None, None)]

    # Optimization of the objective function
    res = minimize(perpendicular_distance_sum, ctrl_init.flatten(), args=(x, y), method="L-BFGS-B", bounds=bounds)
    # the algorithm moves the control points to minimize the distance between the Bezier and the shape of the airfoil
    
    # Final Bezier curve (set of its optimal control points)
    return res.x.reshape(-1, 2) # res.x -> 1D vector with all coordinates optimized
    # reshape(-1, 2) -> converts it back into a matrix (n_ctrl, 2)

# Function to fit the two Bezier curves

def fit_airfoil_bezier(airfoil_xy, n_ctrl=11):
    
    # Split the airfoil into two halves
    n_half = len(airfoil_xy)//2 
    upper, lower = airfoil_xy[:n_half], airfoil_xy[n_half-1:][::-1] # upper/lower surface

    # Coordinates of the control points for the upper/lower surface
    cu = fit_bezier_surface(upper[:,0], upper[:,1], n_ctrl) 
    cl = fit_bezier_surface(lower[:,0], lower[:,1], n_ctrl)
    
    # Shared LE/TE in x (C0) + vertical tangents at LE (C1)
    LE_y = 0.5 * (cu[0, 1] + cl[0, 1]) # common LE_y
    cu[0, 0] = cl[0, 0] = 0 # upper LE_x = lower LE_x = 0
    cu[0, 1] = cl[0, 1] = LE_y # upper LE_y = lower LE_y = common LE_y
    
    TE_y = 0.5 * (cu[-1, 1] + cl[-1, 1]) # common TE_y
    cu[-1, 0] = cl[-1, 0] = 1 # upper TE_x = lower TE_x = 1
    cu[-1, 1] = cl[-1, 1] = TE_y # upper TE_y = lower TE_y = common TE_y
    
    cu[1, 0] = cl[1, 0] = 0 # the second point has the same x as the LE (x = 0)

    ctrl_all = np.vstack((cu, cl[::-1][1:])) # combines the upper/lower surface control points into a single array
    # avoid the duplicate -> len (ctrl_all) = 21

    return cu, cl, ctrl_all 

# Function to convert a Bezier curve to an equivalent B-spline curve

def bezier_to_bspline(ctrl):
 
    n = len(ctrl)
    k = n - 1  
    
    # Two knot values (0 and 1), each with multiplicity n
    t = np.concatenate((np.zeros(n), np.ones(n))) # to clamp the B-spline endpoints
    
    return BSpline(t, ctrl, k)

# Function to create the final composite B-spline

def composite_bspline(cu, cl):

    ctrl_all = np.vstack((cu, cl[::-1][1:])) # exclude duplicate LE
    
    n = len(cu)
    k = n - 1

    # Knot vector: clamp at 0 and 1, add interior knot for LE with multiplicity k
    t = np.concatenate((
        np.zeros(n),     # clamp start 
        np.full(k, 0.5), # interior LE, where the two curves meet
        np.ones(n)       # clamp end
    ))

    return BSpline(t, ctrl_all, k), ctrl_all, t

# Function for the plot 

def plot_airfoil_bspline(name, airfoil_xy, bspline, ctrl_all):
    
    t = np.linspace(0, 1, 1000) # generates 1000 equally spaced values ​​in the parametric domain of the curve, t in [0, 1]
    curve = bspline(t) # evaluates the BSpline object (BSpline(t, ctrl_all, k))
    # curve is an (800, 2) matrix, where each row is a pair (x, y) which represents a point on the profile surface

    plt.figure(figsize=(10, 6))
    plt.plot(curve[:, 0], curve[:, 1], 'g', lw=2.3, label='Airfoil Surface')
    plt.scatter(airfoil_xy[:, 0], airfoil_xy[:, 1], c='orange', s=30, marker='x', label='Airfoil Points')
    # plt.plot(ctrl_all[:, 0], ctrl_all[:, 1], 'b-o', lw=1.9, ms=6, label='Bezier Control Points')
    plt.title(f"{name} B-spline Fit", fontsize=14)
    plt.xlabel("Normalized Location Along Chordline (x/c)", fontsize=12)
    plt.ylabel("Normalized Thickness (t/c)", fontsize=12)
    plt.axis("equal")
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

# Function to run the pipeline

def run_full_pipeline(airfoil_xy, name="Airfoil", n_ctrl=11):
    
    # Fit Bezier curves (upper and lower)
    cu, cl, ctrl_all = fit_airfoil_bezier(airfoil_xy)

    # Conversion into clamped B-splines
    bs_upper = bezier_to_bspline(cu)
    bs_lower = bezier_to_bspline(cl)

    # Combination in a composite B-spline 
    bs_composite, ctrl_all, knots = composite_bspline(cu, cl)

    # Final plot
    plot_airfoil_bspline(name, airfoil_xy, bs_composite, ctrl_all)

    return bs_composite, ctrl_all, knots, bs_upper, bs_lower

# Generate 2 new airfoils

# 2 latent vectors with values different from the z_i means
w1 = np.array([-2.2, -1.6, -3.1, 0.1, 0.2, -0.5, 0.25, -2.2]) 
w2 = np.array([-2.3, -1.7, -3.3, 0.2, 0.4, -1.0, 0.27, -1.7])

Z1 = torch.tensor(w1, dtype=torch.float32).unsqueeze(0) 
Z2 = torch.tensor(w2, dtype=torch.float32).unsqueeze(0)
# .unsqueeze(0) adds a batch size (transforms them from [8] -> [1, 8])
# dec expects inputs of the form [batch_size, latent_dim]

with torch.no_grad():
    
    y1 = dec(Z1).squeeze().numpy() # 199 y ordinates of the generated airfoil
    y2 = dec(Z2).squeeze().numpy() # squeeze() eliminates the batch size
    
airfoil1 = np.column_stack((x_norm, y1)) # coordinates of the first generated airfoil
airfoil2 = np.column_stack((x_norm, y2)) # coordinates of the second generated airfoil

# Raw plot + radar plot

# Latent variables means
z_mean = np.mean(Z, axis=0)

# Amplify the vectors by a common factor
stretch = 1.6
v1 = w1 * stretch
v2 = w2 * stretch

# Global r_max -> both plots use the same radial scale
global_rmax = max(np.max(np.abs(v1)), np.max(np.abs(v2))) * 1.1 # 10% margin
r_ticks = np.linspace(-global_rmax, global_rmax, 5)

# Prepare the radar values
def radar_vals(w):
    vals = (w * stretch).tolist()
    return vals + [vals[0]] # adds the first value to close the polygon

n_dim = len(w1)
labels = [f"z{i}" for i in range(n_dim)]
angles = np.linspace(0, 2*np.pi, n_dim, endpoint=False).tolist() # takes n_dim equidistant angles between 0 and 2pi (2pi excluded)
angles += angles[:1] # adds the first angle at the end -> this also closes the polygon

mean_vals = z_mean.tolist() + [z_mean[0]] # adds the first element at the end -> same trick as before
airfoil1_vals = radar_vals(w1) # distances from the center for each angle of the radar (first airfoil)
airfoil2_vals = radar_vals(w2) # distances from the center for each angle of the radar (second airfoil)

# Generated airfoil 1
fig = plt.figure(figsize=(11, 4))
ax1 = fig.add_axes([0.05, 0.15, 0.42, 0.75])
ax1.plot(airfoil1[:, 0], airfoil1[:, 1], 'r-', lw=1.8)
ax1.set_title("Generated Airfoil #1 (Raw)")
ax1.set_xlabel("x (chordwise)")
ax1.set_ylabel("y")
ax1.axis("equal")
ax1.grid(True)

# Radar plot 1
ax2 = fig.add_axes([0.5, 0.12, 0.35, 0.76], polar=True)
ax2.plot(angles, mean_vals, color='black', linestyle='--', linewidth=1.8, label='Mean')
ax2.fill(angles, mean_vals, color='black', alpha=0.18)
ax2.plot(angles, airfoil1_vals, color='red', linewidth=1.8, label='Airfoil #1')
ax2.fill(angles, airfoil1_vals, color='red', alpha=0.35)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(labels, fontsize=10)
ax2.set_yticks(r_ticks)
ax2.set_yticklabels([f"{t:.2f}" for t in r_ticks], fontsize=9)
ax2.set_ylim(-global_rmax, global_rmax)
ax2.set_rlabel_position(90)
ax2.set_title("Generated Airfoil #1 — Latent vs Mean", size=13, y=1.12)
ax2.grid(True, alpha=0.35)
ax2.legend(loc='upper right', bbox_to_anchor=(1.27, 1.1))

plt.show()

# Generated airfoil 2
fig = plt.figure(figsize=(11, 4))
ax1 = fig.add_axes([0.05, 0.15, 0.42, 0.75])
ax1.plot(airfoil2[:, 0], airfoil2[:, 1], 'r-', lw=1.8)
ax1.set_title("Generated Airfoil #2 (Raw)")
ax1.set_xlabel("x (chordwise)")
ax1.set_ylabel("y")
ax1.axis("equal")
ax1.grid(True)

# Radar plot 2
ax2 = fig.add_axes([0.5, 0.12, 0.35, 0.76], polar=True)
ax2.plot(angles, mean_vals, color='black', linestyle='--', linewidth=1.8, label='Mean')
ax2.fill(angles, mean_vals, color='black', alpha=0.18)
ax2.plot(angles, airfoil2_vals, color='red', linewidth=1.8, label='Airfoil #2')
ax2.fill(angles, airfoil2_vals, color='red', alpha=0.35)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(labels, fontsize=10)
ax2.set_yticks(r_ticks)
ax2.set_yticklabels([f"{t:.2f}" for t in r_ticks], fontsize=9)
ax2.set_ylim(-global_rmax, global_rmax)
ax2.set_rlabel_position(90)
ax2.set_title("Generated Airfoil #2 — Latent vs Mean", size=13, y=1.12)
ax2.grid(True, alpha=0.35)
ax2.legend(loc='upper right', bbox_to_anchor=(1.27, 1.1))

plt.show()

# B-spline on the two generated airfoils

# Airfoil 1
bspline1, ctrl1, knots1, *_ = run_full_pipeline(airfoil1, "Generated Airfoil #1") 

# Airfoil 2
bspline2, ctrl2, knots2, *_ = run_full_pipeline(airfoil2, "Generated Airfoil #2")

# Captured features in the latent dimensions

# Qualitative inspection of latent variables

# Main function without plot: elaborates an airfoil with the composite B-spline without plotting it
def run_full_pipeline_silent(airfoil_xy, name="Airfoil", n_ctrl=11):
    
    cu, cl, ctrl_all = fit_airfoil_bezier(airfoil_xy, n_ctrl)
    
    bs_upper = bezier_to_bspline(cu)
    bs_lower = bezier_to_bspline(cl)
    bs_composite, ctrl_all, knots = composite_bspline(cu, cl)
    
    return bs_composite, ctrl_all, knots, bs_upper, bs_lower
 
# Compute mean and standard deviation for each latent dimension across the entire dataset
z_mean = np.mean(Z, axis=0)
z_std = np.std(Z, axis=0)

# Plot
fig, axes = plt.subplots(4, 2, figsize=(12, 12)) # 8 subplots
axes = axes.flatten()
fig.suptitle("Effect of Each Latent Variable on Airfoil Geometry (μ ± 2σ)", fontsize=15, y=0.93)

for i in range(latent_dim):
    
    ax = axes[i]

    # Case 1: z_i = μ_i - 2σ_i
    z_minus = z_mean.copy()
    z_minus[i] -= 2 * z_std[i]

    # Case 2: z_i = μ_i + 2σ_i
    z_plus = z_mean.copy()
    z_plus[i] += 2 * z_std[i]

    # Decode both airfoils
    with torch.no_grad():
        
        # the decoder generates the output vectors y_minus and y_plus, which represent two airfoils corresponding to z_i = μ_i - 2σ_i and z_i = μ_i + 2σ_i
        y_minus = dec(torch.tensor(z_minus, dtype=torch.float32).unsqueeze(0)).squeeze().numpy()
        y_plus  = dec(torch.tensor(z_plus,  dtype=torch.float32).unsqueeze(0)).squeeze().numpy()

    # Reconstruct and smooth the airfoil geometry
    airfoil_minus = np.column_stack((x_norm, y_minus)) # coordinates of the first generated airfoil
    airfoil_plus = np.column_stack((x_norm, y_plus)) # coordinates of the second generated airfoil

    bspline_minus, *_ = run_full_pipeline_silent(airfoil_minus)
    bspline_plus,  *_ = run_full_pipeline_silent(airfoil_plus)

    t = np.linspace(0, 1, 1000)
    smooth_minus = bspline_minus(t)
    smooth_plus  = bspline_plus(t)

    # Plot both variations
    ax.plot(smooth_minus[:, 0], smooth_minus[:, 1], color='deepskyblue', lw=1.6, label=f"z{i} = μ - 2σ")
    ax.plot(smooth_plus[:, 0],  smooth_plus[:, 1],  color='red', lw=1.6, label=f"z{i} = μ + 2σ")
    ax.axhline(0, color='gray', lw=0.5)
    ax.set_title(f"Latent variable z{i}", fontsize=11)
    axes[i].set_xlabel("x/c", fontsize=11)
    axes[i].set_ylabel("t/c", fontsize=11)
    ax.set_xlim(-0.05, 1.2)
    ax.set_ylim(-0.05, 1.2)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.legend(fontsize=9, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

# Q–Q Plots

from scipy import stats

# Standardization of Z: used to compare each z_i with a standard normal distribution N (0, 1)
Z_standardized = (Z - Z.mean(axis=0)) / Z.std(axis=0)

# Plots
fig, axes = plt.subplots(2, 4, figsize=(12, 6)) # 8 subplots
axes = axes.flatten()

for i in range(Z_standardized.shape[1]): # 8
    
    z_i = Z_standardized[:, i] # extracts the i_th column of Z_standardized, that is, all values ​​of the latent variable z_i for all the airfoils
    (osm, osr), _ = stats.probplot(z_i, dist="norm") # stats.probplot compares the empirical distribution of z_i with the theoretical distribution of a standard normal
    # osm are the theoretical quantiles of the normal distribution
    # osr are the empirical quantiles of the sample z_i
    
    error = np.mean((osr - osm)**2) # computes how much the empirical quantiles differ from the theoretical ones
    
    axes[i].scatter(osm, osr, s=8, color='tab:blue', alpha=0.6) # represents the points (osm, osr)
    axes[i].plot(osm, osm, 'r--', lw=1) # draws the dashed red line y=x, which represents the perfect normal trend
    axes[i].set_title(f"z{i} — Q–Q Error = {error:.3e}", fontsize=9)
    axes[i].set_xlabel("Theoretical Quantiles", fontsize=9)
    axes[i].set_ylabel("Latent Quantiles", fontsize=9)
    axes[i].grid(True)

fig.suptitle("Q–Q Plots", fontsize=13)
plt.tight_layout()
plt.show()

# Correlation analysis

# Correlation metrics: Pearson, Spearman, Kendall-Tau correlation coefficient, Mutual Information I(X; Y)

# Geometric features of the airfoils: 
# maximum thickness (t_max);
# maximum camber (c_max);
# upper and lower crest value (z_u, z_l);
# the chordwise locations of each of these properties (x_t_max, x_c_max, x_z_u, x_z_l);
# the curvatures at the upper and lower crest (k_z_u, k_z_l);
# the leading edge radius (R_LE); 
# the angles of the camber line at the leading and trailing edge (θ_LE, θ_TE);
# the trailing edge wedge angle (γ_TE).

# First type of generated data
# Latent vectors sampled randomly from a uniform distribution

x_ref = x_norm

def generate_latent_random_data(dec, z_mean, z_std, n_samples=500, n_jobs=-1):
    
    latent_dim = len(z_mean)
    
    # Generates 500 Z latent vectors, each of dimension 8, uniformly distributed in the interval (z_mean - 2*z_std, z_mean + 2*z_std) for each dimension i
    Z_sampled = np.random.uniform(z_mean - 2*z_std, z_mean + 2*z_std, size=(n_samples, latent_dim))

    with torch.no_grad():
        
        # Pass all the Z vectors to the decoder in a single call (more efficient)
        y_dec = dec(torch.tensor(Z_sampled, dtype=torch.float32)).numpy() # 500 x 199

    def process_airfoil(y_vec):
        
      airfoil = np.column_stack((x_ref, y_vec))
      features = compute_airfoil_features(airfoil)

      print("\n---------------------------")
      print(f"tmax={features[0]:.5f}, cmax={features[1]:.5f}, zu={features[2]:.5f}, zl={features[3]:.5f}")
      print(f"kzu={features[4]:.5f}, kzl={features[5]:.5f}, xtmax={features[6]:.5f}, xcmax={features[7]:.5f}")
      print(f"xzu={features[8]:.5f}, xzl={features[9]:.5f}, RLE={features[10]:.6f}")
      print(f"θLE={np.degrees(features[11]):.3f}°, θTE={np.degrees(features[12]):.3f}°, γTE={np.degrees(features[13]):.3f}°")
      print("-----------------------------")
        
      return features
      
    F_list = [process_airfoil(y_dec[i]) for i in range(n_samples)]
    # F_list contains a list of arrays (one per airfoil), each with 14 features -> 500 x 14
    
    return Z_sampled, np.vstack(F_list)

# Second type of generated data
# Each latent parameter is varied systematically between z_i = μ_i ± 2σ_i, while the rest is kept at the mean value

def generate_latent_traversal_data(dec, z_mean, z_std, latent_dim, n_points=20, n_jobs=-1):
    
    z_values = np.linspace(-2, 2, n_points)
    Z_list = [] # contains 8 × 20 vectors, each corresponding to a change in a single latent factor

    for i in range(latent_dim):
        
        for val in z_values:
            
            z = z_mean.copy()
            z[i] = z_mean[i] + val * z_std[i] # only the component z_i is modified, leaving all the others at their mean value 
            # the goal is to explore that dimension i by moving from −2σ to +2σ (multiplying val by z_std[i])
            
            Z_list.append(z)

    Z_trav = np.array(Z_list) # (8 x 20, 8)

    with torch.no_grad():
        
        y_dec = dec(torch.tensor(Z_trav, dtype=torch.float32)).numpy() # (8 x 20, 199)

    def process_airfoil(y_vec):
        
        airfoil = np.column_stack((x_ref, y_vec))
        features = compute_airfoil_features(airfoil)

        print("\n---------------------------")
        print(f"tmax={features[0]:.5f}, cmax={features[1]:.5f}, zu={features[2]:.5f}, zl={features[3]:.5f}")
        print(f"kzu={features[4]:.5f}, kzl={features[5]:.5f}, xtmax={features[6]:.5f}, xcmax={features[7]:.5f}")
        print(f"xzu={features[8]:.5f}, xzl={features[9]:.5f}, RLE={features[10]:.6f}")
        print(f"θLE={np.degrees(features[11]):.3f}°, θTE={np.degrees(features[12]):.3f}°, γTE={np.degrees(features[13]):.3f}°")
        print("-----------------------------")
          
        return features

    F_list = [process_airfoil(y_dec[i]) for i in range(len(Z_trav))]

    return Z_trav, np.vstack(F_list)

# Geometric features computation

from scipy.optimize import least_squares

def compute_airfoil_features(airfoil):

    x, y = airfoil[:, 0], airfoil[:, 1]

    # LE and TE
    idx_LE = np.argmin(x)
    idx_TE = np.argmax(x)

    # Same split as in preprocessing
    if idx_LE < idx_TE:
        upper = np.column_stack((x[idx_LE:idx_TE+1], y[idx_LE:idx_TE+1]))
        lower = np.column_stack((np.r_[x[idx_TE:], x[:idx_LE+1]], np.r_[y[idx_TE:], y[:idx_LE+1]]))
    else:
        upper = np.column_stack((np.r_[x[idx_LE:], x[:idx_TE+1]], np.r_[y[idx_LE:], y[:idx_TE+1]]))
        lower = np.column_stack((x[idx_TE:idx_LE+1], y[idx_TE:idx_LE+1]))

    # Upper and lower surface ordered coordinates 
    upper = upper[np.argsort(upper[:, 0])]
    lower = lower[np.argsort(lower[:, 0])]
    xu_raw, yu_raw = upper[:, 0], upper[:, 1]
    xl_raw, yl_raw = lower[:, 0], lower[:, 1]

    # Interpolation
    x_common = x_ref.copy()
    mask_u = (x_common >= xu_raw.min()) & (x_common <= xu_raw.max()) # mask_u = True if x_common is within the range covered by xu_raw
    mask_l = (x_common >= xl_raw.min()) & (x_common <= xl_raw.max()) # mask_l = True if x_common is within the range covered by xl_raw

    yu = np.full_like(x_common, np.nan)
    yl = np.full_like(x_common, np.nan)
    yu[mask_u] = np.interp(x_common[mask_u], xu_raw, yu_raw) # performs a 1D linear interpolation: takes the original points (xu_raw, yu_raw) and estimates y on x_common[mask_u]
    yl[mask_l] = np.interp(x_common[mask_l], xl_raw, yl_raw) # performs a 1D linear interpolation: takes the original points (xl_raw, yl_raw) and estimates y on x_common[mask_l]

    valid = ~np.isnan(yu) & ~np.isnan(yl) # valid is True only if both yu and yl are okay

    # Maximum thickness
    thickness = yu[valid] - yl[valid] # vertical difference between extrados and intrados
    tmax  = thickness.max()
    
    # Maximum (relative) camber
    camber_y  = 0.5 * (yu[valid] + yl[valid]) # average between upper and lower y
    camber_rel = camber_y - camber_y[0]
    cmax  = camber_rel.max() 
    
    # Upper and lower crest value
    zu  = yu[valid].max() # higher y
    zl  = yl[valid].min() # lower y
    
    # Chordwise locations of each of these properties
    xg = x_common[valid]
    xtmax = xg[np.argmax(thickness)] # x where max thickness
    xcmax = xg[np.argmax(camber_rel)] # x where max camber
    xzu = xg[np.argmax(yu[valid])] # x of upper crest value
    xzl = xg[np.argmin(yl[valid])] # x of lower crest value

    # Curvature at at the upper and lower crest
    # Function that takes a neighborhood of points around the index idx (a centered window) and fits a parabola on it
    def local_polyfit(x, y, idx, window=9):
        
        h = window // 2 # 4 on the left + idx + 4 on the right
        i0 = max(idx - h, 0) # window's beginning: if idx is close to the beginning of the array, avoid negative indexes
        i1 = min(idx + h + 1, len(x)) # window's end: if idx is close to the end of the array, avoid going beyond len(x)
        
        return np.polyfit(x[i0:i1], y[i0:i1], 2) # returns the coefficients [a, b, c] of the quadratic polynomial y = ax^2 + bx + c fitted (in a least-squares sense) to the points in a local neighborhood around the point with index idx

    def curvature(p, x0):
        
        a, b = p[0], p[1] # extracts only a and b for the computation of the derivative
        dy  = 2*a*x0 + b # first derivative
        d2y = 2*a # second derivative
        
        return abs(d2y) / (1 + dy**2)**1.5 # curvature formula for plane curves

    # Find the upper/lower crest indexes 
    idx_zu = np.argmax(yu[valid])
    idx_zl = np.argmin(yl[valid])

    # Default curvature values
    kzu = np.nan
    kzl = np.nan

    if 0.05 < xzu < 0.95: # excludes cases where xzu falls too close to LE or TE
        p = local_polyfit(xg, yu[valid], idx_zu) # parabolic fit around the upper crest index
        kzu = curvature(p, xzu) # evaluates the curvature at point xzu

    if 0.05 < xzl < 0.95: # excludes cases where xzl falls too close to LE or TE
        p = local_polyfit(xg, yl[valid], idx_zl) # parabolic fit around the lower crest index
        k = curvature(p, xzl) # evaluates the curvature at point xzl
        kzl = k if k < 50 else np.nan # throws away the estimated curvature if it is > 50

    # LE radius
    def fit_circle(x, y): # function that, given the points (x, y), finds the circumference that best approximates them
        
        # Function for the residuals
        # Each point should satisfy the equation of the circle
        # The residuals are the deviations from this condition: least_squares will try to minimize the sum of these squared residuals to get the best approximation
        def res(p):
            
            a, b, R = p # a, b coordinates of the center of the circumference; R radius
            
            return (x - a)**2 + (y - b)**2 - R**2

        p0 = [x.mean(), y.mean(), 0.05] # initial vector of parameters [a, b, R]
        sol = least_squares(res, p0) # iterative optimization to minimize the residuals
        
        return abs(sol.x[2]) # optimal R

    n_le = 12 # the first 12 points of the xg grid -> area near the LE
    x_le = np.r_[xg[:n_le], xg[:n_le]] 
    y_le = np.r_[yu[valid][:n_le], yl[valid][:n_le]] # concatenate the 12 upper points near the LE and the 12 lower points near the LE
    RLE = fit_circle(x_le, y_le) # 24 points in total for the fit

    # Angles of the camber line at the LE and TE
    LE = xg < 0.05 # boolean mask that takes the camber line points with x < 0.05 (LE zone)
    TE = xg > 0.95 # boolean mask that takes the camber line points with x > 0.95 (TE zone)
    # Fit the line camber_rel(x) ≈ mx+q; np.polyfit returns [m, q]
    θLE = np.arctan(np.polyfit(xg[LE], camber_rel[LE], 1)[0]) # converts slope m to geometric angle for LE
    θTE = np.arctan(np.polyfit(xg[TE], camber_rel[TE], 1)[0]) # converts slope m to geometric angle for TE

    # TE wedge angle
    # np.polyfit returns the slope m_u/m_l of the line that approximates the upper/lower surface near the TE (tangent line)
    γTE = abs(np.arctan(np.polyfit(xg[TE], yu[valid][TE], 1)[0]) - np.arctan(np.polyfit(xg[TE], yl[valid][TE], 1)[0])) # angular difference between the two tangents to the TE

    return np.array([tmax, cmax, zu, zl, kzu, kzl, xtmax, xcmax, xzu, xzl, RLE, θLE, θTE, γTE])

# Filter to remove features containing NaN values

def filter_nan_features(F, feature_labels):

    keep = ~np.any(np.isnan(F), axis=0) # returns a boolean vector of n_features that says True if a column does not contain any NaN values and False otherwise
    F_filt = F[:, keep] # all the rows but only the columns with keep = True (no NaN values)
    labels_filt = [lbl for lbl, k in zip(feature_labels, keep) if k] # contains only the names of the remaining features

    return F_filt, labels_filt, keep

# Correlation metrics between features

def compute_feature_correlations(F): # matrix F (n_samples, 14)

    n_feat = F.shape[1] # 14
    
    # Preallocate three zero-filled square matrices, each of size (14, 14)
    pearson_corr = np.zeros((n_feat, n_feat))
    spearman_corr = np.zeros_like(pearson_corr)
    kendall_corr = np.zeros_like(pearson_corr)
    
    # Double loop on all pairs (i, j) of features
    for i in range(n_feat):
        
        for j in range(n_feat):
            
            # Pearson correlation (linear relationship)
            pearson_corr[i, j], _ = stats.pearsonr(F[:, i], F[:, j])
            
            # Spearman correlation (monotonic relationship)
            spearman_corr[i, j], _ = stats.spearmanr(F[:, i], F[:, j])
            
            # Kendall-Tau correlation (concordance between pairs)
            kendall_corr[i, j], _ = stats.kendalltau(F[:, i], F[:, j])
    
    return pearson_corr, spearman_corr, kendall_corr

# Correlation metrics between airfoil features (F) and latent variables (Z)

from sklearn.feature_selection import mutual_info_regression

def compute_correlations(F, Z):
    
    n_feat = F.shape[1] # 14
    n_latent = Z.shape[1] # 8

    # Preallocate four zero-filled matrices, each of size (14, 8)
    pearson = np.zeros((n_feat, n_latent))
    spearman = np.zeros_like(pearson)
    kendall = np.zeros_like(pearson)
    mi = np.zeros_like(pearson)

    for i in range(n_feat):
        
        for j in range(n_latent):
            
            pearson[i, j], _ = stats.pearsonr(F[:, i], Z[:, j])
            spearman[i, j], _ = stats.spearmanr(F[:, i], Z[:, j])
            kendall[i, j], _ = stats.kendalltau(F[:, i], Z[:, j])
            
            # Mutual Information (linear and non-linear dependencies)
            mi[i, j] = mutual_info_regression(F[:, [i]], Z[:, j], random_state=0)[0] # X (F in this case) -> array 2D -> must be (n_samples, 1); then extracts the first value with [0]

    return pearson, spearman, kendall, mi

# Correlation matrices plots

# Correlation matrices (features, features)

def plot_feature_corr(matrix, title, feature_labels, corr_label):
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': corr_label}, xticklabels=feature_labels, yticklabels=feature_labels)
    plt.title(title, fontsize=13, pad=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
# Correlation matrices (features, latent variables)

def plot_pearson_correlation_matrix(matrix, title, feature_labels, latent_labels):

    plt.figure(figsize=(9, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'ρ'})
    plt.title(title, fontsize=13, pad=12)
    plt.xlabel("Latent Variables (z)", fontsize=11)
    plt.ylabel("Airfoil Features", fontsize=11)
    plt.xticks(ticks=np.arange(len(latent_labels)) + 0.5, labels=latent_labels, rotation=0)
    plt.yticks(ticks=np.arange(len(feature_labels)) + 0.5, labels=feature_labels, rotation=0)
    plt.tight_layout()
    plt.show()
    
def plot_spearman_correlation_matrix(matrix, title, feature_labels, latent_labels):

    plt.figure(figsize=(9, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'ρₛ'})
    plt.title(title, fontsize=13, pad=12)
    plt.xlabel("Latent Variables (z)", fontsize=11)
    plt.ylabel("Airfoil Features", fontsize=11)
    plt.xticks(ticks=np.arange(len(latent_labels)) + 0.5, labels=latent_labels, rotation=0)
    plt.yticks(ticks=np.arange(len(feature_labels)) + 0.5, labels=feature_labels, rotation=0)
    plt.tight_layout()
    plt.show()

def plot_kendall_correlation_matrix(matrix, title, feature_labels, latent_labels):

    plt.figure(figsize=(9, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'τ'})
    plt.title(title, fontsize=13, pad=12)
    plt.xlabel("Latent Variables (z)", fontsize=11)
    plt.ylabel("Airfoil Features", fontsize=11)
    plt.xticks(ticks=np.arange(len(latent_labels)) + 0.5, labels=latent_labels, rotation=0)
    plt.yticks(ticks=np.arange(len(feature_labels)) + 0.5, labels=feature_labels, rotation=0)
    plt.tight_layout()
    plt.show()
    
def plot_mutual_information_matrix(matrix, title, feature_labels, latent_labels):

    plt.figure(figsize=(9, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="OrRd", cbar_kws={'label': 'I(X;Y)'})
    plt.title(title, fontsize=13, pad=12)
    plt.xlabel("Latent Variables (z)", fontsize=11)
    plt.ylabel("Airfoil Features", fontsize=11)
    plt.xticks(ticks=np.arange(len(latent_labels)) + 0.5, labels=latent_labels, rotation=0)
    plt.yticks(ticks=np.arange(len(feature_labels)) + 0.5, labels=feature_labels, rotation=0)
    plt.tight_layout()
    plt.show()
    
# Analysis

latent_dim = len(z_mean)

latent_labels = [f"z{i}" for i in range(latent_dim)]
feature_labels = ["t_max", "c_max", "z_u", "z_l", "k_z_u", "k_z_l", "x_t_max", "x_c_max", "x_z_u", "x_z_l", "R_LE", "θ_LE", "θ_TE", "γ_TE"]

# Generate latent random samples
Z_rand, F_rand = generate_latent_random_data(dec, z_mean, z_std)

# Generate latent traversal samples
Z_trav, F_trav = generate_latent_traversal_data(dec, z_mean, z_std, latent_dim)

# Compute features correlations matrices
F_rand, feature_labels_rand, keep_rand = filter_nan_features(F_rand, feature_labels)
F_trav, feature_labels_trav, keep_trav = filter_nan_features(F_trav, feature_labels)

removed_rand = [l for l, k in zip(feature_labels, keep_rand) if not k]
removed_trav = [l for l, k in zip(feature_labels, keep_trav) if not k]
print("Removed features (rand):", removed_rand)
print("Removed features (trav):", removed_trav)

pearson_rand_feat, spearman_rand_feat, kendall_rand_feat = compute_feature_correlations(F_rand)
pearson_trav_feat, spearman_trav_feat, kendall_trav_feat = compute_feature_correlations(F_trav)

# Compute features, latent variables correlation matrices
pearson_rand, spearman_rand, kendall_rand, mi_rand = compute_correlations(F_rand, Z_rand)
pearson_trav, spearman_trav, kendall_trav, mi_trav = compute_correlations(F_trav, Z_trav)

# Visualize the results

# Latent Random
plot_feature_corr(pearson_rand_feat, "Pearson Correlation – Features (Latent Random)", feature_labels_rand, "ρ")
plot_feature_corr(spearman_rand_feat, "Spearman Correlation – Features (Latent Random)", feature_labels_rand, "ρₛ")
plot_feature_corr(kendall_rand_feat, "Kendall-Tau Correlation – Features (Latent Random)", feature_labels_rand, "τ")

plot_pearson_correlation_matrix(pearson_rand, "Pearson Correlation (Latent Random)", feature_labels_rand, latent_labels)
plot_spearman_correlation_matrix(spearman_rand, "Spearman Correlation (Latent Random)", feature_labels_rand, latent_labels)
plot_kendall_correlation_matrix(kendall_rand, "Kendall-Tau Correlation (Latent Random)", feature_labels_rand, latent_labels)
plot_mutual_information_matrix(mi_rand, "Mutual Information (Latent Random)", feature_labels_rand, latent_labels)

# Latent Traversal
plot_feature_corr(pearson_trav_feat, "Pearson Correlation – Features (Latent Traversal)", feature_labels_trav, "ρ")
plot_feature_corr(spearman_trav_feat, "Spearman Correlation – Features (Latent Traversal)", feature_labels_trav, "ρₛ")
plot_feature_corr(kendall_trav_feat, "Kendall-Tau Correlation – Features (Latent Traversal)", feature_labels_trav, "τ")

plot_pearson_correlation_matrix(pearson_trav, "Pearson Correlation (Latent Traversal)", feature_labels_trav, latent_labels)
plot_spearman_correlation_matrix(spearman_trav, "Spearman Correlation (Latent Traversal)", feature_labels_trav, latent_labels)
plot_kendall_correlation_matrix(kendall_trav, "Kendall-Tau Correlation (Latent Traversal)", feature_labels_trav, latent_labels)
plot_mutual_information_matrix(mi_trav, "Mutual Information (Latent Traversal)", feature_labels_trav, latent_labels)

# Aerodynamic properties of the airfoils:

# Data loading

import pandas as pd

path_aero = r"C:/Users/simon/OneDrive/Desktop/UNIFI/Tirocinio - Tesi/data/data/database.dat"

df = pd.read_csv(path_aero, sep=",", comment=None)
print(df.shape)
print(df.head())
print(df.columns)

# CASE -> ID simulation
# DOF -> Design variables (degrees of freedom of the geometry) -> 15
# FUN -> Derived variables -> 22
# OF -> Aerodynamic outputs (CFD results) -> 29

# Input -> DOF + FUN
# Output -> OF

# DOF_* -> the true design parameters of the airfoil:
# DOF_ALFAIN_ -> inlet flow angle (angle of attack at inlet)
# DOF_ALFAEX_ -> outlet flow angle (angle of attack at outlet)
# DOF_BETA1_ -> inlet blade / flow metal angle
# DOF_BETA2_ -> outlet blade / flow metal angle
# DOF_W1_ -> geometric weighting / thickness parameter on pressure side
# DOF_W2_ -> geometric weighting / thickness parameter on suction side
# DOF_RTE_ -> trailing edge radius
# DOF_TMAXU_ -> maximum thickness on the upper surface
# DOF_TMAXL_ -> maximum thickness on the lower surface
# DOF_TMOVXU_ -> chordwise position of maximum thickness (upper surface)
# DOF_TMOVXL_ -> chordwise position of maximum thickness (lower surface)
# DOF_ZW_ -> camber offset / vertical profile shift
# DOF_REYNOLDS_ -> Reynolds number based on chord
# DOF_M2IS_	-> isentropic Mach number at outlet
# DOF_DARATIO_ -> diffusion / area ratio parameter

# FUN_* -> physical properties or operating conditions:
# FUN_VISCOSITY_ -> dynamic viscosity
# FUN_RO_ -> fluid density
# FUN_CHORDX_ -> airfoil chord length
# FUN_GAMMA_ -> specific heat ratio
# FUN_REMISES_ -> reference Reynolds number
# FUN_ALFAEXL_ -> outlet flow angle on the lower surface
# FUN_ALFAEXU_ -> outlet flow angle on the upper surface
# FUN_SOLIDITY_	-> blade solidity
# FUN_SOLIDITYBB_ -> blade-to-blade solidity
# FUN_PITCH_ -> blade pitch
# FUN_VX_ -> axial velocity component
# FUN_GAP1_	-> geometric gap parameter 1
# FUN_GAP2_	-> geometric gap parameter 2
# FUN_ALFAIN_M_	-> mean inlet flow angle
# FUN_ALFAEX_M_	-> mean outlet flow angle
# FUN_NBLADE_ -> number of blades
# FUN_RAD_ -> radial position
# FUN_P2ONP0_ -> static-to-total pressure ratio
# FUN_M1_ -> inlet Mach number
# FUN_PSI_M2_ -> loading coefficient at outlet
# FUN_PSI_M1_ -> loading coefficient at inlet
# FUN_ARATIO_ -> area ratio

# OF_* -> CFD-computed aerodynamic performance indicators for a specific operating point (OP_01):
# OF_alfa_in_OP_01 -> effective inlet flow angle
# OF_alfa_ex_OP_01 -> effective outlet flow angle
# OF_Re2is_OP_01 -> isentropic Reynolds number at outlet
# OF_Tu_in_OP_01 -> inlet turbulence intensity
# OF_Cpt_OP_01 -> total pressure coefficient
# OF_CSI_OP_01 -> loss coefficient
# OF_phi_OP_01 -> flow coefficient
# OF_psi_OP_01 -> loading coefficient
# OF_Zwi_OP_01 -> internal viscous loss
# OF_Zwc_OP_01 -> wake loss
# OF_fstar_OP_01 -> separation parameter
# OF_Ds_mis_OP_01 -> boundary layer thickness (mis formulation)
# OF_Ds_mis_max_OP_01 -> maximum boundary layer thickness
# OF_Ds_mis_max_TE_OP_01 -> maximum boundary layer thickness at trailing edge
# OF_DFss_mis_OP_01 -> suction-side separation factor
# OF_Ds_cp_OP_01 -> boundary layer thickness (cp formulation)
# OF_Ds_cp_max_OP_01 -> maximum cp-based boundary layer thickness
# OF_Ds_cp_max_TE_OP_01	-> maximum cp-based boundary layer thickness at trailing edge
# OF_DFss_cp_OP_01 -> pressure-side separation factor
# OF_Mis_peak_OP_01	-> peak Mach number
# OF_s_peak_OP_01 -> chordwise position of Mach peak
# OF_s_diff_dim_OP_01 -> separation length (dimensional)
# OF_s_tot_SS_OP_01	-> total suction-side length
# OF_Tmax_OP_01	-> maximum temperature
# OF_X_Tmax_OP_01 -> chordwise position of maximum temperature
# OF_BB_length_OP_01 -> boundary-layer length
# OF_Area_OP_01	-> flow area
# OF_Icsi_OP_01	-> integrated loss coefficient
# OF_Ieta_OP_01	-> integrated efficiency index

# Select only the DOFs
dof_cols = [c for c in df.columns if c.startswith("DOF_")]
print(f"Number of DOFs: {len(dof_cols)}")
print(dof_cols)

DOF = df[dof_cols].values   
print(DOF.shape) # (922, 15)

# Deterministic latent representation (mu only)
full_loader = DataLoader(dataset, batch_size=64, shuffle=False)

enc.eval()

Z_mu_all = []

with torch.no_grad():
    for (x_batch,) in full_loader:  
        x_batch = x_batch.float()
        mu, _ = enc(x_batch)
        Z_mu_all.append(mu.numpy())

Z_mu_all = np.vstack(Z_mu_all)    

print("Z_mu_all shape:", Z_mu_all.shape) # (922, 8)

# Keep only DOFs with standard deviation > 0
tol = 1e-12
keep_dof = np.std(DOF, axis=0) > tol 

DOF_final = DOF[:, keep_dof] 
dof_cols_f = [c for c, k in zip(dof_cols, keep_dof) if k]

print(f"Selected DOF: {len(dof_cols_f)} -> {dof_cols_f}")

# Compute correlations
n_dof = DOF_final.shape[1]
n_lat = Z_mu_all.shape[1]

pearson = np.zeros((n_dof, n_lat))
spearman = np.zeros((n_dof, n_lat))
kendall = np.zeros((n_dof, n_lat))
mi = np.zeros((n_dof, n_lat))

for i in range(n_dof):
    for j in range(n_lat):
        pearson[i, j], _ = stats.pearsonr(DOF_final[:, i], Z_mu_all[:, j])
        spearman[i, j], _ = stats.spearmanr(DOF_final[:, i], Z_mu_all[:, j])
        kendall[i, j], _ = stats.kendalltau(DOF_final[:, i], Z_mu_all[:, j])
        mi[i, j] = mutual_info_regression(DOF_final[:, [i]], Z_mu_all[:, j], random_state=20)[0]

# Correlation matrices
plot_pearson_correlation_matrix(pearson, "Pearson Correlation (DOF - Latent Variables)", dof_cols_f, latent_labels)
plot_spearman_correlation_matrix(spearman, "Spearman Correlation (DOF - Latent Variables)", dof_cols_f, latent_labels)
plot_kendall_correlation_matrix(kendall, "Kendall-Tau Correlation (DOF - Latent Variables)", dof_cols_f, latent_labels)
plot_mutual_information_matrix(mi, "Mutual Information (DOF - Latent Variables)", dof_cols_f, latent_labels)


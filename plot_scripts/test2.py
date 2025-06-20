import numpy as np
import matplotlib.pyplot as plt

# Generate data
x1 = np.linspace(-0.5, 1.0, 100)  # Linear scale
x2 = np.linspace(1.0, 10, 100)    # Inverse scale (1/x)

y = np.linspace(0, 10, 100)
X1, Y = np.meshgrid(x1, y)
X2, _ = np.meshgrid(x2, y)

Z1 = np.sin(X1) * np.cos(Y)
Z2 = np.cos(1 / X2) * np.sin(Y)  # Applying inverse scale

# Create figure with two subplots that share the y-axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True, gridspec_kw={'wspace': 0})

# Define global color limits
vmin, vmax = min(Z1.min(), Z2.min()), max(Z1.max(), Z2.max())

# First plot (linear scale)
im1 = ax1.imshow(Z1, extent=[x1.min(), x1.max(), y.min(), y.max()], 
                 origin='lower', aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
ax1.set_xlabel("Linear Scale (X)")
ax1.set_ylabel("Y")
ax1.set_xlim(x1.min(), x1.max())

# Second plot (1/x scale)
im2 = ax2.imshow(Z2, extent=[1/x2.max(), 1/x2.min(), y.min(), y.max()], 
                 origin='lower', aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
ax2.set_xlabel("Reciprocal Scale (1/X)")
ax2.set_xlim(1/x2.max(), 1/x2.min())  # Flip axis to match continuity

# Remove extra space between subplots
ax2.set_yticklabels([])  

# Add a single colorbar
cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical', fraction=0.05, pad=0.02)
cbar.set_label("Z Value")

# Add a vertical split line at x=1 for clarity
ax1.axvline(x=1.0, color="black", linestyle="--", lw=1)
ax2.axvline(x=1.0, color="black", linestyle="--", lw=1)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Example data
x = np.linspace(0, 10, 100)
y = np.sin(x)
z = np.random.rand(10, 100)

# Create the figure
fig = plt.figure(figsize=(8, 8))
gs = GridSpec(2, 2, width_ratios=[50, 1], height_ratios=[1, 1])  # Reserve space for the colorbar

# First subplot with pcolor
ax1 = fig.add_subplot(gs[0, 0])  # Main plot
p = ax1.pcolor(x, np.arange(10), z, shading='auto')
ax1.set_title('pcolor plot')

# Colorbar
cbar_ax = fig.add_subplot(gs[0, 1])  # Colorbar subplot
fig.colorbar(p, cax=cbar_ax)

# Second subplot with plot
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)  # Second plot
ax2.plot(x, y, label='Sine wave')
ax2.set_title('plot')
ax2.legend()

# Adjust layout
plt.tight_layout()
plt.show()


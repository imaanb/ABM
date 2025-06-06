import numpy as np

# 2D Gaussian sugar hill
width, height = 50, 50
x = np.linspace(-1, 1, width)
y = np.linspace(-1, 1, height)
xv, yv = np.meshgrid(x, y)
sugar_map = np.exp(-5 * (xv**2 + yv**2)) * 10  # Peak value around 10
sugar_map = sugar_map.astype(int)

# Save to file
np.savetxt("sugar-map.txt", sugar_map, fmt='%d')

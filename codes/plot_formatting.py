import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
def plot_fading_line_3d(x, y, z, time, tau, color='blue', lw=1, ax=None):
    """
    Plot a 3D line with fading transparency over time.
    
    Parameters:
    x, y, z (array-like): 3D coordinates
    time (array-like): Time values for each point
    tau (float): Decay parameter (higher = slower fade)
    color: Base color of the line
    lw (float): Line width
    ax (mpl_toolkits.mplot3d.Axes3D): Existing 3D axes (optional)
    """
    # Convert base color to RGBA
    base_rgba = to_rgba(color)
    
    # Create figure and axes if none provided
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Calculate maximum time for decay reference
    max_time = np.max(time)
    
    # Plot each segment individually with its own alpha
    for i in range(len(x)-1):
        # Calculate alpha using exponential decay
        alpha = np.exp(-0.5*(max_time - time[i+1]) / (tau))
        
        # Create color with updated alpha
        seg_color = (*base_rgba[:3], alpha)
        
        # Plot individual segment
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], 
                color=seg_color, lw=lw)
    
    # Set axis limits
    ax.set_xlim(np.min(x), np.max(x))
    ax.set_ylim(np.min(y), np.max(y))
    ax.set_zlim(np.min(z), np.max(z))
    
    return fig, ax
def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates to Cartesian"""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z
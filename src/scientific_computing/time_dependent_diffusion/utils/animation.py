import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from scientific_computing.time_dependent_diffusion import (
    time_dependent_diffusion,
)


def plot_3d_cylinder(grid, ax):
    """Maps the 2D grid onto a 3D cylinder and plots it."""
    Ny, Nx = grid.shape
    theta = np.linspace(0, 2 * np.pi, Nx + 1)
    z = np.linspace(1, 0, Ny + 1)
    Theta, Z = np.meshgrid(theta, z)

    X = 0.5 * (np.cos(Theta) + 1) * (1 / np.pi)
    Y = 0.5 * (np.sin(Theta) + 1) * (1 / np.pi)

    ax.clear()
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.coolwarm(grid), cmap="coolwarm")
    ax.axis("equal")
    ax.set_zlim(0, 1)
    ax.set_title("Time Dependent Diffusion with Cylindrical Boundaries")
    ax.set_xlabel("X")
    ax.set_ylabel("X")
    ax.set_zlabel("Y")


def animate_diffusion(grid_history, dpi=300, fps=50):
    """Creates an animation of the diffusion process on a 3D cylinder."""
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        plot_3d_cylinder(grid_history[frame], ax)

    ani = animation.FuncAnimation(fig, update, frames=len(grid_history), interval=100)
    save_path = "diffusion_cylinder.mp4"
    ani.save(save_path, writer="ffmpeg", fps=fps)
    print(f"Animation saved as {save_path}")


diffusivity = 1
dt = 0.001
time_steps = 1000
intervals = 15

_, grid_history = time_dependent_diffusion(time_steps, intervals, dt, diffusivity)

animate_diffusion(grid_history)

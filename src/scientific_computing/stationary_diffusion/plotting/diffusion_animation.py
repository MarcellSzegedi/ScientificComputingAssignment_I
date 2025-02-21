import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def create_stat_diff_animation_for_all_iter(
        history_len: int,
        save_dir: str,
        file_name: str,
        jac_grid_hist: np.ndarray,
        gs_grid_hist: np.ndarray,
        sor_grid_hist: np.ndarray
) -> None:
    jac_grid_hist, gs_grid_hist, sor_grid_hist = fill_grid_histories(history_len,
                                                                     jac_grid_hist,
                                                                     gs_grid_hist,
                                                                     sor_grid_hist)
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    vmin, vmax = 0, 1

    # Initialize the plots for the two grids
    image_1 = axs[0].imshow(jac_grid_hist[0], cmap="coolwarm", animated=True, vmin=vmin, vmax=vmax)
    axs[0].set_title("Jacobian Iteration")

    image_2 = axs[1].imshow(gs_grid_hist[0], cmap="coolwarm", animated=True, vmin=vmin, vmax=vmax)
    axs[1].set_title("Gauss - Seidel Iteration")

    image_3 = axs[2].imshow(sor_grid_hist[0], cmap="coolwarm", animated=True, vmin=vmin, vmax=vmax)
    axs[2].set_title("Successive Over Relaxation Iteration")

    def update(frame):
        # Update grid lattice for the different iteration approaches
        image_1.set_array(jac_grid_hist[frame])
        image_2.set_array(gs_grid_hist[frame])
        image_3.set_array(sor_grid_hist[frame])

        return image_1, image_2, image_3

    # Create the animation
    ani = FuncAnimation(fig, update, frames=history_len, interval=100, blit=True)

    # Save the animation as HTML
    ani.save(os.path.join(save_dir, file_name), writer='html', fps=5)

    # Display the animation
    plt.tight_layout()
    plt.show()


def create_stat_diff_animation(
        grid_hist: np.ndarray,
        save_dir: str,
        file_name: str,
        title: str
) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    vmin, vmax = 0, 1

    # Initialize the plots for the two grids
    image = ax.imshow(grid_hist[0], cmap="coolwarm", animated=True, vmin=vmin, vmax=vmax)
    ax.set_title(title)

    def update(frame):
        # Update grid lattice for the different iteration approaches
        image.set_array(grid_hist[frame])

        return image,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=grid_hist.shape[0], interval=100, blit=True)

    # Save the animation as HTML
    ani.save(os.path.join(save_dir, file_name), writer='html', fps=5)

    # Display the animation
    plt.tight_layout()
    plt.show()


def fill_grid_histories(
        history_len: int,
        jac_grid_hist: Optional[np.ndarray],
        gs_grid_hist: Optional[np.ndarray],
        sor_grid_hist: Optional[np.ndarray]
) -> (np.ndarray, np.ndarray, np.ndarray):
    if jac_grid_hist.shape[0] != history_len:
        last_slice = jac_grid_hist[-1][np.newaxis, :, :]
        jac_grid_hist = np.concatenate([jac_grid_hist,
                                        np.repeat(last_slice, history_len - jac_grid_hist.shape[0],
                                                  axis=0)],
                                       axis=0)
    if gs_grid_hist.shape[0] != history_len:
        last_slice = gs_grid_hist[-1][np.newaxis, :, :]
        gs_grid_hist = np.concatenate([gs_grid_hist,
                                       np.repeat(last_slice, history_len - gs_grid_hist.shape[0],
                                                 axis=0)],
                                      axis=0)
    if sor_grid_hist.shape[0] != history_len:
        last_slice = sor_grid_hist[-1][np.newaxis, :, :]
        sor_grid_hist = np.concatenate([sor_grid_hist,
                                        np.repeat(last_slice, history_len - sor_grid_hist.shape[0],
                                                  axis=0)],
                                       axis=0)
    return jac_grid_hist, gs_grid_hist, sor_grid_hist


def _check_grid_histories(
        history_len: int,
        jac_grid_hist: Optional[np.ndarray],
        gs_grid_hist: Optional[np.ndarray],
        sor_grid_hist: Optional[np.ndarray]
) -> None:
    if jac_grid_hist.shape[0] != history_len:
        raise ValueError(f"Grid history of the Jacobian iteration approach {jac_grid_hist.shape[0]} does not match with "
                         f"the required history length {history_len}.")
    if gs_grid_hist.shape[0] != history_len:
        raise ValueError(f"Grid history of the Gauss - Seidel iteration approach {gs_grid_hist.shape[0]} does not match "
                         f"with the required history length {history_len}.")
    if sor_grid_hist.shape[0] != history_len:
        raise ValueError(f"Grid history of the SOR iteration approach {sor_grid_hist.shape[0]} does not match with the "
                         f"required history length {history_len}.")

import numpy as np
import numpy.typing as npt


def my_func(x: npt.NDArray) -> npt.NDArray:
    rng = np.random.default_rng(42)
    return x.copy() + rng.random(size=x.shape)

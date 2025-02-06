import numpy as np
import numpy.typing as npt


def my_func(x: npt.NDArray) -> npt.NDArray:
    return x.copy() + np.ones_like(x)

import numpy as np

from scientific_computing.example import my_func


def test_my_func():
    """Should add 1 to an array."""
    x1 = np.array([1, 2, 3], dtype=np.int64)
    y1 = np.array([2, 3, 4], dtype=np.int64)

    x2 = np.array([1, 2, 3], dtype=np.float64)
    y2 = np.array([2, 3, 4], dtype=np.float64)

    x3 = np.array([False, False, False], dtype=np.bool_)
    y3 = np.array([True, True, True], dtype=np.bool_)

    y1_res = my_func(x1)
    y2_res = my_func(x2)
    y3_res = my_func(x3)

    assert (y1 == y1_res).all()
    assert np.isclose(y2, y2_res).all()
    assert (y3 == y3_res).all()

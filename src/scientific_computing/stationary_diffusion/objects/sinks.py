import numpy as np

FIRST_HOR_REC_Y_REL_COORD = 0.1


def create_hor_single_sink_filter(grid: np.ndarray) -> np.ndarray:
    """
    Creates a 2D numpy array consisting of boolean values, where every boolean value representing whether the cell
    belongs to the sink(s) or not.

    Args:
        grid: Existing grid (2D numpy array) where every cell value represents a delta
                    step in the discretised square field with side interval [0, 1].

    Returns:
        Filter (2D numpy array)
    """
    # Initialise filter array
    filter_array = np.zeros(grid.shape, dtype=bool)

    # Create coordinates to place the sink
    sink_width = int(0.5 * grid.shape[1])
    sink_y_position = int(FIRST_HOR_REC_Y_REL_COORD * grid.shape[0])
    sink_x_start = int((filter_array.shape[1] - sink_width) / 2)

    # Assign the True values to the place of the cells that are part of the sink
    filter_array[sink_y_position, sink_x_start:filter_array.shape[1] - sink_x_start + 1] = True

    return filter_array


def create_hor_double_sink_filter(grid: np.ndarray) -> np.ndarray:
    """
    Creates a 2D numpy array consisting of boolean values, where every boolean value representing whether the cell
    belongs to the sink(s) or not.

    Args:
        grid: Existing grid (2D numpy array) where every cell value represents a delta
                    step in the discretised square field with side interval [0, 1].

    Returns:
        Filter (2D numpy array)
    """
    # Initialise filter array
    filter_array = np.zeros(grid.shape, dtype=bool)

    # Create coordinates to place the sink
    sink_width = int(0.2 * grid.shape[1])
    sink_y_position = int(FIRST_HOR_REC_Y_REL_COORD * grid.shape[0])
    sink_x_start = int(filter_array.shape[1] * 0.2)

    # Assign the True values to the place of the cells that are part of the sink
    filter_array[sink_y_position, sink_x_start:sink_x_start + sink_width + 1] = True
    filter_array[sink_y_position, filter_array.shape[1] - sink_x_start - sink_width:filter_array.shape[1] - sink_x_start + 1] = True

    return filter_array

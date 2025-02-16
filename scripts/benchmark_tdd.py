from cProfile import Profile
from pstats import SortKey, Stats

from scientific_computing.time_dependent_diffusion import (
    time_dependent_diffusion_numba,
)

if __name__ == "__main__":
    ITERATIONS = 10
    TIME_STEPS = 1000
    INTERVALS = 250
    DT = 0.00001
    D = 0.5

    with Profile() as profile:
        for _ in range(ITERATIONS):
            # grid = td_diffusion_cylinder([TIME_STEPS], INTERVALS, DT, D)
            grid = time_dependent_diffusion_numba(TIME_STEPS, INTERVALS, DT, D)
            # grid = time_dependent_diffusion(TIME_STEPS, INTERVALS, DT, D)
        Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats(25)

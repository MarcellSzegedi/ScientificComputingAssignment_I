from cProfile import Profile
from pstats import SortKey, Stats

from scientific_computing.time_dependent_diffusion import time_dependent_diffusion

if __name__ == "__main__":
    ITERATIONS = 5
    TIME_STEPS = 500
    INTERVALS = 50
    DT = 0.0001
    D = 0.5

    with Profile() as profile:
        for _ in range(ITERATIONS):
            grid = time_dependent_diffusion(TIME_STEPS, INTERVALS, DT, D)
        Stats(profile).strip_dirs().sort_stats(SortKey.TIME).print_stats(25)

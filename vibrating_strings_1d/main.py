# from vibrating_strings_1d.utils.constants import c, dt, dx, TIME_STEPS, SPATIAL_POINTS
from vibrating_strings_1d.utils.animation import animate_wave

if __name__ == "__main__":
    case = 1
    spatial_intervals = 50
    time_steps = 1000
    frame_rate = 5
    animation = animate_wave(spatial_intervals, time_steps, case, frame_rate)
    animation.save("discretized_wave.mp4")

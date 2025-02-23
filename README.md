# Scientific Computing Assignment 1

Repository for assignment 1 code for Scientific Computing.

To contribute to this repository, see the [contributing docs](CONTRIBUTING.md).

To reproduce our results, read on :) 

## Setup

This repository uses _uv_ to manage Python and its dependencies, and contains Rust code exposed as Python functions via language bindings. As such, there are a few required dependencies:
- [Rust](https://www.rust-lang.org/tools/install)
- [uv](https://github.com/astral-sh/uv)

Once these are installed, we need to set up the environment.

Install [maturin](https://www.maturin.rs/), the tool we will use to compile the rust bindings:
```bash
uv tool install maturin
```

Create a Python environment with the correct Python, compile rust bindings, and install Python packages:
```bash
uv sync
```

## Running the code
Experiments are run via a command-line interface (CLI), called `scicomp`. To view all available commands (experiments), run:

```bash
uv run scicomp --help
```

### Vibrating string experiments
Plot the vibrating string at multiple timesteps for each initial condition:

```bash
uv run scicomp string1d plot -m 0 -m 110 -m 220 -m 330 --dt "0.001" -c 1
```

Animate the vibrating string. Case is either low-freq, high-freq, or bounded-high-freq:

```bash
uv run scicomp string1d animate -si 50 -ti 100 -t 1 -s 1 --case [CASE]
```

### 2D time-dependent diffusion experiments
Compare time-dependent diffusion with analytic solution:

```bash
uv run scicomp td-diffusion compare --mode numba
```

Plot diffusion state at multiple points in time:

```bash
uv run scicomp td-diffusion plot-timesteps -m 0 -m "0.001" -m "0.01" -m "0.1" -m "1" -d 1 --dx 0.01 --dt 0.00001 --mode rust
```

Animate diffusion on a cylinder until equilibrium:

```bash
uv run scicomp td-diffusion animate --intervals 50 --dt 0.0001 --mode rust --time-steps 2500
```

Animate diffusion on a cylinder with two rectangular sinks at x=0.7, y in (0.1, 0.3), width=0.2 and height=0.05:

```bash
uv run scicomp td-diffusion animate --intervals 50 --dt 0.0001 --mode numba --time-steps 2500 --sink-rect "0.7 0.1 0.2 0.05" --sink-rect "0.7 0.3 0.2 0.05"
```

Animate diffusion on a cylinder with two rectangular insulators at x=0.7, y in (0.1, 0.3), width=0.2 and height=0.05:

```bash
uv run scicomp td-diffusion animate --intervals 50 --dt 0.0001 --mode numba --time-steps 2500 --ins-rect "0.7 0.1 0.2 0.05" --ins-rect "0.7 0.3 0.2 0.05"
```

> [!IMPORTANT]
> Insulators are only implemented in the Numba version.


### 2D time-independent diffusion experiments
The results from these experiments are generated from a script. Please note that the script takes 30-60min to run.

```bash
uv run src/scientific_computing/stationary_diffusion/main.py
```

## Running tests
To run the test suite:

```bash
uv run pytest
```

## License

[MIT](LICENSE.md)

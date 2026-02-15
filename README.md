# Numerical Methods Visualizer

A professional-grade desktop application built with Python and CustomTkinter for computing and visualizing key numerical methods:

- Newton-Raphson Method
- Regula Falsi Method
- Gauss-Jacobi Method
- Gauss-Seidel Method

The app includes dynamic inputs, convergence tracking, iteration tables, and convergence plotting in a modern dark-themed interface.

## Features

- Single-file executable-ready architecture in `main.py`
- OOP-based numerical engine with shared abstractions
- Safe symbolic parsing via `sympy`
- Automatic symbolic derivative for Newton-Raphson
- Interval validation and sign-checking for Regula Falsi
- Diagonal dominance warnings for iterative linear solvers
- Divergence detection and convergence monitoring
- Precision-controlled output formatting
- Dynamic UI forms by selected method
- Scrollable iteration result display
- Embedded matplotlib convergence graph

## Installation

1. Clone the repository:

```bash
git clone https://github.com/DevInfinix/numerical-methods-visualizer.git
cd numerical-methods-visualizer
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python main.py
```

## Mathematical Methods

### Newton-Raphson

Uses the update:

`x_(n+1) = x_n - f(x_n) / f'(x_n)`

Derivative is computed symbolically via `sympy.diff`.

### Regula Falsi

Uses bracketing and false-position update:

`c = (a f(b) - b f(a)) / (f(b) - f(a))`

Requires `f(a) * f(b) < 0`.

### Gauss-Jacobi

For `Ax = b`, updates all components from previous iteration values:

`x_i^(k+1) = (b_i - sum(a_ij x_j^(k), j != i)) / a_ii`

### Gauss-Seidel

For `Ax = b`, uses immediate in-iteration updates:

`x_i^(k+1) = (b_i - sum(a_ij x_j^(k+1), j < i) - sum(a_ij x_j^(k), j > i)) / a_ii`

## Convergence Discussion

- Methods stop when either iteration error or residual falls below tolerance.
- Divergence safeguards detect non-finite values and persistent error growth.
- Linear methods warn when strict diagonal dominance is not satisfied.
- Output precision is controlled by user-defined decimal precision.

## Architecture Overview

The project was developed modularly (`numerical_methods/`, `ui/`) and consolidated into a single runnable application file:

- `main.py`:
  - Numerical core (base classes + method implementations)
  - Strategy factory for solver selection
  - CustomTkinter UI frames and app router
  - Embedded graph rendering and results table
- `tests/`:
  - Unit tests for core numerical engine behavior and validation logic

## Screenshots

> Add screenshots in `docs/screenshots/` and update paths if needed.

![Landing Screen](docs/screenshots/landing.png)
![Method Selection](docs/screenshots/selection.png)
![Results Screen](docs/screenshots/results.png)

## GIF Demo

> Add demo GIF at `docs/demo/app-demo.gif`.

![App Demo](docs/demo/app-demo.gif)

## Build to Executable

```bash
pyinstaller --onefile --windowed main.py
```

If hidden imports are required:

```bash
pyinstaller --onefile --windowed --hidden-import sympy --hidden-import matplotlib main.py
```

## License

This project is licensed under the MIT License.

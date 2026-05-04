# py_mesa_helper
A basic package for changing Modules for Experiments in Stellar Astrophysics (MESA) inlists, running simulations, and analyzing the results.

## Installation
To install the package, run the following commands in the terminal:
```bash
git clone https://github.com/Henrik-Knierim/py_mesa_helper
cd py_mesa_helper
python setup.py install
```

## Usage
For a quickstarter guide, check out the `examples.ipynb` notebook. A more detailed documentation will follow soon.

### Custom Key Prefixes for Negative Quantities

When working with MESA data, some quantities (like the J2 or J4 moments of inertia) can be negative. The `mesa_reader` library supports the `log_` prefix to automatically compute logarithms (e.g., `log_L` computes log₁₀(L)), but this fails for negative values. 

To handle negative quantities, `mesa_helper` provides two custom key prefixes:

- **`abslog_`**: Computes log₁₀(|x|) for negative values
- **`absln_`**: Computes ln(|x|) for negative values

#### Example Usage

```python
from mesa_helper import Simulation

sim = Simulation("path/to/simulation")

# Access log10 of J4 even though J4 is negative
j4_log = sim.history.data("abslog_J4")

# Use in plotting
fig, ax = sim.history_plot(x="star_age", y="abslog_J4")

# Or in add_history_data
sim.add_history_data("abslog_J4", key_names="log10_J4")
```

These prefixes work seamlessly with all `Simulation` methods that accept keys (e.g., `history_plot()`, `add_history_data()`, `integrate()`, etc.).

## Packaging Notes
- The repository keeps `tests/` for development and validation.
- Release artifacts (source distribution) exclude `tests/` to keep the install footprint small.

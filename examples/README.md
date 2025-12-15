# PyVibrate Examples

This directory contains example circuits demonstrating pyvibrate's capabilities.

## Falstad Circuit Conversions

The `falstad/` directory contains circuits converted from the
[Falstad Circuit Simulator](https://www.falstad.com/circuit/).
See `falstad_compatible_circuits.md` for the complete list of compatible circuits.

### Running Examples

Each example can be run directly:

```bash
python examples/falstad/basic/voltage_divider.py
python examples/falstad/filters/lowpass_rc.py
python examples/falstad/rlc/lrc_resonance.py
```

### Directory Structure

```
examples/
    falstad/
        basic/              # Simple resistor networks
            voltage_divider.py
        filters/            # RC, RL, and active filters
            lowpass_rc.py
        rlc/                # RLC resonance and transients
            lrc_resonance.py
            inductor_kick.py
        opamp/              # Op-amp circuits (using VCVS)
            inverting.py
```

### Example Pattern

Each example follows a consistent structure:

```python
"""
Example: Circuit Name
Converted from Falstad: original_file.txt

Description of what the circuit demonstrates.
"""
from pyvibrate.timedomain import Network, R, C, L, VSource

def build_circuit():
    """Construct the circuit network."""
    ...

def simulate(params=None):
    """Run simulation with given parameters."""
    ...

if __name__ == "__main__":
    # Run and display results
    ...
```

## Requirements

- pyvibrate (this package)
- JAX
- NumPy

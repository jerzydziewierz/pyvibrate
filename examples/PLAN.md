# Falstad Circuit Examples Implementation Plan

## Overview

Convert compatible Falstad circuit simulator examples to pyvibrate implementations.
These serve as both documentation and validation of pyvibrate's capabilities.

## Task 1: Save Compatible Circuits List

**File:** `examples/falstad_compatible_circuits.md`

**Content:**
- List of all 52 compatible Falstad circuits
- Categorized by type (filters, op-amp, RLC, etc.)
- Brief description of each
- Mapping of Falstad element codes to pyvibrate components

## Task 2: Create Examples Folder Structure

```
examples/
    PLAN.md                          # This file
    falstad_compatible_circuits.md   # Reference list
    README.md                        # How to run examples
    falstad/                         # Converted Falstad circuits
        filters/
            lowpass_rc.py            # filt-lopass.txt
            highpass_rc.py           # filt-hipass.txt
            bandpass.py              # bandpass.txt
        opamp/
            inverting.py             # amp-invert.txt
            noninverting.py          # amp-noninvert.txt
            follower.py              # amp-follower.txt
        rlc/
            lrc_resonance.py         # lrc.txt
            inductor_kick.py         # inductkick.txt
        basic/
            voltage_divider.py       # voltdivide.txt
            thevenin.py              # thevenin.txt
```

## Task 3: Select Initial Examples to Implement

Priority criteria:
1. Educational value (demonstrates pyvibrate features)
2. Simplicity (easy to understand)
3. Variety (covers different component types)

**Selected for initial implementation (5 circuits):**

| Priority | Falstad File | PyVibrate Example | Why |
|----------|--------------|-------------------|-----|
| 1 | voltdivide.txt | basic/voltage_divider.py | Simplest possible - just resistors |
| 2 | filt-lopass.txt | filters/lowpass_rc.py | Classic RC filter, shows C behavior |
| 3 | lrc.txt | rlc/lrc_resonance.py | Shows L, C, R together with resonance |
| 4 | amp-invert.txt | opamp/inverting.py | Shows VCVS as op-amp model |
| 5 | inductkick.txt | rlc/inductor_kick.py | Shows switch + inductor transient |

## Task 4: Implementation Pattern

Each example file will follow this pattern:

```python
"""
Example: [Name]
Converted from Falstad circuit: [filename.txt]

[Brief description of what the circuit does]

Components used: R, C, L, VSource, VCVS, etc.
"""
import jax.numpy as jnp
from pyvibrate.timedomain import Network, R, C, L, VSource, ...

def build_circuit():
    """Build the circuit network."""
    net = Network()
    # ... circuit construction
    return net, {node_refs}, {component_refs}

def simulate(params=None):
    """Run simulation with default or custom parameters."""
    net, nodes, components = build_circuit()
    # ... simulation
    return results

def main():
    """Run example and print/plot results."""
    results = simulate()
    # ... output

if __name__ == "__main__":
    main()
```

## Execution Order

1. Create `examples/` folder
2. Write `falstad_compatible_circuits.md` (the full list)
3. Write `examples/README.md` (how to run)
4. Implement `basic/voltage_divider.py` (simplest first)
5. Implement `filters/lowpass_rc.py`
6. Implement `rlc/lrc_resonance.py`
7. Implement `opamp/inverting.py`
8. Implement `rlc/inductor_kick.py`

## Verification

Each example will:
1. Run without errors
2. Produce physically correct results (verified against theory)
3. Demonstrate JAX differentiability where applicable

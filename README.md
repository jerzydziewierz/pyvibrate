# Python SPICE with JAX for equivalent circuit parameter recovery

A JAX-compatible circuit simulator with automatic differentiation support.

Intended for work on piezo transformers, and more widely, for identification of equivalent circuit parameters out of measurable quantities like impedance.

Applications:
* Equivalent circuit parameter recovery (from measurements)
* Control strategy development and numerical optimisation
* In-depth understanding of effects in circuits

## Package Structure

```
pyvibrate/
    timedomain/        # Time-domain transient simulation (MNA + trapezoidal)
        R, C, L        # Passive components
        VSource        # Voltage source
        Switch         # Controllable switch
        VoltageSwitch  # Voltage-controlled switch
        VCVS           # Voltage-controlled voltage source
        VCR            # Voltage-controlled resistor
        DelayLine      # Pure voltage delay
        HBridge        # H-bridge subcircuit
    frequencydomain/   # Steady-state AC analysis with complex phasors
        R, C, L        # Passive components
        ACSource       # AC voltage source with phase
        PhaseShift     # Pure phase delay element
        VCVS           # Voltage-controlled voltage source
        TLine          # Transmission line (Z0, tau)
```

## Time-Domain Simulation

```python
from pyvibrate.timedomain import Network, R, C, VSource

# Build circuit (functional style)
net = Network()
net, n1 = net.node("n1")
net, n2 = net.node("n2")

net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)
net, r1 = R(net, n1, n2, name="R1", value=1000.0)
net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)

# Compile and simulate
sim = net.compile(dt=1e-6)
state = sim.init({})

for _ in range(1000):
    state = sim.step({}, state, {})
    v_out = sim.v(state, n2)
```

## Subcircuits and Building Blocks

PyVibrate supports reusable subcircuits that don't require ground connection. These floating building blocks can be composed to create complex circuits.

### Series and Parallel Operations

The `Series` and `Parallel` functions accept arbitrary two-port elements via factory functions:

```python
from pyvibrate.timedomain import Network, R, C, L
from pyvibrate.timedomain.subcircuits import Series, Parallel

net = Network()
net, n_in = net.node("in")
net, n_out = net.node("out")

# Series RC low-pass filter (floating, no ground reference yet)
net, (r_ref, c_ref, rc_mid) = Series(
    net, n_in, n_out,
    lambda net, a, b: R(net, a, b, name="r1", value=1000.0),
    lambda net, a, b: C(net, a, b, name="c1", value=1e-6),
    prefix="rc_lpf"
)

# Parallel RL impedance
net, (r_ref, l_ref) = Parallel(
    net, n_in, n_out,
    lambda net, a, b: R(net, a, b, name="r2", value=100.0),
    lambda net, a, b: L(net, a, b, name="l1", value=1e-3),
    prefix="rl_par"
)

# Complete circuit by connecting to voltage source and ground
net, vs = VSource(net, n_in, net.gnd, name="vs", value=5.0)
net, load = R(net, n_out, net.gnd, name="load", value=1000.0)
```

### Nesting Subcircuits

Subcircuits can be nested to create complex structures:

```python
# Nested series-parallel combination
# Structure: n1 ──[R1-R2 series]──[C1||R3 parallel]── n2
net, nested = Series(
    net, n1, n2,
    # First element: R1 and R2 in series
    lambda net, a, b: Series(
        net, a, b,
        lambda n, x, y: R(n, x, y, name="r1", value=100.0),
        lambda n, x, y: R(n, x, y, name="r2", value=200.0),
        prefix="r_ser"
    ),
    # Second element: C1 and R3 in parallel
    lambda net, a, b: Parallel(
        net, a, b,
        lambda n, x, y: C(n, x, y, name="c1", value=1e-6),
        lambda n, x, y: R(n, x, y, name="r3", value=1000.0),
        prefix="rc_par"
    ),
    prefix="nested"
)
```

### Frequency-Domain Subcircuits

The same operations work in frequency domain for impedance analysis:

```python
from pyvibrate.frequencydomain import Network, R, C, L, ACSource, Series, Parallel
import math

net = Network()
net, n_in = net.node("in")
net, n_out = net.node("out")

# Series RLC resonator (floating)
net, (r_ref, lc_refs) = Series(
    net, n_in, n_out,
    lambda net, a, b: R(net, a, b, name="r_esr", value=10.0),
    lambda net, a, b: Parallel(
        net, a, b,
        lambda n, x, y: L(n, x, y, name="l1", value=1e-3),
        lambda n, x, y: C(n, x, y, name="c1", value=1e-6),
        prefix="lc_tank"
    ),
    prefix="rlc_res"
)

# Complete circuit
net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0, phase=0.0)
net, r_load = R(net, n_out, net.gnd, name="load", value=50.0)

# Frequency sweep to find resonance
solver = net.compile()
freqs = [100, 1000, 5033, 10000]  # 5033 Hz ≈ 1/(2π√LC)
for f in freqs:
    sol = solver.solve_at(omega=2*math.pi*f)
    z_in = solver.z_in(sol, vs)
    print(f"{f} Hz: |Z| = {abs(z_in):.1f} Ω, ∠{math.degrees(math.atan2(z_in.imag, z_in.real)):.1f}°")
```

### Custom Subcircuits

You can create your own subcircuits following the pattern in `pyvibrate/timedomain/subcircuits.py` or `pyvibrate/frequencydomain/subcircuits.py`:

```python
def MySubcircuit(net, terminal_a, terminal_b, prefix="mysub"):
    """Create custom subcircuit between two terminals."""
    # Create internal nodes if needed
    net, n_internal = net.node(f"{prefix}_int")

    # Add components
    net, r1 = R(net, terminal_a, n_internal, name=f"{prefix}_r1", value=100.0)
    net, c1 = C(net, n_internal, terminal_b, name=f"{prefix}_c1", value=1e-6)

    # Return network and references
    return net, (r1, c1, n_internal)
```

## Features

### Time-Domain (`pyvibrate.timedomain`)
- Step-by-step simulation with Modified Nodal Analysis (MNA)
- Trapezoidal integration for reactive components
- JAX-compatible for autodiff optimization (sensitivity analysis, gradient descent)
- Functional/immutable API

### Frequency-Domain (`pyvibrate.frequencydomain`)
- Steady-state AC analysis at a single frequency
- Complex phasors for voltages and currents
- Components: R, C, L, ACSource, PhaseShift, VCVS, TLine
- JAX-compatible for autodiff optimization (sensitivity analysis, parameter fitting)
- Target: VNA measurement fitting, piezo transducer modeling

## Frequency-Domain Example

```python
from pyvibrate.frequencydomain import Network, R, C, ACSource
import math

# Build RC low-pass filter
net = Network()
net, n1 = net.node("n1")

net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
net, r1 = R(net, n1, net.gnd, name="R1", value=100.0)
net, c1 = C(net, n1, net.gnd, name="C1", value=1e-6)

# Solve at 1 kHz
solver = net.compile()
omega = 2 * math.pi * 1000.0
sol = solver.solve_at(omega)

# Get input impedance
z_in = solver.z_in(sol, vs)
print(f"|Z| = {abs(z_in):.1f} ohm, phase = {math.degrees(phase(z_in)):.1f} deg")
```

## Status

- Time-domain: 74 tests passing
- Frequency-domain: 36 tests passing
- Total: 110 tests passing

See AGENTS.md for development roadmap.

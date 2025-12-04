# Python SPICE with JAX for equivalent circuit parameter recovery

A JAX-compatible circuit simulator with automatic differentiation support.

Intended for work on piezo transformers.

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

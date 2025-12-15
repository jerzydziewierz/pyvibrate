# PyVibrate Frequency-Domain Usage Guide

A JAX-compatible frequency-domain circuit analyzer with automatic differentiation support.

This guide covers the **frequency-domain** module (`pyvibrate.frequencydomain`).
For time-domain transient analysis, see [USAGE_time_domain.md](USAGE_time_domain.md).

## Quick Start

```python
from pyvibrate.frequencydomain import Network, R, C, ACSource
import jax.numpy as jnp

# Build RC low-pass filter: Vs -- R1 -- C1 -- GND
net = Network()
net, n1 = net.node("n1")
net, n2 = net.node("n2")

net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)  # 1V AC source
net, r1 = R(net, n1, n2, name="R1", value=1000.0)           # 1 kOhm
net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)        # 1 uF

# Compile and solve at 1 kHz
solver = net.compile()
omega = 2 * jnp.pi * 1000.0  # angular frequency in rad/s
sol = solver.solve_at(omega)

# Probe results
v_out = solver.v(sol, n2)            # complex voltage phasor at output
z_in = solver.z_in(sol, vs)          # input impedance seen by source
print(f"Output voltage: {jnp.abs(v_out):.3f} V at {jnp.angle(v_out)*180/jnp.pi:.1f}°")
print(f"Input impedance: {z_in.real:.1f} + j{z_in.imag:.1f} Ohm")
```

## Component Values

PyVibrate supports a hybrid API for component values:

### At Construction Time (Defaults)

```python
# Values provided at construction become defaults
net, r1 = R(net, n1, n2, name="R1", value=1000.0)   # Default: 1 kOhm
net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6) # Default: 1 uF

# Run without params - uses defaults
sol = solver.solve_at(omega)
```

### At Solve Time (Override)

```python
# Override defaults via params dict
params = {
    "R1": 2000.0,   # Override R1 to 2 kOhm
    "C1": 2e-6,     # Override C1 to 2 uF
}
sol = solver.solve_at(omega, params)
```

### Mixed Approach

```python
# Some values at construction, others at runtime
net, r1 = R(net, n1, n2, name="R1")                 # No default - MUST provide
net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6) # Has default

params = {"R1": 1000.0}  # Required
sol = solver.solve_at(omega, params)
```

## Available Components

| Component | Function | Parameters | Impedance |
|-----------|----------|------------|-----------|
| Resistor | `R(net, a, b, name="R1", value=1000.0)` | Resistance in Ohms | Z = R (real) |
| Capacitor | `C(net, a, b, name="C1", value=1e-6)` | Capacitance in Farads | Z = -j/(ωC) |
| Inductor | `L(net, a, b, name="L1", value=1e-3)` | Inductance in Henrys | Z = jωL |
| AC Source | `ACSource(net, p, n, name="vs", value=1.0, phase=0.0)` | Magnitude (V), Phase (rad) | - |
| VCVS | `VCVS(net, out_p, out_n, ctrl_p, ctrl_n, name="E1", gain=10.0)` | Voltage gain | - |
| Phase Shift | `PhaseShift(net, in_p, in_n, out_p, out_n, name="PS1", tau=1e-9)` | Delay (s) | V_out = V_in·e^(-jωτ) |
| Transmission Line | `TLine(net, p1_p, p1_n, p2_p, p2_n, name="TL1", Z0=50.0, tau=5e-9)` | Z0 (Ohm), τ (s) | Y-parameter model |

### Component Behavior by Frequency

| Component | Low Frequency (DC) | High Frequency |
|-----------|-------------------|----------------|
| R | Z = R | Z = R |
| C | |Z| → ∞ (open circuit) | |Z| → 0 (short circuit) |
| L | |Z| → 0 (short circuit) | |Z| → ∞ (open circuit) |

## Probing Values

```python
# Probe node voltage (complex phasor)
v_node = solver.v(sol, n2)
magnitude = jnp.abs(v_node)
phase_deg = jnp.angle(v_node) * 180 / jnp.pi

# Probe source current (for ACSource, VCVS, PhaseShift)
i_src = solver.i(sol, vs)

# Probe input impedance at a source
z_in = solver.z_in(sol, vs)
```

## Subcircuits: Series and Parallel

Build complex impedance networks using functional composition:

### Series Connection

```python
from pyvibrate.frequencydomain import Network, R, C, ACSource
from pyvibrate.frequencydomain.subcircuits import Series

net = Network()
net, n_in = net.node("in")

# Series RC: in -- R -- (mid) -- C -- gnd
net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
net, (r_ref, c_ref, mid) = Series(
    net, n_in, net.gnd,
    lambda net, a, b: R(net, a, b, name="r1", value=1000.0),
    lambda net, a, b: C(net, a, b, name="c1", value=1e-6),
    prefix="rc"
)

solver = net.compile()
sol = solver.solve_at(2 * jnp.pi * 1000.0)

# Access the midpoint voltage
v_mid = solver.v(sol, mid)
```

### Parallel Connection

```python
from pyvibrate.frequencydomain.subcircuits import Parallel

# Parallel RC: both R and C between same nodes
net, (r_ref, c_ref) = Parallel(
    net, n_in, net.gnd,
    lambda net, a, b: R(net, a, b, name="r1", value=1000.0),
    lambda net, a, b: C(net, a, b, name="c1", value=1e-6),
    prefix="par"
)
```

### Nested Compositions

```python
# Series RLC resonator
def rl_series(net, a, b):
    return Series(
        net, a, b,
        lambda net, a, b: R(net, a, b, name="r1", value=100.0),
        lambda net, a, b: L(net, a, b, name="l1", value=10e-3),
        prefix="rl"
    )

net, (rl_refs, c_ref, mid) = Series(
    net, n_in, net.gnd,
    rl_series,
    lambda net, a, b: C(net, a, b, name="c1", value=1e-6),
    prefix="rlc"
)
```

## Frequency Sweep

```python
import jax.numpy as jnp

# Define frequency range (logarithmic)
freqs = jnp.logspace(1, 5, 100)  # 10 Hz to 100 kHz
omegas = 2 * jnp.pi * freqs

# Sweep and collect impedance
z_values = []
for omega in omegas:
    sol = solver.solve_at(omega)
    z = solver.z_in(sol, vs)
    z_values.append(z)

z_array = jnp.array(z_values)

# Plot Bode diagram
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.semilogx(freqs, 20 * jnp.log10(jnp.abs(z_array)))
ax1.set_ylabel('|Z| (dB)')
ax2.semilogx(freqs, jnp.angle(z_array) * 180 / jnp.pi)
ax2.set_ylabel('Phase (deg)')
ax2.set_xlabel('Frequency (Hz)')
```

## JAX Autodiff: Sensitivity Analysis

Compute gradients of impedance with respect to component values:

```python
import jax.numpy as jnp
from jax import grad

# Build circuit
net = Network()
net, n1 = net.node("n1")
net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
net, r1 = R(net, n1, net.gnd, name="R1")  # No default
solver = net.compile()

omega = 2 * jnp.pi * 1000.0  # 1 kHz

def get_impedance_magnitude(R_val):
    sol = solver.solve_at(omega, {"R1": R_val})
    z = solver.z_in(sol, vs)
    return jnp.abs(z)

# Compute dZ/dR
R_val = 100.0
dZ_dR = grad(get_impedance_magnitude)(R_val)
print(f"dZ/dR = {dZ_dR:.4f}")  # For single R: dZ/dR = 1.0
```

### Multi-Parameter Sensitivity

```python
from jax import grad

# RC circuit sensitivity
def impedance_magnitude(R_val, C_val):
    sol = solver.solve_at(omega, {"R1": R_val, "C1": C_val})
    z = solver.z_in(sol, vs)
    return jnp.abs(z)

# Partial derivatives
dZ_dR = grad(impedance_magnitude, argnums=0)(1000.0, 1e-6)
dZ_dC = grad(impedance_magnitude, argnums=1)(1000.0, 1e-6)

print(f"dZ/dR = {dZ_dR:.6f}")
print(f"dZ/dC = {dZ_dC:.2e}")
```

### Sensitivity at Multiple Frequencies

```python
from jax import vmap, grad

def impedance_at_freq(omega, R_val, C_val):
    sol = solver.solve_at(omega, {"R1": R_val, "C1": C_val})
    z = solver.z_in(sol, vs)
    return jnp.abs(z)

# Gradient w.r.t. R at multiple frequencies
dZ_dR_fn = grad(impedance_at_freq, argnums=1)

omegas = 2 * jnp.pi * jnp.array([100.0, 1000.0, 10000.0])
for omega in omegas:
    sensitivity = dZ_dR_fn(omega, 1000.0, 1e-6)
    print(f"f={omega/(2*jnp.pi):.0f} Hz: dZ/dR = {sensitivity:.4f}")
```

## Equivalent Circuit Identification

Use gradient descent to fit an equivalent circuit model to measured impedance data.

### Example: Fitting an RC Circuit

```python
import jax.numpy as jnp
from jax import grad, jit
from pyvibrate.frequencydomain import Network, R, C, ACSource

# Simulated "measured" impedance data (target: R=500 ohm, C=2.2 uF)
target_R = 500.0
target_C = 2.2e-6
freqs = jnp.logspace(1, 4, 20)
omegas = 2 * jnp.pi * freqs

# Generate "measured" data
def true_rc_impedance(omega, R, C):
    z_r = R
    z_c = 1 / (1j * omega * C)
    return z_r + z_c  # Series RC

z_measured = jnp.array([true_rc_impedance(w, target_R, target_C) for w in omegas])

# Build model circuit
net = Network()
net, n1 = net.node("n1")
net, n2 = net.node("n2")
net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
net, r1 = R(net, n1, n2, name="R1")
net, c1 = C(net, n2, net.gnd, name="C1")
solver = net.compile()

def model_impedance(R_val, C_val, omega):
    sol = solver.solve_at(omega, {"R1": R_val, "C1": C_val})
    return solver.z_in(sol, vs)

# Loss function: sum of squared errors
def loss(R_val, C_val):
    total = 0.0
    for i, omega in enumerate(omegas):
        z_model = model_impedance(R_val, C_val, omega)
        z_target = z_measured[i]
        # Compare real and imaginary parts
        total += (z_model.real - z_target.real)**2
        total += (z_model.imag - z_target.imag)**2
    return total

grad_loss = jit(grad(loss, argnums=(0, 1)))

# Gradient descent
R_est = 1000.0  # Initial guess
C_est = 1e-6    # Initial guess
lr_R = 1e-1
lr_C = 1e-15    # Different learning rates for different scales

for iteration in range(200):
    dL_dR, dL_dC = grad_loss(R_est, C_est)
    R_est = R_est - lr_R * dL_dR
    C_est = C_est - lr_C * dL_dC

    if iteration % 50 == 0:
        L = loss(R_est, C_est)
        print(f"Iter {iteration}: R={R_est:.1f}, C={C_est*1e6:.3f} uF, Loss={L:.2e}")

print(f"\nFinal: R={R_est:.1f} ohm (target: {target_R})")
print(f"Final: C={C_est*1e6:.3f} uF (target: {target_C*1e6:.3f})")
```

### Advanced: Using Optax for Optimization

```python
import optax
import jax.numpy as jnp
from jax import jit, value_and_grad

# Parameter vector [log(R), log(C)] for better conditioning
def loss_fn(log_params):
    R_val = jnp.exp(log_params[0])
    C_val = jnp.exp(log_params[1])

    total = 0.0
    for i, omega in enumerate(omegas):
        z_model = model_impedance(R_val, C_val, omega)
        z_target = z_measured[i]
        total += jnp.abs(z_model - z_target)**2
    return total

# Adam optimizer
optimizer = optax.adam(learning_rate=0.1)
log_params = jnp.array([jnp.log(1000.0), jnp.log(1e-6)])
opt_state = optimizer.init(log_params)

@jit
def step(log_params, opt_state):
    loss, grads = value_and_grad(loss_fn)(log_params)
    updates, opt_state = optimizer.update(grads, opt_state)
    log_params = optax.apply_updates(log_params, updates)
    return log_params, opt_state, loss

for i in range(500):
    log_params, opt_state, loss = step(log_params, opt_state)
    if i % 100 == 0:
        R_est = jnp.exp(log_params[0])
        C_est = jnp.exp(log_params[1])
        print(f"Iter {i}: R={R_est:.1f}, C={C_est*1e6:.3f} uF, Loss={loss:.2e}")
```

## Example: Series RLC Resonance Analysis

```python
from pyvibrate.frequencydomain import Network, R, L, C, ACSource
import jax.numpy as jnp

# Build series RLC
net = Network()
net, n1 = net.node("n1")
net, n2 = net.node("n2")
net, n3 = net.node("n3")

net, vs = ACSource(net, n1, net.gnd, name="vs", value=1.0)
net, r1 = R(net, n1, n2, name="R1", value=10.0)     # 10 ohm
net, l1 = L(net, n2, n3, name="L1", value=1e-3)     # 1 mH
net, c1 = C(net, n3, net.gnd, name="C1", value=1e-6) # 1 uF

solver = net.compile()

# Resonant frequency: f_0 = 1/(2*pi*sqrt(L*C))
L_val, C_val = 1e-3, 1e-6
f_0 = 1 / (2 * jnp.pi * jnp.sqrt(L_val * C_val))
print(f"Resonant frequency: {f_0:.1f} Hz")

# Quality factor: Q = (1/R) * sqrt(L/C)
R_val = 10.0
Q = (1/R_val) * jnp.sqrt(L_val / C_val)
print(f"Quality factor: Q = {Q:.1f}")

# Impedance at resonance (should equal R)
omega_0 = 2 * jnp.pi * f_0
sol = solver.solve_at(omega_0)
z_res = solver.z_in(sol, vs)
print(f"Z at resonance: {z_res.real:.2f} + j{z_res.imag:.2f} ohm")
```

## Example: Parallel LC Tank (Anti-Resonance)

```python
from pyvibrate.frequencydomain import Network, L, C, R, ACSource
from pyvibrate.frequencydomain.subcircuits import Parallel
import jax.numpy as jnp

net = Network()
net, n_in = net.node("in")

# Add small series resistance for source
net, n_src = net.node("src")
net, vs = ACSource(net, n_src, net.gnd, name="vs", value=1.0)
net, r_src = R(net, n_src, n_in, name="Rs", value=1.0)

# Parallel LC tank
net, (l_ref, c_ref) = Parallel(
    net, n_in, net.gnd,
    lambda net, a, b: L(net, a, b, name="L1", value=10e-3),
    lambda net, a, b: C(net, a, b, name="C1", value=1e-6),
    prefix="tank"
)

solver = net.compile()

# At resonance, parallel LC has maximum impedance
f_0 = 1 / (2 * jnp.pi * jnp.sqrt(10e-3 * 1e-6))
omega_0 = 2 * jnp.pi * f_0

# Compare on-resonance vs off-resonance
sol_res = solver.solve_at(omega_0 * 0.99)  # slightly off to avoid singularity
sol_off = solver.solve_at(omega_0 * 0.5)

z_res = solver.z_in(sol_res, vs)
z_off = solver.z_in(sol_off, vs)

print(f"Near resonance |Z|: {jnp.abs(z_res):.0f} ohm")
print(f"Off resonance |Z|: {jnp.abs(z_off):.1f} ohm")
```

## Example: Transmission Line Effects

```python
from pyvibrate.frequencydomain import Network, R, TLine, ACSource
import jax.numpy as jnp

# 50 ohm source driving 50 ohm line into 50 ohm load (matched)
net = Network()
net, n_src = net.node("src")
net, n_in = net.node("in")
net, n_out = net.node("out")

net, vs = ACSource(net, n_src, net.gnd, name="vs", value=1.0)
net, r_src = R(net, n_src, n_in, name="Rs", value=50.0)  # Source impedance
net, tl1 = TLine(net, n_in, net.gnd, n_out, net.gnd,
                  name="TL1", Z0=50.0, tau=1e-9)  # 1 ns delay
net, r_load = R(net, n_out, net.gnd, name="Rl", value=50.0)  # Load

solver = net.compile()

# At matched conditions, input impedance = Z0
omega = 2 * jnp.pi * 100e6  # 100 MHz
sol = solver.solve_at(omega)
z_in = solver.z_in(sol, vs)

print(f"Input Z at 100 MHz: {z_in.real:.1f} + j{z_in.imag:.1f} ohm")
```

## Performance Tips

1. **Use defaults for static values**: Compile once, solve many times with different frequencies.

2. **JAX JIT compilation**: The solver is JIT-compiled internally. First call compiles, subsequent calls are fast.

3. **Logarithmic frequency spacing**: For Bode plots, use `jnp.logspace()` rather than `jnp.linspace()`.

4. **Parameter scaling in optimization**: Use log-scale parameters for values spanning orders of magnitude.

5. **Vectorization**: While the solver operates at single frequencies, you can parallelize frequency sweeps:
   ```python
   from jax import vmap
   # Define a function for single omega, then vmap over frequency array
   ```

## Physical Formulas Reference

| Circuit | Impedance | Resonance |
|---------|-----------|-----------|
| Series RC | Z = R - j/(ωC) | - |
| Series RL | Z = R + jωL | - |
| Parallel RC | Z = R / (1 + jωRC) | - |
| Parallel RL | Z = jωLR / (R + jωL) | - |
| Series RLC | Z = R + j(ωL - 1/ωC) | f₀ = 1/(2π√LC) |
| Parallel LC | Z = jωL / (1 - ω²LC) | f₀ = 1/(2π√LC) |

**Capacitor phase**: Current leads voltage by 90° (Z has -90° phase)
**Inductor phase**: Voltage leads current by 90° (Z has +90° phase)
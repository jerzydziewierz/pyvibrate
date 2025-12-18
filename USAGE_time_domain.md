# PyVibrate Time-Domain Usage Guide

A JAX-compatible circuit simulator with automatic differentiation support.

This guide covers the **time-domain** module (`pyvibrate.timedomain`).
For frequency-domain analysis, see the frequency-domain guide (planned).

## Quick Start

```python
from pyvibrate.timedomain import Network, R, C, VSource

# Build circuit: Vs -- R1 -- C1 -- GND
net = Network()
net, n1 = net.node("n1")
net, n2 = net.node("n2")

net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)   # 5V source
net, r1 = R(net, n1, n2, name="R1", value=1000.0)           # 1 kOhm
net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)        # 1 uF

# Compile and run
sim = net.compile(dt=1e-6)  # 1 us timestep
state = sim.init({})

for _ in range(1000):  # 1 ms simulation
    state = sim.step({}, state, {})

print(f"Voltage: {sim.v(state, n2):.3f} V")
```

## Component Values

PyVibrate supports a hybrid API for component values:

### At Construction Time (Defaults)

```python
# Values provided at construction become defaults
net, r1 = R(net, n1, n2, name="R1", value=1000.0)   # Default: 1 kOhm
net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6) # Default: 1 uF
net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)  # Default: 5V

# Run without params - uses defaults
state = sim.init({})
state = sim.step({}, state, {})
```

### At Simulation Time (Override)

```python
# Override defaults via params dict
params = {
    "R1": 2000.0,   # Override R1 to 2 kOhm
    "C1": 2e-6,     # Override C1 to 2 uF
}
state = sim.init(params)
state = sim.step(params, state, {})
```

### Mixed Approach

```python
# Some values at construction, others at runtime
net, r1 = R(net, n1, n2, name="R1")                 # No default - MUST provide
net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6) # Has default

params = {"R1": 1000.0}  # Required
state = sim.init(params)
```

### Voltage Sources via Controls

Voltage sources can be controlled dynamically:

```python
net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)  # Default 5V

# Override via controls dict
controls = {"vs": 10.0}  # Apply 10V instead
state = sim.step({}, state, controls)
```

## Available Components

| Component | Function | Parameters | Notes |
|-----------|----------|------------|-------|
| Resistor | `R(net, a, b, name="R1", value=1000.0)` | Resistance in Ohms | |
| Capacitor | `C(net, a, b, name="C1", value=1e-6)` | Capacitance in Farads | |
| Inductor | `L(net, a, b, name="L1", value=1e-3)` | Inductance in Henrys | |
| Voltage Source | `VSource(net, p, n, name="vs", value=5.0)` | Voltage in Volts | Controllable via controls dict |
| VCVS | `VCVS(net, out_p, out_n, ctrl_p, ctrl_n, name="E1")` | Gain via params | Voltage-Controlled Voltage Source |
| Switch | `Switch(net, a, b, name="sw1")` | State via controls (True/False) | r_on, r_off params available |
| Voltage Switch | `VoltageSwitch(net, a, b, ctrl_p, ctrl_n, name="vsw1")` | Voltage-controlled | threshold param |
| Delay Line | `DelayLine(net, in_p, in_n, out_p, out_n, delay_samples=10, name="D1")` | Pure voltage delay | |

## Probing Values

```python
# Probe node voltage
v_node = sim.v(state, n2)

# Probe component current (for VSource, L, VCVS)
i_vs = sim.i(state, vs)  # Current through voltage source
i_L = sim.i(state, l1)   # Current through inductor
```

## JAX Autodiff: Sensitivity Analysis

Compute gradients through the simulation:

```python
import jax.numpy as jnp
from jax import grad

def simulate_and_measure(R_param):
    """Return capacitor voltage after 1ms."""
    params = {"R1": R_param}
    state = sim.init(params)

    for _ in range(1000):
        state = sim.step(params, state, {})

    return sim.v(state, n2)

# Compute dV/dR
dV_dR = grad(simulate_and_measure)(1000.0)
print(f"dV/dR = {dV_dR:.6e} V/Ohm")
```

For an RC circuit with tau = R*C:
- dV/dR is negative (increasing R slows charging)
- dtau/dR = C (analytically)

## JAX Autodiff: Optimization

Use gradient descent to find component values:

```python
from jax import grad

V0 = 5.0
C_val = 1e-6
target_tau = 0.5e-3  # Target time constant

# At t = tau, V = V0 * (1 - 1/e) ~= 3.16V
V_target = V0 * (1.0 - jnp.exp(-1.0))
n_steps = int(target_tau / dt)

def voltage_at_target_time(R_param):
    params = {"R1": R_param}
    state = sim.init(params)
    for _ in range(n_steps):
        state = sim.step(params, state, {})
    return sim.v(state, n2)

def loss(R_param):
    v = voltage_at_target_time(R_param)
    return (v - V_target) ** 2

grad_loss = grad(loss)

# Gradient descent
R_current = 1000.0
learning_rate = 3e4

for _ in range(15):
    g = grad_loss(R_current)
    R_current = R_current - learning_rate * g
    if loss(R_current) < 1e-6:
        break

print(f"Optimized R = {R_current:.1f} Ohm")
# Expected: R = tau / C = 0.5e-3 / 1e-6 = 500 Ohm
```

## Performance Tips

1. **Fixed step counts**: JAX traces through Python control flow. Use fixed loop bounds for differentiable simulations.

2. **JIT compilation**: The simulator uses `@jax.jit` internally. First call compiles, subsequent calls are fast.

3. **Batch parameters**: Keep params dict structure consistent across calls.

## Example: RC Time Constant Measurement

```python
from pyvibrate.timedomain import Network, R, C, VSource
import jax.numpy as jnp

# Build RC circuit
net = Network()
net, n1 = net.node("n1")
net, n2 = net.node("n2")

net, vs = VSource(net, n1, net.gnd, name="vs", value=5.0)
net, r1 = R(net, n1, n2, name="R1", value=1000.0)
net, c1 = C(net, n2, net.gnd, name="C1", value=1e-6)

dt = 1e-6
sim = net.compile(dt=dt)

# Simulate and measure tau by threshold crossing
def measure_tau(R_param):
    V0 = 5.0
    V_target = V0 * (1 - jnp.exp(-1.0))  # 63.2% of V0

    params = {"R1": R_param}
    state = sim.init(params)

    v_prev = 0.0
    t_prev = 0.0

    for _ in range(5000):
        state = sim.step(params, state, {})
        v_curr = sim.v(state, n2)
        t_curr = float(state.time)

        if v_prev < V_target <= v_curr:
            # Linear interpolation
            frac = (V_target - v_prev) / (v_curr - v_prev + 1e-12)
            return t_prev + frac * dt

        v_prev = v_curr
        t_prev = t_curr

    return t_curr

tau = measure_tau(1000.0)
print(f"tau = {tau*1e3:.3f} ms (expected: 1.0 ms)")
```

## Circuit Examples

### RL Circuit (Inductor Current Buildup)

```python
net = Network()
net, n1 = net.node("n1")
net, n2 = net.node("n2")

net, vs = VSource(net, n1, net.gnd, name="vs", value=10.0)
net, r1 = R(net, n1, n2, name="R1", value=100.0)
net, l1 = L(net, n2, net.gnd, name="L1", value=0.1)

sim = net.compile(dt=1e-6)
state = sim.init({})

# Current builds up exponentially: I(t) = (V/R) * (1 - exp(-t*R/L))
for _ in range(10000):
    state = sim.step({}, state, {})

i_final = sim.i(state, l1)
# Expected: 10V / 100 Ohm = 0.1 A
```

### H-Bridge circuit Control

```python
net = Network()
net, vcc = net.node("vcc")
net, out_a = net.node("out_a")
net, out_b = net.node("out_b")

net, vs = VSource(net, vcc, net.gnd, name="vs", value=12.0)

# High-side switches
net, sw_ah = Switch(net, vcc, out_a, name="sw_ah")
net, sw_bh = Switch(net, vcc, out_b, name="sw_bh")

# Low-side switches
net, sw_al = Switch(net, out_a, net.gnd, name="sw_al")
net, sw_bl = Switch(net, out_b, net.gnd, name="sw_bl")

# Motor (modeled as R-L)
net, r_motor = R(net, out_a, out_b, name="R_motor", value=10.0)

sim = net.compile(dt=1e-6)

# Drive forward: A high, B low
controls = {
    "sw_ah": True,  "sw_al": False,
    "sw_bh": False, "sw_bl": True,
}
state = sim.init({})
state = sim.step({}, state, controls)

v_motor = sim.v(state, out_a) - sim.v(state, out_b)  # ~12V
```

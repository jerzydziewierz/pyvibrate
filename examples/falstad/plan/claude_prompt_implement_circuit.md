# Circuit Implementation Task

**Working directory:** You are in the pyvibrate project root. All paths below are relative to this root.

## Target Circuit

**Falstad source file:** `{{FALSTAD_FILE}}`

### Circuit Contents (from Falstad)

```
{{FALSTAD_CONTENTS}}
```

### Falstad Format Reference

Each line is: `element_code x1 y1 x2 y2 flags value1 value2 ...`
- Coordinates (x1,y1)-(x2,y2) define component placement
- Elements sharing coordinates are connected
- `$` line: simulation parameters (ignore)
- `o`/`O` lines: oscilloscope probes (ignore)
- `w` lines: wires connecting points

---

## Project Context

You are implementing a Falstad circuit example for the **pyvibrate** library - a JAX-based differentiable circuit simulator.

### Project Structure

```
./  (pyvibrate root - your working directory)
├── pyvibrate/
│   └── timedomain/           # Core simulation engine
│       ├── __init__.py       # Exports: Network, R, L, C, VSource, Switch, VCVS
│       └── ...
├── examples/
│   └── falstad/              # Falstad circuit implementations
│       ├── basic/            # Basic circuits (voltage divider, etc.)
│       ├── filters/          # Filter circuits (lowpass, highpass, etc.)
│       ├── rlc/              # RLC circuits (resonance, inductor kick, etc.)
│       ├── opamp/            # Op-amp circuits (inverting, etc.)
│       └── plan/             # Planning docs and scripts
```

### Available Components

| Component | Import | Description |
|-----------|--------|-------------|
| `R` | `from pyvibrate.timedomain import R` | Resistor |
| `L` | `from pyvibrate.timedomain import L` | Inductor |
| `C` | `from pyvibrate.timedomain import C` | Capacitor |
| `VSource` | `from pyvibrate.timedomain import VSource` | Voltage source |
| `Switch` | `from pyvibrate.timedomain import Switch` | SPST switch |
| `VCVS` | `from pyvibrate.timedomain import VCVS` | Voltage-controlled voltage source (for op-amps) |
| `Network` | `from pyvibrate.timedomain import Network` | Circuit network container |

### Falstad Element Code Mapping

| Falstad Code | PyVibrate | Notes |
|--------------|-----------|-------|
| `r` | `R` | Resistor |
| `c` | `C` | Capacitor |
| `l` | `L` | Inductor |
| `v`, `R` | `VSource` | Voltage source (DC or AC waveform) |
| `170` | `VSource` | AC sine source |
| `a` | `VCVS` | Op-amp (model as VCVS with high gain ~1000-100000) |
| `s`, `S` | `Switch` | SPST/SPDT switch |
| `w` | - | Wire (just connectivity, use same node) |
| `g` | `net.gnd` | Ground reference |
| `174` | `VCVS` | Voltage-controlled voltage source |

---

## Implementation Requirements

### 1. Analyze the Circuit

The Falstad circuit contents are provided above. Analyze them to understand:
- Circuit topology (elements sharing coordinates are connected)
- Component values (in the element parameters)
- Key characteristics (filter type, resonant frequency, etc.)

### 2. Create the Python Implementation

Create a new Python file in the appropriate subdirectory under `examples/falstad/`.

**File naming:** Use descriptive snake_case, e.g., `bandpass_filter.py`, `rc_highpass.py`

**Choose subdirectory based on circuit type:**
- `basic/` - Simple resistor networks, voltage dividers
- `filters/` - RC, RL, RLC filters
- `rlc/` - RLC resonance, damping, transient circuits
- `opamp/` - Op-amp based circuits

### 3. Follow This Code Template

```python
"""
Example: [Circuit Name]
Converted from Falstad: {{FALSTAD_FILE}}

[Brief description of what this circuit demonstrates]

[Key equations/formulas if applicable]

Components used: [list components]
"""
import math
import jax.numpy as jnp
from jax import grad
from pyvibrate.timedomain import Network, R, L, C, VSource, Switch, VCVS


def build_circuit():
    """Build the circuit.

    Circuit diagram (ASCII art):
        [Draw the circuit topology]
    """
    net = Network()

    # Create nodes
    net, n1 = net.node("n1")
    net, n2 = net.node("n2")
    # ... more nodes as needed

    # Add components
    net, vs = VSource(net, n1, net.gnd, name="vs")
    net, r1 = R(net, n1, n2, name="R1")
    # ... more components

    nodes = {"n1": n1, "n2": n2}
    components = {"vs": vs, "R1": r1}
    return net, nodes, components


def simulate_step_response(param1=default1, param2=default2, ...):
    """Simulate step response.

    Args:
        param1: Description
        param2: Description

    Returns:
        times: List of time values
        outputs: List of output values
        analysis: Dict with derived parameters
    """
    net, nodes, components = build_circuit()

    # Calculate time constants and simulation parameters
    tau = ...  # characteristic time
    dt = tau / 100  # ~100 samples per time constant
    n_steps = int(5 * tau / dt)  # simulate for 5 time constants

    # Compile and initialize
    sim = net.compile(dt=dt)
    params = {"vs": 0.0, "R1": param1, ...}  # component values
    state = sim.init(params)

    # Run simulation
    times = []
    outputs = []

    for i in range(n_steps):
        t = float(state.time)

        # Define control inputs (voltage sources, switches)
        controls = {"vs": 5.0}  # or time-varying: V_amp * math.sin(omega * t)

        state = sim.step(params, state, controls)

        times.append(t)
        outputs.append(float(sim.v(state, nodes["output_node"])))

    return times, outputs, {"tau": tau, ...}


def simulate_frequency_response(param1=default1, frequencies=None):
    """Simulate AC frequency response (Bode plot data).

    Returns magnitude and phase at each frequency.
    """
    if frequencies is None:
        f_c = ...  # cutoff/center frequency
        frequencies = [f_c * mult for mult in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]]

    net, nodes, components = build_circuit()
    results = []

    for freq in frequencies:
        period = 1.0 / freq
        dt = period / 40  # 40 samples per period
        n_cycles = 10  # wait for steady state
        n_steps = int(n_cycles * period / dt)

        sim = net.compile(dt=dt)
        params = {...}
        state = sim.init(params)

        V_amp = 1.0
        omega = 2 * math.pi * freq

        v_out_last = []

        for i in range(n_steps):
            t = float(state.time)
            v_in = V_amp * math.sin(omega * t)
            state = sim.step(params, state, {"vs": v_in})

            # Collect last 2 cycles
            if i >= n_steps - int(2 * period / dt):
                v_out_last.append(float(sim.v(state, nodes["out"])))

        # Calculate magnitude
        v_out_amp = (max(v_out_last) - min(v_out_last)) / 2
        magnitude = v_out_amp / V_amp
        magnitude_db = 20 * math.log10(magnitude) if magnitude > 0 else -100

        results.append({
            "freq": freq,
            "magnitude": magnitude,
            "magnitude_db": magnitude_db,
        })

    return results


def demo_differentiability(base_param=default_value):
    """Demonstrate JAX differentiability.

    Compute gradient of some output metric with respect to a component parameter.
    """
    def objective(param_value):
        """Output metric as function of parameter."""
        net, nodes, components = build_circuit()

        # Use FIXED simulation parameters (dt, n_steps must not depend on param_value)
        dt = 1e-5  # fixed
        n_steps = 500  # fixed

        sim = net.compile(dt=dt)
        params = {"vs": 0.0, "R1": param_value, ...}
        state = sim.init(params)

        for _ in range(n_steps):
            state = sim.step(params, state, {"vs": 5.0})

        return sim.v(state, nodes["out"])

    # Compute gradient
    grad_fn = grad(objective)
    gradient = float(grad_fn(base_param))

    return gradient


def parameter_sweep(param_name, param_values, **fixed_params):
    """Sweep a parameter and collect results.

    Args:
        param_name: Name of parameter to sweep
        param_values: List of values to test
        fixed_params: Other parameters held constant

    Returns:
        List of dicts with param value and results
    """
    results = []

    for val in param_values:
        # Run simulation with this parameter value
        # Collect relevant metrics
        results.append({
            "param_value": val,
            "metric1": ...,
            "metric2": ...,
        })

    return results


def plot_results(times, outputs, title="Circuit Response"):
    """Generate and save plots."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, outputs)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(title)
        ax.grid(True)

        # Save to same directory as script
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(os.path.join(script_dir, f"{title.lower().replace(' ', '_')}.png"), dpi=150)
        plt.close()
        print(f"Plot saved: {title.lower().replace(' ', '_')}.png")

    except ImportError:
        print("matplotlib not available, skipping plots")


def main():
    print("=" * 60)
    print("[Circuit Name] Example")
    print("Converted from Falstad: {{FALSTAD_FILE}}")
    print("=" * 60)

    # 1. Print circuit parameters
    print("\nCircuit Parameters:")
    print(f"   R = ... ohm")
    print(f"   C = ... F")
    # etc.

    # 2. Step response
    print("\n1. Step Response")
    print("-" * 40)
    times, outputs, analysis = simulate_step_response()
    print(f"   Time constant: {analysis['tau']*1000:.3f} ms")
    # Print key metrics

    # 3. Frequency response (if applicable)
    print("\n2. Frequency Response")
    print("-" * 40)
    ac_results = simulate_frequency_response()
    for r in ac_results:
        print(f"   f={r['freq']:.1f} Hz: {r['magnitude_db']:.2f} dB")

    # 4. Parameter sweep
    print("\n3. Parameter Sweep")
    print("-" * 40)
    sweep_results = parameter_sweep("R1", [100, 500, 1000, 5000, 10000])
    for r in sweep_results:
        print(f"   R={r['param_value']}: metric={r['metric1']:.4f}")

    # 5. Differentiability demo
    print("\n4. JAX Differentiability")
    print("-" * 40)
    grad_val = demo_differentiability()
    print(f"   d(output)/d(param): {grad_val:.6f}")
    print("   (Finite gradient demonstrates differentiability)")

    # 6. Generate plots
    print("\n5. Generating Plots")
    print("-" * 40)
    plot_results(times, outputs, "Step Response")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
```

---

## Verification Checklist

After implementation, verify:

### Functional Tests
- [ ] Script runs without errors: `python your_script.py`
- [ ] Output values are physically reasonable
- [ ] Step response settles to expected steady-state value
- [ ] Frequency response shows expected rolloff/resonance behavior

### Differentiability Tests
- [ ] `demo_differentiability()` returns a finite, non-zero gradient
- [ ] Gradient sign makes physical sense (e.g., increasing R should decrease current)

### Code Quality
- [ ] Docstrings explain the circuit and its behavior
- [ ] ASCII circuit diagram in `build_circuit()` docstring
- [ ] Key formulas documented (time constants, cutoff frequencies, etc.)
- [ ] Parameter defaults match or are inspired by original Falstad values

---

## Reference Implementations

Study these existing implementations for patterns (paths relative to project root):

1. **Basic:** `examples/falstad/basic/voltage_divider.py`

2. **Filter:** `examples/falstad/filters/lowpass_rc.py`

3. **RLC:** `examples/falstad/rlc/lrc_resonance.py`

4. **Inductor transient:** `examples/falstad/rlc/inductor_kick.py`

5. **Op-amp:** `examples/falstad/opamp/inverting.py`

6. **Op-amp with study:** `examples/falstad/opamp/inverting_study.py`

---

## Important Notes

### Op-Amp Modeling

Real op-amps need RC dampeners to break algebraic feedback loops. Use this pattern:

```python
def build_opamp_circuit():
    net = Network()
    net, n_in = net.node("in")
    net, n_inv = net.node("inv")      # Inverting input
    net, n_noninv = net.node("noninv")  # Non-inverting input
    net, n_int = net.node("internal")  # Internal node (after RC)
    net, n_out = net.node("out")

    # Input resistor
    net, r_in = R(net, n_in, n_inv, name="Rin")

    # Feedback resistor
    net, r_f = R(net, n_out, n_inv, name="Rf")

    # RC dampener at inverting input (CRITICAL for stability)
    net, r_p = R(net, n_inv, n_int, name="Rp")  # ~100 ohm
    net, c_p = C(net, n_int, net.gnd, name="Cp")  # ~100-500 pF

    # VCVS models op-amp: Vout = A * (V+ - V-)
    # Here V+ is grounded (n_noninv = gnd), V- is n_int
    net, opamp = VCVS(net, net.gnd, n_int, n_out, net.gnd, name="opamp")

    # Load resistor
    net, r_load = R(net, n_out, net.gnd, name="Rload")

    return net, nodes, components
```

### Timestep Selection

```python
# For RC circuits
tau = R * C
dt = tau / 100

# For RLC circuits
omega_0 = 1 / math.sqrt(L * C)
period = 2 * math.pi / omega_0
dt = period / 50

# For AC analysis
dt = period / 40  # at least 40 samples per period

# For op-amp with RC dampener
tau_rc = Rp * Cp
dt = min(period / 50, tau_rc / 100)
```

### Common Pitfalls

1. **Don't make dt depend on traced parameters** - JAX can't trace through dt calculation
2. **Use `float()` when extracting values** for printing/plotting
3. **Ground reference:** Always use `net.gnd` for ground connections
4. **Component naming:** Use descriptive names for debugging

---

## Execution

After creating your implementation (from project root):

```bash
python examples/falstad/[subdirectory]/[your_file].py
```

Verify output looks reasonable, gradients are finite, and plots are generated.

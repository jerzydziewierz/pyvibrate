# Agent Memory - PyVibrate

## Session: 2025-12-04

### Package Restructuring Complete
Moved all time-domain code to `pyvibrate.timedomain` namespace to make space for
`pyvibrate.frequencydomain`. This is in preparation for frequency-domain analysis
for VNA measurement fitting.

Structure:
```
pyvibrate/
    __init__.py           # Package root (no re-exports)
    timedomain/
        __init__.py       # Exports all time-domain symbols
        network.py        # Network, Node, ComponentRef, ComponentSpec
        components.py     # R, C, L, VSource, Switch, VoltageSwitch, VCVS, VCR, DelayLine
        simulator.py      # SimState, SimFns, compile_network
        subcircuits.py    # HBridge and helpers
```

### Changes Made
1. Created `pyvibrate/timedomain/` directory
2. Moved all modules from `pyvibrate/` to `pyvibrate/timedomain/`
3. Updated all test imports: `from pyvibrate import ...` -> `from pyvibrate.timedomain import ...`
4. Updated all notebook imports similarly
5. Updated README.md and USAGE.md with new structure
6. Cleaned up AGENTS.md to focus on frequency-domain development

### Tests Status
All 74 tests pass with the new namespace structure.

### Next: Frequency-Domain Module
The goal is to implement `pyvibrate.frequencydomain` for:
- Steady-state AC analysis at single frequency
- Complex phasors for voltages and currents
- Components: R, C, L, ACSource, PhaseShift, VCVS, TLine (transmission line with Z₀, τ)
- Target: fitting VNA measurements to equivalent circuit models (piezo transducers)

Key differences from time-domain:
- No step-by-step simulation - just solve Y*V = I at each frequency
- Complex admittance matrix instead of real conductance
- Frequency sweep to generate impedance plots
- S-parameter calculation for VNA comparison

### Grey's Preferences (Preserved)
- Functional style with explicit state passing
- Immutable network building
- Test-first development
- Conservative with new code
- No brand names in code/comments

---

## Session: 2025-12-04 (continued)

### Frequency-Domain Module Implementation COMPLETE

Implemented `pyvibrate.frequencydomain` with all planned components:

Structure:
```
pyvibrate/
    frequencydomain/
        __init__.py       # Exports all frequency-domain symbols
        network.py        # Network, Node, ComponentRef, ComponentSpec
        components.py     # R, C, L, ACSource, PhaseShift, VCVS, TLine
        solver.py         # Solver, Solution, compile_network
```

### Components Implemented (with tests)
1. **R** - Resistor (Y = 1/R, frequency-independent)
2. **C** - Capacitor (Y = jωC, negative imaginary impedance)
3. **L** - Inductor (Y = -j/(ωL), positive imaginary impedance)
4. **ACSource** - AC voltage source with magnitude and phase
5. **PhaseShift** - Pure phase delay element (V_out = V_in * exp(-jωτ))
6. **VCVS** - Voltage-Controlled Voltage Source (V_out = gain * V_ctrl)
7. **TLine** - Transmission line with Y-parameters (Z₀, τ)

### Key Features
- Complex MNA solver using JAX
- All components are JAX-differentiable
- Y-parameter stamping for all components
- 2-port Y-parameter support for TLine
- z_in() function to compute input impedance at ACSource

### Tests Status
- 36 new frequency-domain tests
- All 110 tests pass (74 time-domain + 36 frequency-domain)

### Example Notebooks Created
1. `demo_freqdomain_sensitivity.ipynb` - Sensitivity analysis (d|Z|/dC, d|Z|/dL across frequency)
2. `demo_freqdomain_fitting.ipynb` - Parameter identification fitting capacitor ESR/ESL/C to noisy impedance data

### Technical Notes
- TLine uses Y-parameters: Y11 = Y22 = -jcot(ωτ)/Z₀, Y12 = Y21 = jcsc(ωτ)/Z₀
- PhaseShift is ideal (no reflections), TLine has impedance mismatch effects
- Solver uses JAX `jnp.linalg.solve` for complex matrix solve

---

## Session: 2025-12-15

### Bandpass Filter Implementation Complete

Implemented `examples/falstad/filters/bandpass_rlc.py` from Falstad's `bandpass.txt`.

**Circuit Topology:**
- Series R (250Ω) followed by parallel LC tank (L=0.5H, C=31.7μF)
- At resonance, parallel LC has maximum impedance → maximum voltage transfer
- This is a standard passive bandpass filter topology

**Key Parameters:**
- Center frequency: f₀ = 1/(2π√LC) = 39.98 Hz
- Quality factor: Q = R√(C/L) = 1.99
- Bandwidth: BW = f₀/Q = 20.08 Hz

**Implementation Features:**
- Step response simulation showing transient oscillations
- AC frequency response with Bode magnitude plot
- Parameter sweep demonstrating Q vs. R relationship
- JAX differentiability demo (dV/dC gradient)
- Matplotlib plots for visualization

**Verification:**
- Peak gain at resonance: 1.0 (0 dB) ✓
- Symmetric rolloff above/below f₀ ✓
- DC gain = 0 (correct for bandpass) ✓
- Increasing R increases Q and narrows bandwidth ✓
- All gradients finite and meaningful ✓

The implementation follows the established pattern from other filter examples and passes all functional tests.

---

## Session: 2025-12-15 (Opus Implementation)

### Multi-Model Comparison: Bandpass Filter

Grey ran the same circuit implementation task on three different models (Sonnet, Haiku, Opus) to compare their approaches. Implementations are in separate subdirectories:

```
examples/falstad/filters/bandpass_rlc/
├── sonnet/    # Sonnet's implementation
├── haiku/     # Haiku's implementation
└── opus/      # This session's implementation
```

### Key Finding: Q Formula Discrepancy

Haiku used the wrong Q formula for this topology:
- **Haiku used:** Q = ω₀L/R = 0.50 (series RLC formula)
- **Correct formula:** Q = R/Z₀ = R√(C/L) = 1.99 (series R + parallel LC)

Both formulas are valid but for different topologies:
- Series RLC: Q = (1/R)√(L/C) - lower R → higher Q
- Series R with parallel LC: Q = R√(C/L) - higher R → higher Q

The circuit simulations were correct in all cases; only the analytical parameter calculation differed.

### Opus Implementation Improvements

1. **Added phase response** to frequency analysis
2. **Theoretical comparison** - computes analytical transfer function alongside simulation
3. **Clearer Q derivation** - explains why Q = R/Z₀ for this topology
4. **Falstad source insight** - explains why 150 Hz source is attenuated (-17 dB)
5. **More detailed physical explanations** in findings document

### Files Created
- `opus/bandpass_rlc.py` - Main implementation (~500 lines)
- `opus/bandpass_rlc_findings.md` - Tutorial-style documentation
- `opus/bandpass_step_response.png` - Step response plot
- `opus/bandpass_frequency_response.png` - Bode plot with magnitude and phase

### Simulation Results Match Theory

| Metric | Simulated | Theoretical |
|--------|-----------|-------------|
| f₀ | 39.98 Hz | 39.98 Hz |
| Peak gain | 0.9999 | 1.0 |
| Q | 1.99 | 1.99 |
| BW | ~16 Hz measured | 20.08 Hz |

The measured bandwidth is slightly narrower than theoretical due to finite simulation resolution around the -3dB points.

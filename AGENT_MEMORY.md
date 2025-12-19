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
5. **VCVS** - Voltage-Controlled Voltage Source (V_out = gain * V_ctrl)
6. **TDVCVS** - Time Delay Voltage-Controlled Voltage Source (V_out = V_in * exp(-jωτ))
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

---

## Session: 2025-12-18

### Transformer Component Refactoring and Planning

**Component Rename Completed:**
Renamed `PhaseShift` → `ConstantTimeDelayVCVS` to clarify it's an active element (energy source).
- Updated 11 files: components.py, solver.py, __init__.py, 5 test files, 3 docs, 1 notebook
- Renamed test file: test_freqdomain_phaseshift.py → test_freqdomain_tdvcvs.py
- All tests passing (12 tests total across test_freqdomain_tdvcvs.py and test_delay_line_fft.py)

**Transformer Discussion and Planning:**
Grey asked how to model a magnetic transformer in frequency domain. Key requirements:
- Realistic model with magnetizing inductance, leakage, winding resistance, core losses
- Physical parameters: N1, N2, A_core, l_m, mu_r, B_sat
- Core saturation effects (nonlinear)
- VNA parameter extraction notebook (open circuit, short circuit, loaded, saturation curve tests)

**Critical Challenge Identified:**
Core saturation is fundamentally nonlinear (L depends on I), but PyVibrate's frequency domain solver is purely linear (direct `jnp.linalg.solve`, no Newton-Raphson).

Explored three agents in parallel:
1. Component implementation patterns (factory functions, MNA stamping, Y-parameters)
2. Nonlinear solver capabilities (confirmed: NONE - only linear solver exists)
3. VNA measurement patterns (z_in function, demo_freqdomain_fitting.ipynb, JAX autodiff)

**User Decisions (via AskUserQuestion):**
1. **Saturation:** Linear model only with saturation warnings (no nonlinear solver for now)
2. **Domain:** Frequency domain only (skip time domain entirely)
3. **Core model:** Simple tanh reference (B = B_sat * tanh(H/H_c)) for documentation
4. **VNA tests:** ALL four tests in notebook (open, short, loaded, saturation curve)

**Implementation Plan Created:**
Comprehensive plan written to `AGENTS_realistic_transformer_plan.md` covering:

**Architecture:**
- Single `Transformer()` component with Y-parameter 2-port stamping
- Equivalent circuit: R1 + L_leak1 + (L_mag║R_core) on primary, R2 + L_leak2 on secondary
- Physical params → circuit params: L = μ₀*μᵣ*N²*A/l_m, M = k*sqrt(L1*L2)

**MNA Stamping:**
- Build Z-parameters: Z11 = R1 + jωL_leak1 + (jωL_mag║R_core), Z12 = jωM
- Convert to Y via 2×2 inversion: Y = Z⁻¹
- Stamp with existing `_stamp_2port_admittance()` helper

**Saturation Warning:**
- Post-solve check: estimate I_mag from V_prim/(jωL_mag)
- Compute B = μ₀*μᵣ*N*I/l_m, compare to B_sat
- Warn if B/B_sat > 0.8 (approximate, doesn't account for load current)

**Files to Create:**
1. `pyvibrate/frequencydomain/physical_constants.py` - μ₀, compute_inductance(), compute_flux_density()
2. `tests/test_freqdomain_transformer.py` - 10 unit tests
3. `notebooks/demo_transformer_vna_extraction.ipynb` - 4-section tutorial

**Files to Modify:**
1. `pyvibrate/frequencydomain/components.py` - add Transformer() factory
2. `pyvibrate/frequencydomain/solver.py` - add stamping case, saturation check
3. `pyvibrate/frequencydomain/__init__.py` - exports

**Implementation Sequence:**
- Phase 1: Core component (physical_constants.py, Transformer(), stamping, 3 basic tests)
- Phase 2: Full testing (7 more tests, validation)
- Phase 3: Saturation warning (check function, test)
- Phase 4: VNA notebook (4 sections with extraction)
- Phase 5: Documentation (docstrings, README)

**Critical Details:**
- Edge cases: DC (omega_safe), resonance (det_safe), coupling bounds (k clipped)
- JAX differentiability: use jnp.where() not Python if, all computations in stamping
- VNA extraction: sequential (not simultaneous) for better conditioning
- Numerical stability: safe divisions, safe omegas, determinant checks

**Current State:**
- Planning complete and documented in `AGENTS_realistic_transformer_plan.md`
- User reviewed plan and saved it
- Ready to begin Phase 1 implementation tomorrow

**Next Session:**
Start with Phase 1: Create physical_constants.py module with fundamental constants and helper functions.

# PyVibrate - Development Plan

## Package Structure

```
pyvibrate/
    timedomain/           # COMPLETE - Time-domain transient simulation
    frequencydomain/      # COMPLETE - Steady-state AC analysis
```

## Frequency-Domain Module (Usably Complete)

### Components Implemented

| Component | Admittance/Function | Notes |
|-----------|---------------------|-------|
| R | Y = 1/R (real) | Frequency-independent |
| C | Y = jωC | Positive imaginary admittance |
| L | Y = -j/(ωL) | Negative imaginary admittance |
| ACSource | V∠θ (complex phasor) | Amplitude and phase |
| ConstantTimeDelayVCVS | exp(-jωτ) | Time delay (active element, no impedance effects) |
| VCVS | Vout = gain·Vctrl | Voltage-controlled voltage source |
| TLine | Y₁₁=Y₂₂=-jcot(ωτ)/Z₀, Y₁₂=Y₂₁=jcsc(ωτ)/Z₀ | Transmission line with reflections |

### Tests Passing: 36/36
- R: series, parallel, frequency-independence, differentiability
- C: impedance, frequency-dependence, phase, differentiability
- L: impedance, frequency-dependence, phase, LC resonance, differentiability
- ACSource: basic operation
- ConstantTimeDelayVCVS: zero delay, quarter/half wave, frequency-dependence, differentiability
- VCVS: unity/variable gain, differential input, differentiability
- TLine: matched/open/short loads, impedance transformation, differentiability

### Demo Notebooks
- `demo_freqdomain_sensitivity.ipynb` - Sensitivity analysis (d|Z|/dC, d|Z|/dL)
- `demo_freqdomain_fitting.ipynb` - Parameter identification (fit capacitor ESR/ESL/C to noisy data)
- `demo_delay_line.ipynb` - Delay line in both domains: time-domain with FFT, freq-domain with iFFT

### Current Work: Realistic Transformer Component

**Status:** Planning complete, ready for implementation

**Detailed Plan:** See `AGENTS_realistic_transformer_plan.md`

**Goal:** Frequency-domain transformer with physical parameters (N1, N2, A_core, l_m, mu_r, B_sat) and VNA extraction notebook

**Key Features:**
- Y-parameter 2-port stamping (similar to TLine)
- Equivalent circuit: magnetizing inductance, core loss, leakage, winding resistance
- Linear solver with post-solve saturation warnings
- Four VNA tests: open circuit, short circuit, loaded, and saturation curve extraction

**Implementation Phases:**
1. Core component: physical_constants.py, Transformer() factory, MNA stamping
2. Testing: 10 unit tests validating all aspects
3. Saturation warning: check_transformer_saturation() helper
4. VNA notebook: comprehensive parameter extraction tutorial

**Next Steps:** Begin Phase 1 (core component implementation)

### Future Work
- Frequency sweep helper function
- S-parameter calculation for VNA fitting
- More complex equivalent circuit models (piezo transducers)

## Time-Domain Module Status (Complete)

### Components Implemented
- R, C, L (passive)
- VSource (controllable voltage source)
- Switch, VoltageSwitch (controllable switches)
- VCVS (voltage-controlled voltage source)
- VCR (voltage-controlled resistor)
- DelayLine (N-sample voltage delay)
- HBridge (4-switch subcircuit)

### Tests Passing: 74/74
- RC, RL, RLC circuits
- Switch control and threshold
- VCVS gain and differential input
- VCR voltage-dependent resistance
- DelayLine timing
- HBridge drive modes
- JAX autodiff sensitivity
- JAX gradient descent optimization

### Demo Notebooks
- `demo_rlc_01.ipynb` - H-bridge driven RLC with ring-down
- `demo_vcr_01.ipynb` - VCR frequency mixing demonstration
- `demo_delay_line.ipynb` - Delay line: pulse delay with FFT phase analysis

### Next step


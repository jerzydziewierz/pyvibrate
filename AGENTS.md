# PyVibrate - Development Plan

## Package Structure

```
pyvibrate/
    timedomain/           # COMPLETE - Time-domain transient simulation
    frequencydomain/      # COMPLETE - Steady-state AC analysis
```

## Frequency-Domain Module (Complete)

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


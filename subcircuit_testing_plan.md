# Subcircuit Testing Plan

Test plan for `Series` and `Parallel` operations in both time-domain and frequency-domain modules.

## Time-Domain: Series Operation

### Core Functionality
- [x] `test_series_resistors_equivalence` - Series of two resistors should equal their sum (R1 + R2)
- [x] `test_series_voltage_division` - Voltage across each element follows voltage divider rule
- [x] `test_series_vs_manual` - Series(R, C) produces identical results to manually built R-C chain
- [x] `test_series_midpoint_accessible` - Internal mid-node is probeable and voltage is between input/output

### Physical Correctness (Passive Components)
- [x] `test_series_current_continuity` - Current through both elements is identical at all time steps
- [x] `test_series_rc_charging` - Series RC shows correct exponential charging (τ = RC)
- [x] `test_series_rl_current_rise` - Series RL shows correct current rise time (τ = L/R)
- [x] `test_series_lc_resonance` - Series LC oscillates at f = 1/(2π√LC)

### Composability
- [x] `test_series_nested` - Series can be nested: Series(Series(R, R), R) = 3R total
- [ ] `test_multiple_series_in_circuit` - Multiple independent Series blocks in one network
- [ ] `test_series_with_active_components` - Series works with VCVS or other controlled sources

### JAX Integration
- [x] `test_series_jax_differentiable` - Gradients w.r.t. component values (∂V/∂R, ∂V/∂C)

## Time-Domain: Parallel Operation

### Core Functionality
- [x] `test_parallel_resistors_equivalence` - Parallel of two resistors equals 1/(1/R1 + 1/R2)
- [x] `test_parallel_current_division` - Current through each element follows current divider rule
- [x] `test_parallel_vs_manual` - Parallel(R, C) produces identical results to manual construction
- [x] `test_parallel_voltage_same` - Voltage across both elements is identical

### Physical Correctness (Passive Components)
- [ ] `test_parallel_rc_impedance` - Parallel RC shows correct frequency-dependent behavior
- [ ] `test_parallel_rl_impedance` - Parallel RL shows correct phase relationship
- [ ] `test_parallel_lc_antiresonance` - Parallel LC shows high impedance at resonance

### Composability
- [x] `test_parallel_nested` - Parallel can be nested
- [ ] `test_multiple_parallel_in_circuit` - Multiple independent Parallel blocks
- [x] `test_series_parallel_mixed` - Series(Parallel(...), R) and Parallel(Series(...), C)

### JAX Integration
- [x] `test_parallel_jax_differentiable` - Gradients w.r.t. component values

## Frequency-Domain: Series Operation

### Core Functionality
- [ ] `test_freq_series_impedance_sum` - Z_total = Z1 + Z2 at all frequencies
- [ ] `test_freq_series_voltage_division` - Complex voltage division at multiple frequencies
- [ ] `test_freq_series_vs_manual` - Series produces identical results to manual construction
- [ ] `test_freq_series_midpoint_voltage` - Mid-node voltage is correct at various frequencies

### Physical Correctness
- [ ] `test_freq_series_rc_lowpass` - Series RC shows -20dB/decade rolloff and -90° phase
- [ ] `test_freq_series_rl_highpass` - Series RL shows correct high-pass characteristics
- [ ] `test_freq_series_rlc_resonance` - Series RLC shows minimum impedance at resonance

### Composability
- [ ] `test_freq_series_nested` - Nested Series operations
- [ ] `test_freq_multiple_series` - Multiple Series blocks in frequency domain

### JAX Integration
- [ ] `test_freq_series_jax_differentiable` - Gradients of impedance w.r.t. component values

## Frequency-Domain: Parallel Operation

### Core Functionality
- [ ] `test_freq_parallel_admittance_sum` - Y_total = Y1 + Y2 at all frequencies
- [ ] `test_freq_parallel_current_division` - Complex current division at multiple frequencies
- [ ] `test_freq_parallel_vs_manual` - Parallel produces identical results to manual construction
- [ ] `test_freq_parallel_voltage_same` - Voltage across both elements is identical

### Physical Correctness
- [ ] `test_freq_parallel_rc_impedance` - Parallel RC impedance vs frequency
- [ ] `test_freq_parallel_rl_impedance` - Parallel RL phase characteristics
- [ ] `test_freq_parallel_lc_antiresonance` - Parallel LC shows maximum impedance at resonance

### Composability
- [ ] `test_freq_parallel_nested` - Nested Parallel operations
- [ ] `test_freq_series_parallel_mixed` - Mixed Series and Parallel in frequency domain

### JAX Integration
- [ ] `test_freq_parallel_jax_differentiable` - Gradients of impedance w.r.t. component values

## Implementation Priority

### Phase 1: Core Validation (Must Have)
Priority tests to ensure basic correctness before any other use:
- Series/Parallel resistor equivalence (both domains)
- Voltage/current division rules
- Comparison with manual construction
- Mid-node/internal node accessibility

### Phase 2: Physical Validation (Should Have)
Verify correct physical behavior with reactive components:
- RC/RL time constants (time domain)
- LC resonance/antiresonance (both domains)
- Frequency response characteristics (frequency domain)

### Phase 3: Advanced Features (Nice to Have)
Test composability and differentiation:
- Nested subcircuits
- Mixed Series/Parallel combinations
- JAX gradient computation
- Multiple subcircuit blocks

## Notes

- Each test should verify results against analytical solutions where possible
- Time-domain tests should check transient behavior over multiple time constants
- Frequency-domain tests should sweep across 2-3 decades of frequency
- All tests should be added to the existing test suite structure
- Consider parametrized tests for testing with different component values

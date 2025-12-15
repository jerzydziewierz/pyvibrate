# Testing Plan

## Simple components

### Time-Domain Components
- [x] **Resistor (R)**: `test_r_series`, `test_r_parallel`, `test_r_voltage_divider`, `test_r_differentiable`
- [x] **Capacitor (C)**: `test_rc_step_response`, `test_rc_*` (charging, values, overrides)
- [x] **Inductor (L)**: `test_rl_current_buildup`, `test_rl_voltage_across_inductor`
- [x] **RLC Circuit**: `test_rlc_overdamped`, `test_rlc_underdamped_oscillation`, `test_lc_resonance`
- [x] **Voltage Source**: `test_voltage_source_override_via_controls`
- [x] **Switch**: `test_switch_open_blocks_current`, `test_switch_closed_conducts`, `test_switch_control`
- [x] **VoltageSwitch**: `test_voltage_switch_basic`, `test_voltage_switch_comparator`, `test_voltage_switch_inverted`
- [x] **VCVS**: `test_vcvs_unity_gain`, `test_vcvs_gain_10`, `test_vcvs_negative_gain`, `test_vcvs_inverting`, `test_vcvs_differential_input`, `test_vcvs_tracks_input_change`
- [x] **VCR**: `test_vcr_positive_k`, `test_vcr_negative_k`, `test_vcr_k_zero_*`, `test_vcr_dynamic_control`
- [x] **Delay Line**: `test_delay_line_basic`, `test_delay_line_tracks_signal`, `test_delay_line_zero_delay`, `test_delay_line_differential`
- [x] **HBridge**: `test_hbridge_all_off`, `test_hbridge_drive_a_high_b_low`, `test_hbridge_drive_a_low_b_high`, `test_hbridge_freewheel_low`

### Frequency-Domain Components
- [x] **Resistor (R)**: `test_r_impedance_is_real`, `test_r_differentiable`
- [x] **Capacitor (C)**: `test_c_impedance_decreases_with_frequency`, `test_c_impedance_is_negative_imaginary`, `test_c_phase_is_minus_90`, `test_c_differentiable`
- [x] **Inductor (L)**: `test_l_impedance_increases_with_frequency`, `test_l_impedance_is_positive_imaginary`, `test_l_phase_is_plus_90`, `test_l_differentiable`
- [x] **RC/RL Series/Parallel**: `test_rc_series_impedance`, `test_rc_parallel_impedance`, `test_rl_series_impedance`
- [x] **VCVS**: `test_vcvs_frequency_independent`, `test_vcvs_differentiable`
- [x] **Phase Shift**: `test_phaseshift_*` (zero_delay, quarter_wave, half_wave, magnitude_unity, frequency_dependent, differentiable)
- [x] **Transmission Line**: `test_tline_*` (matched_load, open_circuit, short_circuit, phase_delay, impedance_transformation, different_Z0, differentiable)

### Current Probing
- [x] **Inductor current**: `test_inductor_current_matches_state`, `test_inductor_current_physical_behavior`, `test_inductor_current_differentiable`
- [x] **Capacitor current**: `test_capacitor_current_matches_state`, `test_capacitor_current_physical_behavior`, `test_capacitor_current_differentiable`
- [x] **Multiple L/C**: `test_multiple_inductors`, `test_multiple_capacitors`
- [x] **Not implemented**: `test_resistor_current_not_implemented`, `test_vsource_current_not_implemented`, `test_switch_current_not_implemented`, `test_vcvs_current_not_implemented`, `test_vcr_current_not_implemented`, `test_voltage_switch_current_not_implemented`

### JAX Differentiability
- [x] **Component gradients**: `test_dV_dR_finite`, `test_dV_dC_finite`, `test_dV_dL_finite`, `test_dI_dL_finite`, `test_dI_dR_via_ohms_law`, `test_dI_dV_via_ohms_law`
- [x] **Optimization**: `test_gradient_descent_optimize_R`
- [x] **Sensitivity**: `test_sensitivity_dV_dR`, `test_sensitivity_dTau_dR`
- [x] **RLC gradients**: `test_rlc_all_params_differentiable`, `test_rlc_inductor_current_differentiable`

### Cross-Domain
- [x] **Time/Freq consistency**: `test_timedomain_vs_freqdomain_ifft`, `test_cross_domain_pulse_shape_preserved`
- [x] **FFT analysis**: `test_timedomain_delay_fft_phase_response`, `test_timedomain_delay_group_delay`, `test_ifft_impulse_response_peak`

---

# Subcircuits

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
- [x] `test_multiple_series_in_circuit` - Multiple independent Series blocks in one network
- [x] `test_series_with_active_components` - Series works with VCVS or other controlled sources

### JAX Integration
- [x] `test_series_jax_differentiable` - Gradients w.r.t. component values (∂V/∂R, ∂V/∂C)

## Time-Domain: Parallel Operation

### Core Functionality
- [x] `test_parallel_resistors_equivalence` - Parallel of two resistors equals 1/(1/R1 + 1/R2)
- [x] `test_parallel_current_division` - Current through each element follows current divider rule
- [x] `test_parallel_vs_manual` - Parallel(R, C) produces identical results to manual construction
- [x] `test_parallel_voltage_same` - Voltage across both elements is identical

### Physical Correctness (Passive Components)
- [x] `test_parallel_rc_impedance` - Parallel RC shows correct frequency-dependent behavior
- [x] `test_parallel_rl_impedance` - Parallel RL shows correct phase relationship
- [x] `test_parallel_lc_antiresonance` - Parallel LC shows high impedance at resonance

### Composability
- [x] `test_parallel_nested` - Parallel can be nested
- [x] `test_multiple_parallel_in_circuit` - Multiple independent Parallel blocks
- [x] `test_series_parallel_mixed` - Series(Parallel(...), R) and Parallel(Series(...), C)

### JAX Integration
- [x] `test_parallel_jax_differentiable` - Gradients w.r.t. component values

## Frequency-Domain: Series Operation

### Core Functionality
- [x] `test_freq_series_impedance_sum` - Z_total = Z1 + Z2 at all frequencies
- [x] `test_freq_series_voltage_division` - Complex voltage division at multiple frequencies
- [x] `test_freq_series_vs_manual` - Series produces identical results to manual construction
- [x] `test_freq_series_midpoint_voltage` - Mid-node voltage is correct at various frequencies

### Physical Correctness
- [x] `test_freq_series_rc_lowpass` - Series RC shows -20dB/decade rolloff and -90° phase
- [x] `test_freq_series_rl_highpass` - Series RL shows correct high-pass characteristics
- [x] `test_freq_series_rlc_resonance` - Series RLC shows minimum impedance at resonance

### Composability
- [x] `test_freq_series_nested` - Nested Series operations
- [x] `test_freq_multiple_series` - Multiple Series blocks in frequency domain

### JAX Integration
- [x] `test_freq_series_jax_differentiable` - Gradients of impedance w.r.t. component values

## Frequency-Domain: Parallel Operation

### Core Functionality
- [x] `test_freq_parallel_admittance_sum` - Y_total = Y1 + Y2 at all frequencies
- [x] `test_freq_parallel_current_division` - Complex current division at multiple frequencies
- [x] `test_freq_parallel_vs_manual` - Parallel produces identical results to manual construction
- [x] `test_freq_parallel_voltage_same` - Voltage across both elements is identical

### Physical Correctness
- [x] `test_freq_parallel_rc_impedance` - Parallel RC impedance vs frequency
- [x] `test_freq_parallel_rl_impedance` - Parallel RL phase characteristics
- [x] `test_freq_parallel_lc_antiresonance` - Parallel LC shows maximum impedance at resonance

### Composability
- [x] `test_freq_parallel_nested` - Nested Parallel operations
- [x] `test_freq_series_parallel_mixed` - Mixed Series and Parallel in frequency domain

### JAX Integration
- [x] `test_freq_parallel_jax_differentiable` - Gradients of impedance w.r.t. component values

## Notes

- Each test should verify results against analytical solutions where possible
- Time-domain tests should check transient behavior over multiple time constants
- Frequency-domain tests should sweep across 2-3 decades of frequency
- All tests should be added to the existing test suite structure
- Consider parametrized tests for testing with different component values
- For each circuit composed with pyvibrate, use jax compilation and then get a gradient wrt. something to verify that taking a gradient works

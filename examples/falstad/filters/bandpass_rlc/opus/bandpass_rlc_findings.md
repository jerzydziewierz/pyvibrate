# RLC Bandpass Filter: Implementation and Analysis

**Circuit source:** Falstad `bandpass.txt`
**Implementation:** `bandpass_rlc.py`

## Circuit Overview

This document explores a second-order bandpass filter using a parallel LC tank topology. The circuit selectively passes signals within a narrow frequency band centered on its resonant frequency while attenuating all other frequencies.

### Circuit Topology

```
Vin ---[R]---+------- Vout
             |
         +---+---+
         |       |
        [L]     [C]
         |       |
         +---+---+
             |
            GND
```

The series resistor R and parallel LC tank form a voltage divider where the tank impedance varies dramatically with frequency.

### Component Values (from Falstad)

| Component | Value | Unit |
|-----------|-------|------|
| R | 250 | ohm |
| L | 0.5 | H (500 mH) |
| C | 31.7 | uF |

### Calculated Characteristics

| Parameter | Formula | Value | Meaning |
|-----------|---------|-------|---------|
| f_0 | 1/(2π√LC) | 39.98 Hz | Resonant frequency |
| Z_0 | √(L/C) | 125.6 ohm | Characteristic impedance |
| Q | R/Z_0 | 1.99 | Quality factor |
| BW | f_0/Q | 20.08 Hz | 3dB bandwidth |

## Understanding the Bandpass Mechanism

### Parallel LC Tank Behavior

The parallel LC combination presents a frequency-dependent impedance:

1. **At low frequencies (f << f_0):**
   - Inductor: X_L = ωL → small, acts like short circuit
   - Capacitor: X_C = 1/(ωC) → large
   - Tank impedance: dominated by L, very small
   - Result: Vout ≈ 0 (voltage dropped across R)

2. **At resonance (f = f_0):**
   - X_L = X_C (reactances equal but opposite)
   - In an ideal tank, the currents through L and C cancel perfectly
   - Tank impedance → very large (theoretically infinite)
   - Result: Vout ≈ Vin (no voltage dropped across R)

3. **At high frequencies (f >> f_0):**
   - Inductor: X_L = ωL → large
   - Capacitor: X_C = 1/(ωC) → small, acts like short circuit
   - Tank impedance: dominated by C, very small
   - Result: Vout ≈ 0 (voltage dropped across R)

### Transfer Function Derivation

The impedance of the parallel LC tank is:

```
Z_tank = (jωL) || (1/jωC)
       = jωL / (1 - ω²LC)
```

At resonance where ω²LC = 1, the denominator approaches zero and Z_tank → ∞.

The transfer function is:

```
H(ω) = Z_tank / (R + Z_tank)
```

At resonance: H(ω_0) → 1 (unity gain)

### Why Q = R√(C/L)?

For this topology (series R with parallel LC to ground):

The Q factor relates the reactance at resonance to the resistance:
```
Q = R / Z_0 = R / √(L/C) = R × √(C/L)
```

This is the **opposite** of the series RLC case where Q = (1/R)√(L/C).

**Important distinction:**
- Series RLC: Lower R → Higher Q (less damping)
- Parallel LC with series R: Higher R → Higher Q (more selective)

## Simulation Results

### Step Response

When a 5V step is applied:

| Metric | Value | Explanation |
|--------|-------|-------------|
| Peak voltage | 1.78 V | Transient oscillation at f_0 |
| Time to peak | 5.5 ms | About 1/4 period at f_0 |
| Final voltage | 0 V | DC is blocked (bandpass behavior) |

**Physical interpretation:** The step input contains broadband frequency content (DC plus all harmonics). The bandpass filter responds only to components near f_0, producing a damped oscillation that decays as energy dissipates through R.

The decay time constant is approximately:
```
τ_decay ≈ 2LQ/R = 2 × 0.5 × 1.99 / 250 = 8 ms
```

### Frequency Response

The AC simulation validates the bandpass characteristic:

| f (Hz) | f/f_0 | |H| | dB | Notes |
|--------|-------|-----|-----|------|
| 4 | 0.10 | 0.051 | -26 | Strong low-f attenuation |
| 20 | 0.50 | 0.319 | -10 | Approaching passband |
| 32 | 0.80 | 0.747 | -2.5 | Within -3dB band |
| **40** | **1.00** | **1.000** | **0.0** | **Peak at resonance** |
| 48 | 1.20 | 0.805 | -1.9 | Within -3dB band |
| 80 | 2.00 | 0.317 | -10 | Leaving passband |
| 400 | 10.0 | 0.055 | -25 | Strong high-f attenuation |

**Key observations:**

1. **Unity gain at resonance:** |H(f_0)| = 1.0 (0 dB) - no attenuation at center frequency

2. **Symmetric rolloff:** The attenuation at f/10 and 10f is nearly identical (-26 dB), showing the symmetric log-frequency behavior

3. **Rolloff rate:** Approximately -20 dB/decade on each side (first-order on each slope)

4. **Phase behavior:** Phase shifts from -90° (low f) through 0° (at f_0) to +90° (high f)

### Falstad Source Frequency

The original Falstad circuit uses a 150 Hz source, which is about 3.75× the resonant frequency. At this frequency:

```
|H(150 Hz)| = 0.143 = -16.9 dB
```

This means the 150 Hz source is significantly attenuated - Falstad was demonstrating the high-frequency rejection characteristic.

### Parameter Sweep: Effect of R

| R (ohm) | Q | BW (Hz) | Characteristic |
|---------|---|---------|----------------|
| 50 | 0.40 | 100 | Very broad, low selectivity |
| 100 | 0.80 | 50 | Low selectivity |
| 250 | 1.99 | 20 | Moderate selectivity |
| 500 | 3.98 | 10 | Good selectivity |
| 1000 | 7.96 | 5 | High selectivity |
| 2000 | 15.9 | 2.5 | Very sharp, narrow band |

**Design insight:**
- f_0 remains constant regardless of R (set by L and C only)
- Q scales linearly with R
- Bandwidth scales inversely with R

## JAX Differentiability

The gradient computation demonstrates full differentiability:

```
d(Vout)/d(C) = -21,859 V/F
```

**Interpretation:** A 1 nF increase in C decreases Vout by 0.022 mV at t=50ms.

**Applications:**
1. **Sensitivity analysis:** Determine which component variations have the largest effect
2. **Parameter identification:** Fit simulated response to measured data using gradient descent
3. **Optimization:** Find optimal component values for a given objective

Example optimization loop:
```python
def loss(C_val):
    # Simulate and compute error vs target response
    return error

for _ in range(iterations):
    gradient = grad(loss)(C_val)
    C_val -= learning_rate * gradient
```

## Design Guidelines

### Choosing Components

**Step 1: Select center frequency**
```
f_0 = desired_frequency
LC = 1/(2π·f_0)²
```

**Step 2: Select Q (bandwidth)**
```
Q = f_0 / BW_desired
```

**Step 3: Choose L (practical constraint)**
Pick L based on available inductors, physical size, or ESR requirements.

**Step 4: Calculate remaining values**
```
C = 1/((2π·f_0)²·L)
Z_0 = √(L/C)
R = Q × Z_0
```

### Example: 1 kHz filter, Q=5

```
f_0 = 1000 Hz, Q = 5
Choose L = 10 mH

C = 1/((2π×1000)² × 0.01) = 2.53 µF
Z_0 = √(0.01/2.53e-6) = 63 ohm
R = 5 × 63 = 315 ohm
BW = 1000/5 = 200 Hz
```

### Practical Considerations

1. **Inductor ESR:** Real inductors have series resistance that adds loss, reducing achievable Q
2. **Capacitor ESR:** Similarly affects Q at high frequencies
3. **Component tolerance:** 5% tolerance shifts f_0 by ~2.5%
4. **Temperature drift:** LC products vary with temperature, shifting f_0
5. **Loading effects:** Output impedance at resonance ≈ R; connecting a load < R degrades response

## Comparison: Three Implementations

Three AI models (Sonnet, Haiku, Opus) implemented this circuit with interesting differences:

| Aspect | Sonnet | Haiku | Opus |
|--------|--------|-------|------|
| Q formula | R√(C/L) ✓ | ω₀L/R ✗ | R/Z_0 ✓ |
| Q value | 1.99 | 0.50 | 1.99 |
| Phase analysis | No | No | Yes |
| Theoretical comparison | No | No | Yes |
| Physical insights | Basic | Good | Detailed |

**Key insight:** Haiku used the series RLC Q formula instead of the parallel LC formula, resulting in an incorrect Q value (though the simulation itself was correct).

The correct formula for **series R with parallel LC** is:
```
Q = R × √(C/L) = R / √(L/C) = R / Z_0
```

This is the opposite of **series RLC** where:
```
Q = (1/R) × √(L/C)
```

## Physical Insights

### Energy Storage and Dissipation

At resonance, energy oscillates between L and C:
- Maximum inductor energy when current peaks (magnetic field)
- Maximum capacitor energy when voltage peaks (electric field)

The Q factor represents the ratio of stored energy to energy dissipated per cycle:
```
Q = 2π × (energy stored) / (energy dissipated per cycle)
```

Higher Q means less energy loss per cycle, sharper resonance.

### Why This Topology?

**Series R + Parallel LC → Bandpass**
- Tank impedance maximum at f_0 → maximum voltage transfer
- Tank impedance minimum away from f_0 → voltage dropped across R

**If we used Series LC instead:**
- Series LC impedance minimum at f_0 → notch (band-reject) filter
- This would be the opposite behavior

### Real-World Applications

1. **Radio tuning:** Select one station's carrier frequency from the RF spectrum
2. **Audio EQ:** Boost or cut specific frequency bands
3. **Instrument tuners:** Detect if a note is at the correct frequency
4. **Noise filtering:** Pass signal frequency, reject out-of-band noise
5. **Power systems:** Filter harmonics from AC mains

## Implementation Notes

### Time Step Selection

For accurate simulation:
```python
period = 1 / f_0
dt = period / 50  # At least 50 samples per period
```

For frequency sweep, adjust dt per frequency:
```python
dt = (1/freq) / 40  # 40 samples per period at each frequency
```

### Steady-State Detection

For AC response measurements, simulate enough cycles for transients to decay:
```python
n_cycles = max(10, 5 * Q)  # More cycles needed for high-Q filters
```

Collect data only from final cycles to measure true steady-state amplitude.

### Avoiding JAX Pitfalls

**Never make dt depend on traced parameters:**
```python
# WRONG - dt depends on traced C_param
def objective(C_param):
    tau = R * C_param
    dt = tau / 100  # JAX can't trace through dt calculation

# CORRECT - fixed dt
def objective(C_param):
    dt = 1e-5  # Fixed value
    ...
```

## Conclusion

The parallel LC bandpass filter demonstrates fundamental resonance principles:

1. At resonance, reactive components create maximum impedance
2. Q determines the trade-off between selectivity and loss
3. The transfer function approaches unity at f_0
4. Both time-domain (transient) and frequency-domain (steady-state) analysis confirm theory

The PyVibrate implementation enables:
- Accurate time-domain simulation of circuit behavior
- Automatic differentiation for optimization and sensitivity analysis
- Easy parameter sweeps to explore design space

This circuit serves as a building block for more complex filter designs including cascaded stages, active filters, and matched filter networks.

## References

- Original Falstad circuit: `bandpass.txt`
- This implementation: `examples/falstad/filters/bandpass_rlc/opus/`
- Related examples:
  - Low-pass RC: `examples/falstad/filters/lowpass_rc.py`
  - RLC resonance: `examples/falstad/rlc/lrc_resonance.py`
  - Op-amp filters: `examples/falstad/opamp/`

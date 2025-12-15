# RLC Bandpass Filter: Tutorial and Findings

## Overview

This document explores the implementation and behavior of a parallel-LC bandpass filter converted from Falstad's `bandpass.txt` circuit. The circuit demonstrates how a simple RLC network can selectively pass signals within a specific frequency band while attenuating frequencies outside that band.

## Circuit Topology

### Schematic

```
Vin ---[R]---+---+--- Vout
             |   |
            [L] [C]
             |   |
            GND GND
```

### Component Values (from Falstad original)

- **R** = 250 Ω (series resistance)
- **L** = 0.5 H (parallel inductance)
- **C** = 31.7 μF (parallel capacitance)

### How It Works

1. **The LC Tank**: The parallel LC combination acts as a frequency-dependent impedance:
   - At very low frequencies: Capacitor dominates (low impedance)
   - At very high frequencies: Inductor dominates (low impedance)
   - At resonance: Impedances cancel, creating maximum impedance

2. **Voltage Division**: The circuit forms a voltage divider between R and the LC tank:
   - `Vout/Vin = Z_LC / (R + Z_LC)`
   - Maximum voltage transfer occurs when `Z_LC` is maximized (at resonance)
   - Away from resonance, `Z_LC` drops and less voltage is transferred

3. **Bandpass Characteristic**:
   - DC signals (f=0): Inductor is short circuit → Vout = 0
   - Very high frequencies: Capacitor is short circuit → Vout = 0
   - At resonance: Maximum impedance → Maximum Vout

## Mathematical Analysis

### Center Frequency (Resonance)

The resonant frequency occurs when inductive and capacitive reactances are equal:

```
X_L = X_C
ωL = 1/(ωC)
ω₀² = 1/(LC)
f₀ = 1/(2π√LC)
```

For our values:
```
f₀ = 1/(2π√(0.5 × 31.7×10⁻⁶))
   = 1/(2π × 0.00398)
   = 39.98 Hz
```

### Quality Factor

The quality factor Q determines how selective the filter is:

```
Q = R√(C/L)
```

For this topology (series R, parallel LC):
```
Q = 250 × √(31.7×10⁻⁶ / 0.5)
  = 250 × 0.00796
  = 1.99
```

A Q of ~2 means the filter is moderately selective - not too narrow, not too broad.

### Bandwidth

The 3-dB bandwidth is related to Q:

```
BW = f₀ / Q
   = 39.98 / 1.99
   = 20.08 Hz
```

This means frequencies within roughly ±10 Hz of the center frequency will pass with less than 3 dB attenuation.

### Transfer Function

At resonance, the parallel LC impedance is approximately:

```
Z_LC(ω₀) ≈ L/(RC) = 0.5/(250 × 31.7×10⁻⁶) = 63.1 Ω
```

However, the actual peak gain approaches 1.0 (0 dB) because at exact resonance, the tank impedance becomes very large relative to R.

## Simulation Results and Findings

### 1. Step Response Analysis

**Key Observation**: When a 5V step is applied, the output shows:
- Peak voltage: **1.78 V** at t ≈ 5.5 ms
- Final voltage: **0 V**

**Why the low peak?**
The step response of a bandpass filter is fundamentally limited because:
1. A step input contains mostly DC and low-frequency content
2. The bandpass filter blocks DC (no path for DC current through the capacitor)
3. The transient response is a damped oscillation at f₀
4. With Q = 2, damping is significant, limiting overshoot

**Physical Interpretation**:
The 1.78V peak represents the filter "ringing" at its natural frequency (39.98 Hz) in response to the sudden input. The energy is stored temporarily in the LC tank but dissipates through R.

### 2. Frequency Response Analysis

The AC simulation reveals the classic bandpass characteristic:

| Frequency | f/f₀ | Magnitude | Gain (dB) | Interpretation |
|-----------|------|-----------|-----------|----------------|
| 4.0 Hz    | 0.10 | 0.0507    | -25.9 dB  | Strong low-frequency attenuation |
| 8.0 Hz    | 0.20 | 0.1042    | -19.6 dB  | Roll-up toward passband |
| 20.0 Hz   | 0.50 | 0.3185    | -9.9 dB   | Entering passband |
| 32.0 Hz   | 0.80 | 0.7467    | -2.5 dB   | Near center, almost at peak |
| **40.0 Hz** | **1.00** | **1.0000** | **0.0 dB** | **Peak response at resonance** |
| 48.0 Hz   | 1.20 | 0.8045    | -1.9 dB   | Still in passband |
| 80.0 Hz   | 2.00 | 0.3168    | -10.0 dB  | Roll-off begins |
| 200 Hz    | 5.00 | 0.1093    | -19.2 dB  | High-frequency attenuation |
| 400 Hz    | 10.0 | 0.0532    | -25.5 dB  | Strong high-frequency attenuation |

**Key Findings**:

1. **Perfect Peak**: At exactly f₀, the gain is 1.0 (0 dB), meaning the output amplitude equals the input amplitude. This is characteristic of this filter topology with moderate Q.

2. **Symmetric Rolloff**: The attenuation at 0.1×f₀ (-25.9 dB) and 10×f₀ (-25.5 dB) are nearly equal. This symmetry (on a log scale) is a fundamental property of RLC bandpass filters.

3. **Rolloff Rate**:
   - From f₀ to 2×f₀: -10 dB (approximately -20 dB/decade)
   - From 0.5×f₀ to f₀: +10 dB (approximately +20 dB/decade)
   - This is first-order behavior on each side

4. **3-dB Points**:
   - Lower: Around 32 Hz (0.8×f₀)
   - Upper: Around 50 Hz (1.25×f₀)
   - Bandwidth: ~18-20 Hz (matches theoretical BW = 20.08 Hz)

### 3. Parameter Sweep: Effect of Resistance

Varying R while keeping L and C constant reveals how resistance controls selectivity:

| R (Ω) | f₀ (Hz) | Q | BW (Hz) | Characteristic |
|-------|---------|---|---------|----------------|
| 50    | 39.98   | 0.40 | 100.4 | Broad, low selectivity |
| 100   | 39.98   | 0.80 | 50.2  | Moderate selectivity |
| 250   | 39.98   | 1.99 | 20.1  | Good selectivity |
| 500   | 39.98   | 3.98 | 10.0  | High selectivity |
| 1000  | 39.98   | 7.96 | 5.0   | Very sharp, narrow band |

**Important Insights**:

1. **Center frequency independence**: f₀ remains constant at 39.98 Hz regardless of R. Resonance is determined only by L and C.

2. **Q scales linearly with R**: Doubling R doubles Q. This is because:
   ```
   Q = R√(C/L)
   ```
   R appears linearly in the expression.

3. **Bandwidth scales inversely**: Doubling R halves the bandwidth. Since BW = f₀/Q, and Q ∝ R, we have BW ∝ 1/R.

4. **Design Trade-off**:
   - High R → High Q → Narrow band → Very selective, but also more loss
   - Low R → Low Q → Wide band → Less selective, but also less lossy

### 4. JAX Differentiability

The gradient computation yielded:

```
d(Vout)/d(C) = -21,859 V/F
```

**Physical Interpretation**:

1. **Sign**: Negative gradient means increasing C decreases Vout (in this particular simulation snapshot at t=50ms).

2. **Magnitude**: The large value (-21,859 V/F) indicates high sensitivity. A 1 nF change in C would alter Vout by about -0.022 V.

3. **Why it matters**: This demonstrates that the circuit is fully differentiable with respect to all component values. This enables:
   - Gradient-based optimization (find C that maximizes some objective)
   - Sensitivity analysis (which component variations matter most?)
   - Parameter identification (fit measured data to find unknown component values)

**Practical Application**:
If you measured the frequency response of a real bandpass filter and wanted to identify its component values, you could:
1. Define an error metric (measured vs. simulated response)
2. Use JAX's automatic differentiation to compute gradients
3. Apply gradient descent to minimize the error
4. Recover R, L, C values that best match the measurements

## Design Guidelines

Based on these findings, here's how to design a bandpass filter for your application:

### Step 1: Choose Center Frequency

Start with the desired center frequency f₀:

```
f₀ = 1/(2π√LC)
```

This gives you one constraint: the product LC must equal `1/(2πf₀)²`.

### Step 2: Choose Quality Factor

Decide how selective you want the filter:

- Q < 1: Very broad, passes wide frequency range
- Q = 1-2: Moderate selectivity (like our example)
- Q = 5-10: High selectivity, narrow band
- Q > 10: Very sharp, specialized applications

### Step 3: Solve for Components

With f₀ and Q chosen, you have two equations:

```
LC = 1/(2πf₀)²
Q = R√(C/L)
```

From these, you can solve for R, L, C. A practical approach:

1. Pick a convenient value for L (e.g., 100 mH)
2. Calculate C from the first equation: `C = 1/((2πf₀)²L)`
3. Calculate R from Q equation: `R = Q√(L/C)`

### Step 4: Verify Component Availability

Check if the calculated values are:
- Physically realizable (not too large or too small)
- Commercially available
- Within cost/size constraints

If not, adjust L and recalculate.

### Example Design: 1 kHz Bandpass, Q=5

Let's design a 1 kHz bandpass filter with Q=5:

```
f₀ = 1000 Hz
Q = 5

Choose L = 10 mH = 0.01 H

C = 1/((2π×1000)² × 0.01)
  = 1/(39,478,418 × 0.01)
  = 2.53 μF

R = 5 × √(0.01 / 2.53×10⁻⁶)
  = 5 × 63.1
  = 315 Ω

Bandwidth: BW = 1000/5 = 200 Hz
```

This filter would pass frequencies from ~900 Hz to ~1100 Hz.

## Implementation Notes

### Time Step Selection

For accurate simulation, the time step must be small enough to resolve:

1. The oscillation period: `dt < period/50` for smooth waveforms
2. Any fast transients in the circuit

The implementation uses:
```python
period = 1 / f_0
dt = period / 50  # 50 samples per period
```

For our 40 Hz filter, this gives dt = 0.5 ms, which is adequate.

### Steady-State Convergence

For AC frequency response, we simulate 10 cycles before measuring:

```python
n_cycles = 10
n_steps = int(n_cycles * period / dt)
```

This ensures:
- Initial transients have decayed
- The circuit has reached steady-state
- Measurements are accurate

For high-Q filters (Q > 10), you may need more cycles (20-30) for full convergence.

### Avoiding Common Pitfalls

1. **Don't make dt depend on swept parameters**: JAX can't trace through dt calculations. Use fixed dt based on the highest frequency of interest.

2. **Collect enough data for amplitude estimation**: Sample at least 2 complete cycles when measuring AC amplitude to get accurate peak-to-peak values.

3. **Use `float()` when extracting values**: JAX arrays need explicit conversion for printing/plotting.

## Physical Insights

### Why Parallel LC?

This topology uses a parallel LC tank. What if we used series LC instead?

**Series LC** (with series R):
- At resonance: Series LC has minimum impedance (acts like short)
- This would create a **notch filter** (band-stop), not a bandpass
- Voltage is minimum at f₀, not maximum

**Parallel LC** (with series R):
- At resonance: Parallel LC has maximum impedance
- Creates voltage divider that passes signals at f₀
- This gives us the bandpass characteristic we want

### Energy Perspective

At resonance, energy oscillates between L and C:
- When current is maximum: Energy stored in L (magnetic field)
- When voltage is maximum: Energy stored in C (electric field)
- The energy sloshes back and forth at f₀

The resistor R dissipates energy on each cycle:
- Higher R → More damping → Lower Q
- Lower R → Less damping → Higher Q

This is why Q relates to the ratio of stored energy to dissipated energy per cycle.

### Real-World Considerations

In practice, you also need to consider:

1. **Component Parasitics**:
   - Inductors have series resistance (ESR)
   - Capacitors have equivalent series resistance and inductance
   - These limit the achievable Q

2. **Loading Effects**:
   - The output impedance is approximately R at resonance
   - If you connect a load < R, it will affect the response
   - Use a buffer amplifier if needed

3. **Temperature Stability**:
   - Inductance and capacitance vary with temperature
   - This shifts f₀ and changes Q
   - Use temperature-stable components for critical applications

## Advanced Topics

### Cascading Filters

To get sharper rolloff, cascade multiple bandpass stages:
- Two stages: 40 dB/decade rolloff
- Three stages: 60 dB/decade rolloff
- But: Each stage adds loss and shifts the overall response

### Active Bandpass Filters

For better performance, use op-amps to create active filters:
- Can achieve higher Q without high-value inductors
- Provide gain (output > input)
- Better isolation between input and output
- But: More complex, require power supply

See `examples/falstad/opamp/` for active filter implementations.

### Parameter Identification

Use JAX autodiff for fitting measured data:

```python
def loss_function(params):
    R, L, C = params
    predicted = simulate_filter(R, L, C, test_frequencies)
    measured = load_measurement_data()
    return jnp.sum((predicted - measured)**2)

# Gradient descent
grad_fn = jax.grad(loss_function)
params = initial_guess
for _ in range(100):
    params -= learning_rate * grad_fn(params)
```

This recovers R, L, C from measured frequency response data.

## Conclusion

The parallel-LC bandpass filter is a fundamental circuit that demonstrates:
- How reactive components create frequency-selective behavior
- The relationship between component values and filter characteristics
- The power of simulation for understanding circuit behavior
- How JAX enables differentiable circuit simulation

Key takeaways:
1. Resonance occurs when `f₀ = 1/(2π√LC)`
2. Quality factor Q determines selectivity: `Q = R√(C/L)`
3. Bandwidth is inversely related to Q: `BW = f₀/Q`
4. The circuit is fully differentiable, enabling gradient-based optimization
5. Time-domain simulation captures both transient and steady-state behavior

This implementation provides a solid foundation for exploring more complex filter designs and using circuit simulation for real-world applications.

## References

- Original Falstad circuit: `bandpass.txt`
- Implementation: `examples/falstad/filters/bandpass_rlc.py`
- Related examples:
  - Low-pass RC: `examples/falstad/filters/lowpass_rc.py`
  - RLC resonance: `examples/falstad/rlc/lrc_resonance.py`
  - Active filters: `examples/falstad/opamp/`

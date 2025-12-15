# Inverting Op-Amp Study: Finite Bandwidth Effects

This document accompanies `inverting_study.py` and explains the modeling approach,
simulation results, and key insights about realistic op-amp behavior.

## Overview

Real operational amplifiers have finite bandwidth, unlike the ideal op-amp model
taught in textbooks. This study demonstrates how to model these effects using
parasitic RC elements, and analyzes both time-domain and frequency-domain behavior.

## Circuit Topology

```
                   Rf (3k)
    +----------[====]----------+
    |                          |
    |   Rin (1k)   Rp    Cp    |
Vin-+--[===]---(-)--[==]--||---+
                |              |
          GND--(+)        [VCVS]---Vout
                               |
                              [Rload]
                               |
                              GND
```

### Components

| Component | Value | Purpose |
|-----------|-------|---------|
| Rin | 1 kΩ | Input resistor (sets gain with Rf) |
| Rf | 3 kΩ | Feedback resistor (gain = -Rf/Rin = -3) |
| Rp | 100 Ω | Parasitic resistance (models wire/internal resistance) |
| Cp | 500 pF | Parasitic capacitance (models op-amp input capacitance) |
| VCVS | A=1000 | Voltage-controlled voltage source (open-loop gain) |
| Rload | 10 kΩ | Load resistance |

### Why RC Dampeners?

The RC elements (Rp and Cp) serve two critical purposes:

1. **Simulation Stability**: Op-amp feedback creates an algebraic loop in the
   circuit equations. The output depends on the inverting input, which depends
   on the output through Rf. Without delay elements, this creates an
   ill-conditioned system. The RC introduces a time delay that breaks this loop.

2. **Physical Realism**: Real op-amps have finite bandwidth due to:
   - Internal compensation capacitors
   - Input capacitance (typically 1-10 pF)
   - Parasitic wire/trace impedance

   The RC models these effects, creating a dominant pole that limits bandwidth.

## Key Parameters

### RC Time Constant

```
τ_RC = Rp × Cp = 100Ω × 500pF = 50 ns
```

### Open-Loop Pole Frequency

```
f_pole = 1 / (2π × τ_RC) = 1 / (2π × 50ns) ≈ 3.18 MHz
```

### Loop Gain

```
β = Rin / (Rin + Rf) = 1000 / 4000 = 0.25
Loop Gain = A_ol × β = 1000 × 0.25 = 250
```

### Gain Error

With finite open-loop gain, the closed-loop gain deviates slightly from ideal:

```
Gain Error = 1 / (1 + Loop Gain) = 1/251 ≈ 0.4%
```

## Time Domain Analysis

### Step Response Results

| Metric | Value |
|--------|-------|
| Expected ideal Vout | -3.000 V |
| Actual final Vout | -2.988 V |
| Gain error | 0.40% |
| Time to 90% | ~4 ns |
| Time to 99% | ~8 ns |
| Overshoot | 0% |

### Why Is Settling So Fast?

The settling time appears much faster than the raw RC time constant (50 ns)
would suggest. This is because **feedback speeds up the response**.

In a closed-loop system with high loop gain:

```
Effective τ ≈ τ_RC / (1 + Loop Gain)
           ≈ 50 ns / 251
           ≈ 0.2 ns
```

The closed-loop bandwidth is extended by the loop gain factor, which
correspondingly reduces the effective time constant.

For a first-order system:
- 90% settling ≈ 2.3 × τ_effective ≈ 0.5 ns (theoretical)
- 99% settling ≈ 4.6 × τ_effective ≈ 1 ns (theoretical)

The measured values are slightly higher due to simulation discretization
and the non-ideal nature of the circuit.

## Frequency Domain Analysis

### Bode Plot Results

| Frequency | Gain (dB) | Phase |
|-----------|-----------|-------|
| 100 kHz | 9.51 | -180° |
| 1 MHz | 9.51 | -181° |
| 10 MHz | 9.49 | -185° |
| 50 MHz | 8.89 | -209° |
| 100 MHz | 6.78 | -238° |
| 200 MHz | 2.50 | -245° |
| 500 MHz | -4.98 | -266° |

### Bandwidth Analysis

| Parameter | Value |
|-----------|-------|
| DC Gain | 9.51 dB (×2.988) |
| -3dB Bandwidth | ~200 MHz |
| Gain-Bandwidth Product | ~600 MHz |

### Why Is Bandwidth >> Pole Frequency?

The open-loop pole is at 3.2 MHz, but the measured -3dB bandwidth is ~200 MHz.
This is a fundamental property of feedback systems:

```
f_closed_loop ≈ f_open_loop × (1 + Loop Gain)
             ≈ 3.2 MHz × 251
             ≈ 800 MHz (theoretical upper bound)
```

The actual bandwidth is lower because:
1. The simple first-order model breaks down at very high frequencies
2. Other parasitic effects become significant
3. The VCVS model has its own limitations

The key insight is that **negative feedback trades gain for bandwidth**.
With a closed-loop gain of 3 (vs open-loop gain of 1000), the bandwidth
is extended by roughly the same factor.

## Comparison: Ideal vs Realistic

| Property | Ideal Op-Amp | This Model |
|----------|--------------|------------|
| Open-loop gain | ∞ | 1000 |
| Bandwidth | ∞ | ~200 MHz (-3dB) |
| Input impedance | ∞ | Finite (Rp, Cp) |
| Settling time | 0 | ~8 ns (99%) |
| Gain accuracy | Perfect | 0.4% error |

## Physical Interpretation

The RC dampener elements model real physical effects:

- **Rp (100Ω)**: Represents cumulative resistance from:
  - PCB trace resistance
  - Internal op-amp resistance
  - Bond wire resistance

- **Cp (500pF)**: Represents cumulative capacitance from:
  - Op-amp input capacitance (typically 1-10 pF for real devices)
  - PCB pad capacitance
  - Stray capacitance

The values used here (100Ω, 500pF) are larger than typical real-world
parasitics to make the bandwidth effects clearly visible in simulation.
Real op-amps might have:
- Input capacitance: 1-10 pF
- Equivalent series resistance: 10-100 Ω
- Resulting pole: 10-100 MHz range

## Simulation Notes

### Timestep Selection

The simulation timestep must be small enough to:
1. Resolve the RC dynamics (dt << τ_RC)
2. Resolve the signal period for AC analysis (dt << T/50)

For stability, we use:
```python
dt = min(period / 50, tau_rc / 100)
```

### Performance Considerations

Low-frequency AC analysis is expensive because:
- dt is constrained by the RC time constant (0.5 ns)
- Period at 100 kHz is 10 μs
- This requires ~20,000 samples per cycle

Typical simulation times:
- 100 kHz: ~20 seconds
- 10 MHz: ~0.4 seconds
- Full sweep (20 points): ~60 seconds

## Conclusions

1. **RC dampeners are essential** for stable time-domain simulation of
   op-amp feedback circuits. They break algebraic loops while adding
   physically meaningful bandwidth limitations.

2. **Feedback extends bandwidth** significantly beyond the open-loop pole.
   The closed-loop bandwidth is approximately (1 + Loop Gain) times the
   open-loop bandwidth.

3. **Feedback speeds up settling** by the same factor. A 50 ns RC time
   constant results in ~8 ns settling time due to the high loop gain.

4. **Gain accuracy** depends on loop gain. With loop gain of 250, the
   gain error is only 0.4%.

5. **The gain-bandwidth product** is approximately constant. Lower
   closed-loop gain yields higher bandwidth, and vice versa.

## References

- `inverting_study.py`: Source code for this analysis
- `inverting.py`: Basic inverting amplifier examples with multiple approaches
- `step_response.png`: Time-domain step response plot
- `frequency_response.png`: Bode plot of frequency response

# Falstad Circuit Simulator - Compatible Circuits

This document lists all Falstad circuit simulator examples that can be implemented
using pyvibrate's available components.

**Source:** `/home/mib07150/git/zfs/git/private/from-source/circuit-simulator/src/circuits/`

**Analysis:** 52 out of 279 circuits (18.6%) are compatible with pyvibrate.

## Element Code Mapping

| Falstad Code | PyVibrate Component | Notes |
|--------------|---------------------|-------|
| `r` | `R` | Resistor |
| `c` | `C` | Capacitor |
| `l` | `L` | Inductor |
| `v` | `VSource` | Voltage source (DC or AC) |
| `170` | `VSource` | AC voltage source (sine wave) |
| `a` | `VCVS` | Op-amp (modeled as ideal VCVS with high gain) |
| `s`, `S` | `Switch` | SPST switch |
| `w` | - | Wire (just connectivity) |
| `g` | `net.gnd` | Ground reference |
| `$` | - | Simulation parameters (metadata) |
| `o`, `O` | - | Oscilloscope/output probe (display only) |
| `h` | - | Hint text (display only) |
| `x` | - | Text annotation (display only) |

## Compatible Circuits by Category

### Filters (11 circuits)

| File | Description |
|------|-------------|
| `filt-lopass.txt` | RC low-pass filter |
| `filt-lopass-l.txt` | RL low-pass filter |
| `filt-hipass.txt` | RC high-pass filter |
| `filt-hipass-l.txt` | RL high-pass filter |
| `bandpass.txt` | RLC bandpass filter |
| `notch.txt` | Twin-T notch filter |
| `allpass1.txt` | First-order allpass filter (op-amp) |
| `allpass2.txt` | Second-order allpass filter (op-amp) |
| `besselbutter.txt` | Bessel vs Butterworth comparison |
| `butter10lo.txt` | 10th order Butterworth low-pass |
| `crossover.txt` | Audio crossover network |

### RLC Circuits (10 circuits)

| File | Description |
|------|-------------|
| `lrc.txt` | Series RLC resonance |
| `lrc-critical.txt` | Critically damped RLC |
| `inductkick.txt` | Inductor voltage spike when switched |
| `inductkick-snub.txt` | Inductor kick with snubber |
| `induct.txt` | Basic inductor behavior |
| `inductac.txt` | Inductor with AC source |
| `capac.txt` | Basic capacitor behavior |
| `cap.txt` | Capacitor charging |
| `ringing.txt` | RLC ringing/oscillation |
| `impedance.txt` | Impedance demonstration |

### Op-Amp Circuits (9 circuits)

| File | Description |
|------|-------------|
| `amp-invert.txt` | Inverting amplifier |
| `amp-noninvert.txt` | Non-inverting amplifier |
| `amp-follower.txt` | Voltage follower (unity gain buffer) |
| `amp-dfdx.txt` | Differentiator |
| `amp-sum.txt` | Summing amplifier |
| `capmult.txt` | Capacitance multiplier |
| `gyrator.txt` | Gyrator (simulates inductor with cap) |
| `howland.txt` | Howland current source |
| `nic-r.txt` | Negative impedance converter |

### Switching Networks (8 circuits)

| File | Description |
|------|-------------|
| `3way.txt` | 3-way switch |
| `4way.txt` | 4-way switch |
| `resistors.txt` | Resistor network |
| `indseries.txt` | Inductors in series |
| `indpar.txt` | Inductors in parallel |
| `capseries.txt` | Capacitors in series |
| `cappar.txt` | Capacitors in parallel |
| `mr-crossbar.txt` | Crossbar switch network |

### Impedance/Basic Networks (7 circuits)

| File | Description |
|------|-------------|
| `voltdivide.txt` | Voltage divider |
| `thevenin.txt` | Thevenin equivalent demonstration |
| `wheatstone.txt` | Wheatstone bridge |
| `powerfactor1.txt` | Power factor correction |
| `powerfactor2.txt` | Power factor correction (variant) |
| `res-series.txt` | Resistors in series |
| `res-par.txt` | Resistors in parallel |

### Oscillators (4 circuits)

| File | Description |
|------|-------------|
| `sine.txt` | Sine wave oscillator |
| `triangle.txt` | Triangle wave oscillator |
| `relaxosc.txt` | Relaxation oscillator |
| `phaseshiftosc.txt` | Phase shift oscillator |

### Coupled/Reactive Networks (4 circuits)

| File | Description |
|------|-------------|
| `coupled1.txt` | Coupled LC circuits |
| `coupled3.txt` | Three coupled LC circuits |
| `ladder.txt` | LC ladder network |
| `phaseseq.txt` | Phase sequence network |

### Multipliers/Converters (4 circuits)

| File | Description |
|------|-------------|
| `indmultind.txt` | Inductance multiplier |
| `indmultfreq.txt` | Inductance multiplier (frequency demo) |
| `capmultcaps.txt` | Capacitance multiplier (caps) |
| `capmultfreq.txt` | Capacitance multiplier (frequency demo) |

### Other (5 circuits)

| File | Description |
|------|-------------|
| `diff.txt` | Differential circuit |
| `twint.txt` | Twin-T network |
| `grid.txt` | Resistor grid |
| `grid2.txt` | Resistor grid (variant) |
| `blank.txt` | Empty circuit template |

## Incompatible Circuits (227)

These circuits require components not available in pyvibrate:

| Blocking Element | Count | Examples |
|------------------|-------|----------|
| Transistors (BJT `t`, MOSFET `f`) | 80 | `ceamp.txt`, `cmosinverter.txt` |
| Logic elements (`L`, `I`, gates) | 74 | `counter.txt`, `7segdecoder.txt` |
| Diodes (`d`, `z`) | 31 | `diodeclip.txt`, `fullrect.txt` |
| Potentiometers (`p`) | 21 | `pot.txt` |
| Current conveyors (`172`) | 15 | `cc2.txt` |
| 555 Timer IC | 9 | `555square.txt`, `555monostable.txt` |
| Transformers | 6 | `transformer.txt` |
| Current sources (`i`) | various | `currentsrc.txt` |

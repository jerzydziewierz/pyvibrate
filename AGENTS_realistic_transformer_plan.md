# Realistic Transformer Component Implementation Plan

## Overview

Implement a physically-accurate transformer component for PyVibrate's frequency domain solver with:
- Physical parameter interface (N1, N2, A_core, l_m, mu_r, B_sat)
- Realistic equivalent circuit (magnetizing inductance, core loss, leakage, winding resistance)
- Linear operation with saturation warnings
- Comprehensive VNA parameter extraction notebook

## Implementation Approach

### Component Architecture: Single Component with Internal Y-Parameter Stamping

**Rationale:** Simplest user interface, optimal for parameter fitting, JAX-compatible, follows existing TLine pattern.

**Equivalent Circuit Model:**
```
Primary:  n_p_pos ──[R1]──[L_leak1]──┬──[L_mag║R_core]──║ n:1 ║──┬──[L_leak2]──[R2]── n_s_pos
                                      │                             │
Secondary:  n_p_neg ──────────────────┴─────────────────║     ║──┴────────────────── n_s_neg
```

### Physical Parameters → Circuit Parameters
- `L1 = μ₀ * μᵣ * N1² * A_core / l_m` (primary inductance)
- `L2 = μ₀ * μᵣ * N2² * A_core / l_m` (secondary inductance)
- `M = k * sqrt(L1 * L2)` (mutual inductance, k = coupling coefficient)
- `L_mag = k * L1` (magnetizing inductance)
- `L_leak1 = (1-k) * L1`, `L_leak2 = (1-k) * L2` (leakage inductances)

### MNA Stamping Strategy

Transform equivalent circuit to 2-port Z-parameters:
```
Z11 = R1 + jωL_leak1 + (jωL_mag ║ R_core)
Z22 = R2 + jωL_leak2
Z12 = Z21 = jωM
```

Convert to Y-parameters via 2×2 matrix inversion:
```
det = Z11*Z22 - Z12*Z21
Y11 = Z22/det, Y12 = -Z12/det
Y21 = -Z21/det, Y22 = Z11/det
```

Stamp using existing `_stamp_2port_admittance()` helper.

### Saturation Warning (Post-Solve)

After each solve, optionally check saturation:
```python
def check_transformer_saturation(solution, transformer_ref, params):
    # Estimate magnetizing current from primary voltage
    I_mag = V_prim / (jωL_mag)

    # Compute flux density
    B = μ₀ * μᵣ * N1 * |I_mag| / l_m

    # Compare to B_sat
    return {
        "B_current": B,
        "B_sat": B_sat,
        "utilization": B/B_sat,
        "warning": B/B_sat > 0.8
    }
```

**Note:** Approximation only - doesn't account for load current. Document as order-of-magnitude check.

## Files to Create/Modify

### New Files

1. **`pyvibrate/frequencydomain/physical_constants.py`** (~100 lines)
   - Physical constants: `MU_0 = 4π×10⁻⁷ H/m`
   - Core material database (ferrite, powder iron, silicon steel)
   - Helper functions:
     - `compute_inductance(N, A_core, l_m, mu_r) -> L`
     - `compute_flux_density(I, N, l_m, mu_r) -> B`
     - `tanh_core_model(H, H_c, B_sat)` (reference for future nonlinear work)

2. **`tests/test_freqdomain_transformer.py`** (~300 lines)
   - 10 unit tests covering:
     - Turns ratio (n:1 voltage transformation)
     - Impedance transformation (Z_in = n²*Z_load)
     - Open circuit test (measure L_mag)
     - Short circuit test (measure L_leak)
     - Coupling coefficient effect
     - Core loss effect on efficiency
     - Physical parameter computation
     - Saturation warning logic
     - JAX differentiability
     - Frequency response

3. **`notebooks/demo_transformer_vna_extraction.ipynb`** (~500 lines)
   - **Section 1: Open Circuit Test**
     - Theory: secondary open, Z_in ≈ R1 + jωL_mag + R_core effects
     - Generate synthetic measurement data
     - Extract L_mag and R_core from frequency sweep
     - Plot impedance magnitude and phase vs frequency

   - **Section 2: Short Circuit Test**
     - assume the turn count is known on both sides
     - Theory: secondary shorted, Z_in ≈ R1 + R2/n² + jω(L_leak1 + L_leak2/n²)
     - Extract total leakage inductance and winding resistance
     - Compare measured to computed from coupling coefficient k
     - Assume the turn count is known on both sides

   - **Section 3: Loaded Test**
     - assume the turn count is known on both sides
     - Sweep multiple load resistances (10Ω to 500Ω)
     - Measure Z_in(R_load) at fixed frequency
     - Compute coupling coefficient k from measurements

   - **Section 4: Inductance vs DC Bias (Saturation Curve)**
     - Simulate L(I) using reduced μᵣ at each DC bias point
     - Fit to tanh model: `B = B_sat * tanh(H/H_c)`
     - Extract B_sat and H_c parameters
     - Plot L vs I showing saturation onset
     - **Note:** Demonstrates characterization concept even though solver is linear

### Modified Files

1. **`pyvibrate/frequencydomain/components.py`** (+80 lines)
   ```python
   def Transformer(
       net: Network,
       prim_pos: Node, prim_neg: Node,
       sec_pos: Node, sec_neg: Node,
       *,
       name: str,
       # Physical parameters
       N1: float | None = None,        # Primary turns
       N2: float | None = None,        # Secondary turns
       A_core: float | None = None,    # Core area (m²)
       l_m: float | None = None,       # Mean path length (m)
       mu_r: float | None = None,      # Relative permeability
       B_sat: float | None = None,     # Saturation flux (T)
       # Circuit parameters
       k: float = 0.98,                # Coupling coefficient
       R1: float | None = None,        # Primary resistance
       R2: float | None = None,        # Secondary resistance
       R_core: float | None = None,    # Core loss resistance
   ) -> tuple[Network, ComponentRef]:
   ```

   Store all parameters in defaults tuple using suffix pattern: `{name}_N1`, `{name}_A_core`, etc.

2. **`pyvibrate/frequencydomain/solver.py`** (+60 lines)
   - Add `"Transformer"` case to `_stamp_component()`:
     - Extract physical parameters from params dict
     - Compute L1, L2, M from N1, N2, A_core, l_m, mu_r
     - Build Z-parameters with series/parallel elements
     - Convert to Y-parameters (2×2 inversion with det safety check)
     - Stamp using `_stamp_2port_admittance()`

   - Add `check_transformer_saturation()` helper to Solver class:
     - Takes (solution, component_ref, params) → dict
     - Returns saturation metrics and warning flags
     - Non-invasive, optional post-solve check

3. **`pyvibrate/frequencydomain/__init__.py`** (+2 lines)
   - Export `Transformer` in `__all__`
   - Export `physical_constants` module

## Critical Implementation Details

### Edge Case Handling

1. **DC operation (ω=0):**
   ```python
   omega_safe = jnp.where(omega < 1e-6, 1e-6, omega)
   ```

2. **Determinant safety (at resonance):**
   ```python
   det = Z11*Z22 - Z12*Z21
   det_safe = jnp.where(jnp.abs(det) < 1e-15, 1e-15, det)
   ```

3. **Coupling coefficient bounds:**
   ```python
   k = jnp.clip(k, 0.5, 0.9999)
   ```

### JAX Differentiability

- Use `jnp.where()` for all conditionals (no Python `if`)
- All computations in stamping loop (not component factory)
- Test gradients w.r.t. all physical parameters

### Parameter Extraction Strategy (VNA Notebook)

**Sequential extraction avoids ill-conditioning:**
1. Open circuit → L_mag (independent)
2. Short circuit → L_leak_total (independent)
3. Loaded test → n, k (coupled but constrained)
4. Compute R_core, R1, R2 from loss measurements

**Better than full gradient descent:** Each parameter isolated to specific test.

## Testing Strategy

### Unit Test Hierarchy
1. **Basic transformer behavior** (turns ratio, impedance transform)
2. **VNA test validation** (open, short, loaded match theory)
3. **Physical parameter computation** (N, A → L correct)
4. **Edge cases** (DC, high frequency, saturation)
5. **JAX compatibility** (gradients finite, vmap works)

### Validation Sources
- Hand-calculated Z-parameters for simple cases
- Published transformer datasheets (compare to real devices)
- Cross-check with LTspice (if needed for complex cases)

## Implementation Sequence

### Phase 1: Core Component (Priority 1)
1. Create `physical_constants.py` with μ₀, compute functions
2. Add `Transformer()` factory to `components.py`
3. Add stamping case to `solver.py`
4. Update `__init__.py` exports
5. Write 3 basic tests (turns ratio, impedance, open circuit)

### Phase 2: Testing & Validation (Priority 1)
6. Write remaining 7 unit tests
7. Validate against hand calculations
8. Fix bugs, stabilize numerics

### Phase 3: Saturation Warning (Priority 2)
9. Implement `check_transformer_saturation()`
10. Add saturation test
11. Document approximations and limitations

### Phase 4: VNA Notebook (Priority 2)
12. Create notebook skeleton with 4 sections
13. Implement open circuit extraction
14. Implement short circuit extraction
15. Implement loaded test extraction
16. Implement saturation curve extraction
17. Add explanatory markdown, theory sections

### Phase 5: Documentation (Priority 3)
18. Complete all docstrings
19. Add transformer example to README
20. Create simple application example (e.g., flyback converter)

## Success Criteria

**Functional:**
- [ ] Transformer compiles and solves at all frequencies
- [ ] All 10 unit tests pass with <1% error
- [ ] JAX gradients finite for all parameters
- [ ] VNA notebook runs end-to-end
- [ ] Parameter extraction accuracy <5% on synthetic data

**Quality:**
- [ ] Follows existing PyVibrate patterns (factory functions, stamping)
- [ ] No performance regression (solve time <2× baseline)
- [ ] Docstrings complete with equations
- [ ] Notebook is tutorial-quality with clear explanations

**Documentation:**
- [ ] Equivalent circuit diagram in docstring
- [ ] Physical parameter formulas documented
- [ ] VNA test theory explained
- [ ] Saturation warning limitations noted

## Known Limitations (to Document)

1. **Linear solver only** - saturation effects not modeled in solve loop
2. **Saturation warning is approximate** - uses primary voltage, ignores load current
3. **No hysteresis** - single-valued B-H curve (tanh model)
4. **Frequency domain only** - no time-domain transformer yet
5. **Ideal coupling** - no frequency-dependent losses or skin effect

These are acceptable for the initial implementation and clearly documented.

## Reference Files

Key files for implementation patterns:
- `/home/mib07150/git/zfs/git/private/20251203-pyvibrate/pyvibrate/frequencydomain/solver.py:264-283` - TLine Y-parameter stamping
- `/home/mib07150/git/zfs/git/private/20251203-pyvibrate/pyvibrate/frequencydomain/components.py:140-165` - ConstantTimeDelayVCVS multi-parameter pattern
- `/home/mib07150/git/zfs/git/private/20251203-pyvibrate/tests/test_freqdomain_l.py` - Inductor test patterns
- `/home/mib07150/git/zfs/git/private/20251203-pyvibrate/notebooks/demo_freqdomain_fitting.ipynb` - Parameter extraction patterns

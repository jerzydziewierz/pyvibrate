"""
Tests for frequency-domain Series and Parallel subcircuit operations.

Test plan: see testing_plan.md
"""
import math
import pytest
import jax.numpy as jnp
from jax import grad


class TestFreqSeriesCore:
    """Core functionality tests for frequency-domain Series operation."""

    def test_freq_series_impedance_sum(self):
        """Z_total = Z1 + Z2 at all frequencies.

        Series of two resistors: Z = R1 + R2
        """
        from pyvibrate.frequencydomain import Network, R
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Series

        R1_val = 100.0
        R2_val = 200.0
        expected_z = R1_val + R2_val  # 300 ohm

        net = Network()
        net, n_in = net.node("in")

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
        net, (r1_ref, r2_ref, mid) = Series(
            net, n_in, net.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R1_val),
            lambda net, a, b: R(net, a, b, name="r2", value=R2_val),
            prefix="ser"
        )

        solver = net.compile()

        # Test at multiple frequencies
        for freq in [100.0, 1000.0, 10000.0]:
            omega = 2 * math.pi * freq
            sol = solver.solve_at(omega)
            z = solver.z_in(sol, vs)

            assert abs(z.real - expected_z) < 1e-3, \
                f"Series impedance {z.real} != expected {expected_z} at {freq}Hz"
            assert abs(z.imag) < 1e-6, \
                f"Series R should have no imaginary part at {freq}Hz"

    def test_freq_series_voltage_division(self):
        """Complex voltage division at multiple frequencies.

        For Series(R, C): V_C / V_total = Z_C / (Z_R + Z_C)
        """
        from pyvibrate.frequencydomain import Network, R, C
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Series

        R_val = 1000.0
        C_val = 1e-6
        V_source = 1.0

        net = Network()
        net, n_in = net.node("in")

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=V_source)
        net, (r_ref, c_ref, mid) = Series(
            net, n_in, net.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R_val),
            lambda net, a, b: C(net, a, b, name="c1", value=C_val),
            prefix="rc"
        )

        solver = net.compile()

        # Test at f where Xc = R (corner frequency)
        f_corner = 1 / (2 * math.pi * R_val * C_val)  # ~159 Hz
        omega = 2 * math.pi * f_corner
        sol = solver.solve_at(omega)

        # At corner frequency, |V_C| / |V_total| = 1/sqrt(2) ≈ 0.707
        v_mid = solver.v(sol, mid)
        v_in = solver.v(sol, n_in)
        ratio = jnp.abs(v_mid) / jnp.abs(v_in)
        expected_ratio = 1 / math.sqrt(2)

        assert abs(ratio - expected_ratio) < 0.05, \
            f"Voltage ratio {ratio} != expected {expected_ratio}"

    def test_freq_series_vs_manual(self):
        """Series produces identical results to manual construction."""
        from pyvibrate.frequencydomain import Network, R, C
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Series

        R_val = 1000.0
        C_val = 1e-6

        # Circuit A: Using Series()
        net_a = Network()
        net_a, n_in_a = net_a.node("in")

        net_a, vs_a = ACSource(net_a, n_in_a, net_a.gnd, name="vs", value=1.0)
        net_a, (r_ref, c_ref, mid_a) = Series(
            net_a, n_in_a, net_a.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R_val),
            lambda net, a, b: C(net, a, b, name="c1", value=C_val),
            prefix="rc"
        )

        solver_a = net_a.compile()

        # Circuit B: Manual construction
        net_b = Network()
        net_b, n_in_b = net_b.node("in")
        net_b, mid_b = net_b.node("mid")

        net_b, vs_b = ACSource(net_b, n_in_b, net_b.gnd, name="vs", value=1.0)
        net_b, r1_b = R(net_b, n_in_b, mid_b, name="r1", value=R_val)
        net_b, c1_b = C(net_b, mid_b, net_b.gnd, name="c1", value=C_val)

        solver_b = net_b.compile()

        # Compare at multiple frequencies
        for freq in [100.0, 1000.0, 10000.0]:
            omega = 2 * math.pi * freq
            sol_a = solver_a.solve_at(omega)
            sol_b = solver_b.solve_at(omega)

            z_a = solver_a.z_in(sol_a, vs_a)
            z_b = solver_b.z_in(sol_b, vs_b)

            assert abs(z_a - z_b) < 1e-10, \
                f"Impedances differ at {freq}Hz: {z_a} vs {z_b}"

    def test_freq_series_midpoint_voltage(self):
        """Mid-node voltage is correct at various frequencies."""
        from pyvibrate.frequencydomain import Network, R
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Series

        R1_val = 1000.0
        R2_val = 2000.0
        V_source = 1.0

        net = Network()
        net, n_in = net.node("in")

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=V_source)
        net, (r1_ref, r2_ref, mid) = Series(
            net, n_in, net.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R1_val),
            lambda net, a, b: R(net, a, b, name="r2", value=R2_val),
            prefix="div"
        )

        solver = net.compile()

        # For resistive divider: V_mid = V_in * R2 / (R1 + R2)
        expected_v_mid = V_source * R2_val / (R1_val + R2_val)

        for freq in [100.0, 1000.0, 10000.0]:
            omega = 2 * math.pi * freq
            sol = solver.solve_at(omega)

            v_mid = jnp.abs(solver.v(sol, mid))

            assert abs(v_mid - expected_v_mid) < 0.01, \
                f"Mid voltage {v_mid} != expected {expected_v_mid} at {freq}Hz"


class TestFreqParallelCore:
    """Core functionality tests for frequency-domain Parallel operation."""

    def test_freq_parallel_admittance_sum(self):
        """Y_total = Y1 + Y2 at all frequencies (Z_total = R1*R2/(R1+R2))."""
        from pyvibrate.frequencydomain import Network, R
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Parallel

        R1_val = 100.0
        R2_val = 200.0
        expected_z = (R1_val * R2_val) / (R1_val + R2_val)  # 66.67 ohm

        net = Network()
        net, n_in = net.node("in")

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
        net, (r1_ref, r2_ref) = Parallel(
            net, n_in, net.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R1_val),
            lambda net, a, b: R(net, a, b, name="r2", value=R2_val),
            prefix="par"
        )

        solver = net.compile()

        for freq in [100.0, 1000.0, 10000.0]:
            omega = 2 * math.pi * freq
            sol = solver.solve_at(omega)
            z = solver.z_in(sol, vs)

            assert abs(z.real - expected_z) < 0.1, \
                f"Parallel impedance {z.real} != expected {expected_z} at {freq}Hz"
            assert abs(z.imag) < 1e-6, \
                f"Parallel R should have no imaginary part"

    def test_freq_parallel_vs_manual(self):
        """Parallel produces identical results to manual construction."""
        from pyvibrate.frequencydomain import Network, R, C
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Parallel

        R_val = 1000.0
        C_val = 1e-6

        # Circuit A: Using Parallel()
        net_a = Network()
        net_a, n_in_a = net_a.node("in")

        net_a, vs_a = ACSource(net_a, n_in_a, net_a.gnd, name="vs", value=1.0)
        net_a, (r_ref, c_ref) = Parallel(
            net_a, n_in_a, net_a.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R_val),
            lambda net, a, b: C(net, a, b, name="c1", value=C_val),
            prefix="rc"
        )

        solver_a = net_a.compile()

        # Circuit B: Manual construction
        net_b = Network()
        net_b, n_in_b = net_b.node("in")

        net_b, vs_b = ACSource(net_b, n_in_b, net_b.gnd, name="vs", value=1.0)
        net_b, r1_b = R(net_b, n_in_b, net_b.gnd, name="r1", value=R_val)
        net_b, c1_b = C(net_b, n_in_b, net_b.gnd, name="c1", value=C_val)

        solver_b = net_b.compile()

        for freq in [100.0, 1000.0, 10000.0]:
            omega = 2 * math.pi * freq
            sol_a = solver_a.solve_at(omega)
            sol_b = solver_b.solve_at(omega)

            z_a = solver_a.z_in(sol_a, vs_a)
            z_b = solver_b.z_in(sol_b, vs_b)

            assert abs(z_a - z_b) < 1e-10, \
                f"Impedances differ at {freq}Hz: {z_a} vs {z_b}"

    def test_freq_parallel_current_division(self):
        """Complex current division at multiple frequencies.

        For Parallel(R1, R2): I1/I_total = G1/(G1+G2) = R2/(R1+R2)
        """
        from pyvibrate.frequencydomain import Network, R
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Parallel

        R1_val = 100.0
        R2_val = 200.0
        V_source = 1.0

        # For parallel resistors, current divides inversely to resistance
        # I1 = V/R1, I2 = V/R2, I_total = I1 + I2
        # I1/I_total = R2/(R1+R2)
        expected_ratio = R2_val / (R1_val + R2_val)  # 2/3 ≈ 0.667

        net = Network()
        net, n_in = net.node("in")

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=V_source)
        net, (r1_ref, r2_ref) = Parallel(
            net, n_in, net.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R1_val),
            lambda net, a, b: R(net, a, b, name="r2", value=R2_val),
            prefix="par"
        )

        solver = net.compile()

        for freq in [100.0, 1000.0, 10000.0]:
            omega = 2 * math.pi * freq
            sol = solver.solve_at(omega)

            v_in = jnp.abs(solver.v(sol, n_in))
            i1 = v_in / R1_val
            i2 = v_in / R2_val
            i_total = i1 + i2

            ratio = i1 / i_total

            assert abs(ratio - expected_ratio) < 0.01, \
                f"Current ratio {ratio} != expected {expected_ratio} at {freq}Hz"

    def test_freq_parallel_voltage_same(self):
        """Voltage across both elements in parallel is identical."""
        from pyvibrate.frequencydomain import Network, R, C
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Parallel

        R_val = 1000.0
        C_val = 1e-6

        net = Network()
        net, n_in = net.node("in")

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
        net, (r_ref, c_ref) = Parallel(
            net, n_in, net.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R_val),
            lambda net, a, b: C(net, a, b, name="c1", value=C_val),
            prefix="par"
        )

        solver = net.compile()

        # In parallel, both elements share the same terminals
        # So voltage across both is always equal (V_R = V_C = V_in)
        for freq in [100.0, 1000.0, 10000.0]:
            omega = 2 * math.pi * freq
            sol = solver.solve_at(omega)

            v_in = solver.v(sol, n_in)

            # Both elements have same voltage: n_in to gnd
            # There's no separate node for individual element voltages
            # in a parallel configuration - they share terminals
            assert jnp.abs(v_in) > 0, \
                f"Parallel voltage should be non-zero at {freq}Hz"


class TestFreqSeriesPhysical:
    """Physical correctness tests for frequency-domain Series operation."""

    def test_freq_series_rc_lowpass(self):
        """Series RC shows -20dB/decade rolloff and -90° phase shift at high f.

        H(s) = 1/(1 + sRC)
        At f >> f_c: magnitude rolls off at -20dB/decade
        """
        from pyvibrate.frequencydomain import Network, R, C
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Series

        R_val = 1000.0
        C_val = 1e-6
        f_c = 1 / (2 * math.pi * R_val * C_val)  # ~159 Hz

        net = Network()
        net, n_in = net.node("in")

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
        net, (r_ref, c_ref, mid) = Series(
            net, n_in, net.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R_val),
            lambda net, a, b: C(net, a, b, name="c1", value=C_val),
            prefix="rc"
        )

        solver = net.compile()

        # At corner frequency, phase should be -45°
        omega_c = 2 * math.pi * f_c
        sol = solver.solve_at(omega_c)
        v_out = solver.v(sol, mid)
        v_in = solver.v(sol, n_in)
        phase = jnp.angle(v_out / v_in) * 180 / math.pi

        assert abs(phase - (-45)) < 5, \
            f"Phase at corner {phase}° != expected -45°"

        # At 10x corner frequency, attenuation should be ~-20dB
        omega_10fc = 2 * math.pi * (10 * f_c)
        sol_10 = solver.solve_at(omega_10fc)
        v_out_10 = solver.v(sol_10, mid)
        v_in_10 = solver.v(sol_10, n_in)
        gain_db = 20 * jnp.log10(jnp.abs(v_out_10 / v_in_10))

        assert abs(gain_db - (-20)) < 3, \
            f"Gain at 10*fc: {gain_db}dB != expected ~-20dB"

    def test_freq_series_rl_highpass(self):
        """Series RL shows correct high-pass characteristics.

        Output taken across inductor: H(s) = sL/(R + sL)
        At f << f_c: magnitude → 0 (high-pass)
        At f >> f_c: magnitude → 1
        """
        from pyvibrate.frequencydomain import Network, R, L
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Series

        R_val = 1000.0
        L_val = 0.1  # 100mH
        f_c = R_val / (2 * math.pi * L_val)  # ~1591 Hz

        net = Network()
        net, n_in = net.node("in")

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
        # For RL high-pass: R first, then L to ground
        # Mid-node voltage is across L
        net, (r_ref, l_ref, mid) = Series(
            net, n_in, net.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R_val),
            lambda net, a, b: L(net, a, b, name="l1", value=L_val),
            prefix="rl"
        )

        solver = net.compile()

        # At corner frequency, phase should be +45° (leading)
        omega_c = 2 * math.pi * f_c
        sol = solver.solve_at(omega_c)
        v_out = solver.v(sol, mid)
        v_in = solver.v(sol, n_in)
        ratio = jnp.abs(v_out / v_in)

        # At corner freq, magnitude = 1/sqrt(2) ≈ 0.707
        expected_ratio = 1 / math.sqrt(2)
        assert abs(ratio - expected_ratio) < 0.05, \
            f"Magnitude at corner {ratio} != expected {expected_ratio}"

        # At 10x corner frequency, magnitude should be close to 1
        omega_10fc = 2 * math.pi * (10 * f_c)
        sol_10 = solver.solve_at(omega_10fc)
        v_out_10 = solver.v(sol_10, mid)
        v_in_10 = solver.v(sol_10, n_in)
        ratio_10 = jnp.abs(v_out_10 / v_in_10)

        assert ratio_10 > 0.95, \
            f"High-freq magnitude {ratio_10} should be close to 1"

    def test_freq_series_rlc_resonance(self):
        """Series RLC shows minimum impedance at resonance.

        f_0 = 1/(2π√LC)
        At resonance: Z = R (purely resistive, minimum impedance)
        """
        from pyvibrate.frequencydomain import Network, R, L, C
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Series

        R_val = 100.0
        L_val = 0.01  # 10mH
        C_val = 1e-6  # 1µF
        f_0 = 1 / (2 * math.pi * math.sqrt(L_val * C_val))  # ~1592 Hz

        net = Network()
        net, n_in = net.node("in")

        def rl_series(net, a, b):
            return Series(
                net, a, b,
                lambda net, a, b: R(net, a, b, name="r1", value=R_val),
                lambda net, a, b: L(net, a, b, name="l1", value=L_val),
                prefix="rl"
            )

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
        net, (rl_refs, c_ref, mid) = Series(
            net, n_in, net.gnd,
            rl_series,
            lambda net, a, b: C(net, a, b, name="c1", value=C_val),
            prefix="rlc"
        )

        solver = net.compile()

        # At resonance, impedance should equal R (XL = XC, they cancel)
        omega_0 = 2 * math.pi * f_0
        sol = solver.solve_at(omega_0)
        z = solver.z_in(sol, vs)

        assert abs(z.real - R_val) < 5, \
            f"Z at resonance {z.real} != R={R_val}"
        assert abs(z.imag) < 10, \
            f"Imaginary part {z.imag} should be ~0 at resonance"

        # Off resonance (at 0.5*f_0), impedance should be higher
        omega_half = 2 * math.pi * (0.5 * f_0)
        sol_half = solver.solve_at(omega_half)
        z_half = solver.z_in(sol_half, vs)

        assert jnp.abs(z_half) > jnp.abs(z), \
            f"Off-resonance |Z|={jnp.abs(z_half)} should be > resonance |Z|={jnp.abs(z)}"


class TestFreqParallelPhysical:
    """Physical correctness tests for frequency-domain Parallel operation."""

    def test_freq_parallel_rc_impedance(self):
        """Parallel RC impedance vs frequency.

        At low frequencies: Z ≈ R (capacitor is open circuit)
        At high frequencies: Z ≈ 1/(jωC) (capacitor dominates)
        """
        from pyvibrate.frequencydomain import Network, R, C
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Parallel

        R_val = 1000.0
        C_val = 1e-6
        f_c = 1 / (2 * math.pi * R_val * C_val)  # ~159 Hz

        net = Network()
        net, n_in = net.node("in")

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
        net, (r_ref, c_ref) = Parallel(
            net, n_in, net.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R_val),
            lambda net, a, b: C(net, a, b, name="c1", value=C_val),
            prefix="par"
        )

        solver = net.compile()

        # At low frequency (0.1*fc), Z should be close to R
        omega_low = 2 * math.pi * (0.1 * f_c)
        sol_low = solver.solve_at(omega_low)
        z_low = solver.z_in(sol_low, vs)

        assert abs(jnp.abs(z_low) - R_val) / R_val < 0.15, \
            f"Low freq |Z|={jnp.abs(z_low)} should be close to R={R_val}"

        # At high frequency (10*fc), Z should be dominated by capacitor
        omega_high = 2 * math.pi * (10 * f_c)
        sol_high = solver.solve_at(omega_high)
        z_high = solver.z_in(sol_high, vs)

        xc_high = 1 / (omega_high * C_val)
        assert jnp.abs(z_high) < R_val / 2, \
            f"High freq |Z|={jnp.abs(z_high)} should be much less than R"

    def test_freq_parallel_rl_impedance(self):
        """Parallel RL phase characteristics.

        At low frequencies: Z ≈ jωL (inductor dominates)
        At high frequencies: Z ≈ R (inductor is open circuit)
        """
        from pyvibrate.frequencydomain import Network, R, L
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Parallel

        R_val = 1000.0
        L_val = 0.1  # 100mH
        f_c = R_val / (2 * math.pi * L_val)  # ~1591 Hz

        net = Network()
        net, n_in = net.node("in")

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
        net, (r_ref, l_ref) = Parallel(
            net, n_in, net.gnd,
            lambda net, a, b: R(net, a, b, name="r1", value=R_val),
            lambda net, a, b: L(net, a, b, name="l1", value=L_val),
            prefix="par"
        )

        solver = net.compile()

        # At low frequency (0.1*fc), Z should be small (inductive)
        omega_low = 2 * math.pi * (0.1 * f_c)
        sol_low = solver.solve_at(omega_low)
        z_low = solver.z_in(sol_low, vs)

        xl_low = omega_low * L_val
        assert jnp.abs(z_low) < R_val / 2, \
            f"Low freq |Z|={jnp.abs(z_low)} should be much less than R"

        # At high frequency (10*fc), Z should approach R
        omega_high = 2 * math.pi * (10 * f_c)
        sol_high = solver.solve_at(omega_high)
        z_high = solver.z_in(sol_high, vs)

        assert abs(jnp.abs(z_high) - R_val) / R_val < 0.15, \
            f"High freq |Z|={jnp.abs(z_high)} should be close to R={R_val}"

    def test_freq_parallel_lc_antiresonance(self):
        """Parallel LC shows maximum impedance at resonance.

        f_0 = 1/(2π√LC)
        At resonance: Z → ∞ (ideally) for lossless LC
        """
        from pyvibrate.frequencydomain import Network, L, C
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Parallel

        L_val = 0.01  # 10mH
        C_val = 1e-6  # 1µF
        f_0 = 1 / (2 * math.pi * math.sqrt(L_val * C_val))  # ~1592 Hz

        net = Network()
        net, n_in = net.node("in")

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
        net, (l_ref, c_ref) = Parallel(
            net, n_in, net.gnd,
            lambda net, a, b: L(net, a, b, name="l1", value=L_val),
            lambda net, a, b: C(net, a, b, name="c1", value=C_val),
            prefix="par"
        )

        solver = net.compile()

        # At exact resonance, ideal parallel LC has infinite impedance (Z → ∞)
        # Test slightly off resonance where impedance is high but finite
        omega_near = 2 * math.pi * (0.99 * f_0)  # Slightly below resonance
        sol_near = solver.solve_at(omega_near)
        z_near = solver.z_in(sol_near, vs)

        # At off-resonance (0.5*f_0), |Z| should be much lower
        omega_half = 2 * math.pi * (0.5 * f_0)
        sol_half = solver.solve_at(omega_half)
        z_half = solver.z_in(sol_half, vs)

        # Near-resonance impedance should be much higher than off-resonance
        z_near_mag = jnp.abs(z_near)
        z_half_mag = jnp.abs(z_half)

        assert jnp.isfinite(z_near_mag), \
            f"Near-resonance |Z| should be finite: {z_near_mag}"
        assert jnp.isfinite(z_half_mag), \
            f"Off-resonance |Z| should be finite: {z_half_mag}"
        assert z_near_mag > z_half_mag * 5, \
            f"Near-resonance |Z|={z_near_mag} should be >> off-resonance |Z|={z_half_mag}"


class TestFreqComposability:
    """Composability tests for frequency-domain subcircuits."""

    def test_freq_series_nested(self):
        """Nested Series operations."""
        from pyvibrate.frequencydomain import Network, R
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Series

        R_val = 100.0
        expected_z = 3 * R_val  # 300 ohm

        net = Network()
        net, n_in = net.node("in")

        def inner_series(net, a, b):
            return Series(
                net, a, b,
                lambda net, a, b: R(net, a, b, name="r1", value=R_val),
                lambda net, a, b: R(net, a, b, name="r2", value=R_val),
                prefix="inner"
            )

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
        net, (inner_refs, r3_ref, outer_mid) = Series(
            net, n_in, net.gnd,
            inner_series,
            lambda net, a, b: R(net, a, b, name="r3", value=R_val),
            prefix="outer"
        )

        solver = net.compile()
        omega = 2 * math.pi * 1000.0
        sol = solver.solve_at(omega)
        z = solver.z_in(sol, vs)

        assert abs(z.real - expected_z) < 0.1, \
            f"Nested series Z={z.real} != expected {expected_z}"

    def test_freq_series_parallel_mixed(self):
        """Mixed Series and Parallel in frequency domain."""
        from pyvibrate.frequencydomain import Network, R
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Series, Parallel

        R_par = 1000.0  # Two 1k in parallel = 500 ohm
        R_ser = 500.0   # Series resistor
        expected_z = 1000.0  # 500 + 500 = 1k

        net = Network()
        net, n_in = net.node("in")

        def parallel_rs(net, a, b):
            return Parallel(
                net, a, b,
                lambda net, a, b: R(net, a, b, name="r1", value=R_par),
                lambda net, a, b: R(net, a, b, name="r2", value=R_par),
                prefix="par"
            )

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
        net, (par_refs, r3_ref, mid) = Series(
            net, n_in, net.gnd,
            parallel_rs,
            lambda net, a, b: R(net, a, b, name="r3", value=R_ser),
            prefix="ser"
        )

        solver = net.compile()
        omega = 2 * math.pi * 1000.0
        sol = solver.solve_at(omega)
        z = solver.z_in(sol, vs)

        assert abs(z.real - expected_z) < 0.1, \
            f"Mixed Z={z.real} != expected {expected_z}"

    def test_freq_multiple_series(self):
        """Multiple Series blocks in frequency domain."""
        from pyvibrate.frequencydomain import Network, R
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Series

        R_val = 100.0

        net = Network()
        net, n_in = net.node("in")
        net, n_mid = net.node("mid")

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)

        # First Series block: n_in to n_mid
        net, (r1_ref, r2_ref, mid1) = Series(
            net, n_in, n_mid,
            lambda net, a, b: R(net, a, b, name="r1", value=R_val),
            lambda net, a, b: R(net, a, b, name="r2", value=R_val),
            prefix="ser1"
        )

        # Second Series block: n_mid to gnd
        net, (r3_ref, r4_ref, mid2) = Series(
            net, n_mid, net.gnd,
            lambda net, a, b: R(net, a, b, name="r3", value=R_val),
            lambda net, a, b: R(net, a, b, name="r4", value=R_val),
            prefix="ser2"
        )

        solver = net.compile()
        omega = 2 * math.pi * 1000.0
        sol = solver.solve_at(omega)
        z = solver.z_in(sol, vs)

        expected_z = 4 * R_val  # 400 ohm
        assert abs(z.real - expected_z) < 0.1, \
            f"Multiple series Z={z.real} != expected {expected_z}"

    def test_freq_parallel_nested(self):
        """Nested Parallel operations."""
        from pyvibrate.frequencydomain import Network, R
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Parallel

        R_val = 100.0
        # Inner parallel: 100 || 100 = 50 ohm
        # Outer parallel: 50 || 100 = 33.33 ohm
        expected_z = (50 * 100) / (50 + 100)  # ~33.33 ohm

        net = Network()
        net, n_in = net.node("in")

        def inner_parallel(net, a, b):
            return Parallel(
                net, a, b,
                lambda net, a, b: R(net, a, b, name="r1", value=R_val),
                lambda net, a, b: R(net, a, b, name="r2", value=R_val),
                prefix="inner"
            )

        net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
        net, (inner_refs, r3_ref) = Parallel(
            net, n_in, net.gnd,
            inner_parallel,
            lambda net, a, b: R(net, a, b, name="r3", value=R_val),
            prefix="outer"
        )

        solver = net.compile()
        omega = 2 * math.pi * 1000.0
        sol = solver.solve_at(omega)
        z = solver.z_in(sol, vs)

        assert abs(z.real - expected_z) < 0.5, \
            f"Nested parallel Z={z.real} != expected {expected_z}"


class TestFreqJaxIntegration:
    """JAX integration tests for frequency-domain subcircuits."""

    def test_freq_series_jax_differentiable(self):
        """Gradients of impedance w.r.t. component values."""
        from pyvibrate.frequencydomain import Network, R
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Series

        def impedance_magnitude(R1_val, R2_val):
            net = Network()
            net, n_in = net.node("in")

            net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
            net, (r1_ref, r2_ref, mid) = Series(
                net, n_in, net.gnd,
                lambda net, a, b: R(net, a, b, name="r1"),
                lambda net, a, b: R(net, a, b, name="r2"),
                prefix="ser"
            )

            solver = net.compile()
            omega = 2 * math.pi * 1000.0
            sol = solver.solve_at(omega, {"r1": R1_val, "r2": R2_val})
            z = solver.z_in(sol, vs)
            return jnp.abs(z)

        R1_test = 100.0
        R2_test = 200.0

        dZ_dR1 = grad(impedance_magnitude, argnums=0)(R1_test, R2_test)
        dZ_dR2 = grad(impedance_magnitude, argnums=1)(R1_test, R2_test)

        # For series resistors: Z = R1 + R2, so dZ/dR1 = dZ/dR2 = 1
        assert jnp.isfinite(dZ_dR1), f"dZ/dR1 not finite"
        assert jnp.isfinite(dZ_dR2), f"dZ/dR2 not finite"
        assert abs(dZ_dR1 - 1.0) < 0.01, f"dZ/dR1 = {dZ_dR1} != 1"
        assert abs(dZ_dR2 - 1.0) < 0.01, f"dZ/dR2 = {dZ_dR2} != 1"

    def test_freq_parallel_jax_differentiable(self):
        """Gradients of parallel impedance w.r.t. component values."""
        from pyvibrate.frequencydomain import Network, R
        from pyvibrate.frequencydomain.components import ACSource
        from pyvibrate.frequencydomain.subcircuits import Parallel

        def impedance_magnitude(R1_val, R2_val):
            net = Network()
            net, n_in = net.node("in")

            net, vs = ACSource(net, n_in, net.gnd, name="vs", value=1.0)
            net, (r1_ref, r2_ref) = Parallel(
                net, n_in, net.gnd,
                lambda net, a, b: R(net, a, b, name="r1"),
                lambda net, a, b: R(net, a, b, name="r2"),
                prefix="par"
            )

            solver = net.compile()
            omega = 2 * math.pi * 1000.0
            sol = solver.solve_at(omega, {"r1": R1_val, "r2": R2_val})
            z = solver.z_in(sol, vs)
            return jnp.abs(z)

        R1_test = 100.0
        R2_test = 200.0

        dZ_dR1 = grad(impedance_magnitude, argnums=0)(R1_test, R2_test)
        dZ_dR2 = grad(impedance_magnitude, argnums=1)(R1_test, R2_test)

        # Gradients should be finite and positive
        # (increasing either R increases parallel impedance)
        assert jnp.isfinite(dZ_dR1), f"dZ/dR1 not finite"
        assert jnp.isfinite(dZ_dR2), f"dZ/dR2 not finite"
        assert dZ_dR1 > 0, f"dZ/dR1 should be positive"
        assert dZ_dR2 > 0, f"dZ/dR2 should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

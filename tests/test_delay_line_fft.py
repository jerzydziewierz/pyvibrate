"""
Test: Cross-domain validation of delay line behavior.

Verifies that:
1. Time-domain DelayLine + FFT yields correct linear phase response
2. Frequency-domain ConstantTimeDelayVCVS produces transfer function H(f) = exp(-j*omega*tau)
3. Time-domain and frequency-domain methods produce identical pulse responses
"""
import pytest
import numpy as np
import math


def test_timedomain_delay_fft_phase_response():
    """FFT of time-domain DelayLine output shows linear phase slope = -2*pi*tau."""
    from pyvibrate.timedomain import Network, R, VSource, DelayLine

    dt = 1e-6
    delay_samples = 10
    tau = delay_samples * dt
    n_steps = 256

    net = Network()
    net, v_in = net.node("v_in")
    net, v_out = net.node("v_out")

    net, vs = VSource(net, v_in, net.gnd, name="vs")
    net, dly = DelayLine(net, v_in, net.gnd, v_out, net.gnd,
                         delay_samples=delay_samples, name="dly")
    net, r_load = R(net, v_out, net.gnd, name="R_load")

    sim = net.compile(dt=dt)
    params = {"vs": 0.0, "R_load": 1000.0}
    state = sim.init(params)

    # Input pulse
    pulse_width = 8
    input_signal = np.zeros(n_steps)
    input_signal[2:2 + pulse_width] = 1.0

    # Simulate
    output_signal = np.zeros(n_steps)
    for i in range(n_steps):
        controls = {"vs": float(input_signal[i])}
        state = sim.step(params, state, controls)
        output_signal[i] = float(sim.v(state, v_out))

    # FFT analysis
    fft_in = np.fft.fft(input_signal)
    fft_out = np.fft.fft(output_signal)
    freqs = np.fft.fftfreq(n_steps, d=dt)
    positive_freqs = freqs[:n_steps // 2]

    # Transfer function
    eps = 1e-10
    H = fft_out / (fft_in + eps)
    H_mag = np.abs(H[:n_steps // 2])
    H_phase_unwrapped = np.unwrap(np.angle(H[:n_steps // 2]))

    # Check magnitude is ~1 where input has energy
    valid_idx = H_mag > 0.1
    mag_valid = H_mag[valid_idx]
    assert np.all(np.abs(mag_valid - 1.0) < 0.1), \
        f"Magnitude should be ~1, got range [{mag_valid.min():.2f}, {mag_valid.max():.2f}]"

    # Check phase slope: d(phase)/d(f) should equal -2*pi*tau
    # This avoids absolute phase offset issues from unwrapping
    valid_freqs = positive_freqs[valid_idx]
    valid_phases = H_phase_unwrapped[valid_idx]
    phase_slope = np.diff(valid_phases) / np.diff(valid_freqs)
    expected_slope = -2 * np.pi * tau

    slope_errors = np.abs(phase_slope - expected_slope)
    assert np.median(slope_errors) < 1e-8, \
        f"Phase slope should be -2*pi*tau = {expected_slope:.2e}, median error = {np.median(slope_errors):.2e}"


def test_timedomain_delay_group_delay():
    """Group delay extracted from FFT matches the configured delay time."""
    from pyvibrate.timedomain import Network, R, VSource, DelayLine

    dt = 1e-6
    delay_samples = 10
    tau = delay_samples * dt
    n_steps = 256

    net = Network()
    net, v_in = net.node("v_in")
    net, v_out = net.node("v_out")

    net, vs = VSource(net, v_in, net.gnd, name="vs")
    net, dly = DelayLine(net, v_in, net.gnd, v_out, net.gnd,
                         delay_samples=delay_samples, name="dly")
    net, r_load = R(net, v_out, net.gnd, name="R_load")

    sim = net.compile(dt=dt)
    params = {"vs": 0.0, "R_load": 1000.0}
    state = sim.init(params)

    # Input pulse
    pulse_width = 8
    input_signal = np.zeros(n_steps)
    input_signal[2:2 + pulse_width] = 1.0

    # Simulate
    output_signal = np.zeros(n_steps)
    for i in range(n_steps):
        controls = {"vs": float(input_signal[i])}
        state = sim.step(params, state, controls)
        output_signal[i] = float(sim.v(state, v_out))

    # FFT analysis
    fft_in = np.fft.fft(input_signal)
    fft_out = np.fft.fft(output_signal)
    freqs = np.fft.fftfreq(n_steps, d=dt)
    positive_freqs = freqs[:n_steps // 2]

    eps = 1e-10
    H = fft_out / (fft_in + eps)
    H_mag = np.abs(H[:n_steps // 2])
    H_phase_unwrapped = np.unwrap(np.angle(H[:n_steps // 2]))

    # Group delay = -d(phase)/d(omega) = -d(phase)/(2*pi*df)
    df = positive_freqs[1] - positive_freqs[0]
    group_delay = -np.diff(H_phase_unwrapped) / (2 * np.pi * df)

    # Median group delay in valid region
    valid_idx = H_mag[:-1] > 0.1
    measured_delay = np.median(group_delay[valid_idx])

    assert abs(measured_delay - tau) / tau < 0.05, \
        f"Group delay should be {tau * 1e6:.1f} us, got {measured_delay * 1e6:.1f} us"


def test_freqdomain_tdvcvs_sweep():
    """Frequency sweep of ConstantTimeDelayVCVS (TDVCVS) gives linear phase = -omega*tau.

    TDVCVS = Time-Delay Voltage-Controlled Voltage Source (active element)
    """
    from pyvibrate.frequencydomain import Network, R as FD_R, ACSource, ConstantTimeDelayVCVS

    tau = 10e-6  # 10 us

    fd_net = Network()
    fd_net, n_in = fd_net.node("n_in")
    fd_net, n_out = fd_net.node("n_out")

    fd_net, vs = ACSource(fd_net, n_in, fd_net.gnd, name="vs", value=1.0)
    fd_net, ps = ConstantTimeDelayVCVS(fd_net, n_in, fd_net.gnd, n_out, fd_net.gnd,
                                       name="PS", tau=tau)
    fd_net, r_load = FD_R(fd_net, n_out, fd_net.gnd, name="R_load", value=1000.0)

    solver = fd_net.compile()

    # Sweep frequencies
    n_freq = 64
    sweep_freqs = np.linspace(1e3, 400e3, n_freq)

    H_mag = np.zeros(n_freq)
    H_phase = np.zeros(n_freq)

    for i, f in enumerate(sweep_freqs):
        omega = 2 * np.pi * f
        sol = solver.solve_at(omega)
        v_in = solver.v(sol, n_in)
        v_out = solver.v(sol, n_out)
        H = v_out / v_in
        H_mag[i] = np.abs(H)
        H_phase[i] = np.angle(H)

    # Check magnitude is 1
    assert np.all(np.abs(H_mag - 1.0) < 1e-6), \
        f"ConstantTimeDelayVCVS magnitude should be exactly 1"

    # Check phase matches theory
    theoretical_phase = -2 * np.pi * sweep_freqs * tau
    H_phase_unwrapped = np.unwrap(H_phase)
    theoretical_unwrapped = np.unwrap(theoretical_phase)

    phase_error = np.abs(H_phase_unwrapped - theoretical_unwrapped)
    assert np.max(phase_error) < 0.01, \
        f"Phase error should be < 0.01 rad, got max {np.max(phase_error):.4f}"


def test_ifft_impulse_response_peak():
    """iFFT of ideal delay transfer function gives impulse at t=tau."""
    tau = 10e-6  # 10 us
    dt = 1e-6
    n_ifft = 512

    f_sample = 1 / dt
    ifft_freqs = np.fft.fftfreq(n_ifft, d=dt)

    # Ideal delay: H(f) = exp(-j*2*pi*f*tau)
    H_full = np.exp(-1j * 2 * np.pi * ifft_freqs * tau)

    # iFFT to get impulse response
    impulse_response = np.fft.ifft(H_full).real
    t_ifft = np.arange(n_ifft) * dt

    # Peak should be at t = tau
    peak_idx = np.argmax(impulse_response[:n_ifft // 2])
    peak_time = t_ifft[peak_idx]

    assert abs(peak_time - tau) < dt, \
        f"Impulse peak should be at {tau * 1e6:.1f} us, got {peak_time * 1e6:.1f} us"


def test_timedomain_vs_freqdomain_ifft():
    """Time-domain simulation matches frequency-domain + iFFT for same pulse."""
    from pyvibrate.timedomain import Network, R, VSource, DelayLine

    dt = 1e-6
    delay_samples = 10
    tau = delay_samples * dt
    n_steps = 256

    # --- Time-domain simulation ---
    net = Network()
    net, v_in = net.node("v_in")
    net, v_out = net.node("v_out")

    net, vs = VSource(net, v_in, net.gnd, name="vs")
    net, dly = DelayLine(net, v_in, net.gnd, v_out, net.gnd,
                         delay_samples=delay_samples, name="dly")
    net, r_load = R(net, v_out, net.gnd, name="R_load")

    sim = net.compile(dt=dt)
    params = {"vs": 0.0, "R_load": 1000.0}
    state = sim.init(params)

    # Input pulse
    pulse_width = 8
    input_signal = np.zeros(n_steps)
    input_signal[2:2 + pulse_width] = 1.0

    # Simulate
    output_td = np.zeros(n_steps)
    for i in range(n_steps):
        controls = {"vs": float(input_signal[i])}
        state = sim.step(params, state, controls)
        output_td[i] = float(sim.v(state, v_out))

    # --- Frequency-domain + iFFT ---
    n_ifft = 512
    ifft_freqs = np.fft.fftfreq(n_ifft, d=dt)

    # Ideal delay transfer function
    H_full = np.exp(-1j * 2 * np.pi * ifft_freqs * tau)

    # Pad input and compute output via FFT
    input_padded = np.zeros(n_ifft)
    input_padded[:len(input_signal)] = input_signal
    fft_input = np.fft.fft(input_padded)
    fft_output_fd = H_full * fft_input
    output_fd = np.fft.ifft(fft_output_fd).real

    # Compare (trim to same length)
    output_fd_trimmed = output_fd[:n_steps]
    max_error = np.max(np.abs(output_td - output_fd_trimmed))

    assert max_error < 0.01, \
        f"Time-domain and freq-domain outputs should match, max error = {max_error:.4f} V"


def test_cross_domain_pulse_shape_preserved():
    """Delay preserves pulse shape - only shifts in time."""
    from pyvibrate.timedomain import Network, R, VSource, DelayLine

    dt = 1e-6
    delay_samples = 10
    n_steps = 64

    net = Network()
    net, v_in = net.node("v_in")
    net, v_out = net.node("v_out")

    net, vs = VSource(net, v_in, net.gnd, name="vs")
    net, dly = DelayLine(net, v_in, net.gnd, v_out, net.gnd,
                         delay_samples=delay_samples, name="dly")
    net, r_load = R(net, v_out, net.gnd, name="R_load")

    sim = net.compile(dt=dt)
    params = {"vs": 0.0, "R_load": 1000.0}
    state = sim.init(params)

    # Triangular pulse input
    pulse_width = 8
    input_signal = np.zeros(n_steps)
    for i in range(pulse_width):
        input_signal[2 + i] = min(i, pulse_width - i - 1) / (pulse_width // 2 - 1)

    # Simulate
    output_signal = np.zeros(n_steps)
    for i in range(n_steps):
        controls = {"vs": float(input_signal[i])}
        state = sim.step(params, state, controls)
        output_signal[i] = float(sim.v(state, v_out))

    # The output should be the same shape as input, just shifted
    # Compare input[2:2+pulse_width] with output[2+delay:2+delay+pulse_width]
    input_pulse = input_signal[2:2 + pulse_width]
    output_pulse = output_signal[2 + delay_samples:2 + delay_samples + pulse_width]

    shape_error = np.max(np.abs(input_pulse - output_pulse))
    assert shape_error < 0.01, \
        f"Pulse shape should be preserved, max error = {shape_error:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

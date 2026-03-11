import scipy
import numpy as np
from itertools import product
import cupy as cp




def notch_if(x, fs, f0_hz, bw_hz=5.0):
    """
    Suppress very narrow IF tone(s).
    - x: array (..., N), real or complex. Filtered along last axis.
    - fs: sample rate [Hz]
    - f0_hz: float or list of floats (center freq in Hz, >0)
    - bw_hz: 3 dB bandwidth of notch [Hz]
    """
    from scipy.signal import iirnotch, filtfilt
    x_f = np.asarray(x)
    f0_list = f0_hz if np.iterable(f0_hz) else [f0_hz]
    for f0 in f0_list:
        Q = max(10.0, float(f0) / float(bw_hz))  # higher Q = narrower notch
        b, a = iirnotch(w0=f0, Q=Q, fs=fs)
        x_f = filtfilt(b, a, x_f, axis=-1, method="gust")
    return x_f




def find_ref_frq_first_peak(sig, nfft, f_tar_idx=0, phase_ref=0.0, min_height_ratio=0.5, min_prominence_ratio=0.3):
    """
    Align the signal(s) to the target frequency bin and phase, using the first peak in the FFT magnitude.
    Uses thresholding: min_height_ratio and min_prominence_ratio relative to max.

    Args:
        sig: 1D (num_samples,) or 2D (num_signals, num_samples) array
        nfft: FFT length used for bin indexing
        f_tar_idx: target FFT bin index to align peak to
        phase_ref: desired phase at target bin (scalar), default 0
        min_height_ratio: minimum "height" ratio for a peak (relative to global max)
        min_prominence_ratio: minimum "prominence" ratio for a peak (relative to global max)

    Returns:
        df: frequency offset(s), shape () for 1D input or (B,) for batched input
        da: phase offset(s), same shape as df
    """
    from scipy.signal import find_peaks

    x = np.asarray(sig)
    # FFT and get magnitude along the last axis
    S = np.fft.fft(x, axis=-1, n=nfft)
    mag = np.abs(S)

    # Handle both 1D and batched 2D (or higher) inputs by flattening batch dims
    if mag.ndim == 1:
        maxval = float(mag.max()) if mag.size else 0.0
        min_height = float(min_height_ratio * maxval)
        min_prominence = float(min_prominence_ratio * maxval)
        peaks, _ = find_peaks(mag, height=min_height, prominence=min_prominence)
        idx = int(peaks[0]) if peaks.size else int(np.argmax(mag))
        df = (f_tar_idx - idx) / nfft
        phi_peak = np.angle(S[idx])
        da = (phase_ref - phi_peak)
        return df, da

    # Batched: reshape to (B, L)
    B = int(np.prod(mag.shape[:-1]))
    L = mag.shape[-1]
    mag2 = mag.reshape(B, L)
    S2 = S.reshape(B, L)

    idxs = np.empty(B, dtype=int)
    phases = np.empty(B, dtype=float)
    for b in range(B):
        mrow = mag2[b]
        maxval = float(mrow.max()) if mrow.size else 0.0
        min_height = float(min_height_ratio * maxval)
        min_prominence = float(min_prominence_ratio * maxval)
        peaks, _ = find_peaks(mrow, height=min_height, prominence=min_prominence)
        idx = int(peaks[0]) if peaks.size else int(np.argmax(mrow))
        idxs[b] = idx
        phases[b] = float(np.angle(S2[b, idx]))

    df = (f_tar_idx - idxs) / float(nfft)
    da = (phase_ref - phases)
    # Return with the original batch shape (excluding last axis)
    out_shape = x.shape[:-1]
    return df.reshape(out_shape), da.reshape(out_shape)



def find_ref_frq(sig, nfft, f_tar_idx=0, phase_ref=0.0):
    """
    Align the signal(s) to the target frequency bin and phase.

    Args:
        sig: 1D (num_samples,) or 2D (num_signals, num_samples) array
        nfft: FFT length used for bin indexing
        f_tar_idx: target FFT bin index to align peak to
        phase_ref: desired phase at target bin (scalar), default 0

    Returns:
        Aligned signal(s) with same shape as input.
    """
    from scipy.signal import ZoomFFT
    # sig = np.asarray(sig)
    _ns_zoom = 100

    L = sig.shape[-1]
    idx_max = int(np.argmax(np.abs(np.fft.fft(sig, axis=-1, n=nfft))))
    start, end = idx_max - 1, idx_max + 1
    zf = ZoomFFT(L, [start, end], _ns_zoom, fs=nfft)
    S = zf(sig).ravel()
    imax = int(np.argmax(np.abs(S)))
    df = -(imax / _ns_zoom * abs(start - end) + start - f_tar_idx) / nfft
    phi_peak = np.angle(S[imax])
    da = (phase_ref - phi_peak)
    return df, da






def compute_range_angle(frame_ml, fs_hz, slope_hz_per_s, nfft_range=512, nfft_angle=128,
                        window_range=True, window_angle=True):
    """
    2D-FFT range-angle from a single-chirp snapshot across antennas.
    - frame_ml: (num_ant, num_adc) complex array
    Returns: mag (A,R), angle_bins_rad (A,), ranges_m (R,)
    """
    mag = np.fft.fft2(frame_ml, s=(nfft_angle, nfft_range))
    mag = np.fft.fftshift(mag, axes=0)[:, :nfft_range//2]
    mag = np.flip(np.abs(mag), axis=0)
    u = np.fft.fftfreq(nfft_angle, d=1.0)
    u = np.fft.fftshift(u)
    angle_bins_rad = np.arcsin(np.clip(2.0 * u, -1.0, 1.0))

    # Range bins
    k = np.arange(nfft_range//2, dtype=int)
    # ranges_m = (3.0e8 * (k * (fs_hz / nfft_range))) / (2.0 * slope_hz_per_s)
    ranges_m = fs_hz*3e8 /(2* nfft_range*slope_hz_per_s)*k

    return mag, angle_bins_rad.astype(np.float32), ranges_m.astype(np.float32)




def ula_beamform_range(
    adcData,
    angles_rad,
    nfft,
    fc_hz,
    d_m=None,
    center_array=True,
    apply_hann_window=True,
    aperture_window=None,
    two_way=False,
):
    """
    Delay-and-sum beamforming for a ULA of FMCW signals.
    Inputs:
        adcData: complex array (num_ant, num_samples)
        angles_rad: angles to beamform (radians), shape (num_angles,)
        nfft: range FFT length
        fc_hz: carrier frequency (Hz)
        d_m: element spacing (meters)
    Output:
        (num_angles, nfft) array (complex if return_complex else magnitude)
    """

    x = np.asarray(adcData)
    assert x.ndim == 2, "adcData must be (num_ant, num_samples)"
    M, L = x.shape
    if nfft is None:
        nfft = L

    if apply_hann_window:
        x = x * np.hanning(L).astype(x.dtype)[None, :]

    X = np.fft.fft(x, n=nfft, axis=-1)  # (M, nfft)

    c = 299792458.0
    wavelength = c / float(fc_hz)
    k = 2.0 * np.pi / wavelength
    phase_factor = 2.0 if two_way else 1.0

    # ULA x-positions (centered optional)
    idx = np.arange(M, dtype=np.float32)
    # if center_array:
    #     idx = idx - (M - 1) / 2.0
    # if d_m is None:
    #     d_m = wavelength / 2
    # pos_x = idx * float(d_m)  # (M,)

    ang = np.asarray(angles_rad, dtype=np.float32).ravel()
    sin_th = np.sin(ang)[:, None]  # (A,1)

    # steering = np.exp(-1j * (k * phase_factor) * (sin_th * pos_x[None, :])).astype(X.dtype)  # (A,M)

    steering = np.exp(-1j * np.pi * sin_th * idx[None, :]).astype(X.dtype)

    # if aperture_window is not None:
    #     w_ap = np.asarray(aperture_window, dtype=X.dtype).ravel()
    #     assert w_ap.size == M
    #     steering = steering * w_ap[None, :]

    Y = steering @ X  # (A, nfft)
    
    return Y



# def align_signal(sig, f_tar=0):
#     pass



def shift_freq(sig, df, da):
    return sig * np.exp(1j * 2 * np.pi * df * np.arange(sig.shape[0])) * np.exp(1j * da)


def get_fft_bin_val(sig, k, nfft):
    """
    Get the value of the FFT bin at index k
    Args:
        sig: the signal
        k: the index of the FFT bin
        nfft: the number of FFT bins
    Returns:
        the value of the FFT bin at index k
    """
    n = np.arange(sig.shape[0])
    w = np.exp(-1j * 2 * np.pi * k * n / nfft)
    return np.angle(np.sum(sig * w))




def align_signal_argmax(sig, nfft, f_tar_idx=0, phase_ref=0.0):
    """
    Align the signal(s) to the target frequency bin and phase.

    Args:
        sig: 1D (num_samples,) or 2D (num_signals, num_samples) array
        nfft: FFT length used for bin indexing
        f_tar_idx: target FFT bin index to align peak to
        phase_ref: desired phase at target bin (scalar), default 0

    Returns:
        Aligned signal(s) with same shape as input.
    """
    from scipy.signal import ZoomFFT
    sig = np.asarray(sig)
    _ns_zoom = 100

    if sig.ndim == 1:
        L = sig.shape[-1]
        idx_max = int(np.argmax(np.abs(np.fft.fft(sig, axis=-1, n=nfft))))
        start, end = idx_max - 1, idx_max + 1
        zf = ZoomFFT(L, [start, end], _ns_zoom, fs=nfft)
        S = zf(sig)
        imax = int(np.argmax(np.abs(S)))
        df = -(imax / _ns_zoom * abs(start - end) + start - f_tar_idx) / nfft
        phi_peak = np.angle(S[imax])
        da = (phase_ref - phi_peak)
        n = np.arange(L)
        return sig * np.exp(1j * (2 * np.pi * df * n + da))

    if sig.ndim == 2:
        M, L = sig.shape
        X = np.fft.fft(sig, axis=-1, n=nfft)
        idx_max = np.argmax(np.abs(X), axis=-1).astype(int)  # (M,)

        df = np.empty(M, dtype=float)
        da = np.empty(M, dtype=float)
        for i in range(M):
            start, end = int(idx_max[i]) - 1, int(idx_max[i]) + 1
            zf = ZoomFFT(L, [start, end], _ns_zoom, fs=nfft)
            S = zf(sig[i])
            imax = int(np.argmax(np.abs(S)))
            df[i] = -(imax / _ns_zoom * abs(start - end) + start - f_tar_idx) / nfft
            phi_peak = np.angle(S[imax])
            da[i] = (phase_ref - phi_peak)

        n = np.arange(L)[None, :]
        return sig * np.exp(1j * (2 * np.pi * df[:, None] * n + da[:, None]))

    raise ValueError("sig must be 1D or 2D (num_signals, num_samples)")



def align_signal_findpeak(
    sig,
    nfft,
    f_tar_idx=0,
    phase_ref=0.0,
    height=None,
    rel_height=0.5,
    distance=None,
    prominence=None,
):
    """
    Align signal(s) to target FFT bin/phase by selecting the FIRST peak above a threshold.

    Args:
        sig: 1D (num_samples,) or 2D (num_signals, num_samples) complex array
        nfft: FFT length for bin indexing
        f_tar_idx: target FFT bin index to align to
        phase_ref: desired phase at target bin (scalar), default 0
        height: absolute amplitude threshold for peak detection (overrides rel_height if set)
        rel_height: relative threshold as a fraction of per-signal max magnitude (used if height=None)
        distance: optional minimum separation (in bins) between peaks
        prominence: optional prominence for peak detection

    Returns:
        Aligned signal(s), same shape as input.
    """
    from scipy.signal import ZoomFFT
    sig = np.asarray(sig)

    def _align_1d(x):
        L = x.shape[-1]
        X = np.fft.fft(x, n=nfft)
        mag = np.abs(X)
        h = float(height) if height is not None else float(rel_height) * float(mag.max() if mag.size else 0.0)
        peaks, _ = scipy.signal.find_peaks(mag, height=h, distance=distance, prominence=prominence)
        k0 = int(peaks[0]) if peaks.size else int(np.argmax(mag))
        
        start, end = k0 - 1, k0 + 1
        zf = ZoomFFT(L, [start, end], 100, fs=nfft)
        S = zf(x)
        imax = int(np.argmax(np.abs(S)))
        df = -(imax / 100.0 * abs(end - start) + start - f_tar_idx) / float(nfft)
        phi = float(np.angle(S[imax]))
        da = float(phase_ref) - phi

        n = np.arange(L, dtype=float)
        return x * np.exp(1j * (2.0 * np.pi * df * n + da))

    if sig.ndim == 1:
        return _align_1d(sig)

    if sig.ndim == 2:
        M, L = sig.shape
        X = np.fft.fft(sig, axis=-1, n=nfft)
        mag = np.abs(X)

        df = np.empty(M, dtype=float)
        da = np.empty(M, dtype=float)

        for i in range(M):
            h = float(height) if height is not None else float(rel_height) * float(mag[i].max() if mag.shape[1] else 0.0)
            peaks, _ = scipy.signal.find_peaks(mag[i], height=h, distance=distance, prominence=prominence)
            k0 = int(peaks[0]) if peaks.size else int(np.argmax(mag[i]))

            start, end = k0 - 1, k0 + 1
            zf = ZoomFFT(L, [start, end], 100, fs=nfft)
            S = zf(sig[i])
            imax = int(np.argmax(np.abs(S)))
            df[i] = -(imax / 100.0 * abs(end - start) + start - f_tar_idx) / float(nfft)
            da[i] = float(phase_ref) - float(np.angle(S[imax]))

        n = np.arange(L, dtype=float)[None, :]
        return sig * np.exp(1j * (2.0 * np.pi * df[:, None] * n + da[:, None]))

    raise ValueError("sig must be 1D or 2D (num_signals, num_samples)")




    

def align_signals_xcorr(adcData, nfft=4096, f_tar_idx=None, window='hann', return_shifts=False, max_shift=None):
    """
    Align FMCW raw ADC signals (num_signals, num_samples) by correcting range offsets.
    - Estimates relative range-bin shifts via cross-correlation of adjacent signals'
      FFT magnitudes, accumulates shifts w.r.t. the first signal, and frequency-shifts
      time-domain signals accordingly.
    - Optional: phase-align all signals at a target FFT bin f_tar_idx.

    Args:
        adcData: complex array (M, L)
        nfft: FFT length for range profiles
        f_tar_idx: optional int; if set, phase-align all signals at this bin
        window: 'hann', None, or custom window array of length L
        return_shifts: if True, also return cumulative bin shifts per signal
        max_shift: int or None; limit search to +/- max_shift bins (default nfft//4)

    Returns:
        aligned (M, L) [and shifts (M,)]
    """
    M, L = adcData.shape
    if window == 'hann':
        win = np.hanning(L).astype(adcData.dtype)
    elif window is None:
        win = np.ones(L, dtype=adcData.dtype)
    else:
        win = np.asarray(window, dtype=adcData.dtype)
        assert win.shape[0] == L

    # Range profiles (magnitude)
    X = np.fft.fft(adcData * win[None, :], n=nfft, axis=-1)
    S = np.abs(X) 

    # Estimate relative bin shifts via circular cross-correlation between neighbors
    if max_shift is None:
        max_shift = nfft // 4
    shifts = np.zeros(M, dtype=int)
    for i in range(1, M):
        # if i<1000:
        #     continue
        a = S[i-1]; b = S[i]
        aa = a - a.mean()
        bb = b - b.mean()

        A = np.fft.fft(aa, nfft)
        B = np.fft.fft(bb, nfft)

        r = np.fft.ifft(A * np.conj(B)).real  # circular cross-correlation
        # restrict search window to +/- max_shift around zero-lag
        windowed = np.concatenate([r[-max_shift:], r[:max_shift+1]])
        
        lag = np.argmax(windowed) - max_shift   # positive => b is shifted right vs a
        lag = np.argmax(windowed) + max_shift - nfft//2
        shifts[i] = shifts[i-1] + lag
        

    # Apply fractional frequency shift in time domain to correct range offset
    n = np.arange(L, dtype=float)
    aligned = np.empty_like(adcData, dtype=adcData.dtype)
    for i in range(M):
        df = (shifts[i]-1000) / float(nfft)  # cycles per sample
        # df = -shifts[i] / float(nfft)  # cycles per sample
        aligned[i] = adcData[i] * np.exp(1j * 2.0 * np.pi * df * n)


    # # Optional: phase-align at a target bin (match all to signal 0)
    # if f_tar_idx is not None:
    #     def bin_phase(x, k):
    #         n = np.arange(x.size)
    #         w = np.exp(-1j * 2.0 * np.pi * k * n / nfft)
    #         return np.angle(np.sum(x * w))
    #     phi0 = bin_phase(aligned[0], f_tar_idx)
    #     for i in range(M):
    #         phi = bin_phase(aligned[i], f_tar_idx)
    #         aligned[i] *= np.exp(1j * (phi0 - phi))

    return (aligned, shifts) if return_shifts else aligned




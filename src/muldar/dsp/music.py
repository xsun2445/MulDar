

import numpy as np
from numpy.linalg import eigh

def music_2d_range_angle(X, fs, slope, fc, K,
                         theta_grid_rad, range_grid_m,
                         d=None, c=3e8, snapshots_axis=None,
                         eps=1e-9):
    """
    2D MUSIC over angle (ULA) and range (FMCW fast-time).

    Inputs
    - X: complex array with shape
        (M, N)                -> single snapshot (M antennas, N fast-time samples)
        or (M, N, T)          -> T snapshots (e.g., chirps/frames)
      Data should be phase-calibrated per channel.
    - fs: ADC sample rate [Hz]
    - slope: FMCW slope [Hz/s]
    - fc: carrier [Hz]
    - K: model order (number of point targets)
    - theta_grid_rad: (A,) angles (azimuth) in radians
    - range_grid_m: (R,) ranges in meters
    - d: inter-element spacing [m]; default = λ/2
    - c: speed of light
    - snapshots_axis: None or 2 (if X has T snapshots on axis 2)
    - eps: numerical floor

    Returns
    - P: pseudospectrum magnitude, shape (A, R)
    """
    X = np.asarray(X)
    if X.ndim == 2:
        M, N = X.shape
        Z = X.reshape(M*N, 1, order='F')  # vec single snapshot
    elif X.ndim == 3:
        M, N, T = X.shape
        assert snapshots_axis in (None, 2), "snapshots_axis must be None or 2"
        Z = X.reshape(M*N, T, order='F')  # vec each snapshot
    else:
        raise ValueError("X must be (M,N) or (M,N,T)")

    lam = c / float(fc)
    if d is None:
        d = 0.5 * lam

    # Sample covariance of vec(X)
    R = (Z @ Z.conj().T) / Z.shape[1]

    # Eigendecomposition (Hermitian)
    w, V = eigh(R)                       # ascending eigenvalues
    En = V[:, :max(1, M*N - int(K))]     # noise subspace

    # Precompute antenna indices and time indices
    m_idx = np.arange(M, dtype=float)    # 0..M-1
    n_idx = np.arange(N, dtype=float)    # 0..N-1

    # Steering builders
    def a_theta(theta):
        # ULA broadside at 0 rad; sign convention exp(-j 2π d/λ m sinθ)
        return np.exp(-1j * 2.0*np.pi * d/lam * m_idx * np.sin(theta))

    def b_range(Rm):
        # FMCW beat frequency fb = 2*S*R/c; time index n/fs
        fb = 2.0 * float(slope) * float(Rm) / c
        return np.exp(+1j * 2.0*np.pi * fb * (n_idx / float(fs)))

    # Evaluate pseudospectrum
    A = len(theta_grid_rad)
    Rg = len(range_grid_m)
    P = np.empty((A, Rg), dtype=float)

    # Projector onto noise subspace
    EnEnH = En @ En.conj().T

    for ia, th in enumerate(theta_grid_rad):
        a = a_theta(th)                        # (M,)
        for ir, Rm in enumerate(range_grid_m):
            b = b_range(Rm)                    # (N,)
            v = np.kron(b, a)                  # (M*N,), matches vec order='F'
            denom = np.real(v.conj() @ (EnEnH @ v)) + eps
            P[ia, ir] = 1.0 / denom

    return P




import numpy as np
from numpy.linalg import svd

def music_2d_range_angle_fast(X, fs, slope, fc, K,
                              theta_grid_rad, range_grid_m,
                              d=None, c=3e8, tile_ranges=64,
                              dtype=np.complex64, eps=1e-7,
                              sensor_positions_m=None):
    """
    Fast 2D-MUSIC over angle-range using Khatri-Rao tiles and SVD.
    X: (M,N) or (M,N,T) complex array (antennas, fast-time[, snapshots])
    K: model order
    tile_ranges: number of range columns per tile
    """
    X = np.asarray(X).astype(dtype, copy=False)
    if X.ndim == 2:
        M, N = X.shape
        Z = X.reshape(M*N, 1, order='F')
    else:
        M, N, T = X.shape
        Z = X.reshape(M*N, T, order='F')

    lam = c / float(fc)
    # Sensor geometry
    if sensor_positions_m is None:
        if d is None:
            d = 0.5 * lam
        sensor_positions_m = np.arange(M, dtype=np.float32) * d
    pos = np.asarray(sensor_positions_m, dtype=np.float32).reshape(M, 1)

    # Match slow method: noise subspace from covariance eigen-decomposition
    R = (Z @ Z.conj().T) / Z.shape[1]
    w, V = np.linalg.eigh(R)  # ascending
    En = V[:, :max(1, M*N - int(K))].astype(dtype, copy=False)  # (M*N, #noise)
    EnH = En.conj().T

    # Steering matrices
    m = np.arange(M, dtype=np.float32)
    n = np.arange(N, dtype=np.float32)

    # Angle steering with actual x-positions, unit-norm columns
    A = np.exp(-1j * 2.0*np.pi * (pos / lam) * np.sin(theta_grid_rad)[None, :]).astype(dtype)  # (M, A)
    A /= np.sqrt(M, dtype=np.float32)

    # Range steering: ensure slope in Hz/s; unit-norm columns
    S_si = float(slope)
    if S_si < 1e9:
        S_si *= 1e12
    fb = 2.0 * S_si * np.asarray(range_grid_m, dtype=np.float32) / c                             # (R,)
    B = np.exp(+1j * 2.0*np.pi * (n[:, None] / float(fs)) * fb[None, :]).astype(dtype)           # (N, R)
    B /= np.sqrt(N, dtype=np.float32)

    A_cnt = A.shape[1]
    R_cnt = B.shape[1]
    P = np.empty((A_cnt, R_cnt), dtype=np.float32)

    # Tile over range grid to limit (M*N) x (A*tile) working set
    for r0 in range(0, R_cnt, tile_ranges):
        r1 = min(R_cnt, r0 + tile_ranges)
        # All (angle, range) columns at once: kron(B_tile, A) with angle varying fastest
        V = np.kron(B[:, r0:r1], A)  # (M*N, A*(tile))

        # Project onto noise subspace: ||En^H v||^2
        Ublk = EnH @ V
        denom = np.maximum(np.sum(np.abs(Ublk)**2, axis=0).real, eps)
        # reshape as (tile, angles) then transpose → robust against vec ordering
        P_block = (1.0 / denom).reshape(r1 - r0, A_cnt).T
        P[:, r0:r1] = P_block.astype(np.float32, copy=False)

    return P


def music_2d_range_angle_fast_gpu(X, fs, slope, fc, K,
                                  theta_grid_rad, range_grid_m,
                                  d=None, c=3e8, tile_ranges=128,
                                  dtype='complex64', eps=1e-7,
                                  sensor_positions_m=None):
    """
    GPU-accelerated 2D-MUSIC using CuPy.
    Mirrors music_2d_range_angle_fast (CPU), but runs the heavy linear algebra on GPU.

    Returns numpy float32 array P with shape (len(theta_grid_rad), len(range_grid_m)).
    """
    try:
        import cupy as cp
    except Exception as _:
        raise RuntimeError("CuPy is not available. Please install cupy-cudaXX for your CUDA version.")

    # Convert inputs to GPU arrays
    X = cp.asarray(X, dtype=dtype, order='F')
    if X.ndim == 2:
        M, N = X.shape
        Z = X.reshape(M*N, 1, order='F')
    else:
        M, N, T = X.shape
        Z = X.reshape(M*N, T, order='F')

    lam = c / float(fc)
    # Sensor geometry
    if sensor_positions_m is None:
        if d is None:
            d = 0.5 * lam
        sensor_positions_m = cp.arange(M, dtype=cp.float32) * d
    else:
        sensor_positions_m = cp.asarray(sensor_positions_m, dtype=cp.float32)
    pos = sensor_positions_m.reshape(M, 1)

    # Noise subspace from covariance (match slow method)
    R = (Z @ Z.conj().T) / Z.shape[1]
    w, V = cp.linalg.eigh(R)  # ascending
    En = V[:, :max(1, M*N - int(K))].astype(dtype, copy=False)  # (M*N, #noise)
    EnH = En.conj().T

    # Steering matrices (unit-norm columns)
    theta_grid_rad_cp = cp.asarray(theta_grid_rad, dtype=cp.float32)
    range_grid_m_cp = cp.asarray(range_grid_m, dtype=cp.float32)
    n = cp.arange(N, dtype=cp.float32)

    A = cp.exp(-1j * 2.0*cp.pi * (pos / lam) * cp.sin(theta_grid_rad_cp)[None, :]).astype(dtype)
    A /= cp.sqrt(cp.asarray(M, dtype=cp.float32))

    # Slope to Hz/s if necessary
    S_si = float(slope)
    if S_si < 1e9:
        S_si *= 1e12
    fb = 2.0 * S_si * range_grid_m_cp / c
    B = cp.exp(+1j * 2.0*cp.pi * (n[:, None] / float(fs)) * fb[None, :]).astype(dtype)
    B /= cp.sqrt(cp.asarray(N, dtype=cp.float32))

    A_cnt = A.shape[1]
    R_cnt = B.shape[1]
    P = cp.empty((A_cnt, R_cnt), dtype=cp.float32)

    # Tile over range
    for r0 in range(0, int(R_cnt), int(tile_ranges)):
        r1 = min(R_cnt, r0 + tile_ranges)
        Vblk = cp.kron(B[:, r0:r1], A)  # (M*N, A*(tile))
        Ublk = EnH @ Vblk
        denom = cp.maximum(cp.sum(cp.abs(Ublk)**2, axis=0).real, eps)
        P_block = (1.0 / denom).reshape(r1 - r0, A_cnt).T
        P[:, r0:r1] = P_block.astype(cp.float32, copy=False)

    # Return to CPU memory
    return cp.asnumpy(P)



import numpy as np
import cupy as cp





 





def sar_bp_gpu(adcData, positions_ant, gx, gy, nfft, params, return_complex=False, tile_y=None):
    """Mono-static backprojection on a horizontal plane using GPU.
    """
    
    c = float(params['c']); fc = float(params['frequency'])
    fs = float(params['fs']); slope = float(params['slope'])
    wavelength = c / fc

    # to GPU
    x = cp.asarray(adcData)                     # (N_pos, N_samples)
    pos = cp.asarray(positions_ant, dtype=cp.float32).ravel()   # (N_pos,)
    gx = cp.asarray(gx, dtype=cp.float32)       # (Nx,)
    gy = cp.asarray(gy, dtype=cp.float32)       # (Ny,)

    # range FFT
    if nfft is None: nfft = x.shape[-1]
    win = cp.hanning(x.shape[-1]).astype(x.dtype)
    X = cp.fft.fft(x * win[None, :], n=nfft, axis=-1)          # (N_pos, N_range)

    df = fs / nfft
    dr = c * df / (2.0 * slope)

    kz = 2*cp.pi/c*(fc + slope*np.arange(nfft)/fs)

    Nx, Ny = gx.size, gy.size
    Xg = gx[None, None, :]                        # (1,1,Nx)

    def process_y_slice(y_slice):
        # y_slice: (Ny_tile,)
        Yg = y_slice[:, None]                    # (Ny_tile,1)
        # distances r for all positions and pixels: (N_pos, Ny_tile, Nx)
        r = cp.sqrt((Xg - pos[:, None, None])**2 + (Yg[None, :, :])**2)
        idx = cp.clip(r / dr, 0, nfft - 2)
        idx0 = cp.floor(idx).astype(cp.int32)
        frac = (idx - idx0).astype(cp.float32)

        X_exp = X[:, None, None, :]             # (N_pos,1,1,N_range)
        v0 = cp.take_along_axis(X_exp, idx0[..., None], axis=-1).squeeze(-1)
        v1 = cp.take_along_axis(X_exp, (idx0+1)[..., None], axis=-1).squeeze(-1)
        samp = (1.0 - frac) * v0 + frac * v1    # (N_pos, Ny_tile, Nx)

        # phase = cp.exp(1j * (4.0 * np.pi / wavelength) * r)    # two-way
        phase = cp.exp(1j * (2.0 * np.pi / wavelength) * r)    # two-way
        return cp.sum(samp * phase, axis=0)     # (Ny_tile, Nx)

    if tile_y:  # optional tiling to reduce peak memory
        out = []
        for s in range(0, Ny, tile_y):
            out.append(process_y_slice(gy[s:s+tile_y]))
        img = cp.concatenate(out, axis=0)
    else:
        img = process_y_slice(gy)

    return img.astype(cp.complex64) if return_complex else cp.abs(img)



#####################################
# Add to src/dsp/sar.py
def bistatic_sar_bp_numba(adcData, positions_tx, positions_rx, gx, gy, nfft, params):
    import numpy as np
    import numba as nb

    c = float(params['c']); fc = float(params['frequency'])
    fs = float(params['fs']); slope = float(params['slope'])
    ref_idx = int(params.get('ref_idx', 0))
    wavelength = c / fc

    x = np.asarray(adcData)
    Ns = x.shape[1]
    if nfft is None: nfft = Ns
    X = np.fft.fft(x, n=nfft, axis=-1).astype(np.complex64)

    dr = (fs * c) / float(slope * nfft)
    d0 = np.sqrt(np.sum((positions_tx[:, :2] - positions_rx[:, :2])**2, axis=1)).astype(np.float32)

    gx = gx.astype(np.float32); gy = gy.astype(np.float32)
    pos_tx = positions_tx.astype(np.float32)
    pos_rx = positions_rx.astype(np.float32)

    @nb.njit(parallel=True, fastmath=True)
    def kernel(X, gx, gy, pos_tx, pos_rx, d0, dr, ref_idx, wavelength):
        Ny = gy.size; Nx = gx.size; N = pos_tx.shape[0]; nfft = X.shape[1]
        out = np.zeros((Ny, Nx), dtype=np.complex64)
        two_pi_over_lambda = 2.0 * np.pi / wavelength
        for iy in nb.prange(Ny):
            y = gy[iy]
            for ix in range(Nx):
                xg = gx[ix]
                acc_real = 0.0
                acc_imag = 0.0
                for i in range(N):
                    dx_tx = xg - pos_tx[i,0]; dy_tx = y - pos_tx[i,1]
                    dx_rx = xg - pos_rx[i,0]; dy_rx = y - pos_rx[i,1]
                    r = np.hypot(dx_tx, dy_tx) + np.hypot(dx_rx, dy_rx) - d0[i]
                    idxf = r / dr + ref_idx
                    if idxf < 0: idxf = 0.0
                    if idxf > nfft - 2: idxf = nfft - 2.0
                    i0 = int(idxf)
                    frac = idxf - i0
                    v0 = X[i, i0]; v1 = X[i, i0 + 1]
                    samp = (1.0 - frac) * v0 + frac * v1
                    ph = -two_pi_over_lambda * r
                    cph = np.cos(ph) + 1j * np.sin(ph)
                    val = samp * cph
                    acc_real += val.real
                    acc_imag += val.imag
                out[iy, ix] = np.complex64(acc_real + 1j * acc_imag)
        return out

    img = kernel(X.astype(np.complex64), gx, gy, pos_tx, pos_rx, d0, dr, ref_idx, wavelength)
    return np.abs(img)







#####################################
# Add to src/dsp/sar.py
import numpy as np
import cupy as cp

def bistatic_sar_bp_gpu(adcData, positions_tx, positions_rx, gx, gy, nfft, params, return_complex=False, tile_y=None):
    """
    Bistatic backprojection on a horizontal plane using GPU.
    adcData: (N_pairs, N_samples) complex64
    positions_tx, positions_rx: (N_pairs, 2) or (N_pairs, 3) arrays with x,y,(z) in meters
    gx: (Nx,) x-grid (m)
    gy: (Ny,) y-grid (m)
    params: dict with keys 'c','frequency','slope','fs','ref_idx' (optional)
    """
    c = float(params['c']); fc = float(params['frequency'])
    fs = float(params['fs']); slope = float(params['slope'])
    ref_idx = int(params.get('ref_idx', 0))
    wavelength = c / fc

    # To GPU
    x = cp.asarray(adcData)                               # (N, Ns)
    pos_tx = cp.asarray(positions_tx, dtype=cp.float32)   # (N, 2|3)
    pos_rx = cp.asarray(positions_rx, dtype=cp.float32)   # (N, 2|3)
    gx = cp.asarray(gx, dtype=cp.float32)                 # (Nx,)
    gy = cp.asarray(gy, dtype=cp.float32)                 # (Ny,)

    # Range FFT
    Ns = x.shape[-1]
    if nfft is None: nfft = Ns
    win = cp.hanning(Ns).astype(x.dtype)
    X = cp.fft.fft(x * win[None, :], n=nfft, axis=-1)     # (N, N_range)

    # Range bin spacing (bistatic)
    df = fs / float(nfft)
    dr = c * df / float(slope)                            # matches your naive_bistatic_sar_bp

    # Baseline distance |Tx-Rx|
    # d0 = cp.sqrt(cp.sum((pos_tx[:, :2] - pos_rx[:, :2])**2, axis=1)).astype(cp.float32)  # (N,)
    # only removing the distance between the tx and the first rx
    d0 = cp.sqrt(cp.sum((pos_tx[:, :2] - pos_rx[[0], :2])**2, axis=1)).astype(cp.float32)  # (N,)
    # print(d0)

    Nx, Ny = gx.size, gy.size
    Xg = gx[None, None, :]  # (1,1,Nx)

    def process_y_slice(y_slice):
        Yg = y_slice[:, None]                             # (Ny_tile,1)

        # Distances from pixel (x,y) to Tx and Rx
        dx_tx = Xg - pos_tx[:, None, 0:1]                # (N, Ny_tile, Nx)
        dy_tx = Yg[None, :, :] - pos_tx[:, None, 1:2]
        dx_rx = Xg - pos_rx[:, None, 0:1]
        dy_rx = Yg[None, :, :] - pos_rx[:, None, 1:2]

        r_tx = cp.sqrt(dx_tx*dx_tx + dy_tx*dy_tx)
        r_rx = cp.sqrt(dx_rx*dx_rx + dy_rx*dy_rx)
        r = r_tx + r_rx - d0[:, None, None]              # (N, Ny_tile, Nx)

        # Sample complex range profiles with linear interpolation
        idx = cp.clip(r / dr + ref_idx, 0, nfft - 2)
        idx0 = cp.floor(idx).astype(cp.int32)
        frac = (idx - idx0).astype(cp.float32)

        X_exp = X[:, None, None, :]                      # (N,1,1,N_range)
        v0 = cp.take_along_axis(X_exp, idx0[..., None], axis=-1).squeeze(-1)
        v1 = cp.take_along_axis(X_exp, (idx0 + 1)[..., None], axis=-1).squeeze(-1)
        samp = (1.0 - frac) * v0 + frac * v1             # (N, Ny_tile, Nx)

        # Phase compensation (bistatic)
        phase = cp.exp(-1j * (2.0 * np.pi / wavelength) * r)
        return cp.sum(samp * phase, axis=0)              # (Ny_tile, Nx)

    if tile_y:
        out = []
        for s in range(0, Ny, tile_y):
            out.append(process_y_slice(gy[s:s + tile_y]))
        img = cp.concatenate(out, axis=0)
    else:
        img = process_y_slice(gy)

    return img.astype(cp.complex64) if return_complex else cp.abs(img)






#####################################
def bistatic_sar_bp_numpy(adcData, positions_tx, positions_rx, gx, gy, nfft, params, tile_y=64, return_complex=False):
    import numpy as np
    c = float(params['c']); fc = float(params['frequency'])
    fs = float(params['fs']); slope = float(params['slope'])
    ref_idx = int(params.get('ref_idx', 0))
    wavelength = c / fc

    x = np.asarray(adcData)                               # (N, Ns)
    pos_tx = np.asarray(positions_tx, dtype=np.float32)   # (N, 2|3)
    pos_rx = np.asarray(positions_rx, dtype=np.float32)   # (N, 2|3)
    gx = np.asarray(gx, dtype=np.float32)                 # (Nx,)
    gy = np.asarray(gy, dtype=np.float32)                 # (Ny,)

    Ns = x.shape[-1]
    if nfft is None: nfft = Ns
    win = np.hanning(Ns).astype(x.dtype)
    X = np.fft.fft(x * win[None, :], n=nfft, axis=-1)     # (N, N_range)

    df = fs / float(nfft)
    dr = c * df / float(slope)
    # print(dr)

    # d0 = np.sqrt(np.sum((pos_tx[:, :2] - pos_rx[:, :2])**2, axis=1)).astype(np.float32)  # (N,)
    d0 = np.sqrt(np.sum((pos_tx[:, :2] - pos_rx[[0], :2])**2, axis=1)).astype(np.float32)  # (N,)
    

    Nx = gx.size; Ny = gy.size
    Xg = gx[None, None, :]                                # (1,1,Nx)

    out = np.empty((Ny, Nx), dtype=np.complex64)
    for s in range(0, Ny, tile_y):
        y_slice = gy[s:s + tile_y]
        Yg = y_slice[:, None]                             # (Ny_tile,1)

        dx_tx = Xg - pos_tx[:, None, 0:1]
        dy_tx = Yg[None, :, :] - pos_tx[:, None, 1:2]
        dx_rx = Xg - pos_rx[:, None, 0:1]
        dy_rx = Yg[None, :, :] - pos_rx[:, None, 1:2]

        r_tx = np.sqrt(dx_tx*dx_tx + dy_tx*dy_tx)
        r_rx = np.sqrt(dx_rx*dx_rx + dy_rx*dy_rx)
        # r = r_tx + r_rx              # (N, Ny_tile, Nx)
        r = r_tx + r_rx - d0[:, None, None]              # (N, Ny_tile, Nx)

        idx = np.clip(r / dr + ref_idx, 0, nfft - 2)
        idx0 = np.floor(idx).astype(np.int32)
        frac = (idx - idx0).astype(np.float32)

        X_exp = X[:, None, None, :]                      # (N,1,1,N_range)
        v0 = np.take_along_axis(X_exp, idx0[..., None], axis=-1)[..., 0]
        v1 = np.take_along_axis(X_exp, (idx0 + 1)[..., None], axis=-1)[..., 0]
        samp = (1.0 - frac) * v0 + frac * v1             # (N, Ny_tile, Nx)

        phase = np.exp(-1j * (2.0 * np.pi / wavelength) * r)
        out[s:s + y_slice.size] = np.sum(samp * phase, axis=0).astype(np.complex64)

    return out if return_complex else np.abs(out)





def naive_bistatic_sar_bp(adcData, positions_ant_tx, positions_ant_rx, gx, gy, nfft, params):
    img = []

    fftData = np.fft.fft(adcData, axis=-1, n=nfft)
    wavelength = params['c'] / params['frequency']
    dr = params['fs']*params['c'] / (nfft * params['slope'])
    print(wavelength, dr)

    from itertools import product
    for xi, yi in product(gx, gy):
        val = 0
        pos_tar = np.array([xi, yi])
        for pos_ant_tx, pos_ant_rx, sig in zip(positions_ant_tx, positions_ant_rx, fftData):
            r = np.linalg.norm(pos_tar - pos_ant_tx) \
                + np.linalg.norm(pos_tar - pos_ant_rx) \
                - np.linalg.norm(pos_ant_tx - pos_ant_rx)
            # print(np.linalg.norm(pos_ant_tx - pos_ant_rx))
            # return
            # _wavelength = params['c'] / (params['frequency'] + params['slope'] * 2*r / params['c'])
            _wavelength = wavelength
            matched_filter = np.exp(-1j * (2.0 * np.pi / _wavelength) * r)
            try:
                val += matched_filter*sig[params['ref_idx']+int(r/dr)]
            except Exception as e:
                val = np.nan
        img.append(val)
    
    return np.array(img).reshape(len(gx), len(gy)).T


def naive_sar_bp(adcData, positions_ant, gx, gy, nfft, params):
    img = []

    fftData = np.fft.fft(adcData, axis=-1, n=nfft)
    wavelength = params['c'] / params['frequency']
    dr = params['fs']*params['c'] / (2.0 * nfft * params['slope'])
    print(wavelength)

    from itertools import product
    for xi, yi in product(gx, gy):
        val = 0
        for x_ant, sig in zip(positions_ant, fftData):
            r = np.sqrt((xi-x_ant)**2 + yi**2)
            # _wavelength = params['c'] / (params['frequency'] + params['slope'] * 2*r / params['c'])
            _wavelength = wavelength
            matched_filter = np.exp(-1j * (2.0 * np.pi / _wavelength) * r)
            try:
                val += matched_filter*sig[int(r/dr)]
            except Exception as e:
                val = np.nan
        img.append(val)
    
    return np.array(img).reshape(len(gx), len(gy)).T




def get_window(N_syn_ante: int, N_samples: int) -> cp.ndarray:
    """Get hanning window for combining different chirps and FFT."""
    return cp.hanning(N_syn_ante)[:, None] * cp.hanning(N_samples)[None]


#####################################
# 2D Bistatic MUSIC on angle–range (receiver ULA × fast-time)
def bistatic_music_2d(
    X: np.ndarray,
    fs_hz: float,
    slope_hz_per_s: float,
    fc_hz: float,
    K: int,
    theta_grid_rad: np.ndarray,
    rbi_grid_m: np.ndarray,
    rx_sensor_positions_m: np.ndarray = None,
    c: float = 3e8,
    tile_ranges: int = 64,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    CPU 2D-MUSIC for bistatic angle–range map.

    Model: vec(X) ≈ sum_k (a_rx(θ_k) ⊗ b_bi(Rbi_k)) s_k, where
      - a_rx(θ) is the RX ULA steering using actual sensor x-positions
      - b_bi(Rbi) uses FMCW beat fb = S * Rbi / c (bistatic excess path)

    Inputs
    - X: (M_rx, N_fast) or (M_rx, N_fast, T) complex array; T snapshots improve covariance
    - fs_hz: ADC rate [Hz]
    - slope_hz_per_s: FMCW slope [Hz/s] (convert beforehand if not SI)
    - fc_hz: carrier [Hz]
    - K: model order
    - theta_grid_rad: (A,) azimuth angles (radians) for RX ULA
    - rbi_grid_m: (R,) bistatic excess path length grid in meters (R_tx+R_rx−|Tx−Rx|)
    - rx_sensor_positions_m: (M_rx,) x-positions of RX sensors in meters; if None, λ/2 ULA

    Returns
    - P: pseudospectrum with shape (A, R) (angles x ranges)
    """
    X = np.asarray(X, dtype=np.complex64, order='F')
    if X.ndim == 2:
        M, N = X.shape
        Z = X.reshape(M * N, 1, order='F')
    else:
        M, N, T = X.shape
        Z = X.reshape(M * N, T, order='F')

    lam = c / float(fc_hz)
    if rx_sensor_positions_m is None:
        rx_sensor_positions_m = np.arange(M, dtype=np.float32) * (0.5 * lam)
    pos = np.asarray(rx_sensor_positions_m, dtype=np.float32).reshape(M, 1)

    # Covariance and noise subspace (match monolithic/slow formulation)
    R = (Z @ Z.conj().T) / Z.shape[1]
    w, V = np.linalg.eigh(R)  # ascending
    En = V[:, : max(1, M * N - int(K))].astype(np.complex64, copy=False)  # (M*N, #noise)
    EnH = En.conj().T

    # Steering
    A = np.exp(-1j * 2.0 * np.pi * (pos / lam) * np.sin(theta_grid_rad)[None, :]).astype(np.complex64)
    A /= np.sqrt(M, dtype=np.float32)

    n = np.arange(N, dtype=np.float32)
    fb = (float(slope_hz_per_s) * np.asarray(rbi_grid_m, dtype=np.float32)) / float(c)  # bistatic
    B = np.exp(+1j * 2.0 * np.pi * (n[:, None] / float(fs_hz)) * fb[None, :]).astype(np.complex64)
    B /= np.sqrt(N, dtype=np.float32)

    A_cnt = A.shape[1]
    R_cnt = B.shape[1]
    P = np.empty((A_cnt, R_cnt), dtype=np.float32)

    for r0 in range(0, R_cnt, tile_ranges):
        r1 = min(R_cnt, r0 + tile_ranges)
        Vblk = np.kron(B[:, r0:r1], A)  # (M*N, A*(tile))
        Ublk = EnH @ Vblk
        denom = np.maximum(np.sum(np.abs(Ublk) ** 2, axis=0).real, eps)
        P_block = (1.0 / denom).reshape(A_cnt, r1 - r0, order='F')
        P[:, r0:r1] = P_block.astype(np.float32, copy=False)

    return P


def bistatic_music_2d_gpu(
    X: np.ndarray,
    fs_hz: float,
    slope_hz_per_s: float,
    fc_hz: float,
    K: int,
    theta_grid_rad: np.ndarray,
    rbi_grid_m: np.ndarray,
    rx_sensor_positions_m: np.ndarray = None,
    c: float = 3e8,
    tile_ranges: int = 128,
    eps: float = 1e-7,
) -> np.ndarray:
    """
    GPU 2D-MUSIC (CuPy) for bistatic angle–range map.
    Returns numpy float32 P with shape (A, R).
    """
    import cupy as cp

    Xc = cp.asarray(X, dtype=cp.complex64, order='F')
    if Xc.ndim == 2:
        M, N = Xc.shape
        Z = Xc.reshape(M * N, 1, order='F')
    else:
        M, N, T = Xc.shape
        Z = Xc.reshape(M * N, T, order='F')

    lam = c / float(fc_hz)
    if rx_sensor_positions_m is None:
        rx_sensor_positions_m = cp.arange(M, dtype=cp.float32) * (0.5 * lam)
    else:
        rx_sensor_positions_m = cp.asarray(rx_sensor_positions_m, dtype=cp.float32)
    pos = rx_sensor_positions_m.reshape(M, 1)

    # Noise subspace via covariance eigendecomposition
    R = (Z @ Z.conj().T) / Z.shape[1]
    w, V = cp.linalg.eigh(R)  # ascending
    En = V[:, : max(1, M * N - int(K))].astype(cp.complex64, copy=False)
    EnH = En.conj().T

    theta_cp = cp.asarray(theta_grid_rad, dtype=cp.float32)
    rbi_cp = cp.asarray(rbi_grid_m, dtype=cp.float32)

    A = cp.exp(-1j * 2.0 * cp.pi * (pos / lam) * cp.sin(theta_cp)[None, :]).astype(cp.complex64)
    A /= cp.sqrt(cp.asarray(M, dtype=cp.float32))

    n = cp.arange(N, dtype=cp.float32)
    fb = (float(slope_hz_per_s) * rbi_cp) / float(c)
    B = cp.exp(+1j * 2.0 * cp.pi * (n[:, None] / float(fs_hz)) * fb[None, :]).astype(cp.complex64)
    B /= cp.sqrt(cp.asarray(N, dtype=cp.float32))

    A_cnt = int(A.shape[1])
    R_cnt = int(B.shape[1])
    P = cp.empty((A_cnt, R_cnt), dtype=cp.float32)

    for r0 in range(0, R_cnt, tile_ranges):
        r1 = min(R_cnt, r0 + tile_ranges)
        Vblk = cp.kron(B[:, r0:r1], A)
        Ublk = EnH @ Vblk
        denom = cp.maximum(cp.sum(cp.abs(Ublk) ** 2, axis=0).real, eps)
        P_block = (1.0 / denom).reshape(A_cnt, r1 - r0, order='F')
        P[:, r0:r1] = P_block.astype(cp.float32, copy=False)

    return cp.asnumpy(P)


#####################################
# Matched-field 2D MUSIC over x–y using bistatic phase/range model (GPU)
def bistatic_sar_music_gpu(
    adcData,
    positions_tx,
    positions_rx,
    gx,
    gy,
    params,
    tile_y: int = 64,
    eps: float = 1e-7,
):
    """
    Bistatic matched-field MUSIC on a horizontal plane using GPU (CuPy).

    Inputs
    - adcData:
        shape (N_pairs, N_samples)            single snapshot
        or     (N_pairs, N_samples, T)        T snapshots (preferred)
      Complex baseband per Rx channel (time domain, before FFT).
    - positions_tx: (N_pairs, 2) or (N_pairs, 3) Tx positions [m]
    - positions_rx: (N_pairs, 2) or (N_pairs, 3) Rx positions [m]
    - gx: (Nx,) x-grid [m]
    - gy: (Ny,) y-grid [m]
    - params: dict with keys 'c','frequency','slope','fs' (and optionally 'ref_idx')

    Returns
    - P: float32 pseudospectrum with shape (Ny, Nx)

    Notes
    - Steering per pixel p = (x,y) builds length-(N_pairs*N_samples) vector v(p) by
      concatenating, for each pair i:
        v_i(n; p) = exp(-j 2π/λ r_i(p)) * exp(+j 2π (n/fs) fb_i(p)),
        where r_i(p) = |p−Tx_i| + |p−Rx_i| − |Tx_i−Rx_ref| (match backprojection),
              fb_i(p) = slope * r_i(p) / c   (bistatic relation consistent with dr=c·df/slope).
    - MUSIC uses noise subspace from covariance of vec(X_t) across snapshots t.
    """
    import cupy as cp

    c = float(params['c']); fc = float(params['frequency'])
    fs = float(params['fs']); slope = float(params['slope'])
    lam = c / fc

    # To GPU and canonical shapes
    x = cp.asarray(adcData)
    if x.ndim == 2:
        N_pairs, N_samp = x.shape
        X = x[:, :, None]                      # (N_pairs, N_samp, 1)
    elif x.ndim == 3:
        N_pairs, N_samp, T = x.shape
        X = x
    else:
        raise ValueError("adcData must be (N_pairs, N_samples) or (N_pairs, N_samples, T)")

    pos_tx = cp.asarray(positions_tx, dtype=cp.float32)    # (N,2|3)
    pos_rx = cp.asarray(positions_rx, dtype=cp.float32)    # (N,2|3)
    gx = cp.asarray(gx, dtype=cp.float32)                  # (Nx,)
    gy = cp.asarray(gy, dtype=cp.float32)                  # (Ny,)

    # Build covariance of vec(X_t) over snapshots
    # vecF stacking: (N_pairs, N_samp) -> (N_pairs*N_samp, )
    N_pairs = int(X.shape[0]); 
    N_samp = int(X.shape[1]); 
    T = int(X.shape[2])
    Z = X.reshape(N_pairs * N_samp, T, order='F')          # (MN, T)
    R = (Z @ Z.conj().T) / max(1, T)                       # (MN, MN)
    w, V = cp.linalg.eigh(R)                               # ascending
    K = int(params.get('music_K', 2))
    En = V[:, :max(1, N_pairs * N_samp - K)].astype(cp.complex64, copy=False)
    EnH = En.conj().T

    # Precompute time axis for fast-time steering
    n = cp.arange(N_samp, dtype=cp.float32)                # (N_samp,)

    # Geometry helpers
    Nxg = int(gx.size); Nyg = int(gy.size)
    Xg = gx[None, None, :]                                 # (1,1,Nx)

    # Reference baseline: use distance from each Tx to the first Rx (matching bp implementation)
    d0 = cp.sqrt(cp.sum((pos_tx[:, :2] - pos_rx[[0], :2])**2, axis=1)).astype(cp.float32)  # (N,)

    def process_y_slice(y_slice):
        Yg = y_slice[:, None]                               # (Ny_tile,1)
        # Distances per pair to pixel
        dx_tx = Xg - pos_tx[:, None, 0:1]                   # (N, Ny_tile, Nx)
        dy_tx = Yg[None, :, :] - pos_tx[:, None, 1:2]
        dx_rx = Xg - pos_rx[:, None, 0:1]
        dy_rx = Yg[None, :, :] - pos_rx[:, None, 1:2]
        r_tx = cp.sqrt(dx_tx*dx_tx + dy_tx*dy_tx)
        r_rx = cp.sqrt(dx_rx*dx_rx + dy_rx*dy_rx)
        r = r_tx + r_rx - d0[:, None, None]                 # (N, Ny_tile, Nx)

        # Steering across time
        fb = (slope * r) / c                                # (N, Ny_tile, Nx)
        phase_time = cp.exp(+1j * 2.0*cp.pi * (n[None, None, None, :] / fs) * fb[..., None])  # (N, Ny, Nx, N_samp)
        phase_rng = cp.exp(-1j * (2.0*cp.pi/lam) * r)[..., None]                               # (N, Ny, Nx, 1)
        steer = phase_rng * phase_time                      # (N, Ny, Nx, N_samp)

        # Arrange as (N*N_samp, Ny*tile) with Fortran vec order per snapshot
        # mat = steer.transpose(0, 3, 1, 2).reshape(N_pairs * N_samp, -1, order='F')  # (MN, P)
        mat = steer.transpose(0, 3, 1, 2).reshape(N_pairs * N_samp, -1)  # (MN, P)

        # MUSIC denominator
        Ublk = EnH @ mat                                   # (#noise, P)
        denom = cp.maximum(cp.sum(cp.abs(Ublk)**2, axis=0).real, eps)
        Pblock = (1.0 / denom).reshape(y_slice.size, Nxg).astype(cp.float32)  # (Ny_tile, Nx)
        return Pblock

    # Tile over y for memory control
    if tile_y:
        out = []
        for s in range(0, Nyg, tile_y):
            out.append(process_y_slice(gy[s:s+tile_y]))
        P = cp.concatenate(out, axis=0)
    else:
        P = process_y_slice(gy)

    return cp.asnumpy(P)

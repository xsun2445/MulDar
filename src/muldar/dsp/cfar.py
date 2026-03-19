import numpy as np






#######################################################


def _box_sum_same(arr: np.ndarray, ha: int, hr: int) -> np.ndarray:
    """Compute sum over (2*ha+1)x(2*hr+1) box centered at each cell via separable conv.
    Uses zero padding (np.convolve(..., 'same')). Returns array with same shape as arr.
    """
    a = arr.astype(np.float64, copy=False)
    # Convolve along axis 0 (angles)
    if ha > 0:
        kA = np.ones(2 * ha + 1, dtype=np.float64)
        tmp = np.apply_along_axis(lambda m: np.convolve(m, kA, mode='same'), 0, a)
    else:
        tmp = a
    # Convolve along axis 1 (ranges)
    if hr > 0:
        kR = np.ones(2 * hr + 1, dtype=np.float64)
        out = np.apply_along_axis(lambda m: np.convolve(m, kR, mode='same'), 1, tmp)
    else:
        out = tmp
    return out


def ca_cfar_2d(mag: np.ndarray,
               guard_cells: tuple[int, int] = (1, 1),
               training_cells: tuple[int, int] = (8, 8),
               pfa: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Cell-Averaging CFAR on a 2D magnitude map (angle x range).

    Parameters
    - mag: 2D non-negative array (A, R)
    - guard_cells: (ga, gr) guard half-width (angles, ranges)
    - training_cells: (ta, tr) training half-width (angles, ranges)
    - pfa: desired probability of false alarm

    Returns
    - det_mask: bool mask of detections, same shape as mag
    - thresh: threshold map (float32), same shape as mag (zeros where invalid)
    """
    # mag -= np.mean(mag)
    A, R = mag.shape
    ga, gr = int(guard_cells[0]), int(guard_cells[1])
    ta, tr = int(training_cells[0]), int(training_cells[1])

    wa = ga + ta
    wr = gr + tr
    # print('wa', wa)
    # print('wr', wr)

    # Valid region where full CFAR window exists
    a0, a1 = wa, A - wa
    r0, r1 = wr, R - wr
    # print('a0', a0)
    # print('a1', a1)
    # print('r0', r0)
    # print('r1', r1)
    if a1 <= a0 or r1 <= r0:
        # Window is larger than the map
        return np.zeros_like(mag, dtype=bool), np.zeros_like(mag, dtype=np.float32)

    # Average-window (box) sums using separable convolution.
    outer_full = _box_sum_same(mag, wa, wr)  # includes training+guard+CUT
    inner_full = _box_sum_same(mag, ga, gr)  # includes guard+CUT

    # Extract valid region
    outer = outer_full[a0:a1, r0:r1]
    inner = inner_full[a0:a1, r0:r1]
    # print('outer shape', outer.shape)
    # print('inner shape', inner.shape)

    # Training sum per CUT
    train_sum = outer - inner
    # print('train_sum shape', train_sum.shape)
    # Number of training cells (constant)
    num_outer = (2 * wa + 1) * (2 * wr + 1)
    num_inner = (2 * ga + 1) * (2 * gr + 1)
    # print('num_outer', num_outer)
    # print('num_inner', num_inner)
    N = num_outer - num_inner
    N = max(N, 1)
    # print('N', N)

    noise = np.maximum(train_sum / float(N), 1e-12)
    # CA-CFAR scaling factor
    alpha = N * (pfa ** (-1.0 / N) - 1.0)
    print('alpha', alpha)
    alpha = 1.015
    # alpha = np.mean(np.abs(mag))/np.mean(noise)
    # print('alpha', alpha)
    thresh_valid = alpha * noise

    thresh = np.zeros_like(mag, dtype=np.float32)
    thresh[a0:a1, r0:r1] = thresh_valid.astype(np.float32)


    det = np.zeros_like(mag, dtype=bool)
    cut = mag[a0:a1, r0:r1]
    det[a0:a1, r0:r1] = cut > thresh_valid


    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(thresh.T, origin='lower', aspect='auto')
    # plt.title('CFAR threshold')
    # plt.colorbar(label='Threshold')
    # plt.tight_layout()
    # # plt.show()

    # plt.figure()
    # plt.imshow(cut.T, origin='lower', aspect='auto')
    # plt.title('Magnitude')
    # plt.colorbar(label='Magnitude')
    # plt.tight_layout()
    # # plt.show()

    # plt.figure()
    # plt.imshow(det.T, origin='lower', aspect='auto')
    # plt.title('Magnitude')
    # plt.colorbar(label='Magnitude')
    # plt.tight_layout()
    # plt.show()

    return det, thresh


def nms_2d(mag: np.ndarray, mask: np.ndarray, win: tuple[int, int] = (1, 1)) -> np.ndarray:
    """Non-maximum suppression within a (2*wa+1) x (2*wr+1) window around each detection.
    Keeps strict local maxima.
    """
    wa, wr = int(win[0]), int(win[1])
    if wa <= 0 and wr <= 0:
        return mask.copy()

    keep = mask.copy()
    # Compare with neighbors; if any neighbor strictly greater, suppress
    for da in range(-wa, wa + 1):
        for dr in range(-wr, wr + 1):
            if da == 0 and dr == 0:
                continue
            shifted = np.zeros_like(mag)
            src_a = slice(max(0, -da), mag.shape[0] - max(0, da))
            src_r = slice(max(0, -dr), mag.shape[1] - max(0, dr))
            dst_a = slice(max(0, da), mag.shape[0] - max(0, -da))
            dst_r = slice(max(0, dr), mag.shape[1] - max(0, -dr))
            shifted[dst_a, dst_r] = mag[src_a, src_r]
            keep &= mag >= shifted  # keep only if >= all neighbors
    # Also require original CFAR mask
    return keep & mask


def angle_range_cfar_pointcloud(mag: np.ndarray,
                                angles_rad: np.ndarray,
                                ranges_m: np.ndarray,
                                guard_cells: tuple[int, int] = (1, 1),
                                training_cells: tuple[int, int] = (8, 8),
                                pfa: float = 1e-3,
                                nms_win: tuple[int, int] = (1, 1)):
    """Detect points from an angle-range map using 2D CFAR and convert to XY point cloud.

    Parameters
    - mag: 2D magnitude (A, R)
    - angles_rad: (A,) azimuths in radians (centers)
    - ranges_m: (R,) ranges in meters (centers)
    - guard_cells, training_cells, pfa: CFAR parameters
    - nms_win: neighborhood half-size for non-maximum suppression

    Returns
    - points: (N, 3) array [x, y, intensity]
    - idxs: (N, 2) integer indices [ia, ir]
    """
    det, _th = ca_cfar_2d(mag, guard_cells=guard_cells,
                          training_cells=training_cells, pfa=pfa)
    print(det.shape)
    det = nms_2d(mag, det, win=nms_win)

    ia, ir = np.nonzero(det)
    if ia.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.int32), _th, np.zeros((0,)), np.zeros((0,))

    th = angles_rad[ia]
    rr = ranges_m[ir]
    # x = th
    # y = rr
    y = rr * np.cos(th)
    x = rr * np.sin(th)
    inten = mag[ia, ir].astype(np.float32)
    pts = np.vstack([x, y, inten]).T.astype(np.float32)
    idxs = np.vstack([ia, ir]).T.astype(np.int32)
    # print(pts)
    # print(idxs)
    # print(_th)
    return pts, idxs, _th, th, rr


def simple_cfar_avg(mag: np.ndarray,
                    win: tuple[int, int] = (9, 9),
                    factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Simplest 2D CFAR-like detector: local mean as threshold using only numpy.

    Parameters
    - mag: 2D magnitude map (A, R)
    - win: (h, w) odd window size; uses zero-padded box average
    - factor: threshold scale; detection if mag > factor * local_mean

    Returns
    - det: bool mask (A, R)
    - thresh: float32 threshold map (A, R)
    """
    h, w = int(win[0]), int(win[1])
    if h < 1 or w < 1 or h % 2 == 0 or w % 2 == 0:
        raise ValueError("win should be odd sizes, e.g., (9,9)")
    ha, hr = h // 2, w // 2

    # Box average via separable convolution (zero-padded)
    a = mag.astype(np.float64, copy=False)
    kA = np.ones(h, dtype=np.float64)
    kR = np.ones(w, dtype=np.float64)
    tmp = np.apply_along_axis(lambda m: np.convolve(m, kA, mode='same'), 0, a)
    sum_map = np.apply_along_axis(lambda m: np.convolve(m, kR, mode='same'), 1, tmp)
    area = float(h * w)
    mean_map = (sum_map / area).astype(np.float32)

    thresh = factor * mean_map
    det = mag > thresh
    return det, thresh




def ca(x, *argv, **kwargs):
    """Detects peaks in signal using Cell-Averaging CFAR (CA-CFAR).

    Args:
        x (~numpy.ndarray): Signal.
        *argv: See mmwave.dsp.cfar.ca
        **kwargs: See mmwave.dsp.cfar.ca

    Returns:
        ~numpy.ndarray: Boolean array of detected peaks in x.

    Examples:
        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> det = mm.dsp.ca(signal, l_bound=20, guard_len=1, noise_len=3)
        >>> det
            array([False, False,  True, False, False, False, False,  True, False,
                    True])

        Perform a non-wrapping CFAR

        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> det =  mm.dsp.ca(signal, l_bound=20, guard_len=1, noise_len=3, mode='constant')
        >>> det
            array([False,  True,  True, False, False, False, False,  True,  True,
                    True])

    """
    if isinstance(x, list):
        x = np.array(x)
    threshold, _ = ca_(x, *argv, **kwargs)
    ret = (x > threshold)
    return ret, threshold

def ca_(x, guard_len=4, noise_len=8, mode='wrap', l_bound=4000):
    """Uses Cell-Averaging CFAR (CA-CFAR) to calculate a threshold that can be used to calculate peaks in a signal.

    Args:
        x (~numpy.ndarray): Signal.
        guard_len (int): Number of samples adjacent to the CUT that are ignored.
        noise_len (int): Number of samples adjacent to the guard padding that are factored into the calculation.
        mode (str): Specify how to deal with edge cells. Examples include 'wrap' and 'constant'.
        l_bound (float or int): Additive lower bound while calculating peak threshold.

    Returns:
        Tuple [ndarray, ndarray]
            1. (ndarray): Upper bound of noise threshold.
            #. (ndarray): Raw noise strength.

    Examples:
        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> threshold = mm.dsp.ca_(signal, l_bound=20, guard_len=1, noise_len=3)
        >>> threshold
            (array([70, 76, 64, 79, 81, 91, 74, 71, 70, 79]), array([50, 56, 44, 59, 61, 71, 54, 51, 50, 59]))

        Perform a non-wrapping CFAR thresholding

        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> threshold = mm.dsp.ca_(signal, l_bound=20, guard_len=1, noise_len=3, mode='constant')
        >>> threshold
            (array([44, 37, 41, 65, 81, 91, 67, 51, 34, 46]), array([24, 17, 21, 45, 61, 71, 47, 31, 14, 26]))

    """
    from scipy.ndimage import convolve1d
    if isinstance(x, list):
        x = np.array(x)
    assert type(x) == np.ndarray

    kernel = np.ones(1 + (2 * guard_len) + (2 * noise_len), dtype=x.dtype) / (2 * noise_len)
    kernel[noise_len:noise_len + (2 * guard_len) + 1] = 0

    noise_floor = convolve1d(x, kernel, mode=mode)
    threshold = noise_floor + l_bound

    return threshold, noise_floor

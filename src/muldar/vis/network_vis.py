import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import muldar.utils as utils
import muldar.dsp.bistatic as bistatic
import muldar.dsp.sar as sar
from collections import deque


def _default_grid():
    # Match figures/table defaults but slightly reduced for realtime
    img_x_size = 192
    img_y_size = 192
    gx = np.linspace(-0.6939 - 0.5, 0.7285 + 0.5, img_x_size)
    gy = np.linspace(-0.1, 2.4224 - 0.1, img_y_size)
    return gx, gy


def _build_params(base_params, ave_win=4, sar_nfft=1024, align_params=None):
    rp = base_params['ramp_cfg']
    p = {
        'frequency': rp['f0'],
        'fs': rp['Fs'],
        'slope': rp['slope'],
        'c': 299792458.0,
        'sar_nfft': sar_nfft,
        'ref_idx': 256,
        'ave_win': int(ave_win),
        # Force offline-aligned sync defaults for robust bistatic alignment
        'align_params': {
            'nfft': 1024,
            'f_tar_idx': 256,
            'phase_ref': 0.0,
            'min_height_ratio': 0.7,
            'min_prominence_ratio': 0.5,
        },
        'activated_radar': base_params['activated_radar'],
    }
    return p


def _load_dataset_ramp_cfg(mgr):
    """
    If replaying from recorded files, prefer the dataset's saved configs.yml
    to ensure Fs/slope/f0 match the capture session.
    """
    try:
        import os, yaml
        for rs in getattr(mgr, 'radars', []):
            dca = getattr(rs, 'dca', None)
            file_path = getattr(dca, 'file_path', None)
            if not file_path:
                continue
            cfg_path = os.path.join(os.path.dirname(file_path), 'configs.yml')
            if os.path.isfile(cfg_path):
                ds = yaml.load(open(cfg_path), Loader=yaml.FullLoader)
                rc = ds.get('ramp_cfg', {})
                if all(k in rc for k in ('f0', 'Fs', 'slope')):
                    return {
                        'frequency': rc['f0'],
                        'fs': rc['Fs'],
                        'slope': rc['slope'],
                    }
        return None
    except Exception:
        return None


def start_combined_visualization(
    mgr,
    gx=None,
    gy=None,
    ave_win=4,
    lambda_mono_factor=1.5,
    sar_nfft=None,
    interval_ms=None,
):
    """
    Real-time 3-panel XY view: Mono sum, Multi sum, All sum, computed from 3x3 (tx,rx) bistatic images.
    - Uses the latest ave_win frames from each RX radar stream.
    - Runs GPU backprojection when available, with fallback to CPU.
    """
    if gx is None or gy is None:
        gx, gy = _default_grid()
    # Resolve sar_nfft from runtime params if not provided
    if sar_nfft is None:
        try:
            sar_nfft = int(mgr.params.get('sar_nfft', 1024))
        except Exception:
            sar_nfft = 1024
    params = _build_params(mgr.params, ave_win=ave_win, sar_nfft=int(sar_nfft))
    # Prefer dataset ramp_cfg if replaying from files
    rc_override = _load_dataset_ramp_cfg(mgr)
    if rc_override:
        params.update(rc_override)
    radar_count = len(mgr.radars)
    assert radar_count >= 1, "No radars are active."
    # figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    axes = axes[0]
    titles = ("Mono sum", "Multi sum", "All sum")
    ims = []
    for i, ax in enumerate(axes):
        im = ax.imshow(
            np.zeros((gy.size, gx.size), dtype=np.float32),
            origin='lower',
            extent=[gx[0], gx[-1], gy[0], gy[-1]],
            aspect='equal',
            vmin=0.0,
            vmax=1.0,
            cmap='viridis',
        )
        ax.set_title(titles[i])
        ax.set_xlabel('x [m]')
        if i == 0:
            ax.set_ylabel('y [m]')
        ims.append(im)
    # leave room for suptitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    title_text = fig.suptitle("Frame -", fontsize=12)

    # Fixed global color limits after first valid frame
    vmax_fixed = None
    vmin_fixed = None

    # Fixed global vmax after first valid frame
    vmax_fixed = None

    # Precompute RX masks for FOV gating
    poses = [np.array(r['pose']) for r in mgr.params['activated_radar']]
    rx_masks = [bistatic.img_mask(poses[ri], gx, gy).astype(np.float32) for ri in range(min(3, radar_count))]

    # TX -> list of its two mono chirp indices (fallback to [2*idx, 2*idx+1])
    tx_cfg_indices = []
    for i in range(min(3, radar_count)):
        idxs = getattr(mgr.radars[i], 'cfg_idx', None)
        if isinstance(idxs, (list, tuple)) and len(idxs) >= 2:
            tx_cfg_indices.append((int(idxs[0]), int(idxs[1])))
        else:
            tx_cfg_indices.append((i * 2, i * 2 + 1))

    # Histories per RX radar per needed config: store last frames to support ave_win>1 even during file replay
    needed_cfgs = sorted(set([c for pair in tx_cfg_indices for c in pair]))
    N_view = min(3, radar_count)
    histories = [
        {cfg: deque(maxlen=max(1, int(ave_win))) for cfg in needed_cfgs}
        for _ in range(N_view)
    ]

    def compute_pair_image(rx_idx, tx_idx, rx_data):
        """
        Returns complex image (H,W) for given rx,tx using the latest ave_win frames.
        """
        # Follow offline logic: use contiguous pair [tx*2, tx*2+1] from RX data
        cfg0 = tx_idx * 2
        cfg1 = cfg0 + 1
        if rx_data.shape[2] <= cfg1:
            return None
        # Take last local_ave frames from time dimension
        T = int(rx_data.shape[1])
        local_ave = max(1, min(int(ave_win), T))
        # print(rx_idx, tx_idx, cfg0, cfg1)
        f0 = rx_data[:, -local_ave:, cfg0, :]  # (num_ch, local_ave, num_adc)
        f1 = rx_data[:, -local_ave:, cfg1, :]
        # Geometry
        pos_tx = utils.antenna_positions(poses[tx_idx])[[4, 6]]
        pos_rx = utils.antenna_positions(poses[rx_idx])[[0, 1, 2, 3]]
        ref_ang = np.deg2rad(poses[rx_idx][2]) - np.arctan2(
            poses[tx_idx][1] - poses[rx_idx][1], poses[tx_idx][0] - poses[rx_idx][0]
        )
        imgs = []
        for idx, frameData in enumerate((f0, f1)):
            ch_select = 0 if idx == 0 else 1
            # Use a local params copy with ave_win adjusted to current buffer
            p_local = dict(params)
            p_local['ave_win'] = local_ave
            try:
                # print(tx_idx, rx_idx, ch_select)
                img_list = bistatic._genBiStaticSAR(
                    frameData, tx_idx, rx_idx, pos_tx[[ch_select]], pos_rx, gx, gy, p_local, ref_ang, p_local['align_params']
                )
                imgs.append(img_list[0])
            except Exception:
                print('Error in compute_pair_image')
                # Fallback to CPU: average across time then backproject
                avesig = frameData.reshape(frameData.shape[0], -1, local_ave, frameData.shape[-1]).mean(axis=2)  # (num_ch, 1, num_adc)
                nfft_bp = int(p_local.get('sar_nfft', sar_nfft))
                comp = sar.bistatic_sar_bp_numpy(avesig[:, 0, :], pos_tx[[ch_select]], pos_rx, gx, gy, nfft_bp, p_local, return_complex=True)
                imgs.append(comp)
        # Average the two TX elements (complex) to match offline behavior
        return (0.5 * (imgs[0] + imgs[1])) * (rx_masks[rx_idx] if rx_idx < len(rx_masks) else 1.0)

    def update(_):
        nonlocal vmax_fixed, vmin_fixed
        # Build imgs[tx, rx] complex images
        N = min(3, radar_count)
        imgs = [[None for _ in range(N)] for _ in range(N)]
        frame_counts = []
        frame_counters = []
        for rx_idx in range(N):
            rs = mgr.radars[rx_idx]
            data = rs.get_latest()
            if data is None:
                continue
            # frames in current buffer
            try:
                frame_counts.append(int(data.shape[1]))
            except Exception:
                pass
            # total frames seen by source
            try:
                cnt = int(getattr(rs, 'dca', None).frame_counter)
                frame_counters.append(cnt)
            except Exception:
                pass
            for tx_idx in range(N):
                img = compute_pair_image(rx_idx, tx_idx, data)
                if img is not None:
                    imgs[tx_idx][rx_idx] = img

        # Update frame title from counters first (always update, even if images missing)
        if frame_counters:
            frame_disp = int(max(frame_counters))
        elif frame_counts:
            frame_disp = int(max(frame_counts))
        else:
            frame_disp = 0
        title_text.set_text(f"Frame {frame_disp}")

        # If we do not have all diagonal pairs, we still want the title updated
        try:
            # Incoherent mono sum (match offline: sum of magnitudes, not magnitude of sum)
            mono_sum = None
            for k in range(N):
                if imgs[k][k] is None:
                    return [title_text]
                mag_k = np.abs(imgs[k][k])
                mono_sum = mag_k if mono_sum is None else (mono_sum + mag_k)
            mono_sum = mono_sum * float(lambda_mono_factor)

            # Multi-static symmetric pairs
            def pair(a, b):
                return (imgs[a][b] if imgs[a][b] is not None else 0) + (imgs[b][a] if imgs[b][a] is not None else 0)

            multi = 0
            if N >= 2:
                multi += np.abs(pair(0, 1))
            if N >= 3:
                multi += np.abs(pair(0, 2))
                multi += np.abs(pair(1, 2))
            all_sum = mono_sum + multi

            # Fixed global scaling for visualization
            stack = np.stack([mono_sum, multi, all_sum], axis=0)
            if vmax_fixed is None:
                _v = float(np.percentile(stack, 99.99)) if np.isfinite(stack).all() else float(np.max(stack))
                vmax_fixed = max(_v, 1e-6)
                vmin_fixed = max(0.02 * vmax_fixed, 0.0)
            planes = (mono_sum, multi, all_sum)
            for im, p in zip(ims, planes):
                im.set_data(p)
                im.set_clim(vmin_fixed if vmin_fixed is not None else 0.0, vmax_fixed)
            return ims + [title_text]
        except Exception:
            return [title_text]

    if interval_ms is None:
        interval_ms = int(mgr.params.get('period_frame', 100)) * max(1, int(ave_win))
    ani = FuncAnimation(fig, update, interval=interval_ms, blit=False, cache_frame_data=False)
    plt.show(block=True)
    return fig, axes, ims


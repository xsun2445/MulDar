import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import muldar.dsp.dsp as dsp
import traceback

########################################Static Vis
def vis_beamforming(adcData, angle_range=[-90,90,181], nfft=512, doplot=True, title=''):
    temp_bf = []
    for a in np.linspace(*angle_range):
        steering_vec = np.exp(-1j*np.pi*np.sin(np.deg2rad(a))*np.arange(adcData.shape[0]))[None, :]
        temp_bf.append((steering_vec@adcData).ravel())
    temp_bf = np.array(temp_bf)
    if doplot:
        plt.figure()
        plt.imshow(
            np.abs(np.fft.fft(temp_bf, axis=-1, n=nfft)).T
            , origin='lower', aspect='auto',
            extent=[angle_range[0], angle_range[1], 0, nfft],)
        plt.title(title)

    return temp_bf


########################################Dynamic Vis
# Visualizing the whole network
def _edges(v):
    v = np.asarray(v).ravel()
    if v.size == 1:
        dv = np.pi/180.0
        return np.array([v[0]-dv/2, v[0]+dv/2])
    dv = np.diff(v)
    return np.concatenate(([v[0]-dv[0]/2], v[:-1]+dv/2, [v[-1]+dv[-1]/2]))


def build_overlays(extent, ranges_m, angle_bins_rad, poses, cmaps, alphas,
                   fig_size=(12,12), num_range_bin=None, vis_range=None):
    xmin, xmax, ymin, ymax = extent
    fig = plt.figure(figsize=fig_size, facecolor="white")
    ax_bg = fig.add_axes([0,0,1,1])
    ax_bg.set_aspect('equal', adjustable='box')
    ax_bg.imshow(np.ones((10, 10)), origin="lower", extent=[xmin, xmax, ymin, ymax], cmap="gray", vmin=0, vmax=1, aspect='equal')
    ax_bg.set_axis_off()

    th_e = _edges(angle_bins_rad)
    ranges_base = ranges_m if (num_range_bin is None or num_range_bin >= len(ranges_m)) else ranges_m[:int(num_range_bin)]
    if vis_range is not None and float(vis_range) > 0:
        ranges_plot = ranges_base[ranges_base <= float(vis_range)]
    else:
        ranges_plot = ranges_base
    r_e = _edges(ranges_plot)

    overlays = []
    for i, p in enumerate(poses):
        ThE, RE = np.meshgrid(th_e, r_e, indexing='ij')
        ThG = ThE + float(p['yaw'])
        X_e = float(p['x']) + RE * np.cos(ThG)
        Y_e = float(p['y']) + RE * np.sin(ThG)
        ax = fig.add_axes([0,0,1,1])
        ax.set_axis_off()
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # light FOV wedge background
        try:
            from matplotlib.patches import Wedge
            rmax = float(ranges_plot.max()) if ranges_plot.size else 1.0
            ang_min = float(np.min(angle_bins_rad))
            ang_max = float(np.max(angle_bins_rad))
            th1 = np.rad2deg(ang_min + float(p['yaw']))
            th2 = np.rad2deg(ang_max + float(p['yaw']))
            base_color = plt.get_cmap(cmaps[i % len(cmaps)])(0.6)
            wedge = Wedge((float(p['x']), float(p['y'])), rmax, th1, th2,
                          facecolor=base_color, edgecolor=None, alpha=0.08)
            ax.add_patch(wedge)
        except Exception:
            # print(f"Error in building overlays: {p['name']}")
            # print(traceback.format_exc())
            pass
        C0 = np.zeros((len(angle_bins_rad), len(ranges_plot)), dtype=np.float32)
        qm = ax.pcolormesh(X_e, Y_e, C0, shading="auto", cmap=cmaps[i % len(cmaps)], vmin=0, vmax=1, alpha=alphas[i % len(alphas)])
        overlays.append({'qm': qm, 'X_e': X_e, 'Y_e': Y_e, 'r_cols': C0.shape[1]})
    return fig, overlays


def init_visualization(params):
    adc_shape = params['chirp_cfg']
    # Range–Angle settings
    fs_hz = float(params['ramp_cfg']['Fs'])
    slope_hz_per_s = float(params['ramp_cfg']['slope'])
    nfft_range = params['visual_cfg']['nfft_range']
    nfft_angle = params['visual_cfg']['nfft_angle']
    num_range_bin = 120  # plot only first N range bins

    print(f"resolution (m/bin): {fs_hz*3e8 /(2* nfft_range*slope_hz_per_s):.6f}")

    # Poses for 3 radars (set actual positions/yaws here)
    poses = []
    cmaps = []
    alphas = []
    for r in params['activated_radar']:
        x, y, yaw_deg = r['pose']
        poses.append({'x': float(x), 'y': float(y), 'yaw': np.deg2rad(float(yaw_deg))})
        cmaps.append(r.get('cmap', 'viridis'))
        alphas.append(float(r.get('alpha', 0.5)))

    # Precompute bins for placeholder init
    dummy_frame = np.zeros((8, adc_shape['num_adc']), dtype=np.complex64)
    _, ang_bins, rngs = dsp.compute_range_angle(dummy_frame, fs_hz, slope_hz_per_s, nfft_range, nfft_angle)

    # Build global extent around poses and max range (use visual_cfg.max_range if provided)
    rmax_cfg = float(params.get('visual_cfg', {}).get('max_range', 0) or 0)
    rmax_cfg = 0
    rmax_fft = float(rngs[:max(1, num_range_bin)].max()) if rngs[:num_range_bin].size else 1.0
    rmax = rmax_cfg if rmax_cfg > 0 else rmax_fft
    extent = (-1.5,1.5,-0.7,-0.7+3)

    print(f"max_range used (m): {rmax:.3f}")

    vis_range = rmax  # enforce cutoff in the overlay grid
    fig, overlays = build_overlays(extent, rngs, ang_bins, poses, cmaps, alphas,
                                   fig_size=(12,12), num_range_bin=num_range_bin, vis_range=vis_range)

    return fig, overlays


def start_visualization(mgr):
    fs_hz = mgr.params['ramp_cfg']['Fs']
    slope_hz_per_s = mgr.params['ramp_cfg']['slope']
    nfft_range = mgr.params['visual_cfg']['nfft_range']
    nfft_angle = mgr.params['visual_cfg']['nfft_angle']
    
    def update(_):
        artists = []
        for rs, overlay in zip(mgr.radars, overlays):
            data = rs.get_latest()
            if data is None:
                continue                
            try:
                idxs = rs.cfg_idx
                i0, i1 = int(idxs[0]), int(idxs[1])

                frame = np.concatenate([data[:, -1, i0, :], data[:, -1, i1, :]], axis=0)

                mag, _, _ = dsp.compute_range_angle(frame, fs_hz, slope_hz_per_s, nfft_range, nfft_angle)
                # apply range crop to match overlay grid
                r_cols = overlay.get('r_cols', mag.shape[1])
                if mag.shape[1] > r_cols:
                    mag = mag[:, :r_cols]
                
                vmax = np.max(mag)
                thr = 0.3
                
                norm = np.clip(mag / vmax, 0.0, 1.0)
                
                norm[norm < thr] = np.nan
                # norm = norm**2
                # Set array for color mapping
                overlay['qm'].set_array(norm.ravel(order='C'))
                overlay['qm'].set_clim(thr, 1.0)
                # Create hard alpha mask: 0 below threshold, 1 above
                alpha_mask = (norm >= 0.5).astype(np.float32)
                rgba = overlay['qm'].get_cmap()(norm)
                rgba[..., 3] = alpha_mask
                # overlay['qm'].set_alpha(1.0)
                overlay['qm'].set_alpha(0.6)
                overlay['qm'].set_facecolors(rgba.reshape(-1, 4))
                artists.append(overlay['qm'])
            except Exception:
                # print out the error
                print(f"Error radar visualization: {rs.name}")
                print(traceback.format_exc())
                continue
        return artists
    
    fig, overlays = init_visualization(mgr.params)
    ani = FuncAnimation(fig, update, interval=300, blit=False, cache_frame_data=False)
    plt.show(block=True)



def visualize_waveform(mgr):
    """
    Realtime FFT of IF signals for each radar (rows) and config (cols).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    n_radars = len(mgr.radars)
    n_cfg = mgr.params['chirp_cfg']['num_config']
    num_ch = int(mgr.params['chirp_cfg']['num_ch'])
    nfft = 1024

    fig, axs = plt.subplots(
        n_radars, n_cfg, figsize=(4 + 3 * n_cfg, 3 + 2 * n_radars), squeeze=False
    )
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    fig.suptitle("Realtime Waveform FFT per Radar/Config", fontsize=16)

    # list of line handles per subplot: lines[row][col] -> [Line2D, ...] length = num_ch
    lines = [[None for _ in range(n_cfg)] for _ in range(n_radars)]
    x = np.arange(nfft)

    for i in range(n_radars):
        for j in range(n_cfg):
            ax = axs[i, j]
            lns = [ax.plot(x, np.zeros(nfft), lw=1)[0] for _ in range(num_ch)]
            ax.set_title(f"Radar {i} | Cfg {j}")
            ax.set_xlabel("FFT bin")
            ax.set_ylabel("Magnitude")
            ax.set_xlim(0, nfft - 1)
            ax.set_ylim(0, 1)
            lines[i][j] = lns

    def update(_):
        artists = []
        for i, rs in enumerate(mgr.radars):
            data = rs.get_latest()
            if data is None:
                continue
            for k in range(n_cfg):
                try:
                    # data shape: (ant, time, cfg, samples)
                    wave = data[:, -1, k, :]                              # (num_ch, samples)
                    fft_mag = np.abs(np.fft.fft(wave[:,16:-96], n=nfft, axis=-1))   # (num_ch, nfft)

                    # update each channel's line
                    for ch, ln in enumerate(lines[i][k]):
                        ln.set_ydata(fft_mag[ch])
                        artists.append(ln)

                    ax = axs[i, k]
                    ax.set_ylim(0, max(1e-6, float(fft_mag.max())) * 1.1)
                    ax.set_xlim(0, nfft - 1)
                except Exception as e:
                    print("error in visualization:", e)
                    continue
        return artists

    ani = FuncAnimation(fig, update, interval=mgr.params['period_frame'], blit=False, cache_frame_data=False)
    plt.show(block=True)
    return fig, axs


def visualize_waveform_clean(mgr, save_gif=False, gif_path="waveform.gif", gif_frames=50, gif_fps=10, figsize=None, dpi=100, configs=None):
    """
    Realtime FFT of IF signals — no tick values, only axis labels
    (Amplitude / IF Frequency).

    Args:
        save_gif: If True, save the animation as a GIF.
        gif_path: Output GIF file path.
        gif_frames: Number of frames to capture for the GIF.
        gif_fps: Frames per second in the saved GIF.
        configs: List of config indices to display. None = all configs.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    n_radars = len(mgr.radars)
    n_cfg = mgr.params['chirp_cfg']['num_config']
    num_ch = int(mgr.params['chirp_cfg']['num_ch'])
    nfft = 1024
    plot_cfgs = configs if configs is not None else list(range(n_cfg))
    n_plot_cfg = len(plot_cfgs)

    if figsize is None:
        figsize = (4 + 3 * n_plot_cfg, 3 + 2 * n_radars)
    fig, axs = plt.subplots(
        n_radars, n_plot_cfg, figsize=figsize, squeeze=False
    )
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    fig.suptitle("Realtime Waveform FFT per Radar/Config", fontsize=16)

    lines = [[None for _ in range(n_plot_cfg)] for _ in range(n_radars)]
    x = np.arange(nfft)

    for i in range(n_radars):
        for col, cfg_idx in enumerate(plot_cfgs):
            ax = axs[i, col]
            lns = [ax.plot(x, np.zeros(nfft), lw=1)[0] for _ in range(num_ch)]
            ax.set_title(f"Radar {i} | Cfg {cfg_idx}")
            ax.set_xlabel("IF Frequency")
            ax.set_ylabel("Amplitude")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='both', length=0)
            ax.set_xlim(0, nfft - 1)
            ax.set_ylim(0, 1)
            lines[i][col] = lns

    def update(_):
        artists = []
        for i, rs in enumerate(mgr.radars):
            data = rs.get_latest()
            if data is None:
                continue
            for col, cfg_idx in enumerate(plot_cfgs):
                try:
                    wave = data[:, -1, cfg_idx, :]
                    fft_mag = np.abs(np.fft.fft(wave[:, 16:-96], n=nfft, axis=-1))

                    for ch, ln in enumerate(lines[i][col]):
                        ln.set_ydata(fft_mag[ch])
                        artists.append(ln)

                    ax = axs[i, col]
                    ax.set_ylim(0, max(1e-6, float(fft_mag.max())) * 1.1)
                    ax.set_xlim(0, nfft - 1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                except Exception as e:
                    print("error in visualization:", e)
                    continue
        return artists

    captured_frames = []

    def update_and_capture(frame_num):
        artists = update(frame_num)
        if save_gif and len(captured_frames) < gif_frames:
            fig.canvas.draw()
            captured_frames.append(np.array(fig.canvas.buffer_rgba()).copy())
        return artists

    ani = FuncAnimation(fig, update_and_capture, interval=mgr.params['period_frame'], blit=False, cache_frame_data=False)
    plt.show(block=True)

    if save_gif and captured_frames:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in captured_frames]
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                           loop=0, duration=int(1000 / gif_fps))
        print(f"GIF saved to {gif_path} ({len(captured_frames)} frames)")

    return fig, axs


def visualize_waveform_synced(mgr, save_gif=False, gif_path="waveform_synced.gif", gif_frames=50, gif_fps=10,
                              figsize=None, dpi=100, configs=None,
                              save_coherent_avg=False, coherent_avg_path="coherent_avg.png"):
    """
    Realtime FFT of IF signals after reference-path synchronization.
    Uses bistatic.sync_signal logic: beamform towards TX, find reference peak,
    then frequency/phase-shift all channels to align.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import muldar.dsp.dsp as dsp_mod

    n_radars = len(mgr.radars)
    n_cfg = mgr.params['chirp_cfg']['num_config']
    num_ch = int(mgr.params['chirp_cfg']['num_ch'])
    nfft = 1024
    plot_cfgs = configs if configs is not None else list(range(n_cfg))
    n_plot_cfg = len(plot_cfgs)

    align_params = {
        'nfft': 1024,
        'f_tar_idx': 256,
        'phase_ref': 0.0,
        'min_height_ratio': 0.7,
        'min_prominence_ratio': 0.5,
    }

    # Build TX-config mapping from radar_network (all radars, not just activated)
    network = mgr.params.get('radar_network', mgr.params['activated_radar'])
    network_poses = {int(r['idx']): np.array(r['pose']) for r in network}
    # Map config -> TX radar idx via mono_chirp_idx
    cfg_to_tx = {}
    for r in network:
        for c in r.get('mono_chirp_idx', []):
            cfg_to_tx[int(c)] = int(r['idx'])
    # Map mgr.radars index -> radar idx in network
    active_idx_map = {}
    for i, r in enumerate(mgr.params['activated_radar']):
        active_idx_map[i] = int(r['idx'])

    def get_ref_angle(rx_mgr_idx, cfg_idx):
        """Compute steering angle from RX toward TX for a given config."""
        tx_radar_idx = cfg_to_tx.get(cfg_idx)
        if tx_radar_idx is None:
            return None
        rx_radar_idx = active_idx_map.get(rx_mgr_idx)
        if rx_radar_idx is None or tx_radar_idx == rx_radar_idx:
            return None  # monostatic, no sync needed
        rx_pose = network_poses[rx_radar_idx]
        tx_pose = network_poses[tx_radar_idx]
        ref_ang = np.deg2rad(rx_pose[2]) - np.arctan2(
            tx_pose[1] - rx_pose[1],
            tx_pose[0] - rx_pose[0]
        )
        return ref_ang

    if figsize is None:
        figsize = (4 + 3 * n_plot_cfg, 3 + 2 * n_radars)
    fig, axs = plt.subplots(
        n_radars, n_plot_cfg, figsize=figsize, squeeze=False
    )
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    fig.suptitle("Synced Waveform FFT per Radar/Config", fontsize=16)

    lines = [[None for _ in range(n_plot_cfg)] for _ in range(n_radars)]
    x = np.arange(nfft)

    for i in range(n_radars):
        for col, cfg_idx in enumerate(plot_cfgs):
            ax = axs[i, col]
            lns = [ax.plot(x, np.zeros(nfft), lw=1)[0] for _ in range(num_ch)]
            ax.set_title(f"Radar {i} | Cfg {cfg_idx}")
            ax.set_xlabel("IF Frequency")
            ax.set_ylabel("Amplitude")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='both', length=0)
            ax.set_xlim(0, nfft - 1)
            ax.set_ylim(0, 1)
            lines[i][col] = lns

    # Buffer for coherent averaging: dict keyed by (radar_mgr_idx, col) -> list of complex arrays
    coherent_buffer = {}
    coherent_frame_count = [0]  # mutable counter

    def sync_frame(wave, ref_ang):
        """Sync a single frame (num_ch, num_adc) using reference path."""
        chirp = wave[:, 16:-96]
        steering_vec = np.exp(-1j * np.pi * np.sin(ref_ang) * np.arange(chirp.shape[0]))[None, :]
        ref_sig = steering_vec @ chirp  # (1, num_adc_cropped)
        df, da = dsp_mod.find_ref_frq_first_peak(ref_sig, **align_params)
        # Apply shift to each channel
        n_samp = chirp.shape[-1]
        shift = np.exp(1j * (2 * np.pi * df * np.arange(n_samp) + da))
        return chirp * shift[None, :]

    def update(_):
        artists = []
        for i, rs in enumerate(mgr.radars):
            data = rs.get_latest()
            if data is None:
                continue
            for col, cfg_idx in enumerate(plot_cfgs):
                try:
                    wave = data[:, -1, cfg_idx, :]  # (num_ch, num_adc)
                    ref_ang = get_ref_angle(i, cfg_idx)
                    if ref_ang is not None:
                        synced = sync_frame(wave, ref_ang)
                    else:
                        synced = wave[:, 16:-96]

                    # Accumulate for coherent averaging
                    if save_coherent_avg and coherent_frame_count[0] < gif_frames:
                        key = (i, col)
                        if key not in coherent_buffer:
                            coherent_buffer[key] = []
                        coherent_buffer[key].append(synced.copy())

                    fft_mag = np.abs(np.fft.fft(synced, n=nfft, axis=-1))

                    for ch, ln in enumerate(lines[i][col]):
                        ln.set_ydata(fft_mag[ch])
                        artists.append(ln)

                    ax = axs[i, col]
                    ax.set_ylim(0, max(1e-6, float(fft_mag.max())) * 1.1)
                    ax.set_xlim(0, nfft - 1)
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                except Exception as e:
                    print("error in synced visualization:", e)
                    continue
        return artists

    captured_frames = []

    def update_and_capture(frame_num):
        artists = update(frame_num)
        if save_gif and len(captured_frames) < gif_frames:
            fig.canvas.draw()
            captured_frames.append(np.array(fig.canvas.buffer_rgba()).copy())
        if save_coherent_avg and coherent_frame_count[0] < gif_frames:
            coherent_frame_count[0] += 1
        return artists

    ani = FuncAnimation(fig, update_and_capture, interval=mgr.params['period_frame'], blit=False, cache_frame_data=False)
    plt.show(block=True)

    if save_gif and captured_frames:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in captured_frames]
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                           loop=0, duration=int(1000 / gif_fps))
        print(f"GIF saved to {gif_path} ({len(captured_frames)} frames)")

    # Save coherent average as static plot
    if save_coherent_avg and coherent_buffer:
        fig_ca, axs_ca = plt.subplots(
            n_radars, n_plot_cfg, figsize=figsize, squeeze=False
        )
        plt.subplots_adjust(hspace=0.35, wspace=0.25)
        n_frames_used = 0
        for i in range(n_radars):
            for col, cfg_idx in enumerate(plot_cfgs):
                ax = axs_ca[i, col]
                key = (i, col)
                if key in coherent_buffer and len(coherent_buffer[key]) > 0:
                    frames_complex = np.array(coherent_buffer[key])  # (N, num_ch, num_adc_cropped)
                    n_frames_used = frames_complex.shape[0]
                    # Channel 0 coherent average
                    avg_ch0 = np.mean(frames_complex[:, 0, :], axis=0)  # (num_adc_cropped,)
                    fft_ch0 = np.abs(np.fft.fft(avg_ch0, n=nfft))
                    ax.plot(np.arange(nfft), fft_ch0, lw=1, color='gray', label='Ch 0')
                    # All-channel coherent average
                    avg_complex = np.mean(frames_complex, axis=0)  # (num_ch, num_adc_cropped)
                    avg_complex = np.mean(avg_complex, axis=0)  # (num_adc_cropped,)
                    fft_mag = np.abs(np.fft.fft(avg_complex, n=nfft))
                    ax.plot(np.arange(nfft), fft_mag, lw=2, color='black', label='Avg')
                    ax.legend(fontsize=8)
                    ymax = max(float(fft_mag.max()), float(fft_ch0.max()))
                    ax.set_ylim(0, max(1e-6, ymax) * 1.1)
                    ax.set_xlim(0, nfft - 1)
                ax.set_title(f"Radar {i} | Cfg {cfg_idx}")
                ax.set_xlabel("IF Frequency")
                ax.set_ylabel("Amplitude")
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.tick_params(axis='both', length=0)
        fig_ca.suptitle(f"Coherent Average ({n_frames_used} frames)", fontsize=16)
        fig_ca.savefig(coherent_avg_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig_ca)
        print(f"Coherent average saved to {coherent_avg_path} ({n_frames_used} frames)")

    return fig, axs


def visualize_2d_fft(mgr, nfft_range=256, nfft_angle=128):
    """
    Visualize the 2D FFT (range-angle) of the received signals for each radar and config.
    Shows one subplot per (radar, config) pair, where each subplot has:
        - X axis: range bin (0...nfft_range-1)
        - Y axis: angle bin (0...nfft_angle-1)
        - Color: magnitude of 2D FFT for latest frame, summed or max over channels
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    n_radars = len(mgr.radars)
    num_ch = mgr.params['chirp_cfg']['num_ch']  # assuming all radars use same num channels
    n_cfg = mgr.params['chirp_cfg']['num_config']  # assuming all radars use same configs

    r_res = mgr.params['ramp_cfg']['Fs']*3e8 /(2* nfft_range*mgr.params['ramp_cfg']['slope'])

    fig, axs = plt.subplots(
        n_radars, n_cfg, figsize=(4 + 3 * n_cfg, 3 + 2 * n_radars), squeeze=False
    )
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    fig.suptitle("Realtime 2D FFT (Range-Angle) per Radar/Config", fontsize=16)

    # Store the image handles
    images = [[None for _ in range(n_cfg)] for _ in range(n_radars)]

    # Prepare axes & first images
    for i in range(n_radars):
        for j in range(n_cfg):
            ax = axs[i, j]
            img = ax.imshow(np.zeros((nfft_angle, nfft_range)).T,
                            origin='lower', aspect='auto',
                            extent=[0, nfft_angle-1, 0, (nfft_range-1)*r_res],
                            vmin=0, vmax=1)
            ax.set_title(f"Radar {i} | Cfg {j}")
            ax.set_ylabel("Range [m]")
            ax.set_xlabel("Angle bin")
            images[i][j] = img

    def update(_):
        artists = []
        vmax = 1e-6  # update per frame to auto-scale
        for i, rs in enumerate(mgr.radars):
            data = rs.get_latest()
            if data is None:
                continue
            for k in range(n_cfg):
                try:
                    # data shape: (ant, time, cfg, samples)
                    wave = data[:, -1, k, :]  # (num_ch, samples)
                    # Optionally, crop useful ADC samples
                    crop_wave = wave[:, 16:-96] if wave.shape[-1] > (16+96) else wave

                    # 2D FFT -- axes: (ant, adc)
                    mag = np.abs(np.fft.fft2(crop_wave, s=(nfft_angle, nfft_range)))
                    mag = np.fft.fftshift(mag, axes=0)  # shift angle axis
                    # print(mag.shape)

                    # Reduce over antenna dimension for general display (sum, max, or ch0)
                    if mag.ndim == 2:
                        mag_disp = np.max(mag, axis=0, keepdims=False)  # [angle, range] -> [range]
                        mag_disp = mag
                    else:
                        mag_disp = mag

                    img = images[i][k]
                    img.set_data(mag_disp.T)
                    img.set_clim(vmin=0, vmax=float(mag_disp.max())*1.1 if mag_disp.max() > 1e-6 else 1)
                    artists.append(img)
                    vmax = max(vmax, float(mag_disp.max()))
                except Exception as e:
                    print("error in 2D FFT visualization:", e)
                    continue
        return artists

    ani = FuncAnimation(fig, update, interval=mgr.params['period_frame'], blit=False, cache_frame_data=False)
    plt.show(block=True)
    return fig, axs



def visualize_2d_fft_clean(mgr, nfft_range=256, nfft_angle=128,
                           save_gif=False, gif_path="2d_fft.gif", gif_frames=50, gif_fps=10,
                           figsize=None, dpi=100, configs=None):
    """
    Realtime 2D FFT (Range-Angle) — no tick values, only axis labels.
    Supports config filtering and GIF saving with live display.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    n_radars = len(mgr.radars)
    n_cfg = mgr.params['chirp_cfg']['num_config']
    plot_cfgs = configs if configs is not None else list(range(n_cfg))
    n_plot_cfg = len(plot_cfgs)

    r_res = mgr.params['ramp_cfg']['Fs'] * 3e8 / (2 * nfft_range * mgr.params['ramp_cfg']['slope'])

    if figsize is None:
        figsize = (4 + 3 * n_plot_cfg, 3 + 2 * n_radars)
    fig, axs = plt.subplots(
        n_radars, n_plot_cfg, figsize=figsize, squeeze=False
    )
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    fig.suptitle("Realtime 2D FFT (Range-Angle) per Radar/Config", fontsize=16)

    images = [[None for _ in range(n_plot_cfg)] for _ in range(n_radars)]

    for i in range(n_radars):
        for col, cfg_idx in enumerate(plot_cfgs):
            ax = axs[i, col]
            img = ax.imshow(np.zeros((nfft_angle, nfft_range)).T,
                            origin='lower', aspect='auto',
                            extent=[0, nfft_angle - 1, 0, (nfft_range - 1) * r_res],
                            vmin=0, vmax=1)
            ax.set_title(f"Radar {i} | Cfg {cfg_idx}")
            ax.set_ylabel("Range")
            ax.set_xlabel("Angle")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(axis='both', length=0)
            images[i][col] = img

    def update(_):
        artists = []
        for i, rs in enumerate(mgr.radars):
            data = rs.get_latest()
            if data is None:
                continue
            for col, cfg_idx in enumerate(plot_cfgs):
                try:
                    wave = data[:, -1, cfg_idx, :]
                    crop_wave = wave[:, 16:-96] if wave.shape[-1] > (16 + 96) else wave

                    mag = np.abs(np.fft.fft2(crop_wave, s=(nfft_angle, nfft_range)))
                    mag = np.fft.fftshift(mag, axes=0)

                    img = images[i][col]
                    img.set_data(mag.T)
                    img.set_clim(vmin=0, vmax=float(mag.max()) * 1.1 if mag.max() > 1e-6 else 1)
                    artists.append(img)
                except Exception as e:
                    print("error in 2D FFT visualization:", e)
                    continue
        return artists

    captured_frames = []

    def update_and_capture(frame_num):
        artists = update(frame_num)
        if save_gif and len(captured_frames) < gif_frames:
            fig.canvas.draw()
            captured_frames.append(np.array(fig.canvas.buffer_rgba()).copy())
        return artists

    ani = FuncAnimation(fig, update_and_capture, interval=mgr.params['period_frame'], blit=False, cache_frame_data=False)
    plt.show(block=True)

    if save_gif and captured_frames:
        from PIL import Image
        pil_frames = [Image.fromarray(f) for f in captured_frames]
        pil_frames[0].save(gif_path, save_all=True, append_images=pil_frames[1:],
                           loop=0, duration=int(1000 / gif_fps))
        print(f"GIF saved to {gif_path} ({len(captured_frames)} frames)")

    return fig, axs


def visualize_angle_range_polar(angle_range_mag, angles_rad, ranges_m,
                                title='Angle-Range (Polar)', cmap='viridis',
                                db=False, vmin=None, vmax=None, ax=None,
                                clean=True):
    """
    Plot an angle–range heatmap in polar coordinates.
    angle_range_mag: (n_angles, n_ranges)
    angles_rad: (n_angles,) centers in rad
    ranges_m: (n_ranges,) centers in m
    """
    import numpy as np
    import matplotlib.pyplot as plt

    A = np.asarray(angle_range_mag)
    th = np.asarray(angles_rad).ravel()
    r = np.asarray(ranges_m).ravel()
    assert A.shape == (th.size, r.size), "angle_range_mag must be (len(angles), len(ranges))"

    # bin edges
    th_e = _edges(th)
    r_e = _edges(r)
    ThE, RE = np.meshgrid(th_e, r_e, indexing='ij')

    # magnitude or dB
    M = np.abs(A)
    if db:
        M = 20.0 * np.log10(M + 1e-12)

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = plt.subplot(1, 1, 1, projection='polar')
        created_fig = True
    else:
        fig = ax.figure

    # orientation
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # restrict to valid wedge and radius (expects radians)
    thmin, thmax = th_e[0], th_e[-1]
    ax.set_thetalim(thmin, thmax)
    ax.set_rlim(float(r_e[0]), float(r_e[-1]))

    pcm = ax.pcolormesh(ThE, RE, M, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)

    if not clean:
        ax.set_title(title, pad=12)
        fig.colorbar(pcm, ax=ax, pad=0.1, shrink=0.9, label=('Magnitude (dB)' if db else 'Magnitude'))
    else:
        # hide grids, ticks, labels; keep only data in the valid wedge
        ax.grid(False)
        ax.set_title('')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xticklabels([]); ax.set_yticklabels([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        # remove extra padding
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if created_fig:
        plt.show(block=False)

    return fig, ax, pcm
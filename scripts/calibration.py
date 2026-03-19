"""
Calibrate the radar orientations using 3 rods in the scene.

1. Generate MUSIC detection results of each radar using imaging_methods.py
2. Point cloud generating with thresholding (CFAR don't work for range-azimuth heatmap)
3. NMS to reduce get the points with highes intensity
4. cluttering those points with its neighborhood points
5. optimize the radar orientation with the estimated rods center
"""

import os
import yaml
from typing import List, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt
# calibration uses torch for optimization, but we want to keep it as an 
# optional dependency since it's only needed for this script
import torch

import muldar.utils as utils

def genMUSIC(folderName, theta_grid_rad, range_grid_m):
    import muldar.dsp.music as music
    params = yaml.load(open(os.path.join(folderName, 'configs.yml')), Loader=yaml.FullLoader)
    adc_shape = params['chirp_cfg']
    # theta_grid_rad=np.deg2rad(np.linspace(*param_theta_grid_rad))
    # range_grid_m=np.linspace(*param_range_grid_m)
    img_music_list_all = []
    for radar_idx in range(3):
        fileName = os.path.join(folderName, f'radar_{radar_idx}.bin')
        # frames = utils.readDCA1000(fileName).reshape(
        #             adc_shape['num_ch'],
        #             -1,
        #             adc_shape['num_config'],
        #             adc_shape['num_adc']
        #         )
        frames = utils.readDCA1000Robust(fileName, adc_shape)
        print(frames.shape)
        X = np.concatenate([frames[:,:,radar_idx*2], frames[:,:,radar_idx*2+1]], axis=0)
        X = np.moveaxis(X, 1, -1)
        # Flip antenna order so index 0 corresponds to the physical left-most element
        X = X[::-1, ...]
        P = music.music_2d_range_angle_fast_gpu(X, fs=params['ramp_cfg']['Fs'],
                            slope=params['ramp_cfg']['slope'],
                            fc=params['ramp_cfg']['f0'],
                            K=2,
                            theta_grid_rad=theta_grid_rad,
                            range_grid_m=range_grid_m,
                            )
        img_music_list_all.append(P)
    img_music_list_all = np.array(img_music_list_all)
    return img_music_list_all



def prepare_calibration(folderName, *argv, **kwargs):
    import muldar.dsp.cfar as cfar
    theta_grid_rad = np.deg2rad(np.linspace(*kwargs['param_theta_grid_rad']))
    range_grid_m = np.linspace(*kwargs['param_range_grid_m'])
    imgs = genMUSIC(folderName, theta_grid_rad, range_grid_m)
    imgs_remove_mean = imgs - np.mean(imgs,axis=0,keepdims=True)
    hard_thresh = np.max(imgs_remove_mean,axis=(-2,-1),keepdims=True)/kwargs['factor_of_max']

    dets = imgs_remove_mean > hard_thresh

    pts = []
    dets_nms = np.zeros_like(dets)

    for i in range(dets.shape[0]):
        dets_nms[i] = cfar.nms_2d(imgs[i], dets[i], win=kwargs['nms_win'])
        # dets_nms[i] = dets[i]
        iangle, irange = np.nonzero(dets_nms[i])
        inten = imgs[i][iangle, irange]
        _th = theta_grid_rad[iangle]  # Note: imgs were built with P[::-1] upstream; here we are consistent within this file
        _rr = range_grid_m[irange]
        curr_pts = np.vstack([_th , _rr, inten]).T
        pts.append(curr_pts)

    # Optional visualization of raw detections
    if kwargs.get('do_visualize', False) and kwargs.get('viz_raw', True):
        for i in range(3):
            plt.figure()
            plt.imshow(imgs[i].T, origin='lower', aspect='equal')
            plt.colorbar(label='Magnitude')
            plt.title('Magnitude')

        for i in range(3):
            plt.figure()
            plt.imshow(dets[i].T, origin='lower', aspect='equal')
            plt.colorbar(label='Detection')
            plt.title('Detection')
        
        for i in range(3):
            plt.figure()
            plt.imshow(
                imgs[i].T, 
                extent=[theta_grid_rad[0], theta_grid_rad[-1], range_grid_m[0], range_grid_m[-1]], 
                origin='lower', aspect='auto')
            plt.scatter(pts[i][:,0], pts[i][:,1], s=20, marker='o', linewidths=0.2)
            # plt.colorbar(label='Magnitude')
            plt.title('Points')
        plt.show()
    

    # Group and reorder across radars if requested
    group = kwargs.get('group', True)
    pts = np.array(pts)
    if not group or pts.shape[1] == 1:
        # keep original behavior
        return pts

    # Read poses from configs.yml to transform to global frame
    params = yaml.load(open(os.path.join(folderName, 'configs.yml')), Loader=yaml.FullLoader)
    poses = [np.array(r['pose'], dtype=float) for r in params['activated_radar']]

    n_targets = kwargs.get('num_targets', None)
    distance_thresh = float(kwargs.get('distance_thresh', 0.15))
    ordered_pts, centers = group_and_reorder_pts(pts, poses, k=n_targets, distance_thresh=distance_thresh)

    if kwargs.get('do_visualize', False) and kwargs.get('viz_grouped', True):
        # visualize clustered centers and per-radar assignments
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        # plot global points and centers
        for ridx, p in enumerate(pts):
            xy_local = polar_to_xy(p[:,0], p[:,1])
            xy_global = utils.radar_to_global(xy_local, poses[ridx])
            ax.scatter(xy_global[:,0], xy_global[:,1], s=10, alpha=0.7, label=f'radar {ridx}', c=colors[ridx])
        ax.scatter(centers[:,0], centers[:,1], c='red', s=60, marker='x', label='centers')
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        ax.set_title('Clustered global points and centers')
        plt.show()

    return ordered_pts


def polar_to_xy(theta_rad: np.ndarray, r_m: np.ndarray) -> np.ndarray:
    """Convert polar (theta, r) to local XY (N,2).
    Convention: theta=0 is broadside (forward, +y), hence:
      x = r * sin(theta), y = r * cos(theta)
    This matches the steering vector usage sin(theta) and utils.radar_to_global convention.
    """
    theta = np.asarray(theta_rad).ravel()
    r = np.asarray(r_m).ravel()
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return np.stack([x, y], axis=1)


def _simple_kmeans(points: np.ndarray, k: int, n_init: int = 8, max_iter: int = 50, rng: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Tiny KMeans (no external deps). Returns (centers, labels)."""
    rng_state = np.random.RandomState(rng)
    best_inertia = np.inf
    best_centers = None
    best_labels = None
    P = np.asarray(points, dtype=np.float64)
    if P.shape[0] < k:
        raise ValueError("Not enough points to cluster")
    for _ in range(n_init):
        idx = rng_state.choice(P.shape[0], size=k, replace=False)
        centers = P[idx].copy()
        labels = np.zeros(P.shape[0], dtype=np.int32)
        for _it in range(max_iter):
            # assign
            d2 = np.sum((P[:, None, :] - centers[None, :, :])**2, axis=2)  # (N,k)
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            # update
            for j in range(k):
                mask = (labels == j)
                if np.any(mask):
                    centers[j] = P[mask].mean(axis=0)
                else:
                    # re-seed an empty cluster
                    centers[j] = P[rng_state.randint(0, P.shape[0])]
        inertia = np.sum((P - centers[labels])**2)
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
            best_labels = labels.copy()
    return best_centers.astype(np.float32), best_labels.astype(np.int32)


def _order_centers_polar(centers: np.ndarray) -> np.ndarray:
    """Return index order of centers by angle around centroid (rotation-invariant)."""
    cxy = centers.mean(axis=0, keepdims=True)
    vec = centers - cxy
    ang = np.arctan2(vec[:,1], vec[:,0])
    order = np.argsort(ang)
    return order


def group_and_reorder_pts(pts_per_radar: List[np.ndarray],
                          poses: List[np.ndarray],
                          k: int = None,
                          distance_thresh: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Group detections across radars into common physical targets and reorder.
    Inputs:
      - pts_per_radar: list length R; each (Ni, 3) -> [theta, range, intensity]
      - poses: list length R; each [x, y, yaw_deg]
      - k: number of targets; if None, use median count across radars
      - distance_thresh: max distance (m) to accept assignment to a cluster
    Returns:
      - ordered_pts: (R, K, 3) in original [theta, range, intensity] format, consistently ordered
      - centers: (K, 2) cluster centers in global XY
    """
    R = len(pts_per_radar)
    counts = [p.shape[0] for p in pts_per_radar]
    if k is None:
        k = int(np.median(counts)) if counts else 0
    if k <= 0:
        return np.zeros((R, 0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    # Build global XY for all points and keep index mapping
    all_global = []
    idx_map = []  # (radar_idx, local_idx)
    per_radar_global = []
    for ridx, p in enumerate(pts_per_radar):
        if p.size == 0:
            per_radar_global.append(np.zeros((0, 2), dtype=np.float32))
            continue
        xy_local = polar_to_xy(p[:,0], p[:,1])  # (Ni,2)
        xy_global = utils.radar_to_global(xy_local, poses[ridx])  # (Ni,2)
        per_radar_global.append(xy_global.astype(np.float32))
        all_global.append(xy_global)
        idx_map.extend([(ridx, i) for i in range(p.shape[0])])
    if not all_global:
        return np.zeros((R, 0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    all_global = np.vstack(all_global)

    # Cluster into K targets
    centers, labels = _simple_kmeans(all_global, k=k)
    # Order centers for stable index mapping
    order = _order_centers_polar(centers)
    centers = centers[order]
    # Remap labels to ordered cluster indices
    label_remap = {old: new for new, old in enumerate(order)}
    labels = np.array([label_remap[l] for l in labels], dtype=np.int32)

    # Prepare per-radar assignment: choose nearest point to each center (one-to-one per radar)
    ordered_pts = []
    start = 0
    for ridx in range(R):
        local_pts = pts_per_radar[ridx]
        G = per_radar_global[ridx]
        if local_pts.shape[0] == 0:
            ordered_pts.append(np.full((k, 3), np.nan, dtype=np.float32))
            continue
        # compute distance matrix between (Ni,2) and (K,2)
        dists = np.sqrt(((G[:, None, :] - centers[None, :, :])**2).sum(axis=2))  # (Ni, K)
        # Greedy one-to-one: for each cluster, pick nearest unused point under gate
        taken = set()
        out = np.full((k, 3), np.nan, dtype=np.float32)
        # Process clusters by increasing cluster spread (optional); here simple order 0..K-1
        for c in range(k):
            # candidate indices sorted by distance
            order_idx = np.argsort(dists[:, c])
            chosen = None
            for li in order_idx:
                if li in taken:
                    continue
                if dists[li, c] <= distance_thresh:
                    chosen = li
                    break
            if chosen is not None:
                taken.add(chosen)
                out[c] = local_pts[chosen]  # [theta, range, intensity]
            else:
                # keep NaN if no match for this radar
                pass
        ordered_pts.append(out)
    ordered_pts = np.stack(ordered_pts, axis=0)  # (R,K,3)
    return ordered_pts, centers



def transform(pts, orientation):
    """transform the pts from radar coordinate system to global coordinate system
    using pytorch
    pts: (N, 2) [x, y]
    orientation: [x, y, yaw] radar's orientation
    """
    # Ensure torch tensors
    if not torch.is_tensor(pts):
        pts = torch.as_tensor(pts, dtype=torch.float32)
    if not torch.is_tensor(orientation):
        orientation = torch.as_tensor(orientation, dtype=pts.dtype, device=pts.device)
    ang = -torch.deg2rad(orientation[2] - 90.0)
    c = torch.cos(ang)
    s = torch.sin(ang)
    rot_mat = torch.stack([torch.stack([c, -s]), torch.stack([s, c])], dim=0)
    phase_center_offset = torch.tensor([-(1.9e-3*19/8), 1.9e-3*19/8], dtype=pts.dtype, device=pts.device)
    phase_center_offset = torch.tensor([0,0], dtype=pts.dtype, device=pts.device)
    translation = orientation[:2] + phase_center_offset
    return (pts + phase_center_offset) @ rot_mat + translation


def transform_batched(pts_batched, orientations):
    """Vectorized transform for multiple radars.
    pts_batched: (R, P, 2)
    orientations: (R, 3) -> [x, y, yaw]
    returns global_pts: (R, P, 2)
    """
    pts_batched = pts_batched.float()
    orientations = orientations.float()
    ang = -torch.deg2rad(orientations[:, 2] - 90.0)  # (R,)
    cos_ang = torch.cos(ang)
    sin_ang = torch.sin(ang)
    rot_mats = torch.stack(
        [
            torch.stack([cos_ang, -sin_ang], dim=-1),
            torch.stack([sin_ang,  cos_ang], dim=-1),
        ],
        dim=-2,
    )  # (R, 2, 2)
    phase_center_offset = torch.tensor([-(1.9e-3*19/8), 1.9e-3*19/8], dtype=pts_batched.dtype, device=pts_batched.device)
    phase_center_offset = torch.tensor([0,0], dtype=pts_batched.dtype, device=pts_batched.device)
    translations = orientations[:, :2]  # (R, 2)
    global_pts = torch.matmul(pts_batched + phase_center_offset, rot_mats) + translations[:, None, :]
    return global_pts


def compute_alignment_loss_masked(global_pts, valid_mask):
    """Variance loss across radars for each object, masking missing entries.
    global_pts: (R, P, 2)
    valid_mask: (R, P) boolean tensor; False entries ignored.
    """
    # Compute per-object means over valid radars
    mask = valid_mask.float().unsqueeze(-1)  # (R,P,1)
    weighted_sum = (global_pts * mask).sum(dim=0)  # (P,2)
    counts = mask.sum(dim=0).clamp_min(1.0)        # (P,1)
    mean_per_object = (weighted_sum / counts).unsqueeze(0)  # (1,P,2)
    diffs = (global_pts - mean_per_object) * mask
    # mean squared error over valid entries
    loss = (diffs.pow(2).sum(dim=-1)).sum() / mask.sum().clamp_min(1.0)
    return loss


def optimize_radar_orientations(
    local_pts_xy,               # (R,P,2) torch.Tensor
    valid_mask,                 # (R,P) torch.BoolTensor
    initial_orientations,       # (R,3) torch.Tensor [x,y,yaw_deg]
    steps=1200,
    lr=5e-2,
    # reg_weight=1e-3,
    reg_weight=0,
    device=None,
    verbose_every=100,
    scheduler_type="plateau",
    scheduler_kwargs=None,
    fixed_indices=None,
    fixed_orientations=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    local_pts_xy = local_pts_xy.to(device).float()
    valid_mask = valid_mask.to(device)
    init_orients = initial_orientations.to(device).float()

    params = torch.nn.Parameter(init_orients.clone())
    optimizer = torch.optim.Adam([params], lr=lr)
    if scheduler_kwargs is None:
        scheduler_kwargs = {}
    scheduler = None
    if scheduler_type == "plateau":
        factor = scheduler_kwargs.get("factor", 0.5)
        patience = scheduler_kwargs.get("patience", 500)
        min_lr = scheduler_kwargs.get("min_lr", 1e-6)
        threshold = scheduler_kwargs.get("threshold", 1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr, threshold=threshold, verbose=False
        )
    elif scheduler_type == "step":
        step_size = scheduler_kwargs.get("step_size", max(1, steps // 3))
        gamma = scheduler_kwargs.get("gamma", 0.5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "cosine":
        T_max = scheduler_kwargs.get("T_max", steps)
        eta_min = scheduler_kwargs.get("eta_min", 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    fixed_idx_tensor = None
    fixed_targets = None
    if fixed_indices is not None:
        fixed_idx_tensor = torch.as_tensor(fixed_indices, dtype=torch.long, device=device)
        fixed_targets = (initial_orientations[fixed_idx_tensor] if fixed_orientations is None else fixed_orientations).to(device).to(init_orients.dtype)

    best_params = params.detach().clone()
    best_loss = float("inf")

    for step in range(steps):
        optimizer.zero_grad()
        params_eff = params.clone()
        if fixed_idx_tensor is not None:
            params_eff[fixed_idx_tensor] = fixed_targets
        global_pts = transform_batched(local_pts_xy, params_eff)  # (R,P,2)
        loss_align = compute_alignment_loss_masked(global_pts, valid_mask)
        loss_reg = reg_weight * (params_eff - init_orients).pow(2).mean()
        loss = loss_align + loss_reg
        loss.backward()
        if fixed_idx_tensor is not None and params.grad is not None:
            params.grad[fixed_idx_tensor] = 0.0
        optimizer.step()
        if fixed_idx_tensor is not None:
            with torch.no_grad():
                params[fixed_idx_tensor] = fixed_targets
        with torch.no_grad():
            params_eval = params.clone()
            if fixed_idx_tensor is not None:
                params_eval[fixed_idx_tensor] = fixed_targets
            global_pts_eval = transform_batched(local_pts_xy, params_eval)
            loss_eval = compute_alignment_loss_masked(global_pts_eval, valid_mask) + reg_weight * (params_eval - init_orients).pow(2).mean()
            if scheduler is not None:
                if scheduler_type == "plateau":
                    scheduler.step(loss_eval)
                else:
                    scheduler.step()
            if loss_eval.item() < best_loss:
                best_loss = loss_eval.item()
                best_params = params.detach().clone()
        if verbose_every and (step+1) % max(1, verbose_every) == 0:
            print(f"[opt {step+1}/{steps}] loss={loss.item():.6f}")
    return best_params.cpu(), best_loss



def run_orientation_optimization(pts, params):
# === Optimize radar orientations similar to calibration2.py ===
    # Build local XY from polar detections; create mask for valid points
    R = pts.shape[0]
    K = pts.shape[1] if pts.ndim >= 2 else 0
    theta_mat = pts[:, :, 0] if K > 0 else np.zeros((R,0), dtype=np.float32)
    range_mat = pts[:, :, 1] if K > 0 else np.zeros((R,0), dtype=np.float32)
    valid_mask_np = np.isfinite(theta_mat) & np.isfinite(range_mat)
    local_xy_list = []
    for r in range(R):
        local_xy_list.append(polar_to_xy(theta_mat[r], range_mat[r]))
    local_xy_np = np.stack(local_xy_list, axis=0).astype(np.float32)  # (R,K,2)
    valid_mask_t = torch.from_numpy(valid_mask_np)
    local_xy_t = torch.from_numpy(local_xy_np)

    # Load initial orientations from config
    # params = yaml.load(open(os.path.join(folderName, 'configs.yml')), Loader=yaml.FullLoader)
    init_orients_np = np.array([params['activated_radar'][i]['pose'] for i in range(R)], dtype=np.float32)
    init_orients_t = torch.from_numpy(init_orients_np)

    # Fix the first radar to its current pose to remove gauge ambiguity
    fixed_idx = [1]
    fixed_orients_t = torch.tensor(init_orients_np[fixed_idx], dtype=init_orients_t.dtype)

    best_orients_t, best_loss = optimize_radar_orientations(
        local_pts_xy=local_xy_t,
        valid_mask=valid_mask_t,
        initial_orientations=init_orients_t,
        steps=1000,
        lr=5e-1,
        reg_weight=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose_every=100,
        # scheduler_type=None,
        scheduler_type="plateau",
        scheduler_kwargs={"patience": 200, "factor": 0.5, "min_lr": 1e-6},
        fixed_indices=fixed_idx,
        fixed_orientations=fixed_orients_t,
    )
    print("Optimized orientations (x, y, yaw-deg):")
    print(best_orients_t)

    # Optional: visualize optimized global points
    optimized_global = transform_batched(local_xy_t.to(best_orients_t.dtype), best_orients_t).cpu().numpy()
    
    # show original points
    plt.figure()
    colors = ['r','g','b']
    for r in range(min(R,3)):
        mask_r = valid_mask_np[r]
        # do global transformation with torch tensors
        global_xy = transform(
            torch.from_numpy(local_xy_np[r, mask_r]).to(dtype=best_orients_t.dtype),
            torch.from_numpy(init_orients_np[r]).to(dtype=best_orients_t.dtype),
        ).cpu().numpy()
        plt.scatter(global_xy[:,0], global_xy[:,1], s=14, c=colors[r], label=f"radar {r}")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title("Original global points (by radar)")
    
    plt.figure()
    colors = ['r','g','b']
    for r in range(min(R,3)):
        mask_r = valid_mask_np[r]
        plt.scatter(optimized_global[r, mask_r, 0], optimized_global[r, mask_r, 1], s=14, c=colors[r], label=f"radar {r}")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title("Optimized global points (by radar)")
    plt.show()

    return best_orients_t





if __name__ == "__main__":
    folderList = [
        './adcData/demo_20260129/calibration/20260129_224147',
        './adcData/demo_20260129/calibration/20260129_224212',
        './adcData/demo_20260129/calibration/20260129_224241',
        ]


    radius = 0.20/2
    param_theta_grid_rad = (-90, 90, 512)
    param_range_grid_m = (0.2, 1.3, 512)

    params = yaml.load(open(os.path.join(folderList[0], 'configs.yml')), Loader=yaml.FullLoader)
    
    pts = []
    for folderName in folderList:
        print(folderName)
        _pts = prepare_calibration(
            folderName = folderName, 
            param_theta_grid_rad = param_theta_grid_rad,
            param_range_grid_m = param_range_grid_m,
            factor_of_max = 5,
            nms_win=(1,1),
            do_visualize=False,
            )
        print(_pts.shape)
        print(_pts)
        # shape is 3, N, 3
        # pts.append(_pts)
        pts.append(_pts)
    pts = np.concatenate(pts, axis=1)
    print(pts.shape)

    pts[:,:,1] = pts[:,:,1] + radius

    run_orientation_optimization(pts, params)








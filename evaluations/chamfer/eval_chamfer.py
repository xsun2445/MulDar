from re import M
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys
import scipy.constants

# Make sure the repository root is importable and set as working directory
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)
os.chdir(PROJECT_ROOT_DIR)

import muldar.utils as utils
import muldar.dsp.sar as sar
import muldar.dsp.dsp as dsp
import muldar.dsp.sar as sar
import muldar.vis as vis
import muldar.dsp.bistatic as bistatic

import json

import labels
import muldar.eval.chamfer as chamfer


def chamfer_eval(angle, label_dict, lambda_mono_factor=3, thresh_from_max=0.5, eval_type='planar', show_plot=True):
    
    thresh = thresh_from_max
    print(angle, label_dict[angle])

    folderName = label_dict[angle]

    params = yaml.load(open(os.path.join(folderName, 'configs.yml')), Loader=yaml.FullLoader)
    adc_shape = params['chirp_cfg']


    params['frequency'] = params['ramp_cfg']['f0']
    params['fs'] = params['ramp_cfg']['Fs']
    params['slope'] = params['ramp_cfg']['slope']
    params['c'] = scipy.constants.c
    params['sar_nfft'] = 1024
    params['ref_idx'] = 256
    # params['ave_win'] = 100
    params['align_params'] = {
        'nfft': 4096,
        'f_tar_idx': 1024,
        'phase_ref': 0.0,
        'min_height_ratio': 0.7,
        'min_prominence_ratio': 0.5,
    }

    img_x_size = 128
    img_y_size = 128

    gx = np.linspace(-0.834, 0.72, img_x_size)
    gy = np.linspace(-0.2+0.2, 2-0.3, img_y_size)

    import itertools

    img_list_all = []

    for tx_idx, rx_idx in itertools.product(range(3), range(3)):
        fileName = os.path.join(folderName, f'radar_{rx_idx}.bin')
        frames = utils.readDCA1000Robust(fileName, adc_shape)
        params['ave_win'] = frames.shape[1]
        # print(frames.shape)
        _img, _ ,_ = bistatic.genBiStaticSAR(frames, tx_idx, rx_idx, gx, gy, params)
        # print(_img.shape)
        img_list_all.append(_img.squeeze())
    img_list_all = np.array(img_list_all)
    # print(img_list_all.shape)


    mag_all = np.mean(img_list_all, axis=1).reshape(3,3,img_y_size,img_x_size)

    vmax = float(np.percentile(np.abs(mag_all), 100))
    vmin = 0.0
    fig, axes = plt.subplots(3, 3, figsize=(9, 9), squeeze=False)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    for tx in range(3):
        for col in range(3):
            ax = axes[tx, col]
            img = mag_all[tx, col]
            rx = col // 2
            txch = col % 2  # 0 or 1 (two TX elements/slots)
            im = ax.imshow(
                # np.angle(img),
                np.abs(img),
                origin='lower',
                extent=[gx[0], gx[-1], gy[0], gy[-1]],
                aspect='equal',
                vmin=vmin, vmax=vmax, 
                cmap='viridis')
            ax.set_title(f"TX {tx} | RX {rx} | TXch {txch}")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
    # single colorbar on the right
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, pad=0.02)
    cbar.set_label("Amplitude")
    img_list_all = img_list_all.reshape(-1,2,img_y_size,img_x_size)

    mono_sum = np.abs(mag_all[0,0]) + np.abs(mag_all[1,1]) + np.abs(mag_all[2,2])
    multi_sum = np.abs(mag_all[2,0]+mag_all[0,2]) + \
                np.abs(mag_all[1,0]+mag_all[0,1]) +\
                np.abs(mag_all[1,2]+mag_all[2,1])

    all_sum = mono_sum * lambda_mono_factor + multi_sum


    if show_plot:
        plt.figure()
        plt.imshow(mono_sum, origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
        plt.title(f'{label_dict['name']} {angle} {eval_type} mono sum')
        plt.savefig(f'evaluations/chamfer/img_results/{label_dict['name']}_{angle}_{eval_type}_mono_sum.png')

        plt.figure()
        plt.imshow(multi_sum, origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
        plt.title(f'{label_dict['name']} {angle} {eval_type} multi sum')
        plt.savefig(f'evaluations/chamfer/img_results/{label_dict['name']}_{angle}_{eval_type}_multi_sum.png')
        plt.figure()
        plt.imshow(all_sum, origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
        plt.title(f'{label_dict['name']} {angle} {eval_type} all sum')
        plt.savefig(f'evaluations/chamfer/img_results/{label_dict['name']}_{angle}_{eval_type}_all_sum.png')
    
    mono_sum = mono_sum > (mono_sum.max() * thresh)
    multi_sum = multi_sum > (multi_sum.max() * thresh)
    all_sum = all_sum > (all_sum.max() * thresh)

    if show_plot:
        plt.figure()
        plt.imshow(mono_sum, origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
        plt.title(f'{label_dict['name']} {angle} mono sum > thresh')
        plt.savefig(f'evaluations/chamfer/img_results/{label_dict['name']}_{angle}_{eval_type}_mono_sum_thresh.png')
        plt.figure()
        plt.imshow(multi_sum, origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
        plt.title(f'{label_dict['name']} {angle} multi sum > thresh')
        plt.savefig(f'evaluations/chamfer/img_results/{label_dict['name']}_{angle}_{eval_type}_multi_sum_thresh.png')
        plt.figure()
        plt.imshow(all_sum, origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
        plt.title(f'{label_dict['name']} {angle} all sum > thresh')
        plt.savefig(f'evaluations/chamfer/img_results/{label_dict['name']}_{angle}_{eval_type}_all_sum_thresh.png')
    # all_sum = mono_sum + multi_sum
    # plt.figure()
    # plt.imshow(all_sum, origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
    # plt.title('mono sum > thresh + multi sum > thresh')
    # plt.show()



    # numpy is already imported at module scope as np
    def get_pointcloud(all_sum, gx, gy):
        # First, get indices where all_sum > 0 (True)
        yy, xx = np.where(all_sum > 0)
        # Map mask indices to gx, gy values
        # Note: xx indices are x-axis (columns), yy are y-axis (rows), for imshow origin='lower'
        # gx and gy are both 1D and correspond to image axes
        pointcloud_x = gx[xx]
        pointcloud_y = gy[yy]
        pointcloud = np.stack([pointcloud_x, pointcloud_y], axis=-1)  # shape (num_points, 2)
        return pointcloud

    # pointcloud = get_pointcloud(all_sum, gx, gy)
    pointcloud_mono = get_pointcloud(mono_sum, gx, gy)
    pointcloud_multi = get_pointcloud(multi_sum, gx, gy)
    pointcloud_all = get_pointcloud(all_sum, gx, gy)

    # print(f"Extracted point cloud of shape: {pointcloud_all.shape}")

    # # Optionally visualize as scatter
    # plt.figure()
    # plt.scatter(pointcloud_all[:,0], pointcloud_all[:,1], s=2, c='r')
    # plt.xlim(gx[0], gx[-1])
    # plt.ylim(gy[0], gy[-1])
    # plt.title(f'{folderName} {angle} Extracted Point Cloud from all_sum Mask')
    # plt.xlabel('x [m]')
    # plt.ylabel('y [m]')
    # plt.show()

    import labels as labels
    if eval_type == 'planar':
        pc_gt = labels.planar_gt_pc(
        angle, 
        labels.metal_planar_label['loc'], 
        labels.metal_planar_label['length'],
        )
    elif eval_type == 'curvature':
        pc_gt = labels.curvature_gt_pc(
        angle, 
        )

    if show_plot:
        plt.figure()
        plt.scatter(pc_gt[:,0], pc_gt[:,1], s=2, c='r', label='gt')
        plt.scatter(pointcloud_all[:,0], pointcloud_all[:,1], s=2, label='all')
        plt.scatter(pointcloud_mono[:,0], pointcloud_mono[:,1], s=2, label='mono')
        plt.scatter(pointcloud_multi[:,0], pointcloud_multi[:,1], s=2, label='multi')
        plt.legend()
        plt.xlim(gx[0], gx[-1])
        plt.ylim(gy[0], gy[-1])
        plt.title(f'{label_dict['name']} {angle} {eval_type} Point Cloud Comparison')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.savefig(f'evaluations/chamfer/img_results/{label_dict['name']}_{angle}_{eval_type}_point_cloud.png')
        # plt.show()

    chamfer_metrics = {
        'all': chamfer.evaluate_point_clouds(pointcloud_all, pc_gt),
        'mono': chamfer.evaluate_point_clouds(pointcloud_mono, pc_gt),
        'multi': chamfer.evaluate_point_clouds(pointcloud_multi, pc_gt)
    }
    print(chamfer_metrics)
    plt.close('all')

    return chamfer_metrics




def planar_eval():
    if not os.path.exists('evaluations/chamfer/img_results'):
        os.makedirs('evaluations/chamfer/img_results')

    # Metal Planar
    results = {}
    angle_keys = [k for k in labels.metal_planar_label.keys() if isinstance(k, (int, float))]
    for angle in angle_keys:
        results |= {angle: chamfer_eval(angle, labels.metal_planar_label, show_plot=True)}
    print(results)
    with open('evaluations/chamfer/results_metal_planar.json', 'w') as f:
        json.dump(results, f)

    # Plastic Planar
    results = {}
    angle_keys = [k for k in labels.plastic_planar_label.keys() if isinstance(k, (int, float))]
    for angle in angle_keys:
        results |= {angle: chamfer_eval(angle, labels.plastic_planar_label, show_plot=True)}
    print(results)
    with open('evaluations/chamfer/results_plastic_planar.json', 'w') as f:
        json.dump(results, f)

    # Fabrics Planar
    results = {}
    angle_keys = [k for k in labels.fabrics_planar_label.keys() if isinstance(k, (int, float))]
    for angle in angle_keys:
        results |= {angle: chamfer_eval(angle, labels.fabrics_planar_label, show_plot=True)}
    print(results)
    with open('evaluations/chamfer/results_fabrics_planar.json', 'w') as f:
        json.dump(results, f)

    # Drywall Planar
    results = {}
    angle_keys = [k for k in labels.drywall_planar_label.keys() if isinstance(k, (int, float))]
    for angle in angle_keys:
        results |= {angle: chamfer_eval(angle, labels.drywall_planar_label, show_plot=True)}
    print(results)
    with open('evaluations/chamfer/results_drywall_planar.json', 'w') as f:
        json.dump(results, f)

    # Wood Planar
    results = {}
    angle_keys = [k for k in labels.wood_planar_label.keys() if isinstance(k, (int, float))]
    for angle in angle_keys:
        results |= {angle: chamfer_eval(angle, labels.wood_planar_label, show_plot=True)}
    print(results)
    with open('evaluations/chamfer/results_wood_planar.json', 'w') as f:
        json.dump(results, f)



def planar_vis():
    if not os.path.exists('evaluations/chamfer/chamfer_results'):
        os.makedirs('evaluations/chamfer/chamfer_results')

    names = ['metal_planar', 'plastic_planar', 'fabrics_planar', 'drywall_planar', 'wood_planar']
    
    for n in names:
        resutls = json.load(open(f'evaluations/chamfer/results_{n}.json'))

        plt.figure()
        plt.plot(resutls.keys(), [resutls[k]['all']['cd'] for k in resutls.keys()], label='all_cd')
        plt.plot(resutls.keys(), [resutls[k]['mono']['cd'] for k in resutls.keys()], label='mono_cd')
        plt.plot(resutls.keys(), [resutls[k]['multi']['cd'] for k in resutls.keys()], label='multi_cd')
        plt.legend()
        plt.xlabel('Angle (deg)')
        plt.ylabel('Chamfer Distance')
        plt.title(f'{n} Chamfer Distance CD Comparison')
        plt.savefig(f'evaluations/chamfer/chamfer_results/chamfer_distance_{n}_cd.png')

        plt.figure()
        plt.plot(resutls.keys(), [resutls[k]['all']['cd_pred_to_gt'] for k in resutls.keys()], label='all')
        plt.plot(resutls.keys(), [resutls[k]['mono']['cd_pred_to_gt'] for k in resutls.keys()], label='mono')
        plt.plot(resutls.keys(), [resutls[k]['multi']['cd_pred_to_gt'] for k in resutls.keys()], label='multi')
        plt.legend()
        plt.xlabel('Angle (deg)')
        plt.ylabel('Chamfer Distance')
        plt.title(f'{n} Chamfer Distance Pred to GT Comparison')
        plt.savefig(f'evaluations/chamfer/chamfer_results/chamfer_distance_{n}_pred_to_gt.png')

        plt.figure()
        plt.plot(resutls.keys(), [resutls[k]['all']['cd_gt_to_pred'] for k in resutls.keys()], label='all')
        plt.plot(resutls.keys(), [resutls[k]['mono']['cd_gt_to_pred'] for k in resutls.keys()], label='mono')
        plt.plot(resutls.keys(), [resutls[k]['multi']['cd_gt_to_pred'] for k in resutls.keys()], label='multi')
        plt.legend()
        plt.xlabel('Angle (deg)')
        plt.ylabel('Chamfer Distance')
        plt.title(f'{n} Chamfer Distance GT to Pred Comparison')
        plt.savefig(f'evaluations/chamfer/chamfer_results/chamfer_distance_{n}_gt_to_pred.png')
        # plt.show()


def curvature_eval():
    results = {}
    curvatures = [k for k in labels.curvature_label.keys() if isinstance(k, (int, float))]
    for c in curvatures:
        results |= {c: chamfer_eval(c, labels.curvature_label, eval_type='curvature', show_plot=True)}
    print(results)
    with open('evaluations/chamfer/results_curvature.json', 'w') as f:
        json.dump(results, f)


def curvature_vis():
    results = json.load(open(f'evaluations/chamfer/results_curvature.json'))

    plt.figure()
    plt.plot(results.keys(), [results[k]['all']['cd'] for k in results.keys()], label='all_cd')
    plt.plot(results.keys(), [results[k]['mono']['cd'] for k in results.keys()], label='mono_cd')
    plt.plot(results.keys(), [results[k]['multi']['cd'] for k in results.keys()], label='multi_cd')
    plt.legend()
    plt.xlabel('Curvature')
    plt.ylabel('Chamfer Distance')
    plt.title('Curvature Chamfer Distance CD Comparison')
    plt.savefig('evaluations/chamfer/chamfer_results/chamfer_distance_curvature_cd.png')

    plt.figure()
    plt.plot(results.keys(), [results[k]['all']['cd_pred_to_gt'] for k in results.keys()], label='all_cd')
    plt.plot(results.keys(), [results[k]['mono']['cd_pred_to_gt'] for k in results.keys()], label='mono_cd')
    plt.plot(results.keys(), [results[k]['multi']['cd_pred_to_gt'] for k in results.keys()], label='multi_cd')
    plt.legend()
    plt.xlabel('Curvature')
    plt.ylabel('Chamfer Distance')
    plt.title('Curvature Chamfer Distance Pred to GT Comparison')
    plt.savefig('evaluations/chamfer/chamfer_results/chamfer_distance_curvature_pred_to_gt.png')

    plt.figure()
    plt.plot(results.keys(), [results[k]['all']['cd_gt_to_pred'] for k in results.keys()], label='all_cd')
    plt.plot(results.keys(), [results[k]['mono']['cd_gt_to_pred'] for k in results.keys()], label='mono_cd')
    plt.plot(results.keys(), [results[k]['multi']['cd_gt_to_pred'] for k in results.keys()], label='multi_cd')
    plt.legend()
    plt.xlabel('Curvature')
    plt.ylabel('Chamfer Distance')
    plt.title('Curvature Chamfer Distance GT to Pred Comparison')
    plt.savefig('evaluations/chamfer/chamfer_results/chamfer_distance_curvature_gt_to_pred.png')
    # plt.show()


if __name__ == '__main__':
    import argparse
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--object', type=str, default='planar', choices=['planar', 'curvature'])
    parser.add_argument('--mode', type=str, default='eval', choices=['eval', 'vis'])
    args = parser.parse_args()

    if args.object == 'planar' and args.mode == 'eval':
        planar_eval()
    elif args.object == 'planar' and args.mode == 'vis':
        planar_vis()
    elif args.object == 'curvature' and args.mode == 'eval':
        curvature_eval()
    elif args.object == 'curvature' and args.mode == 'vis':
        curvature_vis()

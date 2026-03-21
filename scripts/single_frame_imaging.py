from re import M
import numpy as np
import yaml
import os
import sys

# Make sure the repository root is importable and set as working directory
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)
os.chdir(PROJECT_ROOT_DIR)

import muldar.utils as utils
import muldar.dsp.sar as sar
import muldar.dsp.dsp as dsp
import muldar.dsp.sar as sar
import muldar.vis as vis
import muldar.dsp.bistatic as bistatic

import scipy.constants
import matplotlib.pyplot as plt


def single_frame_imaging(folderName, lambda_mono=3, show_plot=False, dataName='', saveFolder=None):

    params = yaml.load(open(os.path.join(folderName, 'configs.yml')), Loader=yaml.FullLoader)
    adc_shape = params['chirp_cfg']


    params['frequency'] = params['ramp_cfg']['f0']
    params['fs'] = params['ramp_cfg']['Fs']
    params['slope'] = params['ramp_cfg']['slope']
    params['c'] = scipy.constants.c
    params['sar_nfft'] = 1024
    params['ref_idx'] = 256
    params['ave_win'] = 100
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
        print(frames.shape)
        _img, _ ,_ = bistatic.genBiStaticSAR(frames, tx_idx, rx_idx, gx, gy, params)
        print(_img.shape)
        img_list_all.append(_img.squeeze())
    img_list_all = np.array(img_list_all)
    print(img_list_all.shape)

    mag_all = np.abs(img_list_all).reshape(3,6,img_y_size,img_x_size)

    # per channel (tx-rx pair) visualization
    if show_plot:
        vmax = float(np.percentile(mag_all, 100))
        vmin = 0.0

        fig, axes = plt.subplots(3, 6, figsize=(18, 9), squeeze=False)
        plt.subplots_adjust(hspace=0.35, wspace=0.25)

        for tx in range(3):
            for col in range(6):
                ax = axes[tx, col]
                img = mag_all[tx, col]
                rx = col // 2
                txch = col % 2  # 0 or 1 (two TX elements/slots)
                im = ax.imshow(np.abs(img),
                            origin='lower',
                            extent=[gx[0], gx[-1], gy[0], gy[-1]],
                            aspect='equal',
                            vmin=vmin, vmax=vmax, cmap='viridis')
                ax.set_title(f"TX {tx} | RX {rx} | TXch {txch}")
                ax.set_xlabel("x [m]")
                ax.set_ylabel("y [m]")

        # single colorbar on the right
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, pad=0.02)
        cbar.set_label("Amplitude")
        plt.show()


    mag_all = np.mean(img_list_all, axis=1).reshape(3,3,img_y_size,img_x_size)

    mono_sum = np.abs(mag_all[0,0]) + np.abs(mag_all[1,1]) + np.abs(mag_all[2,2])
    multi_sum = np.abs(mag_all[2,0]+mag_all[0,2]) + \
                np.abs(mag_all[1,0]+mag_all[0,1]) +\
                np.abs(mag_all[1,2]+mag_all[2,1])

    all_sum = mono_sum*lambda_mono + multi_sum


    plt.figure()
    plt.imshow(mono_sum, origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
    plt.title(f'{dataName}: mono sum')
    if saveFolder is not None:
        plt.savefig(os.path.join(saveFolder, f'{dataName}_mono_sum.png'))
    plt.figure()
    plt.imshow(multi_sum, origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
    plt.title(f'{dataName}: multi sum')
    if saveFolder is not None:
        plt.savefig(os.path.join(saveFolder, f'{dataName}_multi_sum.png'))
    plt.figure()
    plt.imshow(all_sum, origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
    plt.title(f'{dataName}: all sum')
    if saveFolder is not None:
        plt.savefig(os.path.join(saveFolder, f'{dataName}_all_sum.png'))
         # plt.show()
        print(f"Saved images to {os.path.join(saveFolder, f'{dataName}_all_sum.png')}")
    if show_plot:
        plt.show()



if __name__ == '__main__':
    folderName = './adcData/real_object/20251121_155913'
    single_frame_imaging(folderName, show_plot=False, dataName='20251121_155913', saveFolder='.')
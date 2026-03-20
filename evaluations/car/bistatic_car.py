import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys
import matplotlib.pyplot as plt

# Make sure the repository root is importable and set as working directory
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)
os.chdir(PROJECT_ROOT_DIR)

import muldar.utils as utils
import muldar.dsp.bistatic as bistatic

import scipy.constants

import car_labels as labels

radar_names = [0,2]
radar_name2idx = {
    0:0,
    2:1,
}

img_x_size = 512
img_y_size = 512
gx = np.linspace(-3, 3.1, img_x_size)
gy = np.linspace(-2.1, 3, img_y_size)


def genBiStaticSAR(adcData, tx_radar_idx, rx_radar_idx, pos_radar_tx, pos_radar_rx, gx, gy, params):
    """Calculate for bistatic sar image given the synced adcData
    adcData [num_ch, num_frames, num_configs, num_adc]
    positions_tx [num_antenna, 2]
    positions_rx [num_antenna, 2]
    gx [num_range on x]
    gy [num_angle on y]
    nfft [num_range, num_angle]
    params [num_chirp, num_antenna, num_range, num_angle]
    """

    pos_tx = utils.antenna_positions(pos_radar_tx)[[4,6]]
    pos_rx = utils.antenna_positions(pos_radar_rx)[[0,1,2,3]]

    # direct path angle
    ref_ang = np.deg2rad(pos_radar_rx[2]) - np.arctan2(pos_radar_tx[1]-pos_radar_rx[1], pos_radar_tx[0]-pos_radar_rx[0])
    # print(np.rad2deg(ref_ang))
    frameData = adcData[:, :, tx_radar_idx*2:tx_radar_idx*2+2, :]

    align_params = params['align_params']
    img_list = []
    for i in range(2):
        _img_list = bistatic._genBiStaticSAR(frameData[:,:,i,:], tx_radar_idx, rx_radar_idx, pos_tx[[i]], pos_rx, gx, gy, params, ref_ang, align_params)
        img_list.append(_img_list)
    img_list = np.array(img_list)

    return img_list, pos_tx, pos_rx


def gen_images(collection_idx, show_images=False):

    name = labels.car_labels[collection_idx]['name']
    radar_poses = labels.car_labels[collection_idx]['radar_poses']
    global_translation = labels.car_labels[collection_idx]['translation_from_car']
    folderName = labels.car_labels[collection_idx]['fileList'][-1]
    lambda_mono_factor = labels.car_labels[collection_idx]['lambda_mono_factor']
    thresh_from_max = labels.car_labels[collection_idx]['thresh_from_max']

    print(name, folderName)

    radar_poses = np.array(radar_poses)
    global_translation = np.array(global_translation)
    radar_poses[:,:2] = radar_poses[:,:2] + global_translation

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

    import itertools

    img_list_all = []

    for tx_name, rx_name in itertools.product(radar_names, radar_names):
        fileName = os.path.join(folderName, f'radar_{rx_name}.bin')
        frames = utils.readDCA1000Robust(fileName, adc_shape)
        params['ave_win'] = frames.shape[1]
        print(frames.shape)
        
        _img, _ ,_ = genBiStaticSAR(
            frames, 
            tx_name, rx_name, 
            radar_poses[radar_name2idx[tx_name]], radar_poses[radar_name2idx[rx_name]], 
            gx, gy, params)

        _mask = bistatic.img_mask(radar_poses[radar_name2idx[rx_name]], gx, gy)[np.newaxis,:,:]
        print(_img.shape)
        img_list_all.append(_img.squeeze() * _mask)
    img_list_all = np.array(img_list_all)
    print(img_list_all.shape)

    out_dir = './evaluations/car/out/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.save(os.path.join(out_dir, f'{name}.npy'), img_list_all)

    mag_all = np.abs(img_list_all).reshape(len(radar_names),2*len(radar_names),img_y_size,img_x_size)
    
    if show_images:
        vmax = float(np.percentile(mag_all, 100))
        vmin = 0.0
        fig, axes = plt.subplots(len(radar_names), 2*len(radar_names), figsize=(18, 9), squeeze=False)
        plt.subplots_adjust(hspace=0.35, wspace=0.25)
        for tx in range(len(radar_names)):
            for col in range(2*len(radar_names)):
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
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, pad=0.02)
        cbar.set_label("Amplitude")
        # uncomment for per channel images
        # plt.show()

    # global color scaling based on all images
    mag_all = np.mean(img_list_all, axis=1).reshape(len(radar_names),len(radar_names),img_y_size,img_x_size)

    print(mag_all.shape)

    if show_images:
        vmax = float(np.percentile(np.abs(mag_all), 100))
        vmin = 0.0
        fig, axes = plt.subplots(len(radar_names), len(radar_names), figsize=(9, 9), squeeze=False)
        plt.subplots_adjust(hspace=0.35, wspace=0.25)
        for tx in range(len(radar_names)):
            for col in range(len(radar_names)):
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
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95, pad=0.02)
        cbar.set_label("Amplitude")

    mono_sum = np.abs(mag_all[0,0]) + np.abs(mag_all[1,1])
    multi_sum = np.abs(mag_all[1,0]) + np.abs(mag_all[0,1])

    all_sum = mono_sum * lambda_mono_factor + multi_sum

    thresh = thresh_from_max

    if show_images:
        plt.figure(dpi=300)
        plt.imshow(mono_sum, origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
        plt.title(f'{name} mono sum')
        plt.savefig(f'./evaluations/car/out/{name}_mono_sum.png')
        plt.figure(dpi=300)
        plt.imshow(multi_sum, origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
        plt.title(f'{name} multi sum')
        plt.savefig(f'./evaluations/car/out/{name}_multi_sum.png')
        plt.figure(dpi=300)
        plt.imshow(all_sum, origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
        plt.title(f'{name} all sum')
        plt.savefig(f'./evaluations/car/out/{name}_all_sum.png')
        # plt.show()

    return all_sum, mono_sum, multi_sum


def eval_all_data():
    collection_index = [x for x in labels.car_labels.keys() if isinstance(x, int)]
    img_list = []
    pc_lsit = []
    for collection_idx in collection_index:
        # if collection_idx not in [1,3]:
        #     continue
        _all_sum, _mono_sum, _multi_sum = gen_images(collection_idx, show_images=True)
        img_list.append([_all_sum, _mono_sum, _multi_sum])
        plt.close('all')
    img_list = np.array(img_list)
    print(img_list.shape)


def vis_data():
    fileList = [
        './evaluations/car/out/0_left_side_0.npy',
        './evaluations/car/out/1_left_side_back_corner_0.npy',
        './evaluations/car/out/2_left_back_corner_0.npy',
        './evaluations/car/out/3_backside_0.npy',
    ]

    imgs = [np.load(file) for file in fileList]
    imgs = np.array(imgs)

    imgs = np.mean(imgs, axis=2)
    # #collection, #radar^2, #y, #x
    print(imgs.shape)

    all_sum = np.zeros_like(imgs[0,0])
    mono_sum = np.zeros_like(imgs[0,0])
    multi_sum = np.zeros_like(imgs[0,0])

    selecting_list = [0,2]

    # for i, curr_img in enumerate(imgs):
    for i, curr_img in zip(selecting_list, imgs[selecting_list]):
        curr_img = curr_img.reshape(2,2,img_y_size,img_x_size)
        print(curr_img.shape)
        lambda_mono_factor = labels.car_labels[i]['lambda_mono_factor']
        # thresh_from_max = labels.car_labels[i]['thresh_from_max']
        _mono_sum = np.abs(curr_img[0,0]) + np.abs(curr_img[1,1])
        _multi_sum = np.abs(curr_img[1,0] + curr_img[0,1])
        _all_sum = _mono_sum * lambda_mono_factor + _multi_sum
        _mono_sum /= _mono_sum.max()
        _multi_sum /= _multi_sum.max()
        _all_sum /= _all_sum.max()
        all_sum += _all_sum
        mono_sum += _mono_sum
        multi_sum += _multi_sum
        plt.figure()
        plt.imshow(np.abs(_all_sum), origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
        plt.title(f'{labels.car_labels[i]["name"]}')
        print(_all_sum.shape)
    
    translations = [label['translation_from_car'] for label in labels.car_labels.values()]
    locs = [label['radar_poses'][:2] for label in labels.car_labels.values()]
    locs_global = [
        [[x[0][0]+y[0], x[0][1]+y[1]],
         [x[1][0]+y[0], x[1][1]+y[1]]] for x, y in zip(locs, translations)]
    locs_global = np.array(locs_global)

    plt.figure()
    plt.imshow(np.abs(all_sum), origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
    plt.savefig(f'./evaluations/car/out/sum_of_all_images.png')

    plt.figure()
    plt.imshow(np.abs(mono_sum), origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
    plt.title(f'{labels.car_labels[selecting_list[0]]["name"]} Sum of mono images')
    plt.savefig(f'./evaluations/car/out/sum_of_mono_images.png')
    plt.figure()
    plt.imshow(np.abs(multi_sum), origin='lower', extent=[gx[0], gx[-1], gy[0], gy[-1]], aspect='equal')
    plt.title(f'{labels.car_labels[selecting_list[0]]["name"]} Sum of multi images')
    plt.savefig(f'./evaluations/car/out/sum_of_multi_images.png')
    plt.show()


if __name__ == '__main__':
    eval_all_data()
    vis_data()

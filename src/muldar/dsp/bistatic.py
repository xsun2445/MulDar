import numpy as np
import muldar.dsp.dsp as dsp
import muldar.dsp.sar as sar
import muldar.utils as utils



def hpf_correction(adcData):
    amp_shift_saveName = f'C:/Workspace/temp_projs/polsar_python/collection/notebook/robustness/RV_rx0_correction.npy'
    amp_shift = np.load(amp_shift_saveName)
    amp_shift /= np.max(amp_shift)
    _fftData = np.fft.fft(adcData, 4096, -1)
    _fftData /= amp_shift
    adcData = np.fft.ifft(_fftData, 4096, -1)[:,:,:256]

    return adcData





def alignBistaticData(adcData, tx_radar_idx, rx_radar_idx, params):
    pos_radar_tx = np.array(params['activated_radar'][tx_radar_idx]['pose'])
    pos_radar_rx = np.array(params['activated_radar'][rx_radar_idx]['pose'])
    ref_ang = np.deg2rad(pos_radar_rx[2]) - np.arctan2(pos_radar_tx[1]-pos_radar_rx[1], pos_radar_tx[0]-pos_radar_rx[0])
    frameData = adcData[:, :, tx_radar_idx*2:tx_radar_idx*2+2, :]
    res = []
    for i in range(2):
        aligned_sig = sync_signal(frameData[:,:,i,:], ref_ang, params['align_params'])
        res.append(aligned_sig)
    return np.array(res)





def genBiStaticSAR(adcData, tx_radar_idx, rx_radar_idx, gx, gy, params):
    """Calculate for bistatic sar image given the synced adcData
    adcData [num_ch, num_frames, num_configs, num_adc]
    positions_tx [num_antenna, 2]
    positions_rx [num_antenna, 2]
    gx [num_range on x]
    gy [num_angle on y]
    nfft [num_range, num_angle]
    params [num_chirp, num_antenna, num_range, num_angle]
    """

    pos_radar_tx = np.array(params['activated_radar'][tx_radar_idx]['pose'])
    pos_radar_rx = np.array(params['activated_radar'][rx_radar_idx]['pose'])
    pos_tx = utils.antenna_positions(pos_radar_tx)[[4,6]]
    pos_rx = utils.antenna_positions(pos_radar_rx)[[0,1,2,3]]

    # direct path angle
    ref_ang = np.deg2rad(pos_radar_rx[2]) - np.arctan2(pos_radar_tx[1]-pos_radar_rx[1], pos_radar_tx[0]-pos_radar_rx[0])
    # print(np.rad2deg(ref_ang))
    frameData = adcData[:, :, tx_radar_idx*2:tx_radar_idx*2+2, :]

    align_params = params['align_params']
    img_list = []
    for i in range(2):
        _img_list = _genBiStaticSAR(frameData[:,:,i,:], tx_radar_idx, rx_radar_idx, pos_tx[[i]], pos_rx, gx, gy, params, ref_ang, align_params)
        img_list.append(_img_list)
    img_list = np.array(img_list)

    # # first channel of tx
    # aligned_sig = sync_signal(frameData[:,:,0,:], ref_ang, align_params)
    # avesig = aligned_sig.reshape(frameData.shape[0],-1,params['ave_win'],256).mean(axis=2)
    # avesig = dsp.notch_if(avesig, align_params['nfft'], align_params['f_tar_idx'], 5e5)
    # # gen bisar
    # _img_list = []
    # for i in range(avesig.shape[1]):
    #     img = sar.bistatic_sar_bp_gpu(avesig[:,i,:], pos_tx[[0]], pos_rx, gx, gy, params['sar_nfft'], params, return_complex=True)
    #     _img_list.append(img.get())
    # _img_list = np.array(_img_list)
    # img_list.append(_img_list)
    
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(np.abs(np.fft.fft(aligned_sig[0,:,:], 1024, -1)).T)
    # plt.plot(np.abs(np.fft.fft(avesig[0,:], 1024, -1)).T, lw=2)
    # plt.show()

    # # second channel of tx
    # aligned_sig = sync_signal(frameData[:,:,1,:], ref_ang, align_params)
    # avesig = aligned_sig.reshape(frameData.shape[0],-1,params['ave_win'],256).mean(axis=2)
    # avesig = dsp.notch_if(avesig, align_params['nfft'], align_params['f_tar_idx'], 5e5)
    # # gen bisar
    # _img_list = []
    # for i in range(avesig.shape[1]):
    #     img = sar.bistatic_sar_bp_gpu(avesig[:,i,:], pos_tx[[1]], pos_rx, gx, gy, params['sar_nfft'], params, return_complex=True)
    #     _img_list.append(img.get())

    

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(np.abs(np.fft.fft(aligned_sig[0,:,:], 1024, -1)).T)
    # plt.show()

    return img_list, pos_tx, pos_rx



def _genBiStaticSAR(frameData, tx_radar_idx, rx_radar_idx, pos_tx, pos_rx, gx, gy, params, ref_ang, align_params):
    if tx_radar_idx == rx_radar_idx:
        avesig = frameData.reshape(frameData.shape[0],-1,params['ave_win'],256).mean(axis=2)
        params = params.copy()
        params['ref_idx'] = 0
    else:
        # import matplotlib.pyplot as plt
        # plt.figure()
        aligned_sig = sync_signal(frameData, ref_ang, align_params)
        avesig = aligned_sig.reshape(frameData.shape[0],-1,params['ave_win'],256).mean(axis=2)
        # plt.plot(np.abs(np.fft.fft(avesig[0,:,:], 1024, -1)).T)
        temp = np.fft.fft(avesig[:,:,:], 256, -1)
        # temp[:,:,:64+15] = 0
        temp[:,:,:64+6] = 0
        # temp[:,:,:64+40] = 0
        # temp[:,:,:72] = 0
        avesig = np.fft.ifft(temp, 256, -1)

        avesig = dsp.notch_if(avesig, align_params['nfft'], align_params['f_tar_idx'], 5e5)
        # plt.plot(np.abs(np.fft.fft(avesig[0,:,:], 1024, -1)).T)
        # plt.show()
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(np.abs(np.fft.fft(avesig[0,:,:], 1024, -1)).T)
        # plt.show()

    _img_list = []
    for i in range(avesig.shape[1]):
        # avesig[1] = 0
        # avesig[2] = 0
        # avesig[3] = 0
        img = sar.bistatic_sar_bp_gpu(avesig[:,i,:], pos_tx, pos_rx, gx, gy, params['sar_nfft'], params, return_complex=True)
        _img_list.append(img.get())

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(np.abs(img.get()), aspect='equal', origin='lower')
        # plt.show()

    _img_list = np.array(_img_list)

    return _img_list


def img_mask(pos_rx, gx, gy, fov_half_angle_deg=None):

    # pos_rx expected as [x, y, angle_deg]
    pos_rx_arr = np.asarray(pos_rx, dtype=float).ravel()
    if pos_rx_arr.size < 3:
        raise ValueError("pos_rx must be [x, y, angle_deg]")
    radar_pos = np.array([pos_rx_arr[0], pos_rx_arr[1]], dtype=float)
    heading_rad = np.deg2rad(pos_rx_arr[2])

    # Build mask purely from gx, gy (shape: (len(gy), len(gx)))
    X, Y = np.meshgrid(gx, gy)
    RX = X - radar_pos[0]
    RY = Y - radar_pos[1]

    ux, uy = np.cos(heading_rad), np.sin(heading_rad)
    dot = RX * ux + RY * uy

    mask = (dot >= 0).astype(np.uint8)  # 1 in front, 0 behind

    # Optional FOV wedge around heading
    if fov_half_angle_deg is not None:
        fov_half = np.deg2rad(float(fov_half_angle_deg))
        ang = np.arctan2(RY, RX)
        d = np.arctan2(np.sin(ang - heading_rad), np.cos(ang - heading_rad))  # wrap to [-pi, pi]
        mask = (mask & (np.abs(d) <= fov_half)).astype(np.uint8)

    return mask


def sync_signal(frames, ref_ang, align_params=None, chirp_clip=[0,-64]):
    """
    adcData [num_ch, num_frames, num_adc]
    """
    import muldar.dsp.dsp as dsp
    if align_params is None:
        align_params = {
            'nfft': 1024,
            'f_tar_idx': 256,
            'phase_ref': 0.0,
            'min_height_ratio': 0.5,
            'min_prominence_ratio': 0.3,
        }
    steering_vec = np.exp(-1j*np.pi*np.sin(ref_ang)*np.arange(4))[None, :]
    df = []
    da = []
    # import matplotlib.pyplot as plt
    # plt.figure()
    for i in range(frames.shape[1]):
        ref_sig = steering_vec@frames[:,i,chirp_clip[0]:chirp_clip[1]]
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(np.abs(np.fft.fft(ref_sig, 1024, -1)).T)
        # plt.show()
        # # if i < 10:
        #     # plt.plot(np.abs(np.fft.fft(ref_sig, 1024, -1)).T)
        _df, _da = dsp.find_ref_frq_first_peak(ref_sig, **align_params)
        df.append(_df)
        da.append(_da)
    # plt.show()
    df = np.array(df).ravel()
    da = np.array(da).ravel()
    aligned_sig = frames*np.exp(1j*(2*np.pi*df[:,None]*np.arange(256)[None,:]+da[:,None]))[None,:,:]
    
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(np.abs(np.fft.fft(aligned_sig[0,:,:], 1024, -1)).T)
    # # plt.plot(np.abs(np.fft.fft(frames[0,:,:], 1024, -1)).T, lw=2)
    # plt.show()
    # # import time
    # # time.sleep(10)
    # # exit()
    
    return aligned_sig
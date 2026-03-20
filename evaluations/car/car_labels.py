car_labels = {
    0: {
        'name': '0_left_side_0',
        'fileList': [
            './adcData/car/0_left_side_0/20251203_103321', # calibration
            './adcData/car/0_left_side_0/20251203_103406',
            './adcData/car/0_left_side_0/20251203_103438',
            './adcData/car/0_left_side_0/20251203_103513',
            './adcData/car/0_left_side_0/20251203_103559', # car
            './adcData/car/0_left_side_0/20251203_103626',
        ],
        'radar_poses': [
            [0, 0, 140.73],
            [-2.3354, 0.1474, 36.25],
        ],
        'translation_from_car': [0, -1.66], # translation from left back tire to radar 0
        'lambda_mono_factor': 1,
        'thresh_from_max': 0.3,
    },

    1: {
        'name': '1_left_side_back_corner_0',
        'fileList': [
            './adcData/car/1_left_side_back_corner_0/20251203_104928', # calibration
            './adcData/car/1_left_side_back_corner_0/20251203_105113',
            './adcData/car/1_left_side_back_corner_0/20251203_105152',
            './adcData/car/1_left_side_back_corner_0/20251203_105236',
            './adcData/car/1_left_side_back_corner_0/20251203_105314', # car
            './adcData/car/1_left_side_back_corner_0/20251203_105401', # car
            './adcData/car/1_left_side_back_corner_0/20251203_105418',
        ],
        'radar_poses': [
            [0, 0, 154.7],
            [-2.12079, -0.9976, 79],
        ],
        'translation_from_car': [1.853, -1.0175], # translation from left back tire to radar 0
        'lambda_mono_factor': 0.4,
        'thresh_from_max': 0.3,
    },

    2: {
        'name': '2_left_back_corner_0',
        'fileList': [
            './adcData/car/2_left_back_corner_0/20251203_110120', # calibration
            './adcData/car/2_left_back_corner_0/20251203_110217',
            './adcData/car/2_left_back_corner_0/20251203_110253',
            './adcData/car/2_left_back_corner_0/20251203_110337',
            './adcData/car/2_left_back_corner_0/20251203_110406', # car
            './adcData/car/2_left_back_corner_0/20251203_110426',
        ],
        'radar_poses': [
            [0, 0, 169.8],
            [-1.5338, -1.3323, 99],
        ],
        'translation_from_car': [3.048, -0.285], # translation from left back tire to radar 0
        'lambda_mono_factor': 0.4,
        'thresh_from_max': 0.3,
    },

    3: {
        'name': '3_backside_0',
        'fileList': [
            './adcData/car/3_backside_0/20251203_111033', # calibration
            './adcData/car/3_backside_0/20251203_111238',
            './adcData/car/3_backside_0/20251203_111342',
            './adcData/car/3_backside_0/20251203_111414',
            './adcData/car/3_backside_0/20251203_111448', # car
            './adcData/car/3_backside_0/20251203_111525',
        ],
        'radar_poses': [
            [0, 0, 209.2],
            [-0.1622, -2.5277, 139.3],
        ],
        'translation_from_car': [2.8722, 2.033], # translation from left back tire to radar 0
        'lambda_mono_factor': 10,
        'thresh_from_max': 0.3,
    },
}


video_label = {
    'name': '4_driving_video',
    'fileList': [
        './adcData/car/4_driving_video/20251203_113624',
        './adcData/car/4_driving_video/20251203_113833',
        './adcData/car/4_driving_video/20251203_113956',
    ],
    'radar_poses': [
        [2.27/2, 0, 110.3],
        [-2.27/2, 0.035, 62],
    ],
    'lambda_mono_factor': 1,
    'thresh_from_max': 0.3,
    'frame_rate_ms': 10,
    'ave_win': 10,
    'translation_from_car': [0, 0],
}

if __name__ == '__main__':
    translations = [label['translation_from_car'] for label in car_labels.values()]
    print(translations)
    locs = [label['radar_poses'][:2] for label in car_labels.values()]
    print(locs)
    locs_global = [
        [[x[0][0]+y[0], x[0][1]+y[1]],
         [x[1][0]+y[0], x[1][1]+y[1]]] for x, y in zip(locs, translations)]
    print(locs_global)


    import numpy as np
    locs_global = np.array(locs_global)
    print(locs_global.shape)

    import matplotlib.pyplot as plt
    plt.figure()
    for i, loc in enumerate(locs_global):
        plt.plot(loc[1,0], loc[1,1], 'o', c='b', label='Radar 0')
        plt.plot(loc[0,0], loc[0,1], 'o', c='r', label='Radar 1')
        plt.text(loc[1,0], loc[1,1], f'{i}', color='b', fontsize=12, va='bottom', ha='left')
        plt.text(loc[0,0], loc[0,1], f'{i}', color='r', fontsize=12, va='bottom', ha='left')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Radar Locations')
    plt.grid(True)
    plt.show()

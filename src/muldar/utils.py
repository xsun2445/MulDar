import numpy as np
import os

def antenna_positions(orientation):
    """Move the antenna positions to the given orientation in 2D space
    orientation: [x, y, yaw]
    """
    return radar_to_global(_antenna_positions(), orientation)


def _antenna_positions():
    """The default antenna positions for the AWR2243 on 2D experiment coordinate system
    """
    ant_pos = __antenna_positions()
    ant_pos[:, 0] = -ant_pos[:, 0]
    ant_pos[:, 1] = 0
    return ant_pos


def __antenna_positions():
    """The default antenna positions for the AWR2243 on radar coordinate system
    """
    d = 1.9e-3
    return np.array([
        [0,0],
        [d,0],
        [2*d,0],
        [3*d,0],
        [6*d,0],
        [8*d,d],
        [10*d,0],
    ])


def radar_to_global(pts, orientation):
    """transform the pts from radar coordinate system to global coordinate system
    pts: (N, 2) [x, y]
    orientation: [x, y, yaw] radar's orientation
    """
    ang = -np.deg2rad(orientation[2]-90)
    # print(orientation[2], ang)
    translation_mat = np.array([orientation[0], orientation[1]])

    rot_mat = np.array([
        [np.cos(ang), -np.sin(ang)],
        [np.sin(ang), np.cos(ang)],
    ])
    return pts @ rot_mat + translation_mat


def render_lua_template(template_path: str, params: dict) -> str:
    from string import Template
    from pathlib import Path
    text = Path(template_path).read_text(encoding="utf-8")
    return Template(text).substitute(**params)


def parseConfigs(config_path='./configs/config_new.yml'):
    """This function is used for the distributed imaging project
    """
    import yaml
    configs = yaml.load(open(config_path), Loader=yaml.FullLoader)
    return configs


def calc_range_resolution(fs_hz, slope_hz_per_s, nfft_range):
    return fs_hz*3e8 /(2*nfft_range*slope_hz_per_s)


def calc_filesize(num_ch, num_chirp, num_config, num_adc):
    """Calculate for the filesize in bytes
    complex values are 2 bytes each
    int16 values are 2 bytes each
    """
    return num_ch * num_chirp * num_config * num_adc * 2 * 2


def readDCA1000(fileName):
    """
    Read DCA1000 data from file, work for AWR2243
    """
    # % globalvariables
    # % change based on sensor config
    numADCBits = 16		# % number of ADC bits per sample
    numLanes = 4		# % do not change. number of lanes is always 4 even if only 1 lane is used. unused lanes
    isReal = 0 			# % set to 1 if real only data, 0 if complex data are populated with 0

    adcData = np.fromfile(open(fileName, 'rb'), np.int16).reshape([2*numLanes,-1], order='F')
    adcData = adcData[:4] + 1j*adcData[4:]
    # adcData = adcData.astype(np.complex64)

    return adcData


def readDCA1000fromBuffer(buffer):
    """
    Read DCA1000 data from buffer, work for AWR2243
    """
    numLanes = 4
    adcData = np.frombuffer(buffer, dtype=np.int16).reshape([2*numLanes,-1], order='F')
    adcData = adcData[:4] + 1j*adcData[4:]

    return adcData


def readDCA1000Robust(fileName, adc_shape):
    adcData = readDCA1000(fileName)
    num_bytes = adcData.shape[1]
    frame_bytes = adc_shape['num_config'] * adc_shape['num_adc']
    num_frames = num_bytes // frame_bytes
    adcData = adcData[:,:num_frames*frame_bytes]
    return adcData.reshape(
        adc_shape['num_ch'],
        -1,
        adc_shape['num_config'],
        adc_shape['num_adc']
    )


def split_path(path):
    # Recursive version to split the path into its components as a list
    head, tail = os.path.split(path)
    if head == '' or head == path:
        return [path] if path else []
    elif tail == '':
        return split_path(head)
    else:
        return split_path(head) + [tail]


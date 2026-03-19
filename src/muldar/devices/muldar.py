"""
Handling muldar radar trigger/retrigger and live data streaming

Hardware triggering can cause MSS error and data loss hence need multiple mmWaveStudio 
retriggering. The retriggering is done by sending a TCP message to the MATLAB studio on 
each radar PC, which will then trigger the radar. The radar data is received in real-time 
and can be saved to a file if needed. The MultiRadarManager class manages multiple 
radars and handles the triggering and data collection process.

The data source can also come from recorded raw data files
"""

import os
import struct
import time
import socket
import threading
import numpy as np

import muldar.utils as utils
from muldar.devices.radar import DCA1000, CMD, CMD_SUCCESS_CODE, MAX_PACKET_SIZE, FILLING_PACKET


def send_to_matlab(matlab_ip, port, message):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.connect((matlab_ip, port))
        client.sendall(message.encode())
        client.close()


def retrigger_studio(retrigger_device_list):
    """use tcp to notify matlab on different devices to retrigger the radar mmwavestudio
    Radar MSS Error will show up without triggering, and no data will be received.
    """
    def notify_matlab(matlab_ip, port, message="Retrigger\n"):
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                # client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # Disable Nagle's algorithm
                print(f"Connecting to MATLAB at {matlab_ip}:{port}...\t", end='', flush=True)
                client.connect((matlab_ip, port))
                # print(f"Connected to MATLAB at {matlab_ip}:{port}")
                client.sendall(message.encode())  # Send notification
                # print(f"Sent: {message.replace('\n', '')}")
                print(f"mmWaveStudio retriggered!")
                client.close()
        except Exception as e:
            print(f"Error: {e}")
    for device in retrigger_device_list:
        _ip = device['IP']
        _port = device['server_access_port']
        notify_matlab(_ip, _port)


def trigger_radar(num_frames: int,
                 period_ms: int,
                 raspi_ip: str = '192.168.50.155',
                 port: int = 5000,
                 delimiter: str = '@$@$',
                 greeting: str = 'hello from server!') -> None:
    """One-shot TCP client: connect → send command → wait for 'Done.' → close.

    Message format: f"{greeting}{delimiter}{num_frames}{delimiter}{period_ms}" (UTF-8)
    """
    msg = f"{greeting}{delimiter}{num_frames}{delimiter}{period_ms}".encode()
    print(f"Connecting to Raspberry Pi at {raspi_ip}:{port}... ", end='', flush=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(10)
        client.connect((raspi_ip, port))
        client.sendall(msg)
        try:
            ack = client.recv(1024)
            print(f"Hardware triggered!")
        except Exception:
            print("No ACK received (continuing)")


class DCA1000_realtime(DCA1000):
    """
    Real-time radar data recording. will continuously record data and update the data attribute.
    
    Handles the binary -> numpy array conversion and reshaping. 
    """
    def __init__(self, radar_params):
        super().__init__(
            system_ip="192.168.33.30", 
            dca_ip=radar_params['ip'], 
            data_port=radar_params['data_port'], 
            config_port=radar_params['cfg_port'])
        self.name = radar_params['name']
        self.idx = radar_params['idx']
        self.adc_shape = radar_params['adc_shape']
        self.frame_bytes = utils.calc_filesize(
            self.adc_shape['num_ch'], 
            self.adc_shape['num_chirp'], 
            self.adc_shape['num_config'], 
            self.adc_shape['num_adc'])
        self.frame_counter = 0

    def configure(self):
        assert self._send_command(CMD.SYSTEM_CONNECT_CMD_CODE).hex() == CMD_SUCCESS_CODE.SYSTEM_CONNECT_SUCCESS
        assert self._send_command(CMD.READ_FPGA_VERSION_CMD_CODE).hex() == CMD_SUCCESS_CODE.READ_FPGA_VERSION_SUCCESS
        assert self._send_command(CMD.CONFIG_FPGA_GEN_CMD_CODE, '0600', '01010102031e').hex() == CMD_SUCCESS_CODE.CONFIG_FPGA_GEN_SUCCESS
        assert self._send_command(CMD.CONFIG_PACKET_DATA_CMD_CODE, '0600', 'c00537000000').hex() == CMD_SUCCESS_CODE.CONFIG_PACKET_DATA_SUCCESS
        assert self._send_command(CMD.RECORD_STOP_CMD_CODE).hex() == CMD_SUCCESS_CODE.RECORD_STOP_SUCCESS
        assert self._send_command(CMD.RECORD_START_CMD_CODE).hex() == CMD_SUCCESS_CODE.RECORD_START_SUCCESS
    
    def __del__(self):
        self.close()

    def record(self, saveName=None, timeout=10):
        self.data_socket.settimeout(timeout)
        if saveName is not None:
            f_data = open(saveName, 'wb')
            logName = os.path.join(os.path.dirname(saveName), os.path.basename(saveName).split('.')[0]+'_log.csv')
            f_log = open(logName, 'w')

        cnt_byte = 0
        prev_seq = 0
        buffer = b''
        
        while True:
            try:
                _data, addr = self.data_socket.recvfrom(MAX_PACKET_SIZE)
                # resolve header
                packet_num = struct.unpack('<1l', _data[:4])[0]
                cnt_byte += len(_data)-10
                _data = _data[10:]
                # check for missing packets
                if packet_num - prev_seq > 1:
                    print(f'Missing packet - radar {self.idx}: {packet_num - prev_seq}')
                prev_seq += 1
                # zero-fill missing packets
                while prev_seq < packet_num:
                    _data = FILLING_PACKET + _data
                    prev_seq += 1
                buffer += _data
                # print progress
                if packet_num%500 == 0:
                    print(f'radar {self.idx} - {self.dca_ip}: {packet_num} bytes')

                if len(buffer) >= self.frame_bytes:
                    buffer = self.update_data(buffer)

                if saveName is not None:
                    f_data.write(_data)
                    f_log.write(','.join([str(packet_num), str(len(_data)-10), str(time.time())]) + '\n')

            except Exception as e:
                print(f"radar {self.idx} Error: {e}")
                if saveName is not None:
                    f_data.close()
                    f_log.close()
                break
        return

    def update_data(self, buffer):
        num_frames = len(buffer) // self.frame_bytes
        num_bytes = num_frames*self.frame_bytes
        self.data = utils.readDCA1000fromBuffer(buffer[:num_bytes]).reshape(
                        self.adc_shape['num_ch'], 
                        -1, 
                        self.adc_shape['num_config'],
                        self.adc_shape['num_adc']
                    )
        self.frame_counter += 1
        # print(f'{self.name}: {self.frame_counter} - {self.data.shape}')
        return buffer[num_bytes:]


class RadarFileReplay:
    """
    File replay source class. will read data from a file and update the data attribute.
    """
    def __init__(self, radar_params):
        import muldar.utils as utils
        self.name = radar_params['name']
        self.adc_shape = radar_params['adc_shape']
        self.timeout = radar_params.get('timeout', 0.01)
        self.file_path = radar_params['from_record_file']
        self.replay_period = radar_params['period_frame'] * 1e-3  # in s
        self.data = None
        self.frame_counter = 0

        self.frame_bytes = utils.calc_filesize(
            self.adc_shape['num_ch'],
            self.adc_shape['num_chirp'],
            self.adc_shape['num_config'],
            self.adc_shape['num_adc']
        )

    def configure(self):
        return

    def _send_command(self, *_args, **_kwargs):
        return b''

    def record(self, *args, **kwargs):
        frames = utils.readDCA1000(self.file_path).reshape(
            self.adc_shape['num_ch'],
            -1,
            self.adc_shape['num_config'],
            self.adc_shape['num_adc']
        )
        print(f'{self.name}: replaying radar file: {self.file_path}, file size: {frames.shape}')

        for i in range(frames.shape[1]):
            self.data = frames[:, i:i+1, :, :]
            time.sleep(self.replay_period)
            self.frame_counter += 1


class RadarPlayer:
    def __init__(self, radar_params):
        self.name = radar_params['name']
        self.idx = radar_params['idx']
        self.adc_shape = radar_params['adc_shape']  # {'num_ch':int,'num_config':int,'num_adc':int}
        self.timeout = radar_params['timeout']
        self.cfg_idx = radar_params['mono_chirp_idx']
        self.save_name = radar_params.get('save_name', None)

        # choose backend by presence of 'file_path'
        if 'from_record_file' in radar_params and radar_params['from_record_file']:
            self.dca = RadarFileReplay(radar_params)
        else:
            self.dca = DCA1000_realtime(radar_params)

        self._stop = threading.Event()
        self._thread = None

        self._lock = threading.Lock()
        self.latest = None   # expected shape (num_ch, num_chirps, num_config, num_adc)

    def start(self):
        self.dca.configure()
        print(f'Radar {self.idx} ({self.name}) is online')
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        self._thread = t

    def stop(self):
        self._stop.set()
        print(f"radar {self.idx} total #frames: {self.dca.frame_counter}")
    
    def join(self):
        self._thread.join()

    def _run(self):
        t = threading.Thread(target=self.dca.record, args=(),
                             kwargs={'saveName': self.save_name, 'timeout': self.timeout}, daemon=True)
        t.start()
        while not self._stop.is_set() and t.is_alive():
            data = getattr(self.dca, 'data', None)
            if isinstance(data, np.ndarray) and data.ndim == 4 and data.shape[3] == self.adc_shape['num_adc']:
                with self._lock:
                    self.latest = data.copy()
            time.sleep(0.01)

    def get_latest(self):
        with self._lock:
            if self.latest is None:
                return None
            return self.latest.copy()

    
class MultiRadarManager:
    def __init__(self, params):
        self.params = params
        if params['flag_save']:
            import yaml
            self.save_dir = os.path.join(params['saving_root_dir'], time.strftime('%Y%m%d_%H%M%S'))
            os.makedirs(self.save_dir, exist_ok=False)
            with open(os.path.join(self.save_dir, 'configs.yml'), 'w') as f:
                yaml.dump(params, f)
        self.radar_cfgs = []
        self.retrigger_device_list = []
        for r in params['activated_radar']:
            if 'from_record_file' not in r:
                self.retrigger_device_list.append(params['comm_cfg'][r['client_name']])
            _r_cfg = params['radar_id'][r['idx']] | r  # b overrides a on key conflicts
            _r_cfg['adc_shape'] = params['chirp_cfg']
            _r_cfg['period_frame'] = params['period_frame']
            _r_cfg['timeout'] = params['radar_timeout']
            if params['flag_save']:
                _r_cfg['save_name'] = os.path.join(self.save_dir, f'radar_{r['idx']}.bin')
            self.radar_cfgs.append(_r_cfg)
        self.radars = [RadarPlayer(cfg) for cfg in self.radar_cfgs]
        self.flag_retrigger = params.get('flag_retrigger', True)
        self.wait_for_threads = params['wait_for_threads']

    def trigger_radar(self, num_loop, period_ms):
        return trigger_radar(
            num_loop, 
            period_ms, 
            self.params['comm_cfg']['trigger'].get('IP', None), 
            self.params['comm_cfg']['trigger'].get('server_access_port', None), 
            self.params['comm_cfg'].get('DELIMITER', None), 
            self.params['comm_cfg']['trigger'].get('greeting', None)
        )
    
    def retrigger_studio(self):
        return retrigger_studio(self.retrigger_device_list)

    def start(self, num_loop=None, period_ms=None):
        num_loop = num_loop if num_loop is not None else self.params['num_trigger']
        period_ms = period_ms if period_ms is not None else self.params['period_frame']
        if self.flag_retrigger and self.retrigger_device_list:
            self.retrigger_studio()
        for r in self.radars:
            r.start()
        time.sleep(1.5) # required otherwise will miss initial triggering pulses
        if self.retrigger_device_list:
            self.trigger_radar(num_loop, period_ms)
    
    def join(self):
        for r in self.radars:
            r.join()
    
    def stop(self):
        if self.wait_for_threads:
            self.join()
        for r in self.radars:
            r.stop()


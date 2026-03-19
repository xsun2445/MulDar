"""
This script configures the radar starting frequency (range offset) for multiple radars and sends the 
configuration to the respective MATLAB studios using Lua scripts. Each children PC will use the LUA 
script to config the radar
"""


MULDAR_CONFIG_PATH = "configs.yml"
RADAR_CONFIG_FILE_PATH = "hardware/radar/config_template.txt"

# Each radar will transmit 2 chirps and all radar will listen.
# monostatic channel should has 0 frequency offset, 
# bistatic channel should have non-zero (Usually >0) frequency offset to make 
# the bi-static channel show up in the IF signal and not affected by LPF
# Each field represent the index of transmitting radar and the frequency offset for 
# each receiving radar (including itself).
# Hence the diagonal entries should be 0, and the non-diagonal entries should be >0.

radar_cfg_list = [
     {
        "RADAR_IDX": 0, # transmitting radar index
        "FREQ_DELTA_SEQ": "0, 0.600, 0.400",  # Receiving radar starting frequency offsets (MHz)
    },
    {
        "RADAR_IDX": 1,
        "FREQ_DELTA_SEQ": "0.9, 0, 0.500",  # MHz
    },
    {
        "RADAR_IDX": 2,
        "FREQ_DELTA_SEQ": "1.2, 1.0, 0",  # MHz
    }
]


# scripts/send_lua.py
from muldar.utils import render_lua_template
import yaml
DELIM = "@$@$"


def send_lua_to_matlab(host, port, lua_text, timeout=10.0):
    import socket
    payload = lua_text.encode("utf-8")
    header = f"RunGivenScript{DELIM}{len(payload)}\n".encode("ascii")
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(header)
        s.sendall(payload)


comm_params = yaml.load(open(MULDAR_CONFIG_PATH), Loader=yaml.FullLoader)

static_params = {
    "START_FREQ": 76.0100000,      # GHz
    "TX_SEQ": "0,1,2",
    "IDLE_TIME": 15.0,
    "FREQ_SLOPE": 38.816929,
    "ADC_START_TIME": 7,
    "TX_START_TIME": 0,
    "ADC_SAMPLES": 256,
    "ADC_SAMPLE_RATE": 2150,
    "RAMP_END_TIME": 128.5,
    "NUM_FRAMES": 0,                # infinite
    "T_FRAME_MS": 1.5
}


for radar_params in radar_cfg_list:

    params = static_params | radar_params
    studio_name = comm_params['activated_radar'][params['RADAR_IDX']]['client_name']
    studio_ip = comm_params['comm_cfg'][studio_name]['IP']
    studio_port = comm_params['comm_cfg'][studio_name]['server_access_port']

    lua_script = render_lua_template(RADAR_CONFIG_FILE_PATH, params)

    send_lua_to_matlab(host=studio_ip, port=studio_port, lua_text=lua_script)


# No ack from the MATLAB side, so just wait for a while to make sure 
# the script is sent before the program exit
import time
time.sleep(5)
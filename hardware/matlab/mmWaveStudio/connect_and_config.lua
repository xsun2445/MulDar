-- Replace the paths and COM port with the ones corresponding to your setup
BSS_PATH = "C:\\ti\\mmwave_studio_03_00_00_14\\rf_eval_firmware\\AWR2243_ES1_1\\radarss\\xwr22xx_radarss.bin"
MSS_PATH = "C:\\ti\\mmwave_studio_03_00_00_14\\rf_eval_firmware\\AWR2243_ES1_1\\masterss\\xwr22xx_masterss.bin"
COM_PORT = 13

-- Radar Configs
RADAR_IDX = 0

local file_path = debug.getinfo(1, "S").source:match("@(.+[\\/])")

connect_file_path = file_path.."\\".."connect_awr2243.lua"
dofile(connect_file_path)

config_file_path = file_path.."\\".."config_awr2243_3radars.lua"
dofile(config_file_path)
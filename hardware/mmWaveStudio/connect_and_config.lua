local file_path = debug.getinfo(1, "S").source:match("@(.+[\\/])")

connect_file_path = file_path.."\\".."connect_awr2243.lua"
dofile(connect_file_path)

config_file_path = file_path.."\\".."config_awr2243_3radars.lua"
dofile(config_file_path)
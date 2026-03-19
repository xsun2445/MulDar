"""
This script is used to visualize multiple monostaticradar data in real-time 
with an option to save the data.
"""
import yaml
from muldar.devices.muldar import MultiRadarManager
from muldar.vis.vis import start_visualization, visualize_waveform, visualize_2d_fft
from muldar.vis.network_vis import start_combined_visualization


def main():
    import argparse
    from argparse import ArgumentParser, BooleanOptionalAction
    parser = ArgumentParser()
    parser.add_argument('--config-path', type=str, default='configs.yml')
    parser.add_argument('--wait-for-threads', action=BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument('--flag-save', action=BooleanOptionalAction, default=argparse.SUPPRESS)
    parser.add_argument('--flag-visualize', action=BooleanOptionalAction, default=argparse.SUPPRESS) # use --no-flag-visualize
    parser.add_argument('--num-trigger', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--period-frame', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--radar-timeout', type=int, default=argparse.SUPPRESS)
    parser.add_argument('--saving-root-dir', type=str, default=argparse.SUPPRESS)
    parser.add_argument('--warmup', action=BooleanOptionalAction, default=argparse.SUPPRESS)
    args = parser.parse_args()

    # args.config_path = 'C:/Workspace/2025_polysight/mmWave_SAR_collection/adcData/eval_data/inthewild/table_video/20251205_034038/configs.yml'


    params = yaml.load(open(args.config_path), Loader=yaml.FullLoader)
    params.update(args.__dict__)
    if params['flag_save'] and not params['flag_visualize']:
        params['wait_for_threads'] = True
    
    if 'warmup' in params and params['warmup']:
        print('warming up...')
        params['period_frame'] = 11
        params['num_trigger'] = 1000
        params['flag_visualize'] = False
        params['flag_save'] = False
        params['wait_for_threads'] = True

    # import time
    # time.sleep(6)

    mgr = MultiRadarManager(params)
    mgr.start()

    if params['flag_visualize']:
        # start_visualization(mgr)
        # # Combined mono/multi/all XY visualization
        start_combined_visualization(mgr)
        # visualize_waveform(mgr)
        # visualize_2d_fft(mgr)

    mgr.stop()


if __name__ == "__main__":
    main()
    
    
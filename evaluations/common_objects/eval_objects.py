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
from scripts.single_frame_imaging import single_frame_imaging

import scipy.constants

from labels import common_objects_label as labels




def eval_all_data():

    saveFolder = './evaluations/common_objects/images'
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    for items in labels.values():
        print(items['name'])
        
        single_frame_imaging(items['fileName'], 
                            lambda_mono=items['lambda_mono_factor'], 
                            show_plot=False, 
                            dataName=items['name'],
                            saveFolder=saveFolder)
        plt.close('all')


if __name__ == '__main__':
    eval_all_data()
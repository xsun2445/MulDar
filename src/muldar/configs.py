ADC_PARAMS = {'chirps': 1,
              'configs': 1,
              'frames': 1,
              'rx': 4,
              'samples': 256,
              'IQ': 2,
              'bytes': 2}

BYTES_IN_FRAME = (ADC_PARAMS['chirps'] * ADC_PARAMS['rx'] * ADC_PARAMS['configs'] * ADC_PARAMS['frames'] *
                  ADC_PARAMS['IQ'] * ADC_PARAMS['samples'] * ADC_PARAMS['bytes'])
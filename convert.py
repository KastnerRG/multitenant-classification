import os
import argparse
import hls4ml
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt

from hls4ml.model.profiling import numerical
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects

from multitenant_fpga_net import yaml_load


def print_dict(d, indent=0):
    align=20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))


def get_hls_config(model, our_config):
    """
    Given keras model, construct config for hls4ml conversion
    """
    # Fix softmax issues? Nope
    # hls4ml.model.optimizer.OutputRoundingSaturationMode.layers = ['Activation']
    # hls4ml.model.optimizer.OutputRoundingSaturationMode.rounding_mode = 'AP_RND'
    # hls4ml.model.optimizer.OutputRoundingSaturationMode.saturation_mode = 'AP_SAT'

    config = hls4ml.utils.config_from_keras_model(model, granularity='name')
    config['Model'] = {}
    config['Model']['ReuseFactor'] = our_config['convert']['ReuseFactor']
    config['Model']['Strategy'] = our_config['convert']['Strategy']
    config['Model']['Precision'] = our_config['convert']['Precision']

    for name in config['LayerName'].keys():
        # config['LayerName'][name]['Trace'] = True
        config['LayerName'][name]['ReuseFactor'] = our_config['convert']['ReuseFactor']
        config['LayerName'][name]['Precision'] = our_config['convert']['Precision']
    # custom config for softmax
    config['LayerName']['activation']['Implementation'] = 'Stable' # NEED THIS
    config['LayerName']['activation']['Strategy'] = 'Stable' # NEEd THIS
    # config['LayerName']['activation']['Precision'] = 'ap_fixed<64,4>' # For ResNet20 only to get accurate CSIM
    # config['LayerName']['activation']['exp_table_t'] = 'ap_fixed<64,4>' # For ResNet20 only to get accurate CSIM
    # config['LayerName']['activation']['inv_table_t'] = 'ap_fixed<64,4>' # For ResNet20 only to get accurate CSIM

    fpga_part = our_config['convert']['XilinxPart']
    cfg = hls4ml.converters.create_backend_config(fpga_part=fpga_part) # conda-env
    # cfg = hls4ml.utils.config.create_vivado_config(fpga_part=fpga_part)
    cfg['HLSConfig'] = config
    cfg['IOType'] = our_config['convert']['IOType']
    cfg['Backend'] = our_config['convert']['Backend']
    cfg['Interface'] = 's_axilite' # or 'm_axi'
    cfg['ClockPeriod'] = our_config['convert']['ClockPeriod']
    cfg['KerasModel'] = model
    cfg['OutputDir'] = our_config['convert']['OutputDir']

    print("-----------------------------------")
    print_dict(cfg)
    print("-----------------------------------")

    return cfg

def main(args):
    # parameters
    our_config = yaml_load(args.config)
    
    model_file_path = our_config['save_dir'] + '/' + our_config['model_name'] + '-best.h5'

    # Load the CIFAR10 data
    
    custom_objects = {}
    _add_supported_quantized_objects(custom_objects)
    
    model = load_model(model_file_path, custom_objects=custom_objects)
   
    model.summary()

    hls_config = get_hls_config(model, our_config)

    # Bitfile time 
    hls_model = hls4ml.converters.keras_to_hls(hls_config)
    hls_model.compile()
    hls_model.build(csim=False,synth=True) #,export=True)
    hls4ml.report.read_vivado_report(our_config['convert']['OutputDir'])

#    print('x_test shape:', x_test.shape)
#    x = np.ascontiguousarray(x_test[:100])
#    y = hls_model.predict(x)
#    print('y')
#    print(np.argmax(y, axis=1))
#    print('y_test')
#    print(np.argmax(y_test[:100], axis=1))
#    # Profiling
#    wp, ap = numerical(model=model, hls_model=hls_model, X=x)
#    wp.savefig('weight_profiling.pdf')
#    ap.savefig('activation_profiling.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="config.yaml", help="specify yaml config")

    args = parser.parse_args()

    main(args)

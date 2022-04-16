import os
import argparse
import hls4ml
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt

from hls4ml.model.profiling import numerical
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects

from convert import get_hls_config
from multitenant_fpga_net import yaml_load, prepare_data, convert_to_one_hot, gen_con_mat


def main(args):
    # parameters
    config = yaml_load(args.config)

    CLASSES = 9
    ROWS = 4095
    CHANNELS = 1

    save_dir = config['save_dir']
    test_dir = config['test_dir'] # Test data directory
    model_file_path = config['save_dir'] + '/' + config['model_name'] + '-best.h5'
   
    
    custom_objects = {}
    _add_supported_quantized_objects(custom_objects)
    
    model = load_model(model_file_path, custom_objects=custom_objects)
   
    model.summary()

    test_images = [test_dir+i for i in os.listdir(test_dir)]
    x_test, test_set_y = prepare_data(test_images, ROWS, CHANNELS)
    y_test = convert_to_one_hot(test_set_y, CLASSES).T

    # Normalize test data
    x_test -= np.mean(x_test, axis=0)
    x_test /= np.std(x_test, axis=0)

    x_test = np.ascontiguousarray(x_test)
    # x_test = x_test[:10]
    # y_test = y_test[:10]
    y_keras = model.predict(x_test.astype(dtype=np.float32))

    hls_config = get_hls_config(model, config)

    hls_model = hls4ml.converters.keras_to_hls(hls_config)
    hls_model.compile()
    # hls_model.build(csim=False,synth=True) #,export=True)
    # hls4ml.report.read_vivado_report(our_config['convert']['OutputDir'])

    print('x_test shape:', x_test.shape)
    
    x = np.ascontiguousarray(x_test)
    # hls_pred, hls4ml_trace = hls_model.trace(x)
    # keras_trace = hls4ml.model.profiling.get_ymodel_keras(model, x)
    y_hls = hls_model.predict(x.astype(dtype=np.float32))

    print('y_test: ', np.argmax(y_test[:20], axis=1))
    print('y_keras:', np.argmax(y_keras[:20], axis=1))
    print('y_hls:  ', np.argmax(y_hls[:20], axis=1))

    from sklearn.metrics import accuracy_score
    print("Keras Accuracy:  {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_keras, axis=1))))
    print("hls4ml Accuracy: {}".format(accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_hls, axis=1))))

    # for key in hls4ml_trace.keys():
    #     # TODO output to file
    #     sample = 0
    #     keras_sample = keras_trace[key][sample] # Just examine 1 sample
    #     hls_sample = hls4ml_trace[key][sample]
    #     diff = abs(keras_sample - hls_sample) > 1e-2
    #     # print(diff)
    #     if np.any(diff):
    #         # print(f'Keras layer {key}, sample 0')
    #         print(f'hls4ml layer {key}, sample 0')
    #         print(f'{(np.sum(diff) / np.size(diff)) * 100}% different\n')
        # print(keras_sample)
        # print(hls_sample)

    # Profiling
    wp, ap = numerical(model=model, hls_model=hls_model, X=x)
    wp.savefig(os.path.join(save_dir, 'weight_profiling.pdf'))
    ap.savefig(os.path.join(save_dir, 'activation_profiling.pdf'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="config.yaml", help="specify yaml config")

    args = parser.parse_args()

    main(args)

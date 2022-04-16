import os
import argparse
import numpy as np

from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects

from multitenant_fpga_net import yaml_load, prepare_data, convert_to_one_hot, gen_con_mat



def main(args):
    config = yaml_load(args.config)

    CLASSES = 9
    ROWS = 4095
    CHANNELS = 1

    test_dir = config['test_dir'] # Test data directory
    batch_size = config['batch_size']
    model_path = config['save_dir'] + '/' + config['model_name'] + '-best.h5'
    
    custom_objects = {}
    _add_supported_quantized_objects(custom_objects)
    
    model = load_model(model_path, custom_objects=custom_objects)

    model.summary()

    test_images = [test_dir+i for i in os.listdir(test_dir)]
    x_test, test_set_y = prepare_data(test_images, ROWS, CHANNELS)
    y_test = convert_to_one_hot(test_set_y, CLASSES).T

    # Normalize test data
    x_test -= np.mean(x_test, axis=0)
    x_test /= np.std(x_test, axis=0)

    # Test model
    test_loss, test_acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size, verbose=1)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default = "config.yaml", help="specify yaml config")

    args = parser.parse_args()

    main(args)
import os
import yaml
import time
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Flatten, Activation
from tensorflow.keras.models import Model, load_model

from qkeras.qlayers import QDense, QActivation
from qkeras.qnormalization import QBatchNormalization
from qkeras.quantizers import quantized_bits, quantized_relu

from sklearn.decomposition import PCA

def yaml_load(config):
    with open(config) as stream:
        param = yaml.safe_load(stream)
    return param

def read_image(file_path):
    fft = np.load(file_path)
    fft = fft.flatten()
    return fft

def prepare_data(images, ROWS, num_classes):
    m = len(images)

    X = np.zeros((m, ROWS)) 

    y = np.zeros((1, m), dtype=np.uint8)
    for i, image_file in enumerate(images):
        fft = read_image(image_file)
        X[i,:] = fft[:ROWS] 
        # -----------^ Truncate fft in case there is an extra freq bin
        if 'base' in image_file.lower():
            y[0, i] = 0
        elif 'ro' in image_file.lower():
            y[0, i] = 2
        elif 'orca-aes' in image_file.lower():
            y[0, i] = 3
        elif 'orca-present' in image_file.lower():
            y[0, i] = 4
        elif 'mb-aes' in image_file.lower():
            y[0, i] = 5
        elif 'mb-present' in image_file.lower():
            y[0, i] = 6
        elif 'pico-aes' in image_file.lower():
            y[0, i] = 7
        elif 'pico-present' in image_file.lower():
            y[0, i] = 8
        elif 'cortex-aes' in image_file.lower():
            y[0, i] = 9
        elif 'cortex-present' in image_file.lower():
            y[0, i] = 10
        elif 'present-hls' in image_file.lower():
            y[0, i] = 11
        elif 'dsp' in image_file.lower():
            y[0, i] = 12
        elif 'aes' in image_file.lower():
            y[0, i] = 1 
            
    return X, y

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y  

def save_history(history, model_name, save_dir, dataset, training_setup):
    save_path = f"{save_dir}plots/{dataset}/{model_name}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print()
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'test_accuracy'], loc='best')
    plt.savefig(f"{save_path}/{training_setup}-acc.pdf")

    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f"{save_path}/{training_setup}-loss.pdf")


def gen_con_mat(predictions, test, num_classes, model_name, save_dir, dataset, training_setup):
    save_path = f"{save_dir}plots/{dataset}/{model_name}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    num_traces = len(test)/num_classes
    con_mat = tf.math.confusion_matrix(tf.argmax(test, 1), predictions=tf.argmax(predictions, 1), num_classes=num_classes, dtype=tf.int32, name=None)

    fig = plt.figure(figsize=(26,25))
    ax = fig.add_subplot(111) 
    ax.matshow(con_mat)
    for (i, j), z in np.ndenumerate(con_mat):
        ax.text(j, i, '{:.2f}'.format(z/num_traces), ha='center', va='center', fontsize=32, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        plt.xlabel("Predictions", size=50)
        plt.ylabel("True Label", size=50)
        if (num_classes == 7):
            ticks = ['base', 'aes', 'ro', 'orca-aes', 'orca-pres', 'mb-aes', 'mb-pres']
        elif (num_classes == 9):
            ticks = ['base', 'aes', 'ro', 'orca-aes', 'orca-pres', 'mb-aes', 'mb-pres', 'pico-aes', 'pico-pres']
        else: # num_classes == 13
            ticks = ['sensor', 'hls-aes', 'ro', 'orca-aes', 'orca-pres', 'mb-aes', 'mb-pres', 'pico-aes', 'pico-pres', 'ctx-aes', 'ctx-pres', 'hls-pres', 'arith']
        ax.set_xticks(np.arange(len(ticks)))
        ax.set_yticks(np.arange(len(ticks)))
        ax.set_xticklabels(ticks, size=50) 
        ax.set_yticklabels(ticks, size=50) 
        ax.tick_params(axis='x', labelrotation=45)
        ax.tick_params(axis='y', labelrotation=45)

    plt.savefig(f"{save_path}/{training_setup}-confusion-matrix.pdf")


def get_NN(rows, num_classes):
    input = Input(shape=(rows,))

    x = input

    # x = BatchNormalization()(x)
    # x = Dense(128, activation="relu")(x)
    # x = BatchNormalization()(x)
    # x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(num_classes, activation="softmax")(x)
    x = Model(inputs=input, outputs=x)
    return x


def get_QNN(rows, num_classes, logit_total_bits, 
    logit_int_bits, activation_total_bits, activation_int_bits):
    input = Input(shape=(rows,))

    x = input

    # x = QBatchNormalization( # beta, gamma, mean, variance quantizers needed
    #     beta_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
    #     gamma_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
    #     mean_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
    #     variance_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1)
    # )(x)
    # x = QDense(
    #     128, 
    #     kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
    #     bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
    #     kernel_initializer='he_normal'
    # )(x)
    # x = QActivation(activation=quantized_relu(activation_total_bits, activation_int_bits))(x)
    x = QBatchNormalization( # beta, gamma, mean, variance quantizers needed
        beta_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
        gamma_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
        mean_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
        variance_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1)
    )(x)
    # Add more hidden layrs here
    x = QDense(
        num_classes,
        kernel_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
        bias_quantizer=quantized_bits(logit_total_bits, logit_int_bits, alpha=1),
        kernel_initializer='he_normal'
    )(x)
    outputs = QActivation("quantized_bits(8, 3)")(x)
    outputs = Activation('softmax')(x)
    model = Model(inputs=input, outputs=outputs)
    return model


def collect_images(dirs):
    images = []
    for dir in dirs:
        cur_imgs = [dir + i for i in os.listdir(dir)]
        images += cur_imgs
    return images


def train(rows, classes, training_dirs, testing_dirs, config, log_file, test_board):
    dataset = config['dataset']

    save_dir = config['save_dir']
    model_name = config['model_name']

    epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']

    print("Collecting Data...")
    train_images = collect_images(training_dirs)
    X_train, train_set_y = prepare_data(train_images, rows, classes)
    Y_train = convert_to_one_hot(train_set_y, classes).T

    # Normalize training data
    X_train -= np.mean(X_train, axis=0)
    X_train /= np.std(X_train, axis=0)

    test_images = collect_images(testing_dirs)
    X_test, test_set_y = prepare_data(test_images, rows, classes)
    Y_test = convert_to_one_hot(test_set_y, classes).T

    # Normalize test data
    X_test -= np.mean(X_test, axis=0)
    X_test /= np.std(X_test, axis=0)
    
    # PCA
    if config['pca']:
        start_time = time.time()
        variance_retained = config['pca_variance']
        print("Performing PCA...")
        pca = PCA(variance_retained)
        pca.fit(X_train)

        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        ROWS = pca.n_components_
        print("Num principal components:", ROWS)
        print('PCA took {}s'.format(time.time() - start_time))


    # train
    if config['quantization']:
        logit_total_bits = config['logit_total_bits']
        logit_int_bits = config['logit_int_bits']
        activation_total_bits = config['activation_total_bits']
        activation_int_bits = config['activation_int_bits']
        model = get_QNN(
            rows, 
            classes,
            logit_total_bits, 
            logit_int_bits, 
            activation_total_bits, 
            activation_int_bits
        )
    else:
        model = get_NN(rows, classes)

    model.summary()
    
    opt = getattr(tf.keras.optimizers, 'Adam')
    model_file_path = save_dir + f'models/{dataset}/' + model_name + '-best.h5'
    checkpoint = ModelCheckpoint(
        filepath=model_file_path,
        monitor='val_acc', 
        verbose=1, 
        save_best_only=True
    )
    callbacks = [checkpoint]
    
    model.compile(optimizer=opt(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['acc'])
    
    history = model.fit(
        X_train, 
        Y_train, 
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True, 
        verbose=1, 
        validation_data=(X_test, Y_test),
        callbacks=callbacks
    )

    print("Evaluating best model...")
    model.load_weights(model_file_path)
    results = model.evaluate(X_test, Y_test) 
    for i in range(len(results)):
        res_str = 'val ' + model.metrics_names[i] + ': ' + str(results[i]) + '\n'
        print(res_str)
        log_file.write(res_str)
    predictions = model.predict(X_test)

    training_setup = 'train' + str(config['num_training']) + '-test-' + test_board
    gen_con_mat(predictions, Y_test, classes, model_name, save_dir, dataset, training_setup)
    save_history(history, model_name, save_dir, dataset, training_setup)


def eval(config, rows, classes):
    dataset = config['dataset']
    save_dir = config['save_dir']
    model_name = config['model_name']
    test_board = config['eval']['test_board']
    test_dir = config[test_board]
    model_file_path = save_dir + f'models/{dataset}/' + model_name + '-best.h5' 
    
    test_images = collect_images(test_dir)
    X_test, test_set_y = prepare_data(test_images, rows, classes)
    Y_test = convert_to_one_hot(test_set_y, classes).T

    # Normalize test data
    X_test -= np.mean(X_test, axis=0)
    X_test /= np.std(X_test, axis=0)

    print(f'Evaluating model {model_file_path}...')
    model = load_model(model_file_path)

    results = model.evaluate(X_test, Y_test) 
    for i in range(len(results)):
        res_str = 'val ' + model.metrics_names[i] + ': ' + str(results[i]) + '\n'
        print(res_str)
    predictions = model.predict(X_test)
    training_setup = 'train' + str(config['num_training']) + '-test-' + test_board + '-eval'
    gen_con_mat(predictions, Y_test, classes, model_name, save_dir, dataset, training_setup)
    

def main(args):
    config = yaml_load(args.config)

    CHANNELS = 1
    ROWS = config['fft_freq_bins'] * CHANNELS
    CLASSES = 13

    log_file = open(config['log_file'], 'a')
    num_training = config['num_training']
    boards = config['boards']
    board_set = set(boards)

    if args.eval_only:
        eval(config, ROWS, CLASSES)
        return

    # All Combos training - cross validation testing on all combinations of 
    # training and testing boards
    if config['all_combo_training']:
        for num_train in range(1, num_training + 1):
            training_combos = list(itertools.combinations(boards, num_train))

            for combo in training_combos:
                training_dirs = []
                testing_dirs = None
                # Get training dirs for specified board traces
                print(f'Training on: {combo}')
                log_file.write(f'\nTraining on: {combo}\n')
                for board in combo:
                    training_dirs += config[board]
                # Get testing dirs for specified board traces
                # combo_set = set(combo)
                # test_boards = board_set.difference(combo_set)
                test_boards = combo 
                # ^-- Uncomment if want to test on same board as combo boards
                for test_board in test_boards:
                    print(f'Testing on: {test_board}')
                    log_file.write(f'\nTesting on: {test_board}\n')
                    testing_dirs = config[test_board]
                    train(ROWS, CLASSES, training_dirs, testing_dirs, config, log_file, test_board)
    else: # Training and testing boards set by user
        training_dirs = []
        test_board = config['fixed_training']['test_board']
        testing_dirs = config[test_board]

        print(f"Training on: {config['fixed_training']['training_boards']}")
        log_file.write(f"Training on: {config['fixed_training']['training_boards']}")
        print(f"Testing on: {config['fixed_training']['test_board']}")
        log_file.write(f"Testing on: {config['fixed_training']['test_board']}")

        for board in config['fixed_training']['training_boards']:
            training_dirs += config[board]
        train(ROWS, CLASSES, training_dirs, testing_dirs, config, log_file, test_board)
        
    log_file.close()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help="specify yaml config")
    parser.add_argument('-e', '--eval-only', action='store_true', help="evaluate model only")

    args = parser.parse_args()

    main(args)

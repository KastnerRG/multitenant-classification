#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 4095
#define N_LAYER_3 9

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,5> input_1_default_t;
typedef nnet::array<ap_fixed<16,5>, 4095*1> input_t;
typedef ap_fixed<16,5> q_batch_normalization_default_t;
typedef nnet::array<ap_fixed<16,5>, 4095*1> layer2_t;
typedef ap_fixed<16,5> q_dense_default_t;
typedef nnet::array<ap_fixed<16,5>, 9*1> layer3_t;
typedef ap_fixed<9,4> weight3_t;
typedef ap_fixed<9,4> bias3_t;
typedef ap_fixed<16,5> activation_default_t;
typedef nnet::array<ap_fixed<16,5>, 9*1> result_t;

#endif

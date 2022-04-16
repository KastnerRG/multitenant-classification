#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/s2.h"
#include "weights/b2.h"
#include "weights/w3.h"
#include "weights/b3.h"
#include "weights/s6.h"
#include "weights/b6.h"
#include "weights/w7.h"
#include "weights/b7.h"

//hls-fpga-machine-learning insert layer-config
// q_batch_normalization
struct config2 : nnet::batchnorm_config {
    static const unsigned n_in = N_INPUT_1_1;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 200;
    static const bool store_weights_in_bram = false;
    typedef q_batch_normalization_default_t bias_t;
    typedef q_batch_normalization_default_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// q_dense
struct config3 : nnet::dense_config {
    static const unsigned n_in = N_INPUT_1_1;
    static const unsigned n_out = N_LAYER_3;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 195;
    static const unsigned n_zeros = 368495;
    static const unsigned n_nonzeros = 155665;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,5> accum_t;
    typedef bias3_t bias_t;
    typedef weight3_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// q_activation
struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_3;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 200;
    typedef ap_fixed<18,8> table_t;
};

// q_batch_normalization_1
struct config6 : nnet::batchnorm_config {
    static const unsigned n_in = N_LAYER_3;
    static const unsigned n_filt = -1;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 200;
    static const bool store_weights_in_bram = false;
    typedef q_batch_normalization_1_default_t bias_t;
    typedef q_batch_normalization_1_default_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// q_dense_1
struct config7 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_3;
    static const unsigned n_out = N_LAYER_7;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 128;
    static const unsigned n_zeros = 190;
    static const unsigned n_nonzeros = 962;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,5> accum_t;
    typedef bias7_t bias_t;
    typedef weight7_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// activation
struct softmax_config9 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_7;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 200;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::stable;
    typedef ap_fixed<18,8,AP_RND,AP_SAT> exp_table_t;
    typedef ap_fixed<18,8,AP_RND,AP_SAT> inv_table_t;
};


#endif

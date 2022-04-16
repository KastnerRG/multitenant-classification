//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &input_1,
    hls::stream<result_t> &layer9_out,
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_1,layer9_out 
    #pragma HLS DATAFLOW 

    const_size_in_1 = N_INPUT_1_1;
    const_size_out_1 = N_LAYER_7;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<q_batch_normalization_default_t, 4095>(s2, "s2.txt");
        nnet::load_weights_from_txt<q_batch_normalization_default_t, 4095>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight3_t, 524160>(w3, "w3.txt");
        nnet::load_weights_from_txt<bias3_t, 128>(b3, "b3.txt");
        nnet::load_weights_from_txt<q_batch_normalization_1_default_t, 128>(s6, "s6.txt");
        nnet::load_weights_from_txt<q_batch_normalization_1_default_t, 128>(b6, "b6.txt");
        nnet::load_weights_from_txt<weight7_t, 1152>(w7, "w7.txt");
        nnet::load_weights_from_txt<bias7_t, 9>(b7, "b7.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=1
    #pragma HLS STABLE variable=layer2_out
    nnet::normalize<input_t, layer2_t, config2>(input_1, layer2_out, s2, b2); // q_batch_normalization

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=1
    #pragma HLS STABLE variable=layer3_out
    nnet::dense<layer2_t, layer3_t, config3>(layer2_out, layer3_out, w3, b3); // q_dense

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=1
    #pragma HLS STABLE variable=layer5_out
    nnet::relu<layer3_t, layer5_t, relu_config5>(layer3_out, layer5_out); // q_activation

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=1
    #pragma HLS STABLE variable=layer6_out
    nnet::normalize<layer5_t, layer6_t, config6>(layer5_out, layer6_out, s6, b6); // q_batch_normalization_1

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=1
    #pragma HLS STABLE variable=layer7_out
    nnet::dense<layer6_t, layer7_t, config7>(layer6_out, layer7_out, w7, b7); // q_dense_1

    nnet::softmax<layer7_t, result_t, softmax_config9>(layer7_out, layer9_out); // activation

}

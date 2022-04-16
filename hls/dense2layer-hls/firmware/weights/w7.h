//Numpy array shape [128, 9]
//Min -0.375000000000
//Max 0.312500000000
//Number of zeros 190

#ifndef W7_H_
#define W7_H_

#ifndef __SYNTHESIS__
weight7_t w7[1152];
#else
weight7_t w7[1152] = {0.00000, -0.12500, 0.00000, 0.00000, -0.06250, -0.06250, -0.18750, -0.06250, 0.00000, -0.25000, -0.25000, -0.25000, 0.00000, 0.00000, 0.00000, 0.00000, -0.12500, 0.25000, 0.00000, -0.12500, 0.06250, 0.25000, 0.06250, 0.12500, -0.06250, -0.18750, -0.18750, 0.31250, 0.06250, 0.06250, 0.12500, 0.18750, 0.12500, -0.18750, 0.18750, -0.12500, 0.06250, -0.06250, 0.00000, 0.06250, 0.25000, -0.12500, 0.18750, 0.12500, 0.25000, -0.12500, 0.06250, 0.31250, 0.12500, -0.12500, 0.00000, 0.18750, 0.06250, -0.12500, -0.12500, 0.12500, 0.06250, -0.06250, 0.00000, 0.06250, -0.25000, 0.18750, -0.12500, 0.00000, 0.25000, -0.18750, 0.06250, 0.18750, -0.12500, -0.06250, 0.18750, 0.00000, 0.06250, -0.12500, -0.06250, 0.12500, -0.18750, -0.06250, 0.06250, 0.12500, 0.18750, 0.00000, -0.12500, -0.12500, 0.12500, 0.12500, 0.00000, 0.00000, -0.25000, -0.12500, -0.25000, -0.06250, 0.25000, -0.06250, 0.00000, 0.06250, 0.06250, -0.18750, 0.12500, -0.06250, -0.06250, 0.25000, -0.06250, 0.06250, 0.00000, 0.00000, -0.12500, 0.00000, -0.12500, 0.00000, -0.25000, 0.18750, 0.25000, 0.12500, 0.00000, -0.25000, 0.18750, -0.12500, -0.06250, -0.06250, -0.18750, 0.00000, 0.06250, 0.06250, 0.18750, -0.06250, 0.12500, -0.12500, -0.12500, 0.18750, -0.12500, -0.06250, 0.00000, 0.25000, -0.06250, 0.12500, 0.18750, 0.25000, -0.25000, 0.12500, -0.18750, -0.12500, -0.25000, 0.18750, 0.18750, 0.00000, 0.00000, -0.12500, -0.18750, -0.12500, 0.06250, -0.12500, 0.00000, 0.12500, 0.18750, 0.12500, 0.06250, -0.18750, 0.06250, 0.18750, 0.18750, 0.00000, 0.00000, 0.06250, -0.06250, -0.18750, 0.06250, 0.06250, 0.00000, 0.31250, 0.06250, 0.00000, -0.12500, -0.12500, 0.25000, 0.18750, -0.06250, 0.00000, 0.06250, -0.31250, 0.25000, 0.12500, 0.25000, -0.06250, 0.00000, -0.12500, -0.18750, -0.18750, -0.25000, 0.12500, 0.00000, -0.25000, 0.00000, -0.12500, -0.06250, 0.06250, -0.06250, 0.18750, 0.06250, -0.12500, -0.25000, 0.00000, -0.06250, 0.06250, 0.00000, 0.12500, -0.12500, 0.12500, 0.12500, 0.00000, 0.18750, -0.18750, -0.12500, -0.18750, 0.06250, 0.00000, 0.00000, -0.18750, 0.06250, 0.06250, 0.00000, -0.06250, -0.18750, 0.12500, 0.06250, 0.18750, 0.18750, 0.06250, -0.12500, -0.12500, 0.00000, -0.18750, 0.00000, 0.12500, 0.06250, -0.06250, -0.18750, 0.06250, 0.12500, 0.06250, 0.18750, 0.18750, 0.00000, -0.12500, -0.12500, 0.06250, -0.06250, -0.12500, -0.18750, 0.06250, 0.06250, -0.06250, 0.06250, 0.00000, -0.18750, -0.06250, 0.06250, -0.06250, 0.25000, 0.00000, -0.31250, -0.25000, -0.06250, -0.18750, 0.12500, -0.06250, 0.00000, 0.06250, -0.25000, -0.12500, 0.00000, 0.12500, 0.06250, 0.12500, 0.00000, 0.06250, 0.25000, -0.12500, 0.06250, 0.12500, 0.12500, -0.06250, 0.06250, -0.18750, 0.06250, 0.12500, -0.25000, -0.06250, -0.25000, 0.12500, 0.06250, 0.25000, -0.18750, -0.06250, -0.25000, -0.25000, 0.06250, -0.06250, 0.12500, -0.06250, 0.18750, 0.00000, 0.00000, -0.06250, -0.12500, -0.06250, -0.18750, -0.06250, -0.25000, 0.06250, -0.12500, -0.06250, -0.18750, 0.12500, 0.12500, -0.12500, -0.12500, -0.18750, -0.12500, 0.12500, -0.06250, 0.00000, 0.00000, 0.18750, 0.25000, -0.06250, 0.25000, -0.18750, 0.06250, 0.12500, 0.00000, 0.06250, -0.18750, 0.18750, -0.18750, 0.12500, -0.25000, 0.06250, -0.06250, 0.06250, 0.12500, -0.18750, -0.18750, -0.12500, 0.06250, -0.12500, -0.12500, 0.12500, -0.06250, -0.18750, 0.12500, 0.06250, 0.00000, 0.18750, 0.06250, 0.12500, 0.25000, 0.00000, -0.06250, 0.25000, 0.00000, 0.00000, -0.06250, -0.18750, -0.06250, -0.18750, -0.06250, -0.18750, 0.25000, -0.25000, 0.00000, 0.00000, 0.00000, -0.25000, 0.12500, 0.25000, 0.06250, 0.00000, -0.31250, 0.18750, 0.12500, -0.06250, 0.06250, -0.06250, -0.18750, 0.06250, -0.06250, 0.12500, 0.12500, -0.25000, 0.00000, 0.00000, -0.12500, 0.31250, -0.06250, -0.06250, -0.12500, 0.00000, -0.18750, 0.18750, 0.31250, 0.06250, 0.18750, 0.25000, 0.06250, 0.06250, -0.12500, 0.12500, 0.06250, -0.06250, 0.00000, 0.06250, -0.12500, 0.00000, 0.12500, -0.06250, 0.18750, 0.12500, 0.25000, -0.12500, 0.00000, 0.06250, -0.06250, 0.06250, 0.00000, -0.18750, 0.06250, -0.06250, -0.12500, -0.25000, 0.00000, 0.25000, 0.06250, 0.18750, 0.06250, -0.06250, -0.06250, 0.00000, -0.12500, -0.12500, -0.18750, 0.06250, 0.06250, 0.12500, 0.25000, 0.06250, 0.18750, 0.00000, 0.12500, 0.31250, 0.06250, -0.18750, -0.06250, 0.31250, 0.25000, -0.12500, -0.12500, -0.12500, 0.31250, -0.06250, 0.00000, 0.00000, 0.12500, -0.12500, -0.12500, -0.31250, -0.18750, -0.06250, 0.12500, 0.06250, -0.18750, 0.12500, -0.18750, 0.18750, 0.06250, -0.12500, 0.12500, -0.25000, 0.00000, -0.12500, -0.12500, 0.06250, 0.00000, -0.06250, 0.06250, 0.12500, -0.06250, 0.00000, -0.18750, -0.06250, -0.06250, -0.18750, 0.00000, 0.18750, -0.06250, 0.06250, 0.18750, 0.06250, 0.06250, -0.06250, 0.12500, -0.18750, 0.18750, -0.18750, 0.06250, 0.06250, 0.00000, -0.18750, 0.00000, 0.31250, -0.06250, 0.06250, 0.12500, -0.18750, -0.06250, 0.12500, -0.06250, 0.18750, -0.18750, 0.18750, 0.06250, 0.00000, -0.18750, 0.06250, -0.06250, 0.31250, 0.18750, 0.06250, -0.12500, 0.00000, -0.25000, -0.06250, -0.06250, 0.18750, 0.06250, 0.12500, -0.12500, 0.12500, 0.12500, 0.00000, -0.06250, 0.12500, -0.06250, -0.06250, 0.00000, -0.31250, -0.18750, -0.25000, 0.18750, -0.18750, 0.06250, -0.12500, 0.12500, 0.18750, 0.18750, -0.18750, 0.25000, -0.18750, 0.12500, 0.06250, 0.12500, 0.18750, -0.18750, 0.12500, 0.31250, 0.18750, 0.06250, -0.18750, -0.12500, 0.12500, -0.06250, -0.12500, -0.18750, -0.25000, 0.18750, 0.25000, 0.12500, 0.18750, 0.06250, -0.25000, -0.06250, 0.00000, -0.06250, -0.18750, -0.12500, 0.00000, -0.12500, -0.06250, 0.31250, 0.12500, 0.18750, 0.25000, 0.06250, -0.25000, -0.12500, 0.25000, -0.25000, 0.18750, 0.06250, -0.12500, -0.06250, 0.06250, 0.00000, -0.12500, 0.06250, 0.00000, 0.12500, -0.12500, 0.00000, -0.06250, -0.18750, -0.06250, -0.12500, -0.25000, 0.00000, -0.18750, -0.06250, 0.12500, 0.00000, 0.06250, 0.00000, 0.00000, -0.18750, 0.25000, -0.18750, 0.18750, 0.00000, 0.06250, 0.12500, -0.06250, -0.12500, -0.06250, -0.06250, 0.25000, 0.06250, 0.06250, 0.06250, 0.06250, -0.06250, 0.00000, 0.06250, -0.18750, 0.18750, 0.18750, 0.00000, 0.00000, -0.06250, 0.12500, 0.00000, 0.06250, 0.00000, -0.06250, 0.00000, 0.31250, -0.18750, 0.12500, 0.06250, 0.18750, -0.06250, -0.25000, 0.00000, 0.18750, -0.25000, -0.12500, -0.18750, 0.18750, -0.31250, 0.06250, 0.25000, -0.06250, 0.06250, 0.12500, 0.00000, 0.06250, 0.06250, 0.18750, 0.25000, -0.25000, -0.06250, 0.12500, -0.12500, -0.12500, -0.18750, 0.00000, 0.06250, -0.12500, 0.06250, 0.00000, -0.06250, 0.12500, -0.12500, -0.18750, -0.12500, 0.00000, -0.06250, -0.25000, 0.25000, 0.00000, -0.18750, 0.12500, 0.06250, 0.18750, 0.12500, -0.25000, -0.18750, 0.12500, 0.00000, -0.37500, 0.18750, 0.12500, 0.12500, 0.00000, 0.06250, -0.06250, 0.18750, -0.12500, 0.18750, -0.25000, 0.00000, -0.18750, -0.12500, 0.18750, -0.18750, -0.12500, -0.25000, 0.12500, 0.12500, -0.12500, 0.06250, -0.12500, -0.06250, -0.06250, 0.06250, -0.25000, -0.06250, -0.25000, -0.18750, -0.18750, 0.12500, -0.18750, -0.18750, -0.31250, 0.18750, -0.25000, 0.06250, 0.00000, 0.06250, 0.18750, -0.12500, 0.06250, -0.06250, -0.12500, 0.00000, 0.25000, -0.25000, 0.00000, 0.00000, -0.31250, 0.06250, 0.06250, 0.00000, 0.31250, -0.12500, 0.06250, 0.00000, 0.06250, 0.25000, -0.18750, 0.00000, -0.12500, -0.18750, 0.18750, 0.25000, 0.18750, 0.25000, 0.00000, 0.00000, 0.06250, 0.00000, 0.06250, 0.25000, 0.06250, -0.18750, 0.25000, -0.18750, 0.18750, -0.12500, 0.00000, 0.00000, 0.00000, -0.12500, 0.00000, -0.12500, 0.12500, 0.00000, -0.12500, 0.18750, -0.12500, 0.18750, -0.31250, 0.06250, -0.12500, 0.25000, 0.25000, -0.18750, 0.00000, 0.25000, 0.18750, 0.00000, -0.06250, -0.12500, -0.25000, 0.18750, 0.00000, -0.18750, -0.12500, 0.25000, 0.06250, -0.12500, -0.25000, -0.12500, 0.00000, 0.12500, 0.12500, -0.06250, 0.25000, -0.06250, -0.18750, -0.25000, 0.06250, -0.06250, 0.00000, 0.00000, 0.18750, -0.18750, 0.00000, 0.00000, 0.12500, -0.18750, 0.06250, -0.06250, -0.18750, 0.12500, 0.00000, -0.12500, -0.18750, -0.12500, 0.06250, 0.25000, 0.18750, -0.12500, 0.18750, -0.25000, -0.06250, -0.18750, 0.12500, 0.18750, -0.06250, 0.00000, 0.06250, 0.18750, 0.18750, 0.06250, -0.18750, 0.00000, 0.06250, -0.18750, 0.12500, 0.06250, 0.00000, -0.18750, 0.00000, -0.18750, -0.18750, -0.06250, -0.25000, 0.06250, 0.18750, -0.06250, -0.18750, 0.06250, 0.12500, -0.31250, 0.00000, 0.00000, 0.18750, -0.12500, 0.06250, -0.06250, -0.06250, -0.06250, 0.06250, -0.18750, 0.00000, 0.06250, 0.06250, -0.06250, 0.00000, 0.06250, -0.06250, 0.06250, 0.00000, -0.06250, -0.06250, 0.12500, 0.06250, 0.25000, 0.18750, 0.18750, 0.00000, 0.06250, -0.25000, -0.06250, 0.00000, -0.06250, -0.25000, -0.18750, 0.12500, 0.12500, 0.12500, 0.06250, 0.00000, 0.25000, 0.12500, -0.06250, -0.25000, 0.00000, 0.00000, 0.06250, -0.06250, 0.00000, -0.12500, -0.12500, 0.00000, -0.18750, -0.18750, 0.12500, -0.06250, 0.25000, -0.06250, -0.18750, 0.12500, 0.06250, 0.12500, 0.00000, 0.06250, 0.00000, 0.12500, -0.25000, -0.18750, 0.18750, 0.12500, 0.06250, 0.00000, -0.12500, -0.25000, 0.12500, 0.18750, 0.18750, 0.12500, 0.06250, 0.12500, -0.12500, -0.06250, 0.12500, 0.00000, -0.12500, 0.00000, 0.12500, -0.12500, 0.00000, 0.25000, -0.06250, 0.18750, -0.06250, 0.00000, -0.06250, 0.25000, -0.06250, 0.06250, -0.25000, 0.12500, 0.00000, -0.06250, -0.06250, 0.06250, 0.18750, -0.06250, -0.18750, 0.25000, -0.25000, 0.00000, 0.00000, -0.25000, -0.25000, 0.06250, -0.25000, 0.18750, -0.06250, 0.00000, 0.25000, -0.18750, -0.25000, -0.06250, -0.25000, 0.12500, -0.18750, 0.00000, 0.31250, 0.25000, 0.00000, -0.12500, -0.12500, 0.00000, 0.00000, -0.12500, 0.12500, -0.06250, 0.12500, -0.06250, -0.12500, 0.06250, 0.18750, 0.00000, -0.25000, 0.12500, -0.12500, -0.25000, 0.00000, 0.18750, 0.31250, -0.18750, 0.06250, -0.12500, -0.18750, -0.06250, 0.25000, -0.12500, 0.06250, -0.18750, -0.25000, -0.06250, 0.00000, -0.06250, 0.00000, -0.06250, 0.18750, -0.06250, 0.12500, 0.18750, 0.18750, 0.00000, 0.06250, -0.06250, 0.06250, -0.06250, -0.12500, 0.00000, 0.06250, 0.06250, -0.06250, 0.00000, 0.06250, -0.25000, -0.06250, -0.12500, 0.00000, 0.12500, 0.00000, 0.06250, 0.06250, -0.12500, -0.12500, 0.12500, 0.06250, 0.06250, 0.00000, 0.06250, -0.18750, 0.06250, 0.18750, 0.00000, 0.18750, 0.00000, 0.12500, -0.12500, 0.06250, 0.12500, 0.00000, 0.25000, 0.00000, 0.00000, 0.25000, -0.18750, 0.06250, 0.00000, -0.12500, 0.00000, -0.06250, 0.31250, -0.12500, 0.06250, -0.18750, -0.12500, 0.12500, -0.18750, 0.00000, 0.12500, 0.12500, -0.18750, -0.06250, 0.00000, 0.18750, 0.12500, -0.25000, 0.12500, -0.06250, 0.12500, 0.12500, 0.25000, -0.18750, 0.06250, 0.12500, -0.25000, 0.18750, 0.18750, 0.25000, -0.18750, 0.25000, 0.31250, -0.06250, 0.06250, 0.00000, 0.25000, -0.06250, 0.06250, -0.25000, 0.00000, 0.12500, 0.18750, 0.12500, 0.00000, 0.18750, 0.12500, 0.00000, 0.00000, 0.00000, 0.06250, -0.06250, -0.12500, 0.18750, -0.25000};
#endif

#endif
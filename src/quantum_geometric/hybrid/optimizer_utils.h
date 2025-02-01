#ifndef OPTIMIZER_UTILS_H
#define OPTIMIZER_UTILS_H

#include "quantum_geometric/hybrid/classical_optimization_engine.h"

// Convert integer to optimizer type
static inline optimizer_type_t convert_optimizer_type(int type) {
    switch (type) {
        case 0:
            return OPTIMIZER_ADAM;
        case 1:
            return OPTIMIZER_SGD;
        case 2:
            return OPTIMIZER_RMSPROP;
        case 3:
            return OPTIMIZER_ADAGRAD;
        case 4:
            return OPTIMIZER_ADADELTA;
        case 5:
            return OPTIMIZER_NADAM;
        case 6:
            return OPTIMIZER_LBFGS;
        case 7:
            return OPTIMIZER_NATURAL_GRADIENT;
        default:
            return OPTIMIZER_CUSTOM;
    }
}

#endif // OPTIMIZER_UTILS_H

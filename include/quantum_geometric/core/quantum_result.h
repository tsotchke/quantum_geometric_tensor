/**
 * @file quantum_result.h
 * @brief Quantum measurement result types and functions
 */

#ifndef QUANTUM_RESULT_H
#define QUANTUM_RESULT_H

#include <stddef.h>
#include "quantum_geometric/core/quantum_types.h"

// Quantum measurement result structure
typedef struct quantum_result {
    size_t num_measurements;     // Number of measurements
    double* measurements;        // Array of measurement results
} quantum_result;

#endif // QUANTUM_RESULT_H

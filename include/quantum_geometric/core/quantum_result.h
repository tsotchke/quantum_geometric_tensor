/**
 * @file quantum_result.h
 * @brief Quantum measurement result types and functions
 *
 * This is the canonical definition of quantum_result used throughout
 * the library. All other headers should include this file.
 */

#ifndef QUANTUM_RESULT_H
#define QUANTUM_RESULT_H

#include <stddef.h>
#include "quantum_geometric/core/quantum_types.h"

// Quantum measurement result structure
#ifndef QUANTUM_RESULT_DEFINED
#define QUANTUM_RESULT_DEFINED
typedef struct quantum_result {
    double* measurements;        // Array of measurement results
    size_t num_measurements;     // Number of measurements
    double* probabilities;       // Measurement probabilities (optional)
    size_t shots;               // Number of shots (optional)
    void* backend_data;         // Backend-specific data (optional)
} quantum_result;
#endif

// Result management functions
quantum_result* create_quantum_result(void);
void cleanup_quantum_result(quantum_result* result);

#endif // QUANTUM_RESULT_H

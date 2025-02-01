#ifndef STABILIZER_MEASUREMENT_H
#define STABILIZER_MEASUREMENT_H

#include "quantum_geometric/physics/quantum_stabilizer.h"
#include "quantum_geometric/physics/stabilizer_types.h"
#include "quantum_geometric/hardware/quantum_ibm_backend.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdbool.h>

// Measurement result structure
typedef struct {
    double expectation;           // Measured expectation value
    bool is_eigenstate;          // Whether state was eigenstate
    double eigenvalue;           // Eigenvalue if eigenstate
    double error_probability;    // Probability of measurement error
    void* auxiliary_data;        // Additional measurement data
} stabilizer_measurement_t;

// Initialize measurement structure
qgt_error_t measurement_create(stabilizer_measurement_t** measurement);

// Destroy measurement structure
void measurement_destroy(stabilizer_measurement_t* measurement);

// Perform stabilizer measurement
qgt_error_t measurement_perform(stabilizer_measurement_t* measurement,
                              const quantum_stabilizer_t* stabilizer,
                              const quantum_geometric_state_t* state);

// Check if measurement indicates error
qgt_error_t measurement_has_error(bool* has_error,
                                 const stabilizer_measurement_t* measurement,
                                 double threshold);

// Get measurement reliability
qgt_error_t measurement_reliability(double* reliability,
                                  const stabilizer_measurement_t* measurement);

// Compare two measurements
qgt_error_t measurement_compare(bool* equal,
                              const stabilizer_measurement_t* measurement1,
                              const stabilizer_measurement_t* measurement2,
                              double tolerance);

// Validate measurement result
qgt_error_t measurement_validate(const stabilizer_measurement_t* measurement);

#endif // STABILIZER_MEASUREMENT_H

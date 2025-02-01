#ifndef QUANTUM_GEOMETRIC_CALCULATIONS_H
#define QUANTUM_GEOMETRIC_CALCULATIONS_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/error_codes.h"
#include <stddef.h>
#include <stdbool.h>

// Configuration flags
#define QG_FLAG_OPTIMIZE (1 << 0)
#define QG_FLAG_ERROR_CORRECT (1 << 1)

// Forward declarations
typedef struct quantum_geometric_state_t quantum_geometric_state_t;
typedef struct quantum_system_t quantum_system_t;

// Geometric calculations
qgt_error_t calculate_metric(const quantum_geometric_state_t* state,
                           size_t i,
                           size_t j,
                           double* result);

qgt_error_t calculate_connection(const quantum_geometric_state_t* state,
                               size_t i,
                               size_t j,
                               size_t k,
                               double* result);

qgt_error_t calculate_curvature(const quantum_geometric_state_t* state,
                               size_t i,
                               size_t j,
                               size_t k,
                               size_t l,
                               double* result);

// Geometric encoding operations
qgt_error_t encode_geometric_state(quantum_geometric_state_t** encoded_state,
                                 const quantum_geometric_state_t* input_state,
                                 const geometric_encoding_config_t* config);

qgt_error_t decode_geometric_state(quantum_geometric_state_t** decoded_state,
                                 const quantum_geometric_state_t* encoded_state,
                                 const geometric_encoding_config_t* config);

// Geometric validation
qgt_error_t validate_geometric_encoding(const quantum_geometric_state_t* state,
                                      const geometric_encoding_config_t* config,
                                      bool* is_valid);

qgt_error_t validate_geometric_metric(const quantum_geometric_state_t* state,
                                    bool* is_valid);

// Resource estimation
qgt_error_t estimate_geometric_resources(const quantum_geometric_state_t* state,
                                       size_t* memory_required,
                                       size_t* operations_required);

// Hardware acceleration checks
bool is_gpu_available(void);
bool is_accelerator_available(void);

#endif // QUANTUM_GEOMETRIC_CALCULATIONS_H

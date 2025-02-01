#ifndef QUANTUM_GEOMETRIC_CORE_H
#define QUANTUM_GEOMETRIC_CORE_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_tensor.h"
#include <stddef.h>
#include <stdbool.h>

// Core initialization
qgt_error_t geometric_core_initialize(void);
void geometric_core_shutdown(void);
qgt_error_t geometric_core_reset(void);

// Memory management
qgt_error_t geometric_core_allocate(void** ptr, size_t size);
void geometric_core_free(void* ptr);
qgt_error_t geometric_core_memcpy(void* dest, const void* src, size_t size);
qgt_error_t geometric_core_memset(void* ptr, int value, size_t size);
qgt_error_t geometric_core_get_memory_stats(size_t* total, size_t* peak, size_t* count);

// Device management
qgt_error_t geometric_core_get_device_count(size_t* count);
qgt_error_t geometric_core_set_device(size_t device);
qgt_error_t geometric_core_get_device_properties(size_t device,
                                               void* properties);
qgt_error_t geometric_core_synchronize_device(void);

// Stream management
qgt_error_t geometric_core_create_stream(void** stream);
void geometric_core_destroy_stream(void* stream);
qgt_error_t geometric_core_synchronize_stream(void* stream);

// Event management
qgt_error_t geometric_core_create_event(void** event);
void geometric_core_destroy_event(void* event);
qgt_error_t geometric_core_record_event(void* event, void* stream);
qgt_error_t geometric_core_synchronize_event(void* event);

// Core operations
qgt_error_t geometric_core_add(void* result,
                             const void* a,
                             const void* b,
                             size_t size);
qgt_error_t geometric_core_subtract(void* result,
                                  const void* a,
                                  const void* b,
                                  size_t size);
qgt_error_t geometric_core_multiply(void* result,
                                  const void* a,
                                  const void* b,
                                  size_t size);
qgt_error_t geometric_core_divide(void* result,
                                const void* a,
                                const void* b,
                                size_t size);

// Core linear algebra
qgt_error_t geometric_core_matrix_multiply(void* result,
                                         const void* a,
                                         const void* b,
                                         size_t m,
                                         size_t n,
                                         size_t k);
qgt_error_t geometric_core_matrix_transpose(void* result,
                                          const void* matrix,
                                          size_t rows,
                                          size_t cols);
qgt_error_t geometric_core_matrix_inverse(void* result,
                                        const void* matrix,
                                        size_t size);
qgt_error_t geometric_core_solve_linear_system(void* result,
                                             const void* a,
                                             const void* b,
                                             size_t size);

// Core tensor operations
qgt_error_t geometric_core_tensor_contract(void* result,
                                         const void* a,
                                         const void* b,
                                         const size_t* dims_a,
                                         const size_t* dims_b,
                                         size_t rank_a,
                                         size_t rank_b,
                                         const size_t* contract_a,
                                         const size_t* contract_b,
                                         size_t num_contractions);
qgt_error_t geometric_core_tensor_transpose(void* result,
                                          const void* tensor,
                                          const size_t* dims,
                                          size_t rank,
                                          const size_t* perm);
qgt_error_t geometric_core_tensor_decompose(void* u,
                                          void* s,
                                          void* v,
                                          const void* tensor,
                                          const size_t* dims,
                                          size_t rank);

// Core optimization
qgt_error_t geometric_core_minimize(void* result,
                                  void (*objective)(void*, const void*),
                                  void (*gradient)(void*, const void*),
                                  const void* initial,
                                  size_t size,
                                  size_t max_iterations);
qgt_error_t geometric_core_maximize(void* result,
                                  void (*objective)(void*, const void*),
                                  void (*gradient)(void*, const void*),
                                  const void* initial,
                                  size_t size,
                                  size_t max_iterations);

// Core validation
qgt_error_t geometric_core_validate_memory(const void* ptr, size_t size);
qgt_error_t geometric_core_validate_device(size_t device);
qgt_error_t geometric_core_validate_stream(const void* stream);
qgt_error_t geometric_core_validate_event(const void* event);

// Core profiling
qgt_error_t geometric_core_start_profiling(void);
qgt_error_t geometric_core_stop_profiling(void);
qgt_error_t geometric_core_reset_profiling(void);
qgt_error_t geometric_core_get_profile_data(void* data, size_t* size);

// Core error handling
const char* geometric_core_get_error_string(qgt_error_t error);
qgt_error_t geometric_core_get_last_error(void);
void geometric_core_clear_error(void);

// Core utility functions
qgt_error_t geometric_core_get_version(size_t* major,
                                     size_t* minor,
                                     size_t* patch);
qgt_error_t geometric_core_get_capabilities(void* capabilities);
qgt_error_t geometric_core_set_debug_mode(bool enable);
qgt_error_t geometric_core_set_log_level(size_t level);

#endif // QUANTUM_GEOMETRIC_CORE_H

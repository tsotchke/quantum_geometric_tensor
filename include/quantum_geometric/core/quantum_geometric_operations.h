#ifndef QUANTUM_GEOMETRIC_OPERATIONS_H
#define QUANTUM_GEOMETRIC_OPERATIONS_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/quantum_geometric_error.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"

// Optional OpenMP support
#if defined(_OPENMP)
#include <omp.h>
#define OMP_PARALLEL _Pragma("omp parallel")
#define OMP_FOR _Pragma("omp for")
#define OMP_CRITICAL _Pragma("omp critical")
#define OMP_FOR_COLLAPSE2 _Pragma("omp for collapse(2)")
#define OMP_FOR_COLLAPSE3 _Pragma("omp for collapse(3)")
#define OMP_FOR_COLLAPSE4 _Pragma("omp for collapse(4)")
#define OMP_FOR_GUIDED _Pragma("omp for schedule(guided)")
#define OMP_FOR_GUIDED_NOWAIT _Pragma("omp for schedule(guided) nowait")
#define OMP_PARALLEL_FOR _Pragma("omp parallel for")
#define OMP_PARALLEL_FOR_COLLAPSE2 _Pragma("omp parallel for collapse(2)")
#define OMP_PARALLEL_FOR_COLLAPSE3 _Pragma("omp parallel for collapse(3)")
#define OMP_PARALLEL_FOR_COLLAPSE4 _Pragma("omp parallel for collapse(4)")
#define OMP_PARALLEL_FOR_IF(cond) _Pragma("omp parallel for if(" #cond ")")
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#define omp_in_parallel() 0
#define OMP_PARALLEL
#define OMP_FOR
#define OMP_CRITICAL
#define OMP_FOR_COLLAPSE2
#define OMP_FOR_COLLAPSE3
#define OMP_FOR_COLLAPSE4
#define OMP_FOR_GUIDED
#define OMP_FOR_GUIDED_NOWAIT
#define OMP_PARALLEL_FOR
#define OMP_PARALLEL_FOR_COLLAPSE2
#define OMP_PARALLEL_FOR_COLLAPSE3
#define OMP_PARALLEL_FOR_COLLAPSE4
#define OMP_PARALLEL_FOR_IF(cond)
#endif

// Optimized block sizes for modern hardware
#define QGT_BLOCK_SIZE QGT_MAX_POOL_BLOCK
#define QGT_TILE_SIZE (QGT_MIN_POOL_BLOCK * 2)
#define QGT_WARP_SIZE 32
#define QGT_PREFETCH_DISTANCE QGT_POOL_PREFETCH
#define QGT_PARALLEL_THRESHOLD 1000

// Memory management
qgt_error_t geometric_initialize(void);
void geometric_shutdown(void);
qgt_error_t geometric_reset(void);

// Core geometric operations
qgt_error_t geometric_compute_metric(quantum_geometric_metric_t* metric,
                                   const quantum_state_t* state);

qgt_error_t geometric_compute_connection(quantum_geometric_connection_t* connection,
                                       const quantum_geometric_metric_t* metric);

qgt_error_t geometric_compute_curvature(quantum_geometric_curvature_t* curvature,
                                      const quantum_geometric_connection_t* connection);

// State management
qgt_error_t geometric_create_state(quantum_geometric_state_t** state,
                                 geometric_state_type_t type,
                                 size_t dimension,
                                 HardwareType hardware_type);

void geometric_destroy_state(quantum_geometric_state_t* state);

qgt_error_t geometric_clone_state(quantum_geometric_state_t** dest,
                                const quantum_geometric_state_t* src);

// Resource management
qgt_error_t geometric_estimate_resources(const quantum_geometric_state_t* state,
                                       size_t* memory,
                                       size_t* operations);

// Error correction
qgt_error_t geometric_error_correct(quantum_geometric_state_t* state,
                                  const quantum_geometric_metric_t* metric);

// Utility functions
qgt_error_t geometric_print_state(const quantum_geometric_state_t* state);

// Optimization operations
qgt_error_t geometric_create_optimization(quantum_geometric_optimization_t** optimization,
                                        geometric_optimization_type_t type,
                                        size_t dimension);

void geometric_destroy_optimization(quantum_geometric_optimization_t* optimization);

qgt_error_t geometric_optimize_parameters(quantum_geometric_optimization_t* optimization,
                                        const quantum_geometric_metric_t* metric,
                                        const quantum_geometric_connection_t* connection,
                                        const quantum_geometric_curvature_t* curvature);

qgt_error_t geometric_check_convergence(const quantum_geometric_optimization_t* optimization,
                                      bool* converged);

qgt_error_t geometric_optimize_step(quantum_geometric_optimization_t* optimization,
                                  const quantum_geometric_metric_t* metric,
                                  const quantum_geometric_connection_t* connection,
                                  const quantum_geometric_curvature_t* curvature);

// Additional geometric operations
qgt_error_t geometric_transform(quantum_geometric_state_t* result,
                              const quantum_geometric_state_t* state,
                              const quantum_geometric_tensor_t* transform);

qgt_error_t geometric_parallel_transport(quantum_geometric_state_t* result,
                                       const quantum_geometric_state_t* state,
                                       const quantum_geometric_connection_t* connection);

qgt_error_t geometric_project(quantum_geometric_state_t* result,
                            const quantum_geometric_state_t* state,
                            const quantum_geometric_state_t* subspace);

// Metric operations
qgt_error_t geometric_create_metric(quantum_geometric_metric_t** metric,
                                  geometric_metric_type_t type,
                                  size_t dimension);

void geometric_destroy_metric(quantum_geometric_metric_t* metric);

qgt_error_t geometric_distance(double* distance,
                             const quantum_geometric_state_t* state1,
                             const quantum_geometric_state_t* state2,
                             const quantum_geometric_metric_t* metric);

// Connection operations
qgt_error_t geometric_create_connection(quantum_geometric_connection_t** connection,
                                      geometric_connection_type_t type,
                                      size_t dimension);

void geometric_destroy_connection(quantum_geometric_connection_t* connection);

qgt_error_t geometric_transport(quantum_geometric_state_t* result,
                              const quantum_geometric_state_t* state,
                              const quantum_geometric_connection_t* connection,
                              const quantum_geometric_state_t* path);

// Curvature operations
qgt_error_t geometric_create_curvature(quantum_geometric_curvature_t** curvature,
                                     geometric_curvature_type_t type,
                                     size_t dimension);

void geometric_destroy_curvature(quantum_geometric_curvature_t* curvature);

qgt_error_t geometric_sectional_curvature(double* curvature,
                                        const quantum_geometric_curvature_t* R,
                                        const quantum_geometric_state_t* u,
                                        const quantum_geometric_state_t* v);

// Validation operations
qgt_error_t geometric_validate_state(const quantum_geometric_state_t* state);

qgt_error_t geometric_validate_metric(const quantum_geometric_metric_t* metric,
                                    geometric_validation_flags_t flags,
                                    validation_result_t* result);

qgt_error_t geometric_validate_connection(const quantum_geometric_connection_t* connection,
                                        geometric_validation_flags_t flags,
                                        validation_result_t* result);

qgt_error_t geometric_validate_curvature(const quantum_geometric_curvature_t* curvature,
                                       geometric_validation_flags_t flags,
                                       validation_result_t* result);

// Hardware operations
qgt_error_t geometric_to_device(quantum_geometric_state_t* state,
                              HardwareType hardware);

qgt_error_t geometric_from_device(quantum_geometric_state_t* state,
                                HardwareType hardware);

bool geometric_is_on_device(const quantum_geometric_state_t* state,
                          HardwareType hardware);

#endif // QUANTUM_GEOMETRIC_OPERATIONS_H

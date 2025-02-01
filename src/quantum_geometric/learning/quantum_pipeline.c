#include "quantum_geometric/learning/quantum_pipeline.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/memory_pool.h"
#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"
#include "quantum_geometric/hardware/quantum_geometric_tensor_gpu.h"
#include "quantum_geometric/hardware/quantum_geometric_tensor_ops.h"
#include "quantum_geometric/hardware/quantum_geometric_tensor_perf.h"

#include <stdlib.h>
#include <string.h>

// Forward declarations of C++ implementation functions
#ifdef __cplusplus
extern "C" {
#endif

void* quantum_pipeline_create_impl(const float* config);
int quantum_pipeline_train_impl(void* pipeline, const float* data, const int* labels, size_t num_samples);
int quantum_pipeline_evaluate_impl(void* pipeline, const float* data, const int* labels, size_t num_samples, float* results);
int quantum_pipeline_save_impl(void* pipeline, const char* filename);
void quantum_pipeline_destroy_impl(void* pipeline);

#ifdef __cplusplus
}
#endif

// C wrapper functions that forward to C++ implementations
quantum_pipeline_handle_t quantum_pipeline_create(const float* config) {
    return (quantum_pipeline_handle_t)quantum_pipeline_create_impl(config);
}

int quantum_pipeline_train(quantum_pipeline_handle_t pipeline,
                         const float* data,
                         const int* labels,
                         size_t num_samples) {
    return quantum_pipeline_train_impl(pipeline, data, labels, num_samples);
}

int quantum_pipeline_evaluate(quantum_pipeline_handle_t pipeline,
                           const float* data,
                           const int* labels,
                           size_t num_samples,
                           float* results) {
    return quantum_pipeline_evaluate_impl(pipeline, data, labels, num_samples, results);
}

int quantum_pipeline_save(quantum_pipeline_handle_t pipeline, const char* filename) {
    return quantum_pipeline_save_impl(pipeline, filename);
}

void quantum_pipeline_destroy(quantum_pipeline_handle_t pipeline) {
    quantum_pipeline_destroy_impl(pipeline);
}

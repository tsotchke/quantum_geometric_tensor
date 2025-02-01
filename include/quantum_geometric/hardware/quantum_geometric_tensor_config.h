#ifndef QUANTUM_GEOMETRIC_TENSOR_CONFIG_H
#define QUANTUM_GEOMETRIC_TENSOR_CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

// Hardware acceleration options
#define QGT_ACCELERATION_NONE 0
#define QGT_ACCELERATION_METAL 1
#define QGT_ACCELERATION_CUDA 2
#define QGT_ACCELERATION_AMX 3

// Default to Metal on Apple Silicon
#if defined(__APPLE__) && defined(__aarch64__)
#define QGT_DEFAULT_ACCELERATION QGT_ACCELERATION_METAL
#else
#define QGT_DEFAULT_ACCELERATION QGT_ACCELERATION_NONE
#endif

// Tensor operation configuration
typedef struct {
    int acceleration_type;  // Type of hardware acceleration to use
    int batch_size;        // Batch size for tensor operations
    int use_gpu;          // Whether to use GPU acceleration
    float learning_rate;   // Learning rate for training
} QGTensorConfig;

// Initialize default configuration
QGTensorConfig qgt_get_default_config(void);

// Set global configuration
int qgt_set_config(const QGTensorConfig* config);

// Get current configuration
const QGTensorConfig* qgt_get_config(void);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_TENSOR_CONFIG_H

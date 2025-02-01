#ifndef QUANTUM_GEOMETRIC_TYPES_H
#define QUANTUM_GEOMETRIC_TYPES_H

#include <stddef.h>
#include <stdbool.h>

// Error type for quantum geometric operations
typedef enum {
    QGT_SUCCESS = 0,
    QGT_ERROR_INVALID_ARGUMENT = -1,
    QGT_ERROR_ALLOCATION_FAILED = -2,
    QGT_ERROR_IO_ERROR = -3,
    QGT_ERROR_ALREADY_INITIALIZED = -4,
    QGT_ERROR_NOT_INITIALIZED = -5,
    QGT_ERROR_INVALID_STATE = -6,
    QGT_ERROR_OPERATION_FAILED = -7,
    QGT_ERROR_INVALID_PROPERTY = -25,  // Changed from macro to enum value
    QGT_ERROR_INVALID_DIMENSION = -26,
    QGT_ERROR_INVALID_SIZE = -27,
    QGT_ERROR_INVALID_TYPE = -28,
    QGT_ERROR_INVALID_VALUE = -29,
    QGT_ERROR_INVALID_RANGE = -30,
    QGT_ERROR_INVALID_FORMAT = -31,
    QGT_ERROR_INVALID_OPERATION = -32,
    QGT_ERROR_INVALID_CONFIGURATION = -33,
    QGT_ERROR_INVALID_PARAMETER = -34,
    QGT_ERROR_INVALID_HANDLE = -35
} qgt_error_t;

// Forward declarations of opaque types
typedef struct quantum_geometric_context_t* quantum_geometric_context;
typedef struct quantum_geometric_tensor_t* quantum_geometric_tensor;
typedef struct quantum_geometric_metric_t* quantum_geometric_metric;
typedef struct quantum_geometric_connection_t* quantum_geometric_connection;
typedef struct quantum_geometric_curvature_t* quantum_geometric_curvature;

// Common type definitions
typedef double complex_t[2];
typedef double* matrix_t;
typedef double* vector_t;

// Configuration types
typedef struct {
    size_t dimensions;
    size_t precision;
    bool use_gpu;
    char* device_name;
} quantum_geometric_config_t;

// Memory management types
typedef struct {
    void* data;
    size_t size;
    bool is_gpu_memory;
} quantum_geometric_memory_t;

// Operation status
typedef struct {
    qgt_error_t error;
    char message[256];
    double computation_time;
    size_t memory_used;
} quantum_geometric_status_t;

#endif // QUANTUM_GEOMETRIC_TYPES_H

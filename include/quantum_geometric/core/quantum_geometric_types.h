#ifndef QUANTUM_GEOMETRIC_TYPES_H
#define QUANTUM_GEOMETRIC_TYPES_H

#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/error_codes.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

// Platform-specific optimizations
#ifdef __APPLE__
#include "quantum_geometric/core/accelerate_wrapper.h"
#endif

// Error handling macros
#define QGT_CHECK_NULL(ptr) \
    if ((ptr) == NULL) { \
        return QGT_ERROR_INVALID_PARAMETER; \
    }

#define QGT_CHECK_ARGUMENT(condition) \
    if (!(condition)) { \
        return QGT_ERROR_INVALID_PARAMETER; \
    }

#define QGT_CHECK_STATE(condition) \
    if (!(condition)) { \
        return QGT_ERROR_INVALID_STATE; \
    }

// Process and memory error codes
typedef enum process_error_t {
    QG_PROCESS_SUCCESS = 0,
    QG_PROCESS_ERROR_INIT = -1,
    QG_PROCESS_ERROR_CREATE = -2,
    QG_PROCESS_ERROR_RESOURCE = -3,
    QG_PROCESS_ERROR_TERMINATE = -4,
    QG_PROCESS_ERROR_INVALID_ID = -5,
    QG_PROCESS_ERROR_COMMUNICATION = -6,
    QG_PROCESS_ERROR_NOT_SUPPORTED = -7
} process_error_t;

typedef enum distributed_memory_error_t {
    QG_DISTRIBUTED_MEMORY_SUCCESS = 0,
    QG_DISTRIBUTED_MEMORY_ERROR_INIT = -1,
    QG_DISTRIBUTED_MEMORY_ERROR_ALLOC = -2,
    QG_DISTRIBUTED_MEMORY_ERROR_FREE = -3,
    QG_DISTRIBUTED_MEMORY_ERROR_INVALID = -4
} distributed_memory_error_t;

// Optional SIMD/NEON support
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

// AMX configuration
#ifdef __aarch64__
#define AMX_TILE_M 32
#define AMX_TILE_N 32
#define AMX_TILE_K 32
#define AMX_ALIGNMENT 64
#endif

// Error message length
#define QGT_MAX_ERROR_MESSAGE_LENGTH 1024

// Forward declarations for optimization types
typedef enum geometric_optimization_type_t {
    GEOMETRIC_OPTIMIZATION_GRADIENT,     // Gradient-based optimization
    GEOMETRIC_OPTIMIZATION_NEWTON,       // Newton's method optimization
    GEOMETRIC_OPTIMIZATION_CONJUGATE,    // Conjugate gradient optimization
    GEOMETRIC_OPTIMIZATION_QUASI,        // Quasi-Newton optimization
    GEOMETRIC_OPTIMIZATION_STOCHASTIC,   // Stochastic optimization
    GEOMETRIC_OPTIMIZATION_CUSTOM        // Custom optimization method
} geometric_optimization_type_t;

// Forward declarations
typedef struct quantum_geometric_state_t quantum_geometric_state_t;
typedef struct quantum_geometric_operator_t quantum_geometric_operator_t;
typedef struct quantum_geometric_tensor_t quantum_geometric_tensor_t;
typedef struct quantum_geometric_config_t quantum_geometric_config_t;
typedef struct quantum_geometric_metric_t quantum_geometric_metric_t;
typedef struct quantum_geometric_connection_t quantum_geometric_connection_t;
typedef struct quantum_geometric_curvature_t quantum_geometric_curvature_t;

// Constants for optimization and validation
#define QGT_MAX_PARAMETER_MAGNITUDE 1e6
#define QGT_DEFAULT_CONVERGENCE_THRESHOLD 1e-6
#define QGT_DEFAULT_LEARNING_RATE 0.01
#define QGT_MAX_ITERATIONS 10000
#define QGT_VALIDATION_TOLERANCE 1e-6
#define QGT_MAX_DIMENSIONS 16

// Hardware type definitions
typedef struct quantum_geometric_hardware_t {
    size_t device_id;
    char device_name[256];
    size_t memory_size;
    bool supports_gpu;
    bool supports_metal;
    bool supports_cuda;
    void* device_handle;
    void* context;
    void* command_queue;
    HardwareType type;
    bool is_initialized;
    size_t dimension;
    ComplexFloat* input_buffer;
    ComplexFloat* output_buffer;
    char* connection_string;
} quantum_geometric_hardware_t;

// Optimization structure
typedef struct quantum_geometric_optimization_t {
    geometric_optimization_type_t type;
    size_t dimension;
    ComplexFloat* parameters;
    size_t iterations;
    float convergence_threshold;
    float learning_rate;
    bool converged;
    void* optimizer_state;
} quantum_geometric_optimization_t;

// Pool configuration
typedef struct PoolConfig {
    size_t min_block_size;          // Minimum block size
    size_t alignment;               // Memory alignment
    size_t num_size_classes;        // Number of size classes
    float growth_factor;            // Size class growth factor
    size_t prefetch_distance;       // Prefetch distance for optimization
    bool use_huge_pages;            // Use huge pages for large allocations
    bool cache_local_free_lists;    // Use thread-local free lists
    size_t max_blocks_per_class;    // Maximum blocks per size class
    size_t thread_cache_size;       // Size of thread-local cache
    bool enable_stats;              // Enable memory statistics
} PoolConfig;


// Process status
typedef struct process_status_t {
    int process_id;
    int is_active;
    int error_code;
    char error_message[256];
} process_status_t;

// Process configuration
typedef struct process_config_t {
    int num_processes;
    int threads_per_process;
    bool use_affinity;
    int priority;
    char* working_dir;
    void* mpi_comm;
} process_config_t;

// Process group
typedef struct process_group_t {
    int group_id;
    const char* group_name;
    size_t num_processes;
    int* process_ids;
    void* group_comm;
} process_group_t;

// Memory management types
typedef struct distributed_memory_config_t {
    size_t total_memory;
    size_t local_memory;
    size_t shared_memory;
    bool use_numa;
    size_t numa_node;
} distributed_memory_config_t;

typedef struct memory_distribution_t {
    size_t* node_sizes;
    void** node_ptrs;
    size_t num_nodes;
    bool is_balanced;
} memory_distribution_t;

typedef enum memory_region_type_t {
    MEMORY_REGION_LOCAL,
    MEMORY_REGION_SHARED,
    MEMORY_REGION_DISTRIBUTED,
    MEMORY_REGION_GPU,
    MEMORY_REGION_CUSTOM
} memory_region_type_t;

// System configuration flags
#define QUANTUM_SYSTEM_FLAG_OPTIMIZED    (1 << 0)
#define QUANTUM_SYSTEM_FLAG_PROTECTED    (1 << 1)
#define QUANTUM_SYSTEM_FLAG_DISTRIBUTED  (1 << 2)
#define QUANTUM_SYSTEM_FLAG_CACHED       (1 << 3)
#define QUANTUM_SYSTEM_FLAG_MONITORED    (1 << 4)

// Function declarations for quantum system operations
void quantum_system_destroy(quantum_system_t* system);

// Geometric encoding configuration
typedef struct geometric_encoding_config_t {
    float error_rate;           // Maximum allowed error rate
    uint32_t flags;            // Configuration flags
    void* custom_config;       // Additional custom configuration
} geometric_encoding_config_t;

// Configuration flags
#define QG_FLAG_OPTIMIZE      (1 << 0)  // Enable optimization
#define QG_FLAG_ERROR_CORRECT (1 << 1)  // Enable error correction

// Geometric state types
typedef enum {
    GEOMETRIC_STATE_EUCLIDEAN,    // Euclidean geometric state
    GEOMETRIC_STATE_HYPERBOLIC,   // Hyperbolic geometric state
    GEOMETRIC_STATE_SPHERICAL,    // Spherical geometric state
    GEOMETRIC_STATE_SYMPLECTIC,   // Symplectic geometric state
    GEOMETRIC_STATE_KAHLER,       // Kähler geometric state
    GEOMETRIC_STATE_CALABI_YAU,   // Calabi-Yau geometric state
    GEOMETRIC_STATE_CUSTOM        // Custom geometric state
} geometric_state_type_t;

// Geometric operator types
typedef enum {
    GEOMETRIC_OPERATOR_METRIC,      // Metric operator
    GEOMETRIC_OPERATOR_CONNECTION,  // Connection operator
    GEOMETRIC_OPERATOR_CURVATURE,   // Curvature operator
    GEOMETRIC_OPERATOR_LAPLACIAN,   // Laplacian operator
    GEOMETRIC_OPERATOR_DIRAC,       // Dirac operator
    GEOMETRIC_OPERATOR_CUSTOM       // Custom operator
} geometric_operator_type_t;

// Geometric tensor types
typedef enum {
    GEOMETRIC_TENSOR_SCALAR,     // Scalar tensor
    GEOMETRIC_TENSOR_VECTOR,     // Vector tensor
    GEOMETRIC_TENSOR_COVECTOR,   // Covector tensor
    GEOMETRIC_TENSOR_BIVECTOR,   // Bivector tensor
    GEOMETRIC_TENSOR_TRIVECTOR,  // Trivector tensor
    GEOMETRIC_TENSOR_UNITARY,    // Unitary tensor
    GEOMETRIC_TENSOR_HERMITIAN,  // Hermitian tensor
    GEOMETRIC_TENSOR_SYMMETRIC,  // Symmetric tensor
    GEOMETRIC_TENSOR_CUSTOM      // Custom tensor
} geometric_tensor_type_t;

// Geometric configuration structure
struct quantum_geometric_config_t {
    size_t num_threads;              // Number of threads for parallel execution
    size_t batch_size;              // Batch size for processing
    size_t max_iterations;          // Maximum number of iterations
    float learning_rate;            // Learning rate for optimization
    float convergence_threshold;    // Convergence threshold
    bool use_gpu;                  // Whether to use GPU acceleration
    bool distributed;              // Whether to use distributed computing
    void* custom_config;           // Additional custom configuration
};

// Geometric state structure
struct quantum_geometric_state_t {
    geometric_state_type_t type;    // State type
    size_t dimension;               // State dimension
    size_t manifold_dim;           // Manifold dimension
    ComplexFloat* coordinates;      // State coordinates
    ComplexFloat* metric;          // Metric tensor
    ComplexFloat* connection;      // Connection coefficients
    void* auxiliary_data;          // Additional state data
    bool is_normalized;            // Normalization flag
    HardwareType hardware;   // Hardware location
};

// Geometric operator structure
struct quantum_geometric_operator_t {
    geometric_operator_type_t type; // Operator type
    size_t dimension;              // Operator dimension
    size_t rank;                   // Operator rank
    ComplexFloat* coefficients;    // Operator coefficients
    void* auxiliary_data;          // Additional operator data
    bool is_hermitian;            // Hermiticity flag
    HardwareType hardware;   // Hardware location
};

// Geometric tensor structure
struct quantum_geometric_tensor_t {
    geometric_tensor_type_t type;   // Tensor type
    size_t* dimensions;            // Tensor dimensions
    size_t rank;                   // Tensor rank
    ComplexFloat* components;      // Tensor components
    void* auxiliary_data;          // Additional tensor data
    bool is_symmetric;            // Symmetry flag
    bool is_unitary;              // Unitary flag
    bool is_hermitian;            // Hermitian flag
    HardwareType hardware;         // Hardware location
    size_t total_elements;        // Total number of elements
    size_t aligned_elements;      // Number of elements after alignment
};

// Geometric transformation types
typedef enum {
    GEOMETRIC_TRANSFORM_ROTATION,    // Rotation transformation
    GEOMETRIC_TRANSFORM_TRANSLATION, // Translation transformation
    GEOMETRIC_TRANSFORM_SCALING,     // Scaling transformation
    GEOMETRIC_TRANSFORM_SHEAR,      // Shear transformation
    GEOMETRIC_TRANSFORM_REFLECTION,  // Reflection transformation
    GEOMETRIC_TRANSFORM_CUSTOM      // Custom transformation
} geometric_transform_type_t;

// Geometric metric types
typedef enum {
    GEOMETRIC_METRIC_EUCLIDEAN,    // Euclidean metric
    GEOMETRIC_METRIC_MINKOWSKI,    // Minkowski metric
    GEOMETRIC_METRIC_FUBINI_STUDY, // Fubini-Study metric
    GEOMETRIC_METRIC_KAHLER,       // Kähler metric
    GEOMETRIC_METRIC_CUSTOM        // Custom metric
} geometric_metric_type_t;

// Geometric connection types
typedef enum {
    GEOMETRIC_CONNECTION_LEVI_CIVITA, // Levi-Civita connection
    GEOMETRIC_CONNECTION_SPIN,        // Spin connection
    GEOMETRIC_CONNECTION_YANG_MILLS,  // Yang-Mills connection
    GEOMETRIC_CONNECTION_CUSTOM       // Custom connection
} geometric_connection_type_t;

// Geometric curvature types
typedef enum {
    GEOMETRIC_CURVATURE_RIEMANN,    // Riemann curvature
    GEOMETRIC_CURVATURE_RICCI,      // Ricci curvature
    GEOMETRIC_CURVATURE_SCALAR,     // Scalar curvature
    GEOMETRIC_CURVATURE_WEYL,       // Weyl curvature
    GEOMETRIC_CURVATURE_CUSTOM      // Custom curvature
} geometric_curvature_type_t;

// Geometric symmetry types
typedef enum {
    GEOMETRIC_SYMMETRY_CONTINUOUS,   // Continuous symmetry
    GEOMETRIC_SYMMETRY_DISCRETE,     // Discrete symmetry
    GEOMETRIC_SYMMETRY_GAUGE,        // Gauge symmetry
    GEOMETRIC_SYMMETRY_CUSTOM        // Custom symmetry
} geometric_symmetry_type_t;

// Geometric boundary types
typedef enum {
    GEOMETRIC_BOUNDARY_DIRICHLET,    // Dirichlet boundary
    GEOMETRIC_BOUNDARY_NEUMANN,      // Neumann boundary
    GEOMETRIC_BOUNDARY_PERIODIC,     // Periodic boundary
    GEOMETRIC_BOUNDARY_CUSTOM        // Custom boundary
} geometric_boundary_type_t;

// Geometric validation flags
typedef enum {
    GEOMETRIC_VALIDATION_CHECK_NONE = 0,
    GEOMETRIC_VALIDATION_CHECK_SYMMETRY = 1 << 0,
    GEOMETRIC_VALIDATION_CHECK_POSITIVE_DEFINITE = 1 << 1,
    GEOMETRIC_VALIDATION_CHECK_TORSION_FREE = 1 << 2,
    GEOMETRIC_VALIDATION_CHECK_BIANCHI = 1 << 3,
    GEOMETRIC_VALIDATION_CHECK_BOUNDS = 1 << 4,
    GEOMETRIC_VALIDATION_CHECK_CONVERGENCE = 1 << 5,
    GEOMETRIC_VALIDATION_CHECK_COMPATIBILITY = 1 << 6,
    GEOMETRIC_VALIDATION_CHECK_ALL = (1 << 7) - 1
} geometric_validation_flags_t;

// Validation result structure
typedef struct {
    bool is_valid;                  // Overall validation status
    qgt_error_t error_code;         // Error code if validation fails
    char error_message[QGT_MAX_ERROR_MESSAGE_LENGTH];  // Detailed error message if validation fails
} validation_result_t;

// Geometric metric structure
struct quantum_geometric_metric_t {
    geometric_metric_type_t type;     // Metric type
    size_t dimension;                 // Metric dimension
    ComplexFloat* components;         // Metric components
    bool is_symmetric;                // Symmetry flag
    void* auxiliary_data;             // Additional metric data
    HardwareType hardware;            // Hardware location
};

// Geometric connection structure
struct quantum_geometric_connection_t {
    geometric_connection_type_t type;   // Connection type
    size_t dimension;                   // Connection dimension
    ComplexFloat* coefficients;         // Connection coefficients
    void* auxiliary_data;               // Additional connection data
    bool is_compatible;                 // Compatibility flag
    HardwareType hardware;              // Hardware location
};

// Geometric curvature structure
struct quantum_geometric_curvature_t {
    geometric_curvature_type_t type; // Curvature type
    size_t dimension;                // Curvature dimension
    ComplexFloat* components;        // Curvature components
    void* auxiliary_data;            // Additional curvature data
    bool is_flat;                    // Flatness flag
    HardwareType hardware;           // Hardware location
};

#endif // QUANTUM_GEOMETRIC_TYPES_H

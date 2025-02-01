#ifndef NUMERICAL_OPERATIONS_H
#define NUMERICAL_OPERATIONS_H

#include <stdbool.h>
#include <stddef.h>
#include <complex.h>

// Numerical types
typedef enum {
    NUM_TYPE_REAL,          // Real numbers
    NUM_TYPE_COMPLEX,       // Complex numbers
    NUM_TYPE_QUATERNION,    // Quaternions
    NUM_TYPE_TENSOR,        // Tensors
    NUM_TYPE_QUANTUM       // Quantum states
} numerical_type_t;

// Operation types
typedef enum {
    OP_ADD,                // Addition
    OP_SUBTRACT,           // Subtraction
    OP_MULTIPLY,           // Multiplication
    OP_DIVIDE,             // Division
    OP_TENSOR_PROD,        // Tensor product
    OP_QUANTUM_PROD       // Quantum product
} operation_type_t;

// Precision levels
typedef enum {
    PRECISION_SINGLE,      // Single precision
    PRECISION_DOUBLE,      // Double precision
    PRECISION_EXTENDED,    // Extended precision
    PRECISION_ARBITRARY   // Arbitrary precision
} precision_level_t;

// Computation modes
typedef enum {
    MODE_SEQUENTIAL,       // Sequential computation
    MODE_PARALLEL,         // Parallel computation
    MODE_DISTRIBUTED,      // Distributed computation
    MODE_QUANTUM         // Quantum computation
} computation_mode_t;

// Numerical configuration
typedef struct {
    numerical_type_t type;         // Numerical type
    precision_level_t precision;   // Precision level
    computation_mode_t mode;       // Computation mode
    bool use_gpu;                  // Use GPU flag
    bool enable_validation;        // Enable validation
    size_t workspace_size;        // Workspace size
} numerical_config_t;

// Matrix properties
typedef struct {
    size_t rows;                   // Number of rows
    size_t cols;                   // Number of columns
    bool is_symmetric;             // Symmetry flag
    bool is_hermitian;             // Hermitian flag
    bool is_positive_definite;     // Positive definite flag
    double condition_number;       // Condition number
} matrix_properties_t;

// Tensor properties
typedef struct {
    size_t* dimensions;            // Tensor dimensions
    size_t num_dimensions;         // Number of dimensions
    size_t total_elements;         // Total elements
    numerical_type_t element_type; // Element type
    bool is_contiguous;           // Contiguity flag
    void* tensor_data;           // Additional data
} tensor_properties_t;

// Operation result
typedef struct {
    bool success;                  // Operation success
    double error_estimate;         // Error estimate
    double computation_time;       // Computation time
    size_t iterations;            // Iteration count
    char* error_message;          // Error message
    void* result_data;           // Additional data
} operation_result_t;

// Opaque operations handle
typedef struct numerical_operations_t numerical_operations_t;

// Core functions
numerical_operations_t* create_numerical_operations(const numerical_config_t* config);
void destroy_numerical_operations(numerical_operations_t* operations);

// Basic operations
bool add(numerical_operations_t* ops, const void* a, const void* b, void* result, size_t size);
bool subtract(numerical_operations_t* ops, const void* a, const void* b, void* result, size_t size);
bool multiply(numerical_operations_t* ops, const void* a, const void* b, void* result, size_t size);
bool divide(numerical_operations_t* ops, const void* a, const void* b, void* result, size_t size);

// Matrix operations
bool matrix_multiply(numerical_operations_t* ops, const void* a, const void* b, void* result,
                    size_t m, size_t n, size_t k);
bool matrix_inverse(numerical_operations_t* ops, const void* matrix, void* result, size_t size);
bool matrix_decompose(numerical_operations_t* ops, const void* matrix, void* result,
                     const char* decomp_type, size_t size);
bool solve_linear_system(numerical_operations_t* ops, const void* a, const void* b,
                        void* x, size_t size);

// Tensor operations
bool tensor_contract(numerical_operations_t* ops, const void* a, const void* b,
                    void* result, const int* indices, size_t num_indices);
bool tensor_transpose(numerical_operations_t* ops, const void* tensor, void* result,
                     const int* perm, size_t num_dims);
bool tensor_reshape(numerical_operations_t* ops, const void* tensor, void* result,
                   const size_t* new_dims, size_t num_dims);
bool tensor_decompose(numerical_operations_t* ops, const void* tensor, void* result,
                     const char* decomp_type);

// Quantum operations
bool quantum_tensor_product(numerical_operations_t* ops, const void* a, const void* b,
                          void* result, size_t num_qubits_a, size_t num_qubits_b);
bool quantum_transform(numerical_operations_t* ops, const void* state, const void* operator,
                      void* result, size_t num_qubits);
bool quantum_measure(numerical_operations_t* ops, const void* state, void* result,
                    size_t num_qubits);

// Utility functions
bool get_matrix_properties(numerical_operations_t* ops, const void* matrix,
                         matrix_properties_t* props, size_t size);
bool get_tensor_properties(numerical_operations_t* ops, const void* tensor,
                         tensor_properties_t* props);
bool validate_operation(numerical_operations_t* ops, const void* result,
                       operation_type_t op_type, operation_result_t* op_result);

// Memory management
bool allocate_workspace(numerical_operations_t* ops, size_t size);
bool free_workspace(numerical_operations_t* ops);
bool clear_workspace(numerical_operations_t* ops);

// Error handling
const char* get_last_error(numerical_operations_t* ops);
int get_error_code(numerical_operations_t* ops);
void clear_error(numerical_operations_t* ops);

#endif // NUMERICAL_OPERATIONS_H

#include "quantum_geometric/core/quantum_geometric_tensor.h"
#include "quantum_geometric/core/simd_operations.h"
#include "quantum_geometric/core/numeric_utils.h"
#include "quantum_geometric/core/quantum_rng.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#endif

// Numerical stability thresholds
#define QGT_STABILITY_THRESHOLD 1e-7
#define QGT_UNITARY_THRESHOLD 1e-6
#define QGT_HERMITIAN_THRESHOLD 1e-6

// Cache-friendly block sizes
#define QGT_L1_BLOCK_SIZE 32
#define QGT_L2_BLOCK_SIZE 128
#define QGT_L3_BLOCK_SIZE 512

// Validate quantum geometric properties
static bool validate_unitary(const ComplexFloat* components, size_t dim) {
    // Check if tensor is unitary by verifying T^† T = I
    ComplexFloat* temp = (ComplexFloat*)aligned_alloc(32, dim * dim * sizeof(ComplexFloat));
    if (!temp) return false;
    
    bool is_unitary = true;
    for (size_t i = 0; i < dim && is_unitary; i++) {
        for (size_t j = 0; j < dim && is_unitary; j++) {
            ComplexFloat sum = {0.0f, 0.0f};
            for (size_t k = 0; k < dim; k++) {
                ComplexFloat conj = {
                    components[k * dim + i].real,
                    -components[k * dim + i].imag
                };
                sum = complex_float_add(sum, 
                    complex_float_multiply(conj, components[k * dim + j]));
            }
            if (i == j) {
                float diff = fabsf(sum.real - 1.0f) + fabsf(sum.imag);
                if (diff > QGT_UNITARY_THRESHOLD) {
                    is_unitary = false;
                }
            } else {
                float mag = sqrtf(sum.real * sum.real + sum.imag * sum.imag);
                if (mag > QGT_UNITARY_THRESHOLD) {
                    is_unitary = false;
                }
            }
        }
    }
    
    free(temp);
    return is_unitary;
}

static bool validate_hermitian(const ComplexFloat* components, size_t dim) {
    // Check if tensor is Hermitian by verifying T = T^†
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = i + 1; j < dim; j++) {
            ComplexFloat a = components[i * dim + j];
            ComplexFloat b = components[j * dim + i];
            float real_diff = fabsf(a.real - b.real);
            float imag_diff = fabsf(a.imag + b.imag);
            if (real_diff > QGT_HERMITIAN_THRESHOLD || 
                imag_diff > QGT_HERMITIAN_THRESHOLD) {
                return false;
            }
        }
    }
    return true;
}

bool geometric_tensor_is_hermitian(const quantum_geometric_tensor_t* tensor) {
    if (!tensor || !tensor->components || tensor->rank != 2 || 
        tensor->dimensions[0] != tensor->dimensions[1]) {
        return false;
    }
    return validate_hermitian(tensor->components, tensor->dimensions[0]);
}

// Helper function to validate tensor dimensions
static qgt_error_t validate_dimensions(const size_t* dimensions, size_t rank) {
    if (!dimensions) {
        printf("Error: dimensions pointer is NULL\n");
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    if (rank == 0) {
        printf("Error: rank must be greater than 0\n");
        return QGT_ERROR_INVALID_PARAMETER;
    }

    size_t total_elements = 1;
    for (size_t i = 0; i < rank; i++) {
        if (dimensions[i] == 0) {
            printf("Error: dimension[%zu] is 0\n", i);
            return QGT_ERROR_INVALID_PARAMETER;
        }
        
        // Check for overflow
        if (total_elements > SIZE_MAX / dimensions[i]) {
            printf("Error: dimension overflow at index %zu\n", i);
            return QGT_ERROR_INVALID_PARAMETER;
        }
        total_elements *= dimensions[i];
    }
    
    printf("Dimension validation successful: rank=%zu, total_elements=%zu\n", rank, total_elements);
    return QGT_SUCCESS;
}

// Helper function to initialize tensor properties
static void initialize_tensor_properties(quantum_geometric_tensor_t* tensor,
                                      geometric_tensor_type_t type) {
    tensor->type = type;
    tensor->is_symmetric = false;
    tensor->is_unitary = false;
    tensor->is_hermitian = false;
    tensor->hardware = HARDWARE_TYPE_SIMULATOR;
    tensor->auxiliary_data = NULL;

    // Set quantum geometric properties based on type
    switch (type) {
        case GEOMETRIC_TENSOR_SYMMETRIC:
            tensor->is_symmetric = true;
            tensor->is_hermitian = true;  // Symmetric tensors are also Hermitian
            break;
        case GEOMETRIC_TENSOR_UNITARY:
            tensor->is_unitary = true;
            break;
        case GEOMETRIC_TENSOR_HERMITIAN:
            tensor->is_hermitian = true;
            break;
        case GEOMETRIC_TENSOR_SCALAR:
            tensor->is_symmetric = true;
            tensor->is_hermitian = true;
            break;
        case GEOMETRIC_TENSOR_VECTOR:
        case GEOMETRIC_TENSOR_COVECTOR:
            // These types preserve their default flags
            break;
        case GEOMETRIC_TENSOR_BIVECTOR:
        case GEOMETRIC_TENSOR_TRIVECTOR:
            tensor->is_symmetric = true;  // Multi-vectors are symmetric
            break;
        case GEOMETRIC_TENSOR_CUSTOM:
            // Custom type preserves default flags
            break;
    }
}

qgt_error_t geometric_tensor_create(quantum_geometric_tensor_t** tensor,
                                  geometric_tensor_type_t type,
                                  const size_t* dimensions,
                                  size_t rank) {
    printf("Creating tensor: type=%d, rank=%zu\n", type, rank);
    
    if (!tensor) {
        printf("Error: tensor pointer is NULL\n");
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    qgt_error_t err = validate_dimensions(dimensions, rank);
    if (err != QGT_SUCCESS) {
        printf("Error: dimension validation failed with code %d\n", err);
        return err;
    }

    // Calculate total elements
    size_t total_elements = 1;
    for (size_t i = 0; i < rank; i++) {
        total_elements *= dimensions[i];
    }

    printf("Allocating memory: total_elements=%zu\n", total_elements);

    // Round up for SIMD alignment (AVX-512 requires 64-byte alignment)
    size_t aligned_elements = (total_elements + 7) & ~7;
    printf("Aligned elements: %zu\n", aligned_elements);

    // Allocate all memory with proper alignment
    quantum_geometric_tensor_t* t = (quantum_geometric_tensor_t*)aligned_alloc(64, sizeof(quantum_geometric_tensor_t));
    if (!t) {
        printf("Error: failed to allocate tensor struct\n");
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    size_t* dims = (size_t*)malloc(rank * sizeof(size_t));
    if (!dims) {
        printf("Error: failed to allocate dimensions array\n");
        free(t);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    ComplexFloat* comps = (ComplexFloat*)aligned_alloc(64, aligned_elements * sizeof(ComplexFloat));
    if (!comps) {
        printf("Error: failed to allocate components array\n");
        free(dims);
        free(t);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    printf("Memory allocation successful\n");

    // Initialize components with proper numerical stability
    #pragma omp parallel for simd aligned(comps: 64)
    for (size_t i = 0; i < aligned_elements; i++) {
        comps[i].real = 0.0f;
        comps[i].imag = 0.0f;
    }

    printf("Components initialized to zero\n");

    // Copy dimensions
    memcpy(dims, dimensions, rank * sizeof(size_t));
    printf("Dimensions copied\n");

    // Initialize tensor struct
    t->dimensions = dims;
    t->components = comps;
    t->rank = rank;
    t->total_elements = total_elements;
    t->aligned_elements = aligned_elements;

    printf("Tensor struct initialized\n");

    // Initialize properties
    initialize_tensor_properties(t, type);
    printf("Properties initialized: is_hermitian=%d\n", t->is_hermitian);

    *tensor = t;
    printf("Tensor creation successful\n");
    return QGT_SUCCESS;
}

void geometric_tensor_destroy(quantum_geometric_tensor_t* tensor) {
    if (tensor) {
        if (tensor->dimensions) {
            free(tensor->dimensions);
        }
        if (tensor->components) {
            free(tensor->components);
        }
        if (tensor->auxiliary_data) {
            free(tensor->auxiliary_data);
        }
        free(tensor);
    }
}

qgt_error_t geometric_tensor_initialize_random(quantum_geometric_tensor_t* tensor) {
    QGT_CHECK_NULL(tensor);
    QGT_CHECK_NULL(tensor->components);

    // Initialize QRNG context
    qrng_ctx* rng_ctx = NULL;
    qrng_error rng_err = qrng_init(&rng_ctx, NULL, 0);
    if (rng_err == QRNG_ERROR_NULL_CONTEXT) {
        return QGT_ERROR_NOT_INITIALIZED;
    } else if (rng_err == QRNG_ERROR_NULL_BUFFER) {
        return QGT_ERROR_INVALID_ARGUMENT;
    } else if (rng_err == QRNG_ERROR_INSUFFICIENT_ENTROPY) {
        return QGT_ERROR_RESOURCE_UNAVAILABLE;
    } else if (rng_err == QRNG_ERROR_INVALID_LENGTH) {
        return QGT_ERROR_INVALID_DIMENSION;
    } else if (rng_err == QRNG_ERROR_INVALID_RANGE) {
        return QGT_ERROR_INVALID_ARGUMENT;
    } else if (rng_err != QRNG_SUCCESS) {
        return QGT_ERROR_INITIALIZATION;
    }

    // Fill tensor with random complex numbers
    ComplexFloat* data = tensor->components;
    
    if (tensor->type == GEOMETRIC_TENSOR_HERMITIAN && tensor->rank == 2) {
        size_t dim = tensor->dimensions[0];
        // For Hermitian tensors, we need to ensure H = H†
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = i; j < dim; j++) {
                if (i == j) {
                    // Diagonal elements must be real for Hermitian matrices
                    float real = 2.0f * ((float)qrng_double(rng_ctx)) - 1.0f;
                    data[i * dim + i].real = real;
                    data[i * dim + i].imag = 0.0f;
                } else {
                    // Generate random values for upper triangle
                    float real = 2.0f * ((float)qrng_double(rng_ctx)) - 1.0f;
                    float imag = 2.0f * ((float)qrng_double(rng_ctx)) - 1.0f;
                    
                    // Normalize to ensure unit magnitude
                    float magnitude = sqrtf(real * real + imag * imag);
                    if (magnitude > 0) {
                        real /= magnitude;
                        imag /= magnitude;
                    }
                    
                    // Set upper triangle element
                    data[i * dim + j].real = real;
                    data[i * dim + j].imag = imag;
                    
                    // Set lower triangle element to conjugate
                    data[j * dim + i].real = real;
                    data[j * dim + i].imag = -imag;
                }
            }
        }
    } else {
        // For non-Hermitian tensors, use standard random initialization
        for (size_t i = 0; i < tensor->total_elements; i++) {
            // Generate random values between -1 and 1
            float real = 2.0f * ((float)qrng_double(rng_ctx)) - 1.0f;
            float imag = 2.0f * ((float)qrng_double(rng_ctx)) - 1.0f;
            
            // Normalize to ensure unit magnitude
            float magnitude = sqrtf(real * real + imag * imag);
            if (magnitude > 0) {
                real /= magnitude;
                imag /= magnitude;
            }
            
            data[i].real = real;
            data[i].imag = imag;
        }
    }

    // Cleanup
    qrng_free(rng_ctx);

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_clone(quantum_geometric_tensor_t** dest,
                                 const quantum_geometric_tensor_t* src) {
    qgt_error_t err = geometric_tensor_create(dest, src->type, src->dimensions, src->rank);
    if (err != QGT_SUCCESS) {
        return err;
    }

    // Use SIMD operations for copying with prefetching
    const size_t block_size = 1024; // Process 1KB blocks
    const size_t num_blocks = src->aligned_elements / block_size;
    const size_t remaining = src->aligned_elements % block_size;

    for (size_t i = 0; i < num_blocks; i++) {
        size_t offset = i * block_size;
        // Prefetch next block
        if (i + 1 < num_blocks) {
            __builtin_prefetch(&src->components[(i + 1) * block_size], 0, 3);
            __builtin_prefetch(&(*dest)->components[(i + 1) * block_size], 1, 3);
        }
        simd_complex_copy(&(*dest)->components[offset], 
                         &src->components[offset],
                         block_size);
    }

    // Handle remaining elements
    if (remaining > 0) {
        simd_complex_copy(&(*dest)->components[num_blocks * block_size],
                         &src->components[num_blocks * block_size],
                         remaining);
    }
    
    (*dest)->is_symmetric = src->is_symmetric;
    (*dest)->is_unitary = src->is_unitary;
    (*dest)->is_hermitian = src->is_hermitian;
    (*dest)->hardware = src->hardware;
    (*dest)->auxiliary_data = src->auxiliary_data;
    (*dest)->total_elements = src->total_elements;
    (*dest)->aligned_elements = src->aligned_elements;

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_add(quantum_geometric_tensor_t* result,
                                const quantum_geometric_tensor_t* a,
                                const quantum_geometric_tensor_t* b) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(a);
    QGT_CHECK_NULL(b);
    
    // Validate dimensions match
    QGT_CHECK_ARGUMENT(a->rank == b->rank);
    for (size_t i = 0; i < a->rank; i++) {
        QGT_CHECK_ARGUMENT(a->dimensions[i] == b->dimensions[i]);
    }

    // Process in blocks with prefetching
    const size_t block_size = 1024;
    const size_t num_blocks = a->aligned_elements / block_size;
    const size_t remaining = a->aligned_elements % block_size;

    for (size_t i = 0; i < num_blocks; i++) {
        size_t offset = i * block_size;
        // Prefetch next block
        if (i + 1 < num_blocks) {
            __builtin_prefetch(&a->components[(i + 1) * block_size], 0, 3);
            __builtin_prefetch(&b->components[(i + 1) * block_size], 0, 3);
            __builtin_prefetch(&result->components[(i + 1) * block_size], 1, 3);
        }
        simd_tensor_add(&result->components[offset],
                       &a->components[offset],
                       &b->components[offset],
                       block_size);
    }

    // Handle remaining elements
    if (remaining > 0) {
        simd_tensor_add(&result->components[num_blocks * block_size],
                       &a->components[num_blocks * block_size],
                       &b->components[num_blocks * block_size],
                       remaining);
    }

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_subtract(quantum_geometric_tensor_t* result,
                                     const quantum_geometric_tensor_t* a,
                                     const quantum_geometric_tensor_t* b) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(a);
    QGT_CHECK_NULL(b);
    
    // Validate dimensions match
    QGT_CHECK_ARGUMENT(a->rank == b->rank);
    for (size_t i = 0; i < a->rank; i++) {
        QGT_CHECK_ARGUMENT(a->dimensions[i] == b->dimensions[i]);
    }

    // Process in blocks with prefetching
    const size_t block_size = 1024;
    const size_t num_blocks = a->aligned_elements / block_size;
    const size_t remaining = a->aligned_elements % block_size;

    for (size_t i = 0; i < num_blocks; i++) {
        size_t offset = i * block_size;
        // Prefetch next block
        if (i + 1 < num_blocks) {
            __builtin_prefetch(&a->components[(i + 1) * block_size], 0, 3);
            __builtin_prefetch(&b->components[(i + 1) * block_size], 0, 3);
            __builtin_prefetch(&result->components[(i + 1) * block_size], 1, 3);
        }
        simd_tensor_subtract(&result->components[offset],
                           &a->components[offset],
                           &b->components[offset],
                           block_size);
    }

    // Handle remaining elements
    if (remaining > 0) {
        simd_tensor_subtract(&result->components[num_blocks * block_size],
                           &a->components[num_blocks * block_size],
                           &b->components[num_blocks * block_size],
                           remaining);
    }

    return QGT_SUCCESS;
}

// Helper function to calculate strides for arbitrary rank tensors
static void calculate_strides(size_t* strides, const size_t* dims, size_t rank) {
    strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
}

// Helper function to convert linear index to multi-dimensional indices
static void linear_to_indices(size_t* indices, size_t linear_idx, 
                            const size_t* dims, size_t rank) {
    for (size_t i = 0; i < rank; i++) {
        indices[i] = linear_idx % dims[i];
        linear_idx /= dims[i];
    }
}

// Helper function to convert multi-dimensional indices to linear index
static size_t indices_to_linear(const size_t* indices, const size_t* strides, size_t rank) {
    size_t linear_idx = 0;
    for (size_t i = 0; i < rank; i++) {
        linear_idx += indices[i] * strides[i];
    }
    return linear_idx;
}

// Helper function to check numerical stability
static bool check_numerical_stability(const ComplexFloat* result, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (isnan(result[i].real) || isnan(result[i].imag) ||
            isinf(result[i].real) || isinf(result[i].imag)) {
            return false;
        }
    }
    return true;
}

// Helper function for Strassen multiplication
static void strassen_multiply_recursive(ComplexFloat* C, const ComplexFloat* A, 
                                      const ComplexFloat* B, size_t n, size_t stride) {
    if (n <= QGT_L1_BLOCK_SIZE) {
        // Base case: use standard multiplication for small matrices
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                ComplexFloat sum = {0.0f, 0.0f};
                for (size_t k = 0; k < n; k++) {
                    sum = complex_float_add(sum, 
                        complex_float_multiply(A[i * stride + k], B[k * stride + j]));
                }
                C[i * stride + j] = sum;
            }
        }
        return;
    }

    size_t m = n / 2;
    size_t new_stride = stride / 2;

    // Allocate temporary matrices with proper alignment
    ComplexFloat* M1 = (ComplexFloat*)aligned_alloc(64, m * m * sizeof(ComplexFloat));
    ComplexFloat* M2 = (ComplexFloat*)aligned_alloc(64, m * m * sizeof(ComplexFloat));
    ComplexFloat* M3 = (ComplexFloat*)aligned_alloc(64, m * m * sizeof(ComplexFloat));
    ComplexFloat* M4 = (ComplexFloat*)aligned_alloc(64, m * m * sizeof(ComplexFloat));
    ComplexFloat* M5 = (ComplexFloat*)aligned_alloc(64, m * m * sizeof(ComplexFloat));
    ComplexFloat* M6 = (ComplexFloat*)aligned_alloc(64, m * m * sizeof(ComplexFloat));
    ComplexFloat* M7 = (ComplexFloat*)aligned_alloc(64, m * m * sizeof(ComplexFloat));
    ComplexFloat* temp = (ComplexFloat*)aligned_alloc(64, m * m * sizeof(ComplexFloat));

    if (!M1 || !M2 || !M3 || !M4 || !M5 || !M6 || !M7 || !temp) {
        // Handle allocation failure
        free(M1); free(M2); free(M3); free(M4);
        free(M5); free(M6); free(M7); free(temp);
        return;
    }

    // Compute the 7 Strassen products recursively
    #pragma omp task if(n > 64)
    strassen_multiply_recursive(M1, A, B, m, new_stride);
    
    #pragma omp task if(n > 64)
    strassen_multiply_recursive(M2, &A[m], B, m, new_stride);
    
    #pragma omp task if(n > 64)
    strassen_multiply_recursive(M3, A, &B[m * stride], m, new_stride);
    
    #pragma omp task if(n > 64)
    strassen_multiply_recursive(M4, &A[m * stride + m], &B[m], m, new_stride);
    
    #pragma omp task if(n > 64)
    strassen_multiply_recursive(M5, A, &B[m], m, new_stride);
    
    #pragma omp task if(n > 64)
    strassen_multiply_recursive(M6, &A[m * stride], B, m, new_stride);
    
    #pragma omp task if(n > 64)
    strassen_multiply_recursive(M7, &A[m], &B[m * stride], m, new_stride);
    
    #pragma omp taskwait

    // Combine results with SIMD operations
    #pragma omp parallel for collapse(2) if(n > 64)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            size_t idx = i * stride + j;
            C[idx] = complex_float_add(
                complex_float_add(M1[i * m + j], M4[i * m + j]),
                complex_float_subtract(M7[i * m + j], M5[i * m + j]));
            
            C[idx + m] = complex_float_add(M3[i * m + j], M5[i * m + j]);
            
            C[idx + m * stride] = complex_float_add(M2[i * m + j], M4[i * m + j]);
            
            C[idx + m * stride + m] = complex_float_add(
                complex_float_subtract(M1[i * m + j], M2[i * m + j]),
                complex_float_add(M3[i * m + j], M6[i * m + j]));
        }
    }

    // Cleanup
    free(M1); free(M2); free(M3); free(M4);
    free(M5); free(M6); free(M7); free(temp);
}

qgt_error_t geometric_tensor_multiply(quantum_geometric_tensor_t* result,
                                     const quantum_geometric_tensor_t* a,
                                     const quantum_geometric_tensor_t* b) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(a);
    QGT_CHECK_NULL(b);
    
    // Validate dimensions match for multiplication
    QGT_CHECK_ARGUMENT(a->rank == b->rank);

    // Calculate output dimensions
    size_t* out_dims = (size_t*)aligned_alloc(64, a->rank * sizeof(size_t));
    if (!out_dims) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // For matrix multiplication case
    if (a->rank == 2 && a->dimensions[1] == b->dimensions[0]) {
        const size_t M = a->dimensions[0];
        const size_t K = a->dimensions[1];
        const size_t N = b->dimensions[1];

        // Initialize result
        memset(result->components, 0, M * N * sizeof(ComplexFloat));

        // Use Strassen for large matrices
        if (M >= 128 && K >= 128 && N >= 128) {
            // Round up to next power of 2
            size_t max_dim = max(max(M, K), N);
            size_t padded_size = 1;
            while (padded_size < max_dim) padded_size *= 2;

            // Allocate padded matrices
            ComplexFloat* padded_a = (ComplexFloat*)aligned_alloc(64, padded_size * padded_size * sizeof(ComplexFloat));
            ComplexFloat* padded_b = (ComplexFloat*)aligned_alloc(64, padded_size * padded_size * sizeof(ComplexFloat));
            ComplexFloat* padded_c = (ComplexFloat*)aligned_alloc(64, padded_size * padded_size * sizeof(ComplexFloat));

            if (!padded_a || !padded_b || !padded_c) {
                free(padded_a); free(padded_b); free(padded_c);
                free(out_dims);
                return QGT_ERROR_MEMORY_ALLOCATION;
            }

            // Zero pad matrices
            memset(padded_a, 0, padded_size * padded_size * sizeof(ComplexFloat));
            memset(padded_b, 0, padded_size * padded_size * sizeof(ComplexFloat));
            
            // Copy data with SIMD
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < K; j++) {
                    padded_a[i * padded_size + j] = a->components[i * K + j];
                }
            }

            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < K; i++) {
                for (size_t j = 0; j < N; j++) {
                    padded_b[i * padded_size + j] = b->components[i * N + j];
                }
            }

            // Perform Strassen multiplication
            #pragma omp parallel
            {
                #pragma omp single
                strassen_multiply_recursive(padded_c, padded_a, padded_b, padded_size, padded_size);
            }

            // Copy result back
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < M; i++) {
                for (size_t j = 0; j < N; j++) {
                    result->components[i * N + j] = padded_c[i * padded_size + j];
                }
            }

            free(padded_a);
            free(padded_b);
            free(padded_c);
        } else {
            // Use blocked multiplication for smaller matrices
            #pragma omp parallel for collapse(2) schedule(guided)
            for (size_t i = 0; i < M; i += QGT_L1_BLOCK_SIZE) {
                for (size_t j = 0; j < N; j += QGT_L1_BLOCK_SIZE) {
                    for (size_t k = 0; k < K; k += QGT_L1_BLOCK_SIZE) {
                        size_t i_end = min(i + QGT_L1_BLOCK_SIZE, M);
                        size_t j_end = min(j + QGT_L1_BLOCK_SIZE, N);
                        size_t k_end = min(k + QGT_L1_BLOCK_SIZE, K);

                        // Process block with vectorization
                        for (size_t ii = i; ii < i_end; ii++) {
                            for (size_t jj = j; jj < j_end; jj++) {
                                ComplexFloat sum = {0.0f, 0.0f};
                                for (size_t kk = k; kk < k_end; kk++) {
                                    sum = complex_float_add(sum,
                                        complex_float_multiply(
                                            a->components[ii * K + kk],
                                            b->components[kk * N + jj]));
                                }
                                result->components[ii * N + jj] = 
                                    complex_float_add(result->components[ii * N + jj], sum);
                            }
                        }
                    }
                }
            }
        }

        // Check numerical stability
        if (!check_numerical_stability(result->components, M * N)) {
            free(out_dims);
            return QGT_ERROR_NUMERICAL_INSTABILITY;
        }

        out_dims[0] = M;
        out_dims[1] = N;
        result->dimensions = out_dims;
        result->rank = 2;
        result->total_elements = M * N;
        result->aligned_elements = (M * N + 7) & ~7;

        return QGT_SUCCESS;
    }

    // Handle higher rank tensor multiplication using batched matrix multiplication
    // For tensors with rank > 2, we reshape them into batched matrices
    
    // First validate dimensions
    if (a->dimensions[a->rank - 1] != b->dimensions[0]) {
        free(out_dims);
        return QGT_ERROR_INVALID_DIMENSION;
    }

    // Calculate batch dimensions (all dimensions except the contracting ones)
    size_t batch_size_a = 1;
    for (size_t i = 0; i < a->rank - 1; i++) {
        batch_size_a *= a->dimensions[i];
    }

    size_t batch_size_b = 1;
    for (size_t i = 1; i < b->rank; i++) {
        batch_size_b *= b->dimensions[i];
    }

    // Output dimensions
    size_t out_rank = a->rank + b->rank - 2;
    for (size_t i = 0; i < a->rank - 1; i++) {
        out_dims[i] = a->dimensions[i];
    }
    for (size_t i = 0; i < b->rank - 1; i++) {
        out_dims[a->rank - 1 + i] = b->dimensions[i + 1];
    }

    // Calculate total elements and initialize result
    size_t total_elements = batch_size_a * batch_size_b;
    size_t aligned_elements = (total_elements + 7) & ~7;
    memset(result->components, 0, aligned_elements * sizeof(ComplexFloat));

    // Calculate strides for input tensors
    size_t* a_strides = (size_t*)aligned_alloc(64, a->rank * sizeof(size_t));
    size_t* b_strides = (size_t*)aligned_alloc(64, b->rank * sizeof(size_t));
    if (!a_strides || !b_strides) {
        free(a_strides);
        free(b_strides);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // Calculate strides
    calculate_strides(a_strides, a->dimensions, a->rank);
    calculate_strides(b_strides, b->dimensions, b->rank);

    // Matrix dimensions for the batched multiplication
    const size_t M = batch_size_a;
    const size_t K = a->dimensions[a->rank - 1];
    const size_t N = batch_size_b;

    // Use blocked matrix multiplication for the batched operation
    #pragma omp parallel for collapse(2) schedule(guided)
    for (size_t i = 0; i < M; i += QGT_L1_BLOCK_SIZE) {
        for (size_t j = 0; j < N; j += QGT_L1_BLOCK_SIZE) {
            for (size_t k = 0; k < K; k += QGT_L1_BLOCK_SIZE) {
                size_t i_end = min(i + QGT_L1_BLOCK_SIZE, M);
                size_t j_end = min(j + QGT_L1_BLOCK_SIZE, N);
                size_t k_end = min(k + QGT_L1_BLOCK_SIZE, K);

                // Process block with vectorization
                for (size_t ii = i; ii < i_end; ii++) {
                    for (size_t jj = j; jj < j_end; jj++) {
                        ComplexFloat sum = {0.0f, 0.0f};
                        for (size_t kk = k; kk < k_end; kk++) {
                            // Convert linear indices to tensor indices
                            size_t* a_indices = (size_t*)alloca(a->rank * sizeof(size_t));
                            size_t* b_indices = (size_t*)alloca(b->rank * sizeof(size_t));
                            
                            linear_to_indices(a_indices, ii * K + kk, a->dimensions, a->rank);
                            linear_to_indices(b_indices, kk * N + jj, b->dimensions, b->rank);
                            
                            // Convert back to linear indices with strides
                            size_t a_idx = indices_to_linear(a_indices, a_strides, a->rank);
                            size_t b_idx = indices_to_linear(b_indices, b_strides, b->rank);
                            
                            sum = complex_float_add(sum,
                                complex_float_multiply(
                                    a->components[a_idx],
                                    b->components[b_idx]));
                        }
                        result->components[ii * N + jj] = 
                            complex_float_add(result->components[ii * N + jj], sum);
                    }
                }
            }
        }
    }

    // Clean up
    free(a_strides);
    free(b_strides);

    // Check numerical stability
    if (!check_numerical_stability(result->components, total_elements)) {
        free(out_dims);
        return QGT_ERROR_NUMERICAL_INSTABILITY;
    }

    // Update result metadata
    result->dimensions = out_dims;
    result->rank = out_rank;
    result->total_elements = total_elements;
    result->aligned_elements = aligned_elements;
    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_contract(quantum_geometric_tensor_t* result,
                                     const quantum_geometric_tensor_t* a,
                                     const quantum_geometric_tensor_t* b,
                                     const size_t* indices_a,
                                     const size_t* indices_b,
                                     size_t num_indices) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(a);
    QGT_CHECK_NULL(b);
    QGT_CHECK_NULL(indices_a);
    QGT_CHECK_NULL(indices_b);
    QGT_CHECK_ARGUMENT(num_indices > 0);

    // Validate contraction indices and dimensions
    for (size_t i = 0; i < num_indices; i++) {
        QGT_CHECK_ARGUMENT(indices_a[i] < a->rank);
        QGT_CHECK_ARGUMENT(indices_b[i] < b->rank);
        QGT_CHECK_ARGUMENT(a->dimensions[indices_a[i]] == b->dimensions[indices_b[i]]);
    }

    // Calculate output rank and dimensions
    size_t out_rank = a->rank + b->rank - 2 * num_indices;
    size_t* out_dims = (size_t*)aligned_alloc(64, out_rank * sizeof(size_t));
    if (!out_dims) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // Calculate output dimensions and strides using a more efficient approach
    size_t* dim_map_a = (size_t*)aligned_alloc(64, a->rank * sizeof(size_t));
    size_t* dim_map_b = (size_t*)aligned_alloc(64, b->rank * sizeof(size_t));
    if (!dim_map_a || !dim_map_b) {
        free(out_dims);
        free(dim_map_a);
        free(dim_map_b);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // Initialize dimension maps to -1 (uncontracted)
    memset(dim_map_a, -1, a->rank * sizeof(size_t));
    memset(dim_map_b, -1, b->rank * sizeof(size_t));

    // Mark contracted indices and validate dimensions
    for (size_t i = 0; i < num_indices; i++) {
        dim_map_a[indices_a[i]] = i;
        dim_map_b[indices_b[i]] = i;
    }

    // Build output dimensions
    size_t out_idx = 0;
    for (size_t i = 0; i < a->rank; i++) {
        if (dim_map_a[i] == (size_t)-1) {
            out_dims[out_idx++] = a->dimensions[i];
        }
    }
    for (size_t i = 0; i < b->rank; i++) {
        if (dim_map_b[i] == (size_t)-1) {
            out_dims[out_idx++] = b->dimensions[i];
        }
    }

    // Free dimension maps
    free(dim_map_a);
    free(dim_map_b);

    // Calculate total sizes
    size_t total_out_size = 1;
    for (size_t i = 0; i < out_rank; i++) {
        total_out_size *= out_dims[i];
    }

    size_t contract_size = 1;
    for (size_t i = 0; i < num_indices; i++) {
        contract_size *= a->dimensions[indices_a[i]];
    }

    // Initialize result tensor
    size_t aligned_out_size = (total_out_size + 7) & ~7;
    memset(result->components, 0, aligned_out_size * sizeof(ComplexFloat));
    result->dimensions = out_dims;
    result->rank = out_rank;
    result->total_elements = total_out_size;
    result->aligned_elements = aligned_out_size;

    // Calculate strides for input tensors
    size_t* a_strides = (size_t*)aligned_alloc(64, a->rank * sizeof(size_t));
    size_t* b_strides = (size_t*)aligned_alloc(64, b->rank * sizeof(size_t));
    if (!a_strides || !b_strides) {
        free(out_dims);
        free(a_strides);
        free(b_strides);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // Calculate strides using helper function
    calculate_strides(a_strides, a->dimensions, a->rank);
    calculate_strides(b_strides, b->dimensions, b->rank);

    // Process in blocks for better cache utilization and vectorization
    const size_t tile_size = 8; // Size of SIMD vector
    const size_t l1_block = QGT_L1_BLOCK_SIZE;
    const size_t l2_block = QGT_L2_BLOCK_SIZE;
    const size_t l3_block = QGT_L3_BLOCK_SIZE;
    
    // Pre-identify uncontracted indices using stack allocation
    size_t uncontracted_a_buffer[32];  // Max rank is typically much smaller
    size_t uncontracted_b_buffer[32];
    size_t uncontracted_a_count = 0;
    size_t uncontracted_b_count = 0;
    
    // Identify uncontracted indices once
    for (size_t k = 0; k < a->rank; k++) {
        bool is_contracted = false;
        for (size_t c = 0; c < num_indices; c++) {
            if (k == indices_a[c]) {
                is_contracted = true;
                break;
            }
        }
        if (!is_contracted) {
            uncontracted_a_buffer[uncontracted_a_count++] = k;
        }
    }
    
    for (size_t k = 0; k < b->rank; k++) {
        bool is_contracted = false;
        for (size_t c = 0; c < num_indices; c++) {
            if (k == indices_b[c]) {
                is_contracted = true;
                break;
            }
        }
        if (!is_contracted) {
            uncontracted_b_buffer[uncontracted_b_count++] = k;
        }
    }
    
    // Pre-compute contracted indices lookup table
    size_t contracted_indices[32][32];  // [contract_idx][dim_idx]
    for (size_t j = 0; j < contract_size && j < 32; j++) {
        size_t temp = j;
        for (size_t k = 0; k < num_indices; k++) {
            contracted_indices[j][k] = temp % a->dimensions[indices_a[k]];
            temp /= a->dimensions[indices_a[k]];
        }
    }
    
    // Initialize result tensor
    memset(result->components, 0, aligned_out_size * sizeof(ComplexFloat));
    
    // Allocate thread-local result arrays
    size_t num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    
    ComplexFloat* thread_results = (ComplexFloat*)aligned_alloc(64, num_threads * total_out_size * sizeof(ComplexFloat));
    if (!thread_results) {
        free(a_strides);
        free(b_strides);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    memset(thread_results, 0, num_threads * total_out_size * sizeof(ComplexFloat));
    
    // Process in blocks using a hierarchical blocking strategy
    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        ComplexFloat* local_result = &thread_results[thread_id * total_out_size];
        
        // Thread-local arrays for indices using stack allocation
        size_t a_indices[32];  // Max rank is typically much smaller
        size_t b_indices[32];
        
        // Thread-local buffers for cache blocking
        ComplexFloat l1_buffer[QGT_L1_BLOCK_SIZE] __attribute__((aligned(64)));
        ComplexFloat l2_buffer[QGT_L2_BLOCK_SIZE] __attribute__((aligned(64)));
        ComplexFloat l3_buffer[QGT_L3_BLOCK_SIZE] __attribute__((aligned(64)));
        
        // Use buffers for block processing
        memcpy(l1_buffer, &a->components[0], QGT_L1_BLOCK_SIZE * sizeof(ComplexFloat));
        memcpy(l2_buffer, &b->components[0], QGT_L2_BLOCK_SIZE * sizeof(ComplexFloat));
        memcpy(l3_buffer, result->components, QGT_L3_BLOCK_SIZE * sizeof(ComplexFloat));
        
        // Thread-local buffer for SIMD tiles
        ComplexFloat tile_buffer[tile_size] __attribute__((aligned(64)));
        
        // Process L3 blocks
        #pragma omp for collapse(2) schedule(guided) nowait
        for (size_t out_l3 = 0; out_l3 < total_out_size; out_l3 += l3_block) {
            for (size_t contract_l3 = 0; contract_l3 < contract_size; contract_l3 += l3_block) {
                size_t max_out_l3 = min(out_l3 + l3_block, total_out_size);
                size_t max_contract_l3 = min(contract_l3 + l3_block, contract_size);
                
                // Process L2 blocks within L3 blocks
                for (size_t out_l2 = out_l3; out_l2 < max_out_l3; out_l2 += l2_block) {
                    for (size_t contract_l2 = contract_l3; contract_l2 < max_contract_l3; contract_l2 += l2_block) {
                        size_t max_out_l2 = min(out_l2 + l2_block, max_out_l3);
                        size_t max_contract_l2 = min(contract_l2 + l2_block, max_contract_l3);
                
                        // Process L1 blocks within L2 blocks
                        for (size_t out_l1 = out_l2; out_l1 < max_out_l2; out_l1 += l1_block) {
                            for (size_t contract_l1 = contract_l2; contract_l1 < max_contract_l2; contract_l1 += l1_block) {
                                size_t max_out_l1 = min(out_l1 + l1_block, max_out_l2);
                                size_t max_contract_l1 = min(contract_l1 + l1_block, max_contract_l2);
                        
                                // Process tiles within L1 blocks
                                for (size_t out_tile = out_l1; out_tile < max_out_l1; out_tile += tile_size) {
                                    size_t max_out_tile = min(out_tile + tile_size, max_out_l1);
                                    
                                    // Initialize tile buffer
                                    memset(tile_buffer, 0, tile_size * sizeof(ComplexFloat));
                                    
                                    // Process each element in tile
                                    for (size_t i = out_tile; i < max_out_tile; i++) {
                                        // Initialize uncontracted indices
                                        size_t temp = i;
                                        for (size_t k = 0; k < uncontracted_a_count; k++) {
                                            a_indices[uncontracted_a_buffer[k]] = temp % a->dimensions[uncontracted_a_buffer[k]];
                                            temp /= a->dimensions[uncontracted_a_buffer[k]];
                                        }
                                        
                                        temp = i;
                                        for (size_t k = 0; k < uncontracted_b_count; k++) {
                                            b_indices[uncontracted_b_buffer[k]] = temp % b->dimensions[uncontracted_b_buffer[k]];
                                            temp /= b->dimensions[uncontracted_b_buffer[k]];
                                        }
                                        
                                        // Process contraction indices with vectorization
                                        ComplexFloat sum = {0.0f, 0.0f};
                                        
                                        // Process contraction indices in SIMD tiles
                                        for (size_t j = contract_l1; j < max_contract_l1; j += tile_size) {
                                            size_t max_j = min(j + tile_size, max_contract_l1);
                                            
                                            // Load and prefetch data for the current tile
                                            for (size_t jj = j; jj < max_j; jj++) {
                                                // Set contracted indices from lookup table
                                                for (size_t k = 0; k < num_indices; k++) {
                                                    size_t idx = contracted_indices[jj][k];
                                                    a_indices[indices_a[k]] = idx;
                                                    b_indices[indices_b[k]] = idx;
                                                }
                                                
                                                // Convert linear indices to multi-dimensional indices
                                                linear_to_indices(a_indices, jj, a->dimensions, a->rank);
                                                linear_to_indices(b_indices, jj, b->dimensions, b->rank);
                                                
                                                // Calculate linear indices
                                                size_t a_idx = indices_to_linear(a_indices, a_strides, a->rank);
                                                size_t b_idx = indices_to_linear(b_indices, b_strides, b->rank);
                                                
                                                // Prefetch next elements
                                                if (jj + 1 < max_j) {
                                                    __builtin_prefetch(&a->components[a_idx + 1], 0, 3);
                                                    __builtin_prefetch(&b->components[b_idx + 1], 0, 3);
                                                }
                                                
                                                // Load data into tile buffer
                                                size_t tile_idx = jj - j;
                                                tile_buffer[tile_idx] = complex_float_multiply(
                                                    a->components[a_idx],
                                                    b->components[b_idx]);
                                            }
                                            
                                            // Accumulate tile results using SIMD
                                            #pragma omp simd reduction(+:sum.real,sum.imag) aligned(tile_buffer:64)
                                            for (size_t tile_idx = 0; tile_idx < max_j - j; tile_idx++) {
                                                sum.real += tile_buffer[tile_idx].real;
                                                sum.imag += tile_buffer[tile_idx].imag;
                                            }
                                        }
                                        
                                        // Store result in thread-local buffer
                                        local_result[i].real += sum.real;
                                        local_result[i].imag += sum.imag;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Combine thread results
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total_out_size; i++) {
        ComplexFloat sum = {0.0f, 0.0f};
        for (size_t t = 0; t < num_threads; t++) {
            sum.real += thread_results[t * total_out_size + i].real;
            sum.imag += thread_results[t * total_out_size + i].imag;
        }
        result->components[i] = sum;
    }
    
    free(thread_results);

    // Cleanup
    free(a_strides);
    free(b_strides);

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_scale(quantum_geometric_tensor_t* result,
                                  const quantum_geometric_tensor_t* tensor,
                                  ComplexFloat scalar) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(tensor);
    
    // Process in blocks with prefetching
    const size_t block_size = 1024;
    const size_t num_blocks = tensor->aligned_elements / block_size;
    const size_t remaining = tensor->aligned_elements % block_size;

    for (size_t i = 0; i < num_blocks; i++) {
        size_t offset = i * block_size;
        // Prefetch next block
        if (i + 1 < num_blocks) {
            __builtin_prefetch(&tensor->components[(i + 1) * block_size], 0, 3);
            __builtin_prefetch(&result->components[(i + 1) * block_size], 1, 3);
        }
        simd_tensor_scale(&result->components[offset],
                         &tensor->components[offset],
                         scalar,
                         block_size);
    }

    // Handle remaining elements
    if (remaining > 0) {
        simd_tensor_scale(&result->components[num_blocks * block_size],
                         &tensor->components[num_blocks * block_size],
                         scalar,
                         remaining);
    }

    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_norm(float* norm,
                                 const quantum_geometric_tensor_t* tensor) {
    QGT_CHECK_NULL(norm);
    QGT_CHECK_NULL(tensor);
    
    // Process in blocks with prefetching
    const size_t block_size = 1024;
    const size_t num_blocks = tensor->aligned_elements / block_size;
    const size_t remaining = tensor->aligned_elements % block_size;

    float total_norm = 0.0f;
    
    // Process blocks
    for (size_t i = 0; i < num_blocks; i++) {
        size_t offset = i * block_size;
        // Prefetch next block
        if (i + 1 < num_blocks) {
            __builtin_prefetch(&tensor->components[(i + 1) * block_size], 0, 3);
        }
        float block_norm = simd_tensor_norm(&tensor->components[offset], block_size);
        total_norm += block_norm;
    }

    // Handle remaining elements
    if (remaining > 0) {
        float remaining_norm = simd_tensor_norm(
            &tensor->components[num_blocks * block_size], remaining);
        total_norm += remaining_norm;
    }

    *norm = sqrtf(total_norm);
    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_conjugate(quantum_geometric_tensor_t* result,
                                      const quantum_geometric_tensor_t* tensor) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(tensor);
    
    // Process in blocks with prefetching
    const size_t block_size = 1024;
    const size_t num_blocks = tensor->aligned_elements / block_size;
    const size_t remaining = tensor->aligned_elements % block_size;

    for (size_t i = 0; i < num_blocks; i++) {
        size_t offset = i * block_size;
        // Prefetch next block
        if (i + 1 < num_blocks) {
            __builtin_prefetch(&tensor->components[(i + 1) * block_size], 0, 3);
            __builtin_prefetch(&result->components[(i + 1) * block_size], 1, 3);
        }
        simd_tensor_conjugate(&result->components[offset],
                            &tensor->components[offset],
                            block_size);
    }

    // Handle remaining elements
    if (remaining > 0) {
        simd_tensor_conjugate(&result->components[num_blocks * block_size],
                            &tensor->components[num_blocks * block_size],
                            remaining);
    }

    return QGT_SUCCESS;
}

// Helper function to calculate transposed index
static size_t get_transposed_index(const size_t* indices, const size_t* dims,
                                 const size_t* perm, size_t rank) {
    size_t index = 0;
    size_t stride = 1;
    
    // Calculate index using permuted dimensions
    for (int i = rank - 1; i >= 0; i--) {
        index += indices[perm[i]] * stride;
        stride *= dims[perm[i]];
    }
    return index;
}

qgt_error_t geometric_tensor_transpose(quantum_geometric_tensor_t* result,
                                      const quantum_geometric_tensor_t* tensor,
                                      const size_t* permutation) {
    QGT_CHECK_NULL(result);
    QGT_CHECK_NULL(tensor);
    QGT_CHECK_NULL(permutation);

    // Validate permutation
    bool* used = (bool*)calloc(tensor->rank, sizeof(bool));
    if (!used) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    
    for (size_t i = 0; i < tensor->rank; i++) {
        if (permutation[i] >= tensor->rank || used[permutation[i]]) {
            free(used);
            return QGT_ERROR_INVALID_ARGUMENT;
        }
        used[permutation[i]] = true;
    }
    free(used);

    // Calculate new dimensions
    size_t* new_dims = (size_t*)aligned_alloc(64, tensor->rank * sizeof(size_t));
    if (!new_dims) {
        return QGT_ERROR_MEMORY_ALLOCATION;
    }
    
    for (size_t i = 0; i < tensor->rank; i++) {
        new_dims[i] = tensor->dimensions[permutation[i]];
    }

    // Calculate strides for input and output
    size_t* in_strides = (size_t*)aligned_alloc(64, tensor->rank * sizeof(size_t));
    size_t* out_strides = (size_t*)aligned_alloc(64, tensor->rank * sizeof(size_t));
    if (!in_strides || !out_strides) {
        free(new_dims);
        free(in_strides);
        free(out_strides);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    // Calculate strides
    in_strides[tensor->rank - 1] = 1;
    out_strides[tensor->rank - 1] = 1;
    for (int i = tensor->rank - 2; i >= 0; i--) {
        in_strides[i] = in_strides[i + 1] * tensor->dimensions[i + 1];
        out_strides[i] = out_strides[i + 1] * new_dims[i + 1];
    }

    // Process in blocks for better cache utilization
    const size_t block_size = QGT_L1_BLOCK_SIZE;
    size_t total_size = tensor->total_elements;
    
    // Temporary arrays for indices
    size_t* indices = (size_t*)aligned_alloc(64, tensor->rank * sizeof(size_t));
    if (!indices) {
        free(new_dims);
        free(in_strides);
        free(out_strides);
        return QGT_ERROR_MEMORY_ALLOCATION;
    }

    #pragma omp parallel for schedule(guided)
    for (size_t block_start = 0; block_start < total_size; block_start += block_size) {
        size_t block_end = min(block_start + block_size, total_size);
        size_t local_indices[32]; // Thread-local array for indices
        
        for (size_t i = block_start; i < block_end; i++) {
            // Convert linear index to multi-dimensional indices
            size_t idx = i;
            for (size_t j = 0; j < tensor->rank; j++) {
                local_indices[j] = idx / in_strides[j];
                idx %= in_strides[j];
            }
            
            // Calculate transposed index
            size_t out_idx = get_transposed_index(local_indices, tensor->dimensions,
                                                permutation, tensor->rank);
            
            // Copy element
            result->components[out_idx] = tensor->components[i];
        }
    }

    // Update result dimensions
    memcpy(result->dimensions, new_dims, tensor->rank * sizeof(size_t));
    result->rank = tensor->rank;
    result->total_elements = total_size;
    result->aligned_elements = (total_size + 7) & ~7;

    // Cleanup
    free(new_dims);
    free(in_strides);
    free(out_strides);
    free(indices);

    return QGT_SUCCESS;
}

// Helper function to validate tensor properties
static qgt_error_t validate_tensor_properties(const quantum_geometric_tensor_t* tensor) {
    if (tensor->type == GEOMETRIC_TENSOR_UNITARY && !tensor->is_unitary) {
        if (!validate_unitary(tensor->components, tensor->dimensions[0])) {
            return QGT_ERROR_INVALID_PROPERTY;
        }
    }

    if (tensor->type == GEOMETRIC_TENSOR_HERMITIAN && !tensor->is_hermitian) {
        if (!validate_hermitian(tensor->components, tensor->dimensions[0])) {
            return QGT_ERROR_INVALID_PROPERTY;
        }
    }

    return QGT_SUCCESS;
}

    // Helper function to validate tensor memory alignment
static qgt_error_t validate_memory_alignment(const quantum_geometric_tensor_t* tensor) {
    // Check 64-byte alignment for AVX-512
    if ((uintptr_t)tensor->components % 64 != 0) {
        return QGT_ERROR_INVALID_ALIGNMENT;
    }

    // Dimensions array doesn't need to be aligned
    return QGT_SUCCESS;
}

qgt_error_t geometric_tensor_validate(const quantum_geometric_tensor_t* tensor) {
    QGT_CHECK_NULL(tensor);
    QGT_CHECK_NULL(tensor->dimensions);
    QGT_CHECK_NULL(tensor->components);

    // Validate rank and dimensions
    QGT_CHECK_ARGUMENT(tensor->rank > 0);

    size_t total_elements = 1;
    for (size_t i = 0; i < tensor->rank; i++) {
        QGT_CHECK_ARGUMENT(tensor->dimensions[i] > 0);
        // Check for overflow
        QGT_CHECK_ARGUMENT(total_elements <= SIZE_MAX / tensor->dimensions[i]);
        total_elements *= tensor->dimensions[i];
    }

    // Validate total elements matches stored value
    QGT_CHECK_ARGUMENT(total_elements == tensor->total_elements);

    // Validate aligned elements
    size_t aligned_elements = (total_elements + 7) & ~7;
    QGT_CHECK_ARGUMENT(aligned_elements == tensor->aligned_elements);

    // Validate memory alignment
    qgt_error_t err = validate_memory_alignment(tensor);
    if (err != QGT_SUCCESS) {
        return err;
    }

    // Check numerical stability
    if (!check_numerical_stability(tensor->components, tensor->total_elements)) {
        return QGT_ERROR_NUMERICAL_INSTABILITY;
    }

    // Validate quantum geometric properties
    err = validate_tensor_properties(tensor);
    if (err != QGT_SUCCESS) {
        return err;
    }

    return QGT_SUCCESS;
}

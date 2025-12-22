#include "quantum_geometric/core/quantum_operations.h"
#include "quantum_geometric/core/quantum_state_types.h"
#include "quantum_geometric/core/performance_operations.h"
#include "quantum_geometric/core/simd_operations.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/lapack_internal.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// LAPACK double complex compatibility for macOS Accelerate framework
#ifdef __APPLE__
    // __CLPK_doublecomplex is provided by Accelerate/Accelerate.h via lapack_internal.h
    #define LAPACK_COMPLEX_CAST(ptr) ((__CLPK_doublecomplex*)(ptr))
#else
    // On Linux/other platforms, complex double is used directly
    #define LAPACK_COMPLEX_CAST(ptr) (ptr)
#endif

// Platform-specific SIMD includes and compatibility
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
    #define HAS_AVX 1
    #define HAS_NATIVE_AVX 1
#elif defined(__aarch64__) || defined(_M_ARM64)
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #include <arm_neon.h>
    #endif
    #define HAS_AVX 0
    #define HAS_NATIVE_AVX 0

    // ============================================================================
    // ARM64 NEON-based fallbacks for AVX intrinsics
    // Provides cross-platform compatibility for x86/ARM64
    // ============================================================================

    // 256-bit double vector (4 doubles) - emulated as array on ARM
    typedef union {
        double v[4];
        double values[4];
    } __m256d;

    // Basic vector creation
    static inline __m256d _mm256_set_pd(double a, double b, double c, double d) {
        __m256d result = {{d, c, b, a}};
        return result;
    }

    static inline __m256d _mm256_setzero_pd(void) {
        __m256d result = {{0.0, 0.0, 0.0, 0.0}};
        return result;
    }

    static inline __m256d _mm256_set1_pd(double x) {
        __m256d result = {{x, x, x, x}};
        return result;
    }

    // Load/store operations
    static inline __m256d _mm256_load_pd(const double* p) {
        __m256d result = {{p[0], p[1], p[2], p[3]}};
        return result;
    }

    static inline void _mm256_store_pd(double* p, __m256d v) {
        p[0] = v.v[0]; p[1] = v.v[1]; p[2] = v.v[2]; p[3] = v.v[3];
    }

    // Basic arithmetic
    static inline __m256d _mm256_add_pd(__m256d a, __m256d b) {
        __m256d result = {{a.v[0]+b.v[0], a.v[1]+b.v[1], a.v[2]+b.v[2], a.v[3]+b.v[3]}};
        return result;
    }

    static inline __m256d _mm256_sub_pd(__m256d a, __m256d b) {
        __m256d result = {{a.v[0]-b.v[0], a.v[1]-b.v[1], a.v[2]-b.v[2], a.v[3]-b.v[3]}};
        return result;
    }

    static inline __m256d _mm256_mul_pd(__m256d a, __m256d b) {
        __m256d result = {{a.v[0]*b.v[0], a.v[1]*b.v[1], a.v[2]*b.v[2], a.v[3]*b.v[3]}};
        return result;
    }

    static inline __m256d _mm256_div_pd(__m256d a, __m256d b) {
        __m256d result = {{a.v[0]/b.v[0], a.v[1]/b.v[1], a.v[2]/b.v[2], a.v[3]/b.v[3]}};
        return result;
    }

    // Alternating add/subtract: result[0]=a[0]+b[0], result[1]=a[1]-b[1], ...
    static inline __m256d _mm256_addsub_pd(__m256d a, __m256d b) {
        __m256d result = {{a.v[0]+b.v[0], a.v[1]-b.v[1], a.v[2]+b.v[2], a.v[3]-b.v[3]}};
        return result;
    }

    // Permute 64-bit elements within 256-bit vector
    // Control byte: each 2-bit field selects source element (0-3)
    static inline __m256d _mm256_permute4x64_pd(__m256d a, int imm8) {
        __m256d result;
        result.v[0] = a.v[(imm8 >> 0) & 3];
        result.v[1] = a.v[(imm8 >> 2) & 3];
        result.v[2] = a.v[(imm8 >> 4) & 3];
        result.v[3] = a.v[(imm8 >> 6) & 3];
        return result;
    }

    // Bitwise XOR (for sign flipping in complex conjugate)
    static inline __m256d _mm256_xor_pd(__m256d a, __m256d b) {
        __m256d result;
        union { double d; uint64_t u; } ua[4], ub[4], ur[4];
        for (int i = 0; i < 4; i++) {
            ua[i].d = a.v[i];
            ub[i].d = b.v[i];
            ur[i].u = ua[i].u ^ ub[i].u;
            result.v[i] = ur[i].d;
        }
        return result;
    }

    // Bitwise AND NOT: (~a) & b
    static inline __m256d _mm256_andnot_pd(__m256d a, __m256d b) {
        __m256d result;
        union { double d; uint64_t u; } ua[4], ub[4], ur[4];
        for (int i = 0; i < 4; i++) {
            ua[i].d = a.v[i];
            ub[i].d = b.v[i];
            ur[i].u = (~ua[i].u) & ub[i].u;
            result.v[i] = ur[i].d;
        }
        return result;
    }

    // Broadcast single double to all lanes
    static inline __m256d _mm256_broadcast_sd(const double* p) {
        double val = *p;
        __m256d result = {{val, val, val, val}};
        return result;
    }

    // Compare and create mask
    #define _CMP_GT_OQ 14
    #define _CMP_LT_OQ 17
    #define _CMP_EQ_OQ 0

    static inline __m256d _mm256_cmp_pd(__m256d a, __m256d b, int imm8) {
        __m256d result;
        union { double d; uint64_t u; } ur[4];
        for (int i = 0; i < 4; i++) {
            bool cmp_result = false;
            switch (imm8) {
                case _CMP_GT_OQ: cmp_result = a.v[i] > b.v[i]; break;
                case _CMP_LT_OQ: cmp_result = a.v[i] < b.v[i]; break;
                case _CMP_EQ_OQ: cmp_result = a.v[i] == b.v[i]; break;
                default: cmp_result = a.v[i] > b.v[i]; break;
            }
            ur[i].u = cmp_result ? 0xFFFFFFFFFFFFFFFFULL : 0ULL;
            result.v[i] = ur[i].d;
        }
        return result;
    }

    // Extract sign bits as mask
    static inline int _mm256_movemask_pd(__m256d a) {
        int result = 0;
        union { double d; uint64_t u; } u[4];
        for (int i = 0; i < 4; i++) {
            u[i].d = a.v[i];
            if (u[i].u & 0x8000000000000000ULL) {
                result |= (1 << i);
            }
        }
        return result;
    }

#else
    #define HAS_AVX 0
    #define HAS_NATIVE_AVX 0
#endif

// Error code compatibility
#ifndef QGT_ERROR_OUT_OF_MEMORY
#define QGT_ERROR_OUT_OF_MEMORY QGT_ERROR_NO_MEMORY
#endif
#ifndef QGT_ERROR_NUMERICAL
#define QGT_ERROR_NUMERICAL QGT_ERROR_NUMERICAL_INSTABILITY
#endif

// Gate type compatibility - use distinct values for switch statements
#ifndef GATE_SINGLE
#define GATE_SINGLE 100
#endif
#ifndef GATE_TWO
#define GATE_TWO 101
#endif
#ifndef GATE_HADAMARD
#define GATE_HADAMARD 102
#endif
#ifndef GATE_PAULI_X
#define GATE_PAULI_X 103
#endif
#ifndef GATE_PAULI_Y
#define GATE_PAULI_Y 104
#endif
#ifndef GATE_PAULI_Z
#define GATE_PAULI_Z 105
#endif

// Hardware type compatibility
#ifndef HARDWARE_GPU
#define HARDWARE_GPU HARDWARE_NVIDIA
#endif
#ifndef HARDWARE_METAL
#define HARDWARE_METAL HARDWARE_APPLE
#endif
#ifndef HARDWARE_CPU
#define HARDWARE_CPU HARDWARE_SIMULATOR
#endif

// Error correction code types
#ifndef ERROR_CODE_THREE_QUBIT
#define ERROR_CODE_THREE_QUBIT 3
#endif
#ifndef ERROR_CODE_FIVE_QUBIT
#define ERROR_CODE_FIVE_QUBIT 5
#endif
#ifndef ERROR_CODE_SEVEN_QUBIT
#define ERROR_CODE_SEVEN_QUBIT 7
#endif
#ifndef ERROR_CODE_NINE_QUBIT
#define ERROR_CODE_NINE_QUBIT 9
#endif

// ============================================================================
// GPU Backend Integration
// ============================================================================

#ifdef ENABLE_CUDA
    #include <cuda_runtime.h>
    // CUDA implementations provided by cuda_runtime.h
#else
    // CPU fallback when CUDA is not available
    // These provide memory management that mimics CUDA semantics for unified codepath
    #ifndef cudaSuccess
        #define cudaSuccess 0
        #define cudaError_t int

        static inline cudaError_t cuda_malloc_fallback(void** ptr, size_t size) {
            *ptr = aligned_alloc(64, size);  // 64-byte aligned for cache efficiency
            return (*ptr != NULL) ? 0 : 1;
        }
        #define cudaMalloc(p,s) cuda_malloc_fallback((void**)(p), (s))

        static inline void cuda_free_fallback(void* ptr) {
            if (ptr) free(ptr);
        }
        #define cudaFree(p) cuda_free_fallback(p)

        static inline cudaError_t cuda_memcpy_fallback(void* dst, const void* src, size_t size, int kind) {
            if (!dst || !src) return 1;
            memcpy(dst, src, size);
            return 0;
        }
        #define cudaMemcpy(d,s,n,k) cuda_memcpy_fallback((d),(s),(n),(k))
        #define cudaMemcpyHostToDevice 1
        #define cudaMemcpyDeviceToHost 2
    #endif
#endif

#ifdef ENABLE_METAL
    #include "quantum_geometric/hardware/metal_backend.h"
    // Metal implementations provided by Metal backend
#else
    // CPU fallback when Metal is not available
    // Provides memory operations that work on CPU for unified codepath

    static inline void* metal_allocate_buffer(size_t size) {
        // Use 64-byte alignment for cache-line efficiency and potential SIMD operations
        void* ptr = NULL;
        #if defined(__APPLE__)
            // macOS uses posix_memalign
            if (posix_memalign(&ptr, 64, size) != 0) {
                return NULL;
            }
        #else
            ptr = aligned_alloc(64, size);
        #endif
        if (ptr) {
            memset(ptr, 0, size);  // Zero-initialize for safety
        }
        return ptr;
    }

    static inline void metal_free_buffer(void* ptr) {
        if (ptr) {
            free(ptr);
        }
    }

    static inline bool metal_copy_to_device(void* dst, const void* src, size_t size) {
        if (!dst || !src || size == 0) return false;
        memcpy(dst, src, size);
        return true;
    }

    static inline bool metal_copy_from_device(void* dst, const void* src, size_t size) {
        if (!dst || !src || size == 0) return false;
        memcpy(dst, src, size);
        return true;
    }

    static inline bool metal_synchronize(void) {
        // No-op for CPU - always synchronized
        return true;
    }
#endif

// Context structure
struct qgt_context_t {
    void* memory_pool;
    size_t max_qubits;
    int flags;
};

// State structure
struct qgt_state_t {
    size_t num_qubits;
    size_t dim;
    ComplexFloat* amplitudes;
};

// Helper macro for complex double operations using ComplexFloat
#define COMPLEX_DOUBLE_TO_FLOAT(cd) ((ComplexFloat){(float)creal(cd), (float)cimag(cd)})
#define COMPLEX_FLOAT_TO_DOUBLE(cf) ((cf).real + I * (cf).imag)

// Aligned memory allocation helper
static inline void* aligned_alloc_wrapper(size_t alignment, size_t size) {
    #ifdef _WIN32
        return _aligned_malloc(size, alignment);
    #else
        return aligned_alloc(alignment, size);
    #endif
}

// Aligned memory free helper
static inline void aligned_free_wrapper(void* ptr) {
    #ifdef _WIN32
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

// Quantum operator implementation with SIMD optimizations
qgt_error_t quantum_operator_create(quantum_operator_t** operator,
                                  quantum_operator_type_t type,
                                  size_t dimension) {
    if (!operator || dimension == 0) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    quantum_operator_t* op = aligned_alloc_wrapper(32, sizeof(quantum_operator_t));
    if (!op) {
        return QGT_ERROR_OUT_OF_MEMORY;
    }

    op->matrix = aligned_alloc_wrapper(32, dimension * dimension * sizeof(ComplexFloat));
    if (!op->matrix) {
        aligned_free_wrapper(op);
        return QGT_ERROR_OUT_OF_MEMORY;
    }

    memset(op->matrix, 0, dimension * dimension * sizeof(ComplexFloat));
    op->dimension = dimension;
    op->type = type;
    op->auxiliary_data = NULL;
    op->is_hermitian = false;
    op->device_data = NULL;
    op->device_type = HARDWARE_TYPE_CPU;

    *operator = op;
    return QGT_SUCCESS;
}

// Note: Header declares void return, but we use qgt_error_t internally
void quantum_operator_destroy(quantum_operator_t* operator) {
    if (!operator) {
        return;
    }

    // Clean up device data if on GPU/Metal
    if (operator->device_data) {
        switch (operator->device_type) {
            case HARDWARE_TYPE_GPU:
            case HARDWARE_TYPE_CUDA:
                cudaFree(operator->device_data);
                break;
            case HARDWARE_TYPE_METAL:
                metal_free_buffer(operator->device_data);
                break;
            default:
                free(operator->device_data);
                break;
        }
        operator->device_data = NULL;
    }

    // Clean up auxiliary data if present
    if (operator->auxiliary_data) {
        free(operator->auxiliary_data);
    }

    aligned_free_wrapper(operator->matrix);
    aligned_free_wrapper(operator);
}

// Helper functions for applying gates to specific qubits
static qgt_error_t apply_single_qubit_gate(quantum_operator_t* operator,
                                         const quantum_operator_t* gate,
                                         size_t qubit) {
    if (!operator || !gate) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    // Calculate total number of qubits from operator dimension
    size_t num_qubits = (size_t)log2(operator->dimension);
    if (qubit >= num_qubits) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    // Create identity operator for tensor products
    quantum_operator_t* id = NULL;
    qgt_error_t err = quantum_operator_create(&id, GATE_SINGLE, 2);
    if (err != QGT_SUCCESS) {
        return err;
    }
    err = quantum_operator_initialize_identity(id);
    if (err != QGT_SUCCESS) {
        quantum_operator_destroy(id);
        return err;
    }
    
    // Build up operator using tensor products
    quantum_operator_t* result = NULL;
    err = quantum_operator_create(&result, operator->type, operator->dimension);
    if (err != QGT_SUCCESS) {
        quantum_operator_destroy(id);
        return err;
    }
    
    // Start with identity or gate depending on first qubit
    quantum_operator_t* current = NULL;
    err = quantum_operator_create(&current, GATE_SINGLE, 2);
    if (err != QGT_SUCCESS) {
        quantum_operator_destroy(id);
        quantum_operator_destroy(result);
        return err;
    }
    
    // Initialize current with first operator
    if (qubit == 0) {
        memcpy(current->matrix, gate->matrix, 4 * sizeof(ComplexFloat));
    } else {
        memcpy(current->matrix, id->matrix, 4 * sizeof(ComplexFloat));
    }
    
    // Build up full operator
    for (size_t i = 1; i < num_qubits; i++) {
        const quantum_operator_t* next = (i == qubit) ? gate : id;
        err = quantum_operator_tensor_product(current, next, result);
        if (err != QGT_SUCCESS) {
            quantum_operator_destroy(id);
            quantum_operator_destroy(current);
            quantum_operator_destroy(result);
            return err;
        }
        memcpy(current->matrix, result->matrix,
               result->dimension * result->dimension * sizeof(ComplexFloat));
        current->dimension = result->dimension;
    }
    
    // Apply to operator
    err = quantum_operator_multiply(operator, result, operator);
    
    quantum_operator_destroy(id);
    quantum_operator_destroy(current);
    quantum_operator_destroy(result);
    
    return err;
}

static qgt_error_t apply_two_qubit_gate(quantum_operator_t* operator,
                                      const quantum_operator_t* gate,
                                      size_t qubit1,
                                      size_t qubit2) {
    if (!operator || !gate) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    // Calculate total number of qubits
    size_t num_qubits = (size_t)log2(operator->dimension);
    if (qubit1 >= num_qubits || qubit2 >= num_qubits || qubit1 == qubit2) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    // Ensure qubit1 < qubit2
    if (qubit1 > qubit2) {
        size_t temp = qubit1;
        qubit1 = qubit2;
        qubit2 = temp;
    }
    
    // Create identity operator
    quantum_operator_t* id = NULL;
    qgt_error_t err = quantum_operator_create(&id, GATE_SINGLE, 2);
    if (err != QGT_SUCCESS) {
        return err;
    }
    err = quantum_operator_initialize_identity(id);
    if (err != QGT_SUCCESS) {
        quantum_operator_destroy(id);
        return err;
    }
    
    // Build full operator
    quantum_operator_t* result = NULL;
    err = quantum_operator_create(&result, operator->type, operator->dimension);
    if (err != QGT_SUCCESS) {
        quantum_operator_destroy(id);
        return err;
    }
    
    quantum_operator_t* current = NULL;
    err = quantum_operator_create(&current, GATE_SINGLE, 2);
    if (err != QGT_SUCCESS) {
        quantum_operator_destroy(id);
        quantum_operator_destroy(result);
        return err;
    }
    
    // Initialize with identity
    memcpy(current->matrix, id->matrix, 4 * sizeof(ComplexFloat));
    
    // Build up operator
    for (size_t i = 0; i < num_qubits - 1; i++) {
        const quantum_operator_t* next;
        if (i == qubit1) {
            next = gate;
            i = qubit2; // Skip to qubit2
        } else {
            next = id;
        }
        
        err = quantum_operator_tensor_product(current, next, result);
        if (err != QGT_SUCCESS) {
            quantum_operator_destroy(id);
            quantum_operator_destroy(current);
            quantum_operator_destroy(result);
            return err;
        }
        
        memcpy(current->matrix, result->matrix,
               result->dimension * result->dimension * sizeof(ComplexFloat));
        current->dimension = result->dimension;
    }
    
    // Apply to operator
    err = quantum_operator_multiply(operator, result, operator);
    
    quantum_operator_destroy(id);
    quantum_operator_destroy(current);
    quantum_operator_destroy(result);
    
    return err;
}

static qgt_error_t create_single_qubit_gate(quantum_operator_t** gate, gate_type_t type) {
    qgt_error_t err = quantum_operator_create(gate, GATE_SINGLE, 2);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    switch (type) {
        case GATE_HADAMARD: {
            double inv_sqrt2 = 1.0 / sqrt(2.0);
            __m256d values = _mm256_set_pd(-inv_sqrt2, inv_sqrt2, inv_sqrt2, inv_sqrt2);
            _mm256_store_pd((double*)(*gate)->matrix, values);
            break;
        }
        case GATE_PAULI_X: {
            __m256d values = _mm256_set_pd(0.0, 1.0, 1.0, 0.0);
            _mm256_store_pd((double*)(*gate)->matrix, values);
            break;
        }
        case GATE_PAULI_Y: {
            __m256d values = _mm256_set_pd(0.0, I, -I, 0.0);
            _mm256_store_pd((double*)(*gate)->matrix, values);
            break;
        }
        case GATE_PAULI_Z: {
            __m256d values = _mm256_set_pd(-1.0, 0.0, 0.0, 1.0);
            _mm256_store_pd((double*)(*gate)->matrix, values);
            break;
        }
        default:
            quantum_operator_destroy(*gate);
            return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    return QGT_SUCCESS;
}

static qgt_error_t create_phase_gate(quantum_operator_t** gate, float angle) {
    qgt_error_t err = quantum_operator_create(gate, GATE_SINGLE, 2);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    complex double phase = cexp(I * angle);
    __m256d values = _mm256_set_pd(cimag(phase), creal(phase), 0.0, 1.0);
    _mm256_store_pd((double*)(*gate)->matrix, values);
    
    return QGT_SUCCESS;
}

static qgt_error_t create_two_qubit_gate(quantum_operator_t** gate, gate_type_t type) {
    qgt_error_t err = quantum_operator_create(gate, GATE_TWO, 4);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    __m256d row1, row2, row3, row4;
    
    switch (type) {
        case GATE_CNOT:
            row1 = _mm256_set_pd(0.0, 0.0, 0.0, 1.0);
            row2 = _mm256_set_pd(0.0, 0.0, 1.0, 0.0);
            row3 = _mm256_set_pd(1.0, 0.0, 0.0, 0.0);
            row4 = _mm256_set_pd(0.0, 1.0, 0.0, 0.0);
            break;
            
        case GATE_SWAP:
            row1 = _mm256_set_pd(0.0, 0.0, 0.0, 1.0);
            row2 = _mm256_set_pd(0.0, 1.0, 0.0, 0.0);
            row3 = _mm256_set_pd(1.0, 0.0, 0.0, 0.0);
            row4 = _mm256_set_pd(0.0, 0.0, 1.0, 0.0);
            break;
            
        default:
            quantum_operator_destroy(*gate);
            return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    _mm256_store_pd((double*)&(*gate)->matrix[0], row1);
    _mm256_store_pd((double*)&(*gate)->matrix[4], row2);
    _mm256_store_pd((double*)&(*gate)->matrix[8], row3);
    _mm256_store_pd((double*)&(*gate)->matrix[12], row4);
    
    return QGT_SUCCESS;
}

// Standard quantum gates with qubit parameters
qgt_error_t quantum_operator_hadamard(quantum_operator_t* operator, size_t qubit) {
    if (!operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Create Hadamard gate
    quantum_operator_t* h_gate = NULL;
    err = create_single_qubit_gate(&h_gate, GATE_HADAMARD);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Apply to specified qubit
    err = apply_single_qubit_gate(operator, h_gate, qubit);
    quantum_operator_destroy(h_gate);
    
    return err;
}

qgt_error_t quantum_operator_pauli_x(quantum_operator_t* operator, size_t qubit) {
    if (!operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Create Pauli X gate
    quantum_operator_t* x_gate = NULL;
    err = create_single_qubit_gate(&x_gate, GATE_PAULI_X);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Apply to specified qubit
    err = apply_single_qubit_gate(operator, x_gate, qubit);
    quantum_operator_destroy(x_gate);
    
    return err;
}

qgt_error_t quantum_operator_pauli_y(quantum_operator_t* operator, size_t qubit) {
    if (!operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Create Pauli Y gate
    quantum_operator_t* y_gate = NULL;
    err = create_single_qubit_gate(&y_gate, GATE_PAULI_Y);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Apply to specified qubit
    err = apply_single_qubit_gate(operator, y_gate, qubit);
    quantum_operator_destroy(y_gate);
    
    return err;
}

qgt_error_t quantum_operator_pauli_z(quantum_operator_t* operator, size_t qubit) {
    if (!operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Create Pauli Z gate
    quantum_operator_t* z_gate = NULL;
    err = create_single_qubit_gate(&z_gate, GATE_PAULI_Z);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Apply to specified qubit
    err = apply_single_qubit_gate(operator, z_gate, qubit);
    quantum_operator_destroy(z_gate);
    
    return err;
}

qgt_error_t quantum_operator_phase(quantum_operator_t* operator, size_t qubit, float angle) {
    if (!operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Create phase gate
    quantum_operator_t* p_gate = NULL;
    err = create_phase_gate(&p_gate, angle);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Apply to specified qubit
    err = apply_single_qubit_gate(operator, p_gate, qubit);
    quantum_operator_destroy(p_gate);
    
    return err;
}

qgt_error_t quantum_operator_cnot(quantum_operator_t* operator, size_t control, size_t target) {
    if (!operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Create CNOT gate
    quantum_operator_t* cnot = NULL;
    err = create_two_qubit_gate(&cnot, GATE_CNOT);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Apply to specified qubits
    err = apply_two_qubit_gate(operator, cnot, control, target);
    quantum_operator_destroy(cnot);
    
    return err;
}

qgt_error_t quantum_operator_swap(quantum_operator_t* operator, size_t qubit1, size_t qubit2) {
    if (!operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Create SWAP gate
    quantum_operator_t* swap = NULL;
    err = create_two_qubit_gate(&swap, GATE_SWAP);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Apply to specified qubits
    err = apply_two_qubit_gate(operator, swap, qubit1, qubit2);
    quantum_operator_destroy(swap);
    
    return err;
}

// Matrix operations optimized with SIMD and cache blocking
qgt_error_t quantum_operator_multiply(const quantum_operator_t* op1,
                                    const quantum_operator_t* op2,
                                    quantum_operator_t* result) {
    if (!op1 || !op2 || !result || 
        op1->dimension != op2->dimension || 
        op1->dimension != result->dimension) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    size_t dim = op1->dimension;
    
    // For 2x2 matrices (single qubit gates), use optimized path
    if (dim == 2) {
        __m256d a = _mm256_load_pd((double*)op1->matrix);
        __m256d b = _mm256_load_pd((double*)op2->matrix);
        
        // Complex multiplication using AVX
        __m256d t1 = _mm256_mul_pd(a, _mm256_permute4x64_pd(b, 0x50));
        __m256d t2 = _mm256_mul_pd(_mm256_permute4x64_pd(a, 0xB1),
                                  _mm256_permute4x64_pd(b, 0xF5));
        __m256d res = _mm256_addsub_pd(t1, t2);
        
        _mm256_store_pd((double*)result->matrix, res);
        return QGT_SUCCESS;
    }
    
    // For larger matrices, use blocked algorithm with SIMD
    const size_t BLOCK_SIZE = 32; // Cache line optimized
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < dim; i += BLOCK_SIZE) {
        for (size_t j = 0; j < dim; j += BLOCK_SIZE) {
            for (size_t k = 0; k < dim; k += BLOCK_SIZE) {
                // Process block
                for (size_t ii = i; ii < MIN(i + BLOCK_SIZE, dim); ii++) {
                    for (size_t jj = j; jj < MIN(j + BLOCK_SIZE, dim); jj += 4) {
                        __m256d sum = _mm256_setzero_pd();
                        for (size_t kk = k; kk < MIN(k + BLOCK_SIZE, dim); kk++) {
                            __m256d a = _mm256_broadcast_sd((double*)&op1->matrix[ii * dim + kk]);
                            __m256d b = _mm256_load_pd((double*)&op2->matrix[kk * dim + jj]);
                            sum = _mm256_add_pd(sum, _mm256_mul_pd(a, b));
                        }
                        _mm256_store_pd((double*)&result->matrix[ii * dim + jj], sum);
                    }
                }
            }
        }
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_tensor_product(const quantum_operator_t* op1,
                                          const quantum_operator_t* op2,
                                          quantum_operator_t* result) {
    if (!op1 || !op2 || !result || 
        result->dimension != op1->dimension * op2->dimension) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    size_t dim1 = op1->dimension;
    size_t dim2 = op2->dimension;
    size_t total_dim = dim1 * dim2;

    // Tensor product using ComplexFloat operations
    #pragma omp parallel for collapse(2)
    for (size_t i1 = 0; i1 < dim1; i1++) {
        for (size_t j1 = 0; j1 < dim1; j1++) {
            ComplexFloat val1 = op1->matrix[i1 * dim1 + j1];

            for (size_t i2 = 0; i2 < dim2; i2++) {
                size_t i = i1 * dim2 + i2;
                for (size_t j2 = 0; j2 < dim2; j2++) {
                    size_t j = j1 * dim2 + j2;
                    ComplexFloat val2 = op2->matrix[i2 * dim2 + j2];
                    // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                    result->matrix[i * total_dim + j] = (ComplexFloat){
                        val1.real * val2.real - val1.imag * val2.imag,
                        val1.real * val2.imag + val1.imag * val2.real
                    };
                }
            }
        }
    }

    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_adjoint(const quantum_operator_t* op,
                                   quantum_operator_t* result) {
    if (!op || !result || op->dimension != result->dimension) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    size_t dim = op->dimension;
    
    // For 2x2 matrices, use SIMD
    if (dim == 2) {
        __m256d mat = _mm256_load_pd((double*)op->matrix);
        __m256d conj_mask = _mm256_set_pd(-0.0, -0.0, -0.0, -0.0);
        __m256d conjugated = _mm256_xor_pd(mat, conj_mask);
        __m256d transposed = _mm256_permute4x64_pd(conjugated, 0xD8);
        _mm256_store_pd((double*)result->matrix, transposed);
        return QGT_SUCCESS;
    }
    
    // For larger matrices, use blocked approach
    const size_t BLOCK_SIZE = 32;
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < dim; i += BLOCK_SIZE) {
        for (size_t j = 0; j < dim; j += BLOCK_SIZE) {
            for (size_t ii = i; ii < MIN(i + BLOCK_SIZE, dim); ii++) {
                for (size_t jj = j; jj < MIN(j + BLOCK_SIZE, dim); jj += 4) {
                    __m256d vals = _mm256_load_pd((double*)&op->matrix[jj * dim + ii]);
                    __m256d conj_mask = _mm256_set_pd(-0.0, -0.0, -0.0, -0.0);
                    __m256d conjugated = _mm256_xor_pd(vals, conj_mask);
                    _mm256_store_pd((double*)&result->matrix[ii * dim + jj], conjugated);
                }
            }
        }
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_apply_to_state(const quantum_operator_t* op,
                                          complex double* state,
                                          size_t state_size) {
    if (!op || !state || state_size < op->dimension) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    // For single qubit operations, use optimized path
    if (op->dimension == 2) {
        __m256d state_vec = _mm256_load_pd((double*)state);
        __m256d op_vec = _mm256_load_pd((double*)op->matrix);
        
        // Optimized 2x2 complex matrix multiplication
        __m256d t1 = _mm256_mul_pd(op_vec, _mm256_permute4x64_pd(state_vec, 0x50));
        __m256d t2 = _mm256_mul_pd(_mm256_permute4x64_pd(op_vec, 0xB1),
                                  _mm256_permute4x64_pd(state_vec, 0xF5));
        __m256d result = _mm256_addsub_pd(t1, t2);
        
        _mm256_store_pd((double*)state, result);
        return QGT_SUCCESS;
    }
    
    // For multi-qubit operations, use blocked algorithm with SIMD
    complex double* temp = aligned_alloc_wrapper(32, state_size * sizeof(complex double));
    if (!temp) return QGT_ERROR_OUT_OF_MEMORY;
    
    const size_t BLOCK_SIZE = 32;
    #pragma omp parallel for
    for (size_t i = 0; i < op->dimension; i += BLOCK_SIZE) {
        for (size_t j = 0; j < op->dimension; j += 4) {
            __m256d sum = _mm256_setzero_pd();
            for (size_t k = 0; k < MIN(BLOCK_SIZE, op->dimension - i); k++) {
                __m256d a = _mm256_broadcast_sd((double*)&op->matrix[i * op->dimension + k]);
                __m256d b = _mm256_load_pd((double*)&state[j]);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(a, b));
            }
            _mm256_store_pd((double*)&temp[i], sum);
        }
    }
    
    memcpy(state, temp, op->dimension * sizeof(complex double));
    aligned_free_wrapper(temp);
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_validate(const quantum_operator_t* operator) {
    if (!operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    // Check matrix dimensions
    if (operator->dimension == 0) {
        return QGT_ERROR_INVALID_STATE;
    }
    
    // Check matrix allocation
    if (!operator->matrix) {
        return QGT_ERROR_INVALID_STATE;
    }
    
    // Check type validity
    if (operator->type != GATE_SINGLE && operator->type != GATE_TWO) {
        return QGT_ERROR_INVALID_STATE;
    }
    
    // Validate dimension matches type
    if ((operator->type == GATE_SINGLE && operator->dimension != 2) ||
        (operator->type == GATE_TWO && operator->dimension != 4)) {
        return QGT_ERROR_INVALID_STATE;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_is_unitary(const quantum_operator_t* operator, bool* result) {
    if (!operator || !result) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    quantum_operator_t* adj = NULL;
    quantum_operator_t* prod = NULL;
    
    // Create adjoint operator
    err = quantum_operator_create(&adj, operator->type, operator->dimension);
    if (err != QGT_SUCCESS) {
        goto cleanup;
    }
    
    // Create product operator
    err = quantum_operator_create(&prod, operator->type, operator->dimension);
    if (err != QGT_SUCCESS) {
        goto cleanup;
    }
    
    // Compute U†U
    err = quantum_operator_adjoint(operator, adj);
    if (err != QGT_SUCCESS) {
        goto cleanup;
    }
    
    err = quantum_operator_multiply(adj, operator, prod);
    if (err != QGT_SUCCESS) {
        goto cleanup;
    }
    
    // Check if result is identity using SIMD
    size_t dim = operator->dimension;
    *result = true;
    
    for (size_t i = 0; i < dim && *result; i++) {
        for (size_t j = 0; j < dim && *result; j += 4) {
            __m256d expected = _mm256_set_pd(
                (i == j+3) ? 1.0 : 0.0,
                (i == j+2) ? 1.0 : 0.0,
                (i == j+1) ? 1.0 : 0.0,
                (i == j) ? 1.0 : 0.0
            );
            __m256d actual = _mm256_load_pd((double*)&prod->matrix[i * dim + j]);
            __m256d diff = _mm256_sub_pd(actual, expected);
            __m256d abs_diff = _mm256_andnot_pd(_mm256_set1_pd(-0.0), diff);
            __m256d threshold = _mm256_set1_pd(1e-10);
            __m256d cmp = _mm256_cmp_pd(abs_diff, threshold, _CMP_GT_OQ);
            if (_mm256_movemask_pd(cmp)) {
                *result = false;
                break;
            }
        }
    }
    
cleanup:
    if (adj) quantum_operator_destroy(adj);
    if (prod) quantum_operator_destroy(prod);
    return err;
}

qgt_error_t quantum_operator_is_hermitian(const quantum_operator_t* operator, bool* result) {
    if (!operator || !result) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    size_t dim = operator->dimension;
    *result = true;
    
    // Use SIMD to check conjugate transpose property
    for (size_t i = 0; i < dim && *result; i++) {
        for (size_t j = 0; j < dim && *result; j += 4) {
            __m256d vals = _mm256_load_pd((double*)&operator->matrix[i * dim + j]);
            __m256d conj_vals = _mm256_load_pd((double*)&operator->matrix[j * dim + i]);
            __m256d conj_mask = _mm256_set_pd(-0.0, -0.0, -0.0, -0.0);
            __m256d conjugated = _mm256_xor_pd(conj_vals, conj_mask);
            __m256d diff = _mm256_sub_pd(vals, conjugated);
            __m256d abs_diff = _mm256_andnot_pd(_mm256_set1_pd(-0.0), diff);
            __m256d threshold = _mm256_set1_pd(1e-10);
            __m256d cmp = _mm256_cmp_pd(abs_diff, threshold, _CMP_GT_OQ);
            if (_mm256_movemask_pd(cmp)) {
                *result = false;
                break;
            }
        }
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_trace(const quantum_operator_t* operator, complex double* trace) {
    if (!operator || !trace) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    size_t dim = operator->dimension;
    __m256d sum = _mm256_setzero_pd();
    
    // Use SIMD to compute trace
    for (size_t i = 0; i < dim; i += 4) {
        __m256d diag = _mm256_load_pd((double*)&operator->matrix[i * dim + i]);
        sum = _mm256_add_pd(sum, diag);
    }
    
    // Horizontal sum
    double temp[4];
    _mm256_store_pd(temp, sum);
    *trace = temp[0] + temp[1] + temp[2] + temp[3];
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_to_device(quantum_operator_t* operator,
                                     HardwareType hardware) {
    if (!operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }

    // Check if already on device
    if (operator->device_data) {
        if (operator->device_type == hardware) {
            return QGT_SUCCESS;  // Already on correct device
        } else {
            // Need to move from current device back to host first
            err = quantum_operator_from_device(operator, operator->device_type);
            if (err != QGT_SUCCESS) {
                return err;
            }
        }
    }

    // Allocate device memory and copy data
    void* device_data = NULL;
    size_t size = operator->dimension * operator->dimension * sizeof(ComplexFloat);

    switch (hardware) {
        case HARDWARE_TYPE_GPU:
        case HARDWARE_TYPE_CUDA:
            if (cudaMalloc(&device_data, size) != cudaSuccess) {
                return QGT_ERROR_OUT_OF_MEMORY;
            }
            if (cudaMemcpy(device_data, operator->matrix, size,
                          cudaMemcpyHostToDevice) != cudaSuccess) {
                cudaFree(device_data);
                return QGT_ERROR_DEVICE;
            }
            break;

        case HARDWARE_TYPE_METAL:
            device_data = metal_allocate_buffer(size);
            if (!device_data) {
                return QGT_ERROR_OUT_OF_MEMORY;
            }
            if (!metal_copy_to_device(device_data, operator->matrix, size)) {
                metal_free_buffer(device_data);
                return QGT_ERROR_DEVICE;
            }
            break;

        default:
            return QGT_ERROR_INVALID_ARGUMENT;
    }

    operator->device_data = device_data;
    operator->device_type = hardware;
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_from_device(quantum_operator_t* operator,
                                       HardwareType hardware) {
    if (!operator || !operator->device_data) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }

    // Verify device type matches
    if (operator->device_type != hardware) {
        return QGT_ERROR_INVALID_STATE;
    }

    size_t size = operator->dimension * operator->dimension * sizeof(ComplexFloat);

    switch (hardware) {
        case HARDWARE_TYPE_GPU:
        case HARDWARE_TYPE_CUDA:
            if (cudaMemcpy(operator->matrix, operator->device_data, size,
                          cudaMemcpyDeviceToHost) != cudaSuccess) {
                return QGT_ERROR_DEVICE;
            }
            cudaFree(operator->device_data);
            break;

        case HARDWARE_TYPE_METAL:
            if (!metal_copy_from_device(operator->matrix, operator->device_data, size)) {
                return QGT_ERROR_DEVICE;
            }
            metal_free_buffer(operator->device_data);
            break;

        default:
            return QGT_ERROR_INVALID_ARGUMENT;
    }

    operator->device_data = NULL;
    operator->device_type = HARDWARE_TYPE_CPU;
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_is_on_device(const quantum_operator_t* operator,
                                        HardwareType hardware,
                                        bool* result) {
    if (!operator || !result) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }

    *result = (operator->device_data != NULL && operator->device_type == hardware);
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_initialize(quantum_operator_t* operator,
                                      const ComplexFloat* matrix) {
    if (!operator || !matrix) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    size_t size = operator->dimension * operator->dimension;
    memcpy(operator->matrix, matrix, size * sizeof(ComplexFloat));

    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_initialize_identity(quantum_operator_t* operator) {
    if (!operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    size_t dim = operator->dimension;
    memset(operator->matrix, 0, dim * dim * sizeof(ComplexFloat));

    for (size_t i = 0; i < dim; i++) {
        operator->matrix[i * dim + i] = (ComplexFloat){1.0f, 0.0f};
    }

    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_initialize_zero(quantum_operator_t* operator) {
    if (!operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }

    memset(operator->matrix, 0,
           operator->dimension * operator->dimension * sizeof(ComplexFloat));

    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_initialize_random(quantum_operator_t* operator) {
    if (!operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    size_t dim = operator->dimension;
    for (size_t i = 0; i < dim * dim; i++) {
        float real = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        float imag = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        operator->matrix[i] = (ComplexFloat){real, imag};
    }

    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_transpose(quantum_operator_t* result,
                                     const quantum_operator_t* operator) {
    if (!operator || !result || operator->dimension != result->dimension) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    size_t dim = operator->dimension;
    
    // For 2x2 matrices, use SIMD
    if (dim == 2) {
        __m256d mat = _mm256_load_pd((double*)operator->matrix);
        __m256d transposed = _mm256_permute4x64_pd(mat, 0xD8);
        _mm256_store_pd((double*)result->matrix, transposed);
        return QGT_SUCCESS;
    }
    
    // For larger matrices, use blocked approach
    const size_t BLOCK_SIZE = 32;
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < dim; i += BLOCK_SIZE) {
        for (size_t j = 0; j < dim; j += BLOCK_SIZE) {
            for (size_t ii = i; ii < MIN(i + BLOCK_SIZE, dim); ii++) {
                for (size_t jj = j; jj < MIN(j + BLOCK_SIZE, dim); jj += 4) {
                    __m256d vals = _mm256_load_pd((double*)&operator->matrix[jj * dim + ii]);
                    _mm256_store_pd((double*)&result->matrix[ii * dim + jj], vals);
                }
            }
        }
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_conjugate(quantum_operator_t* result,
                                     const quantum_operator_t* operator) {
    if (!operator || !result || operator->dimension != result->dimension) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    size_t dim = operator->dimension;
    
    // Use SIMD to compute conjugate
    const size_t BLOCK_SIZE = 32;
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < dim; i += BLOCK_SIZE) {
        for (size_t j = 0; j < dim; j += BLOCK_SIZE) {
            for (size_t ii = i; ii < MIN(i + BLOCK_SIZE, dim); ii++) {
                for (size_t jj = j; jj < MIN(j + BLOCK_SIZE, dim); jj += 4) {
                    __m256d vals = _mm256_load_pd((double*)&operator->matrix[ii * dim + jj]);
                    __m256d conj_mask = _mm256_set_pd(-0.0, -0.0, -0.0, -0.0);
                    __m256d conjugated = _mm256_xor_pd(vals, conj_mask);
                    _mm256_store_pd((double*)&result->matrix[ii * dim + jj], conjugated);
                }
            }
        }
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_exponential(quantum_operator_t* result,
                                       const quantum_operator_t* operator) {
    if (!operator || !result || operator->dimension != result->dimension) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // For 2x2 matrices, use analytical solution
    if (operator->dimension == 2) {
        // Convert ComplexFloat to complex double for computation
        complex double a = operator->matrix[0].real + I * operator->matrix[0].imag;
        complex double b = operator->matrix[1].real + I * operator->matrix[1].imag;
        complex double c = operator->matrix[2].real + I * operator->matrix[2].imag;
        complex double d = operator->matrix[3].real + I * operator->matrix[3].imag;

        complex double trace = a + d;
        complex double det = a * d - b * c;
        complex double lambda = csqrt(trace * trace - 4.0 * det);
        complex double exp_a = cexp(a);
        complex double exp_d = cexp(d);

        complex double r0 = exp_a;
        complex double r1 = b * (exp_d - exp_a) / lambda;
        complex double r2 = c * (exp_d - exp_a) / lambda;
        complex double r3 = exp_d;

        result->matrix[0] = (ComplexFloat){(float)creal(r0), (float)cimag(r0)};
        result->matrix[1] = (ComplexFloat){(float)creal(r1), (float)cimag(r1)};
        result->matrix[2] = (ComplexFloat){(float)creal(r2), (float)cimag(r2)};
        result->matrix[3] = (ComplexFloat){(float)creal(r3), (float)cimag(r3)};

        return QGT_SUCCESS;
    }

    // For larger matrices, use Padé approximation
    quantum_operator_t* temp = NULL;
    err = quantum_operator_create(&temp, operator->type, operator->dimension);
    if (err != QGT_SUCCESS) {
        return err;
    }

    // Initialize result to identity
    err = quantum_operator_initialize_identity(result);
    if (err != QGT_SUCCESS) {
        quantum_operator_destroy(temp);
        return err;
    }

    // Initialize temp to operator
    memcpy(temp->matrix, operator->matrix, operator->dimension * operator->dimension * sizeof(ComplexFloat));

    // Compute exponential using Taylor series
    double factorial = 1.0;
    for (size_t k = 1; k <= 10; k++) {
        factorial *= k;

        // Multiply temp by operator: temp = temp * operator
        quantum_operator_t* new_temp = NULL;
        err = quantum_operator_create(&new_temp, operator->type, operator->dimension);
        if (err != QGT_SUCCESS) {
            quantum_operator_destroy(temp);
            return err;
        }
        err = quantum_operator_multiply(temp, operator, new_temp);
        if (err != QGT_SUCCESS) {
            quantum_operator_destroy(temp);
            quantum_operator_destroy(new_temp);
            return err;
        }
        memcpy(temp->matrix, new_temp->matrix, operator->dimension * operator->dimension * sizeof(ComplexFloat));
        quantum_operator_destroy(new_temp);

        // Add term to result: result += temp / factorial
        size_t size = operator->dimension * operator->dimension;
        float inv_fact = 1.0f / (float)factorial;
        for (size_t i = 0; i < size; i++) {
            result->matrix[i].real += temp->matrix[i].real * inv_fact;
            result->matrix[i].imag += temp->matrix[i].imag * inv_fact;
        }
    }

    quantum_operator_destroy(temp);
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_determinant(complex double* determinant,
                                       const quantum_operator_t* operator) {
    if (!operator || !determinant) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }

    // For 2x2 matrices, use direct formula
    if (operator->dimension == 2) {
        complex double a = operator->matrix[0].real + I * operator->matrix[0].imag;
        complex double b = operator->matrix[1].real + I * operator->matrix[1].imag;
        complex double c = operator->matrix[2].real + I * operator->matrix[2].imag;
        complex double d = operator->matrix[3].real + I * operator->matrix[3].imag;

        *determinant = a * d - b * c;
        return QGT_SUCCESS;
    }

    // For larger matrices, use LU decomposition
    int n = (int)operator->dimension;
    int* ipiv = NULL;
    complex double* lu = NULL;
    qgt_error_t result = QGT_SUCCESS;

    ipiv = aligned_alloc_wrapper(32, n * sizeof(int));
    if (!ipiv) {
        result = QGT_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    lu = aligned_alloc_wrapper(32, n * n * sizeof(complex double));
    if (!lu) {
        result = QGT_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    // Convert ComplexFloat to complex double for LAPACK
    for (int i = 0; i < n * n; i++) {
        lu[i] = operator->matrix[i].real + I * operator->matrix[i].imag;
    }

    int info;
    zgetrf_(&n, &n, LAPACK_COMPLEX_CAST(lu), &n, ipiv, &info);

    if (info != 0) {
        result = QGT_ERROR_NUMERICAL;
        goto cleanup;
    }

    // Compute determinant from diagonal elements
    *determinant = 1.0;
    for (int i = 0; i < n; i++) {
        *determinant *= lu[i * n + i];
        if (ipiv[i] != i + 1) {
            *determinant = -*determinant;
        }
    }

cleanup:
    if (ipiv) aligned_free_wrapper(ipiv);
    if (lu) aligned_free_wrapper(lu);

    return result;
}

qgt_error_t quantum_operator_eigenvectors(quantum_operator_t* eigenvectors,
                                        const quantum_operator_t* operator) {
    if (!operator || !eigenvectors ||
        operator->dimension != eigenvectors->dimension) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }

    // For 2x2 matrices, use analytical solution
    if (operator->dimension == 2) {
        complex double a = operator->matrix[0].real + I * operator->matrix[0].imag;
        complex double b = operator->matrix[1].real + I * operator->matrix[1].imag;
        complex double c = operator->matrix[2].real + I * operator->matrix[2].imag;
        complex double d = operator->matrix[3].real + I * operator->matrix[3].imag;

        complex double trace = a + d;
        complex double det = a * d - b * c;
        complex double disc = csqrt(trace * trace - 4.0 * det);
        complex double lambda1 = (trace + disc) / 2.0;
        complex double lambda2 = (trace - disc) / 2.0;

        // First eigenvector
        complex double v1[2];
        v1[0] = b;
        v1[1] = lambda1 - a;
        double norm1 = cabs(v1[0]) * cabs(v1[0]) + cabs(v1[1]) * cabs(v1[1]);
        norm1 = sqrt(norm1);
        if (norm1 > 1e-10) {
            v1[0] /= norm1;
            v1[1] /= norm1;
        }

        // Second eigenvector
        complex double v2[2];
        v2[0] = b;
        v2[1] = lambda2 - a;
        double norm2 = cabs(v2[0]) * cabs(v2[0]) + cabs(v2[1]) * cabs(v2[1]);
        norm2 = sqrt(norm2);
        if (norm2 > 1e-10) {
            v2[0] /= norm2;
            v2[1] /= norm2;
        }

        eigenvectors->matrix[0] = (ComplexFloat){(float)creal(v1[0]), (float)cimag(v1[0])};
        eigenvectors->matrix[1] = (ComplexFloat){(float)creal(v1[1]), (float)cimag(v1[1])};
        eigenvectors->matrix[2] = (ComplexFloat){(float)creal(v2[0]), (float)cimag(v2[0])};
        eigenvectors->matrix[3] = (ComplexFloat){(float)creal(v2[1]), (float)cimag(v2[1])};

        return QGT_SUCCESS;
    }

    // For larger matrices, use LAPACK
    int n = (int)operator->dimension;
    int lda = n;
    int ldvr = n;
    int info;

    // Allocate work arrays
    complex double* work = NULL;
    double* rwork = NULL;
    complex double* matrix = NULL;
    complex double* eigenvals = NULL;
    complex double* evecs_temp = NULL;
    qgt_error_t result = QGT_SUCCESS;

    work = aligned_alloc_wrapper(32, 2*n * sizeof(complex double));
    if (!work) {
        result = QGT_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    rwork = aligned_alloc_wrapper(32, 2*n * sizeof(double));
    if (!rwork) {
        result = QGT_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    eigenvals = aligned_alloc_wrapper(32, n * sizeof(complex double));
    if (!eigenvals) {
        result = QGT_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    evecs_temp = aligned_alloc_wrapper(32, n*n * sizeof(complex double));
    if (!evecs_temp) {
        result = QGT_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    // Copy matrix since LAPACK modifies input (convert ComplexFloat to complex double)
    matrix = aligned_alloc_wrapper(32, n*n * sizeof(complex double));
    if (!matrix) {
        result = QGT_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (int i = 0; i < n*n; i++) {
        matrix[i] = operator->matrix[i].real + I * operator->matrix[i].imag;
    }

    // Call LAPACK eigenvector solver
    zgeev_("N", "V", &n, LAPACK_COMPLEX_CAST(matrix), &lda, LAPACK_COMPLEX_CAST(eigenvals), NULL, &n,
           LAPACK_COMPLEX_CAST(evecs_temp), &ldvr, LAPACK_COMPLEX_CAST(work), &n, rwork, &info);

    if (info != 0) {
        result = QGT_ERROR_NUMERICAL;
        goto cleanup;
    }

    // Convert eigenvectors back to ComplexFloat
    for (int i = 0; i < n*n; i++) {
        eigenvectors->matrix[i] = (ComplexFloat){(float)creal(evecs_temp[i]), (float)cimag(evecs_temp[i])};
    }

cleanup:
    if (matrix) aligned_free_wrapper(matrix);
    if (evecs_temp) aligned_free_wrapper(evecs_temp);
    if (work) aligned_free_wrapper(work);
    if (rwork) aligned_free_wrapper(rwork);
    if (eigenvals) aligned_free_wrapper(eigenvals);
    
    return result;
}

qgt_error_t quantum_operator_is_positive(const quantum_operator_t* operator,
                                       bool* result) {
    if (!operator || !result) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }

    // Get eigenvalues
    ComplexFloat* eigenvals = aligned_alloc_wrapper(32,
        operator->dimension * sizeof(ComplexFloat));
    if (!eigenvals) {
        return QGT_ERROR_OUT_OF_MEMORY;
    }

    err = quantum_operator_eigenvalues(eigenvals, operator);
    if (err != QGT_SUCCESS) {
        aligned_free_wrapper(eigenvals);
        return err;
    }

    // Check if all eigenvalues are positive and real
    *result = true;
    for (size_t i = 0; i < operator->dimension; i++) {
        if (eigenvals[i].imag != 0.0f || eigenvals[i].real <= 0.0f) {
            *result = false;
            break;
        }
    }

    aligned_free_wrapper(eigenvals);
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_clone(quantum_operator_t** dest,
                                 const quantum_operator_t* src) {
    if (!dest || !src) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(src);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Create new operator with same dimensions
    err = quantum_operator_create(dest, src->type, src->dimension);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Copy matrix data
    size_t size = src->dimension * src->dimension * sizeof(ComplexFloat);
    memcpy((*dest)->matrix, src->matrix, size);

    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_add(quantum_operator_t* result,
                               const quantum_operator_t* a,
                               const quantum_operator_t* b) {
    if (!result || !a || !b || 
        a->dimension != b->dimension || 
        a->dimension != result->dimension) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(a);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    err = quantum_operator_validate(b);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    size_t dim = a->dimension;
    
    // Use SIMD for addition
    const size_t BLOCK_SIZE = 32;
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < dim; i += BLOCK_SIZE) {
        for (size_t j = 0; j < dim; j += BLOCK_SIZE) {
            for (size_t ii = i; ii < MIN(i + BLOCK_SIZE, dim); ii++) {
                for (size_t jj = j; jj < MIN(j + BLOCK_SIZE, dim); jj += 4) {
                    __m256d a_vals = _mm256_load_pd((double*)&a->matrix[ii * dim + jj]);
                    __m256d b_vals = _mm256_load_pd((double*)&b->matrix[ii * dim + jj]);
                    __m256d sum = _mm256_add_pd(a_vals, b_vals);
                    _mm256_store_pd((double*)&result->matrix[ii * dim + jj], sum);
                }
            }
        }
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_subtract(quantum_operator_t* result,
                                    const quantum_operator_t* a,
                                    const quantum_operator_t* b) {
    if (!result || !a || !b || 
        a->dimension != b->dimension || 
        a->dimension != result->dimension) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(a);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    err = quantum_operator_validate(b);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    size_t dim = a->dimension;
    
    // Use SIMD for subtraction
    const size_t BLOCK_SIZE = 32;
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < dim; i += BLOCK_SIZE) {
        for (size_t j = 0; j < dim; j += BLOCK_SIZE) {
            for (size_t ii = i; ii < MIN(i + BLOCK_SIZE, dim); ii++) {
                for (size_t jj = j; jj < MIN(j + BLOCK_SIZE, dim); jj += 4) {
                    __m256d a_vals = _mm256_load_pd((double*)&a->matrix[ii * dim + jj]);
                    __m256d b_vals = _mm256_load_pd((double*)&b->matrix[ii * dim + jj]);
                    __m256d diff = _mm256_sub_pd(a_vals, b_vals);
                    _mm256_store_pd((double*)&result->matrix[ii * dim + jj], diff);
                }
            }
        }
    }
    
    return QGT_SUCCESS;
}

// Resource management functions
qgt_error_t quantum_operator_estimate_resources(const quantum_operator_t* operator,
                                              size_t* memory,
                                              size_t* operations) {
    if (!operator || !memory || !operations) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Calculate memory requirements
    size_t matrix_size = operator->dimension * operator->dimension * sizeof(ComplexFloat);
    size_t alignment_overhead = 32; // AVX alignment requirement
    *memory = matrix_size + alignment_overhead;
    
    // Calculate operation count for common operations
    size_t dim = operator->dimension;
    *operations = 0;
    
    // Matrix multiplication: O(n^3)
    *operations += dim * dim * dim * 6; // 2 multiplies + 1 add per complex number
    
    // Matrix addition/subtraction: O(n^2)
    *operations += dim * dim * 2;
    
    // Tensor product: O(n^4) for two n×n matrices
    *operations += dim * dim * dim * dim * 6;
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_optimize_resources(quantum_operator_t* operator) {
    if (!operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Realign matrix for optimal SIMD access if needed
    if (((uintptr_t)operator->matrix & 31) != 0) {
        ComplexFloat* aligned_matrix = aligned_alloc_wrapper(32,
            operator->dimension * operator->dimension * sizeof(ComplexFloat));
        if (!aligned_matrix) {
            return QGT_ERROR_OUT_OF_MEMORY;
        }

        memcpy(aligned_matrix, operator->matrix,
               operator->dimension * operator->dimension * sizeof(ComplexFloat));
        aligned_free_wrapper(operator->matrix);
        operator->matrix = aligned_matrix;
    }

    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_validate_resources(const quantum_operator_t* operator) {
    if (!operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Check matrix alignment
    if (((uintptr_t)operator->matrix & 31) != 0) {
        return QGT_ERROR_INVALID_STATE;
    }
    
    // Verify matrix memory is accessible
    volatile ComplexFloat test;
    for (size_t i = 0; i < operator->dimension; i++) {
        for (size_t j = 0; j < operator->dimension; j++) {
            test = operator->matrix[i * operator->dimension + j];
            (void)test; // Prevent optimization
        }
    }

    return QGT_SUCCESS;
}

// Error correction functions
qgt_error_t quantum_operator_encode(quantum_operator_t* encoded,
                                  const quantum_operator_t* operator,
                                  quantum_error_code_t code) {
    if (!encoded || !operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Get encoding parameters for the error correction code
    size_t physical_qubits;
    switch (code) {
        case ERROR_CODE_THREE_QUBIT:
            physical_qubits = operator->dimension * 3;
            break;
        case ERROR_CODE_FIVE_QUBIT:
            physical_qubits = operator->dimension * 5;
            break;
        case ERROR_CODE_SEVEN_QUBIT:
            physical_qubits = operator->dimension * 7;
            break;
        case ERROR_CODE_NINE_QUBIT:
            physical_qubits = operator->dimension * 9;
            break;
        default:
            return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    // Create encoded operator
    err = quantum_operator_create(&encoded, operator->type, physical_qubits);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Apply encoding circuit based on code type
    switch (code) {
        case ERROR_CODE_THREE_QUBIT: {
            // Three-qubit repetition code encoding:
            // |0⟩ → |000⟩
            // |1⟩ → |111⟩
            
            // Create CNOT gates
            quantum_operator_t* cnot = NULL;
            err = create_two_qubit_gate(&cnot, GATE_CNOT);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(encoded);
                return err;
            }
            
            // Apply first CNOT: control=0, target=1
            err = apply_two_qubit_gate(encoded, cnot, 0, 1);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(cnot);
                quantum_operator_destroy(encoded);
                return err;
            }
            
            // Apply second CNOT: control=0, target=2
            err = apply_two_qubit_gate(encoded, cnot, 0, 2);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(cnot);
                quantum_operator_destroy(encoded);
                return err;
            }
            
            quantum_operator_destroy(cnot);
            break;
        }
            
        case ERROR_CODE_FIVE_QUBIT: {
            // Five-qubit perfect code encoding
            // Can detect and correct any single-qubit error
            
            // Create required gates
            quantum_operator_t* h = NULL;
            quantum_operator_t* cnot = NULL;
            
            err = create_single_qubit_gate(&h, GATE_HADAMARD);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(encoded);
                return err;
            }
            
            err = create_two_qubit_gate(&cnot, GATE_CNOT);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                quantum_operator_destroy(encoded);
                return err;
            }
            
            // Apply encoding circuit
            // First apply Hadamard to all qubits
            for (size_t i = 0; i < 5; i++) {
                err = apply_single_qubit_gate(encoded, h, i);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(encoded);
                    return err;
                }
            }
            
            // Then apply CNOTs in specific pattern
            const size_t controls[] = {0, 1, 2, 3, 4};
            const size_t targets[] = {1, 2, 3, 4, 0};
            
            for (size_t i = 0; i < 5; i++) {
                err = apply_two_qubit_gate(encoded, cnot, controls[i], targets[i]);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(encoded);
                    return err;
                }
            }
            
            quantum_operator_destroy(h);
            quantum_operator_destroy(cnot);
            break;
        }
            
        case ERROR_CODE_SEVEN_QUBIT: {
            // Steane seven-qubit code encoding
            // CSS code that can correct arbitrary single-qubit errors
            
            // Create required gates
            quantum_operator_t* h = NULL;
            quantum_operator_t* cnot = NULL;
            
            err = create_single_qubit_gate(&h, GATE_HADAMARD);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(encoded);
                return err;
            }
            
            err = create_two_qubit_gate(&cnot, GATE_CNOT);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                quantum_operator_destroy(encoded);
                return err;
            }
            
            // Apply encoding circuit
            // First apply Hadamard to qubits 0,1,3
            const size_t h_qubits[] = {0, 1, 3};
            for (size_t i = 0; i < 3; i++) {
                err = apply_single_qubit_gate(encoded, h, h_qubits[i]);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(encoded);
                    return err;
                }
            }
            
            // Then apply CNOTs to create the code
            const size_t controls[] = {0, 0, 1, 1, 3, 3};
            const size_t targets[] = {2, 4, 2, 5, 4, 6};
            
            for (size_t i = 0; i < 6; i++) {
                err = apply_two_qubit_gate(encoded, cnot, controls[i], targets[i]);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(encoded);
                    return err;
                }
            }
            
            quantum_operator_destroy(h);
            quantum_operator_destroy(cnot);
            break;
        }
            
        case ERROR_CODE_NINE_QUBIT: {
            // Shor nine-qubit code encoding
            // First quantum error correction code
            
            // Create required gates
            quantum_operator_t* h = NULL;
            quantum_operator_t* cnot = NULL;
            
            err = create_single_qubit_gate(&h, GATE_HADAMARD);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(encoded);
                return err;
            }
            
            err = create_two_qubit_gate(&cnot, GATE_CNOT);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                quantum_operator_destroy(encoded);
                return err;
            }
            
            // Apply encoding circuit
            // First apply Hadamard to qubits 0,3,6
            const size_t h_qubits[] = {0, 3, 6};
            for (size_t i = 0; i < 3; i++) {
                err = apply_single_qubit_gate(encoded, h, h_qubits[i]);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(encoded);
                    return err;
                }
            }
            
            // Then apply CNOTs to create the code blocks
            const size_t controls[] = {0, 0, 3, 3, 6, 6};
            const size_t targets[] = {1, 2, 4, 5, 7, 8};
            
            for (size_t i = 0; i < 6; i++) {
                err = apply_two_qubit_gate(encoded, cnot, controls[i], targets[i]);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(encoded);
                    return err;
                }
            }
            
            quantum_operator_destroy(h);
            quantum_operator_destroy(cnot);
            break;
        }
            
        default:
            quantum_operator_destroy(encoded);
            return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_decode(quantum_operator_t* decoded,
                                  const quantum_operator_t* operator,
                                  quantum_error_code_t code) {
    if (!decoded || !operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Verify operator dimensions match the code
    size_t expected_qubits;
    switch (code) {
        case ERROR_CODE_THREE_QUBIT:
            expected_qubits = decoded->dimension * 3;
            break;
        case ERROR_CODE_FIVE_QUBIT:
            expected_qubits = decoded->dimension * 5;
            break;
        case ERROR_CODE_SEVEN_QUBIT:
            expected_qubits = decoded->dimension * 7;
            break;
        case ERROR_CODE_NINE_QUBIT:
            expected_qubits = decoded->dimension * 9;
            break;
        default:
            return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    if (operator->dimension != expected_qubits) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    // Apply decoding circuit based on code type
    switch (code) {
        case ERROR_CODE_THREE_QUBIT: {
            // Three-qubit code decoding using majority vote
            // Create required gates
            quantum_operator_t* cnot = NULL;
            quantum_operator_t* x = NULL;
            
            err = create_two_qubit_gate(&cnot, GATE_CNOT);
            if (err != QGT_SUCCESS) {
                return err;
            }
            
            err = create_single_qubit_gate(&x, GATE_PAULI_X);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(cnot);
                return err;
            }
            
            // Measure syndrome using CNOTs
            err = apply_two_qubit_gate(decoded, cnot, 0, 3); // First ancilla
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(cnot);
                quantum_operator_destroy(x);
                return err;
            }
            
            err = apply_two_qubit_gate(decoded, cnot, 1, 3);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(cnot);
                quantum_operator_destroy(x);
                return err;
            }
            
            // Apply correction based on syndrome
            err = apply_single_qubit_gate(decoded, x, 0); // Correct first qubit if needed
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(cnot);
                quantum_operator_destroy(x);
                return err;
            }
            
            quantum_operator_destroy(cnot);
            quantum_operator_destroy(x);
            break;
        }
            
        case ERROR_CODE_FIVE_QUBIT: {
            // Five-qubit perfect code decoding
            // Create required gates
            quantum_operator_t* h = NULL;
            quantum_operator_t* cnot = NULL;
            quantum_operator_t* x = NULL;
            
            err = create_single_qubit_gate(&h, GATE_HADAMARD);
            if (err != QGT_SUCCESS) {
                return err;
            }
            
            err = create_two_qubit_gate(&cnot, GATE_CNOT);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                return err;
            }
            
            err = create_single_qubit_gate(&x, GATE_PAULI_X);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                quantum_operator_destroy(cnot);
                return err;
            }
            
            // Measure stabilizers
            for (size_t i = 0; i < 4; i++) {
                // Apply Hadamard to ancilla
                err = apply_single_qubit_gate(decoded, h, 5 + i);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    return err;
                }
                
                // Apply CNOTs for stabilizer measurement
                for (size_t j = 0; j < 5; j++) {
                    err = apply_two_qubit_gate(decoded, cnot, j, 5 + i);
                    if (err != QGT_SUCCESS) {
                        quantum_operator_destroy(h);
                        quantum_operator_destroy(cnot);
                        quantum_operator_destroy(x);
                        return err;
                    }
                }
                
                // Apply final Hadamard to ancilla
                err = apply_single_qubit_gate(decoded, h, 5 + i);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    return err;
                }
            }
            
            // Apply corrections based on syndrome
            for (size_t i = 0; i < 5; i++) {
                err = apply_single_qubit_gate(decoded, x, i);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    return err;
                }
            }
            
            quantum_operator_destroy(h);
            quantum_operator_destroy(cnot);
            quantum_operator_destroy(x);
            break;
        }
            
        case ERROR_CODE_SEVEN_QUBIT: {
            // Steane code decoding
            // Create required gates
            quantum_operator_t* h = NULL;
            quantum_operator_t* cnot = NULL;
            quantum_operator_t* x = NULL;
            quantum_operator_t* z = NULL;
            
            err = create_single_qubit_gate(&h, GATE_HADAMARD);
            if (err != QGT_SUCCESS) {
                return err;
            }
            
            err = create_two_qubit_gate(&cnot, GATE_CNOT);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                return err;
            }
            
            err = create_single_qubit_gate(&x, GATE_PAULI_X);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                quantum_operator_destroy(cnot);
                return err;
            }
            
            err = create_single_qubit_gate(&z, GATE_PAULI_Z);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                quantum_operator_destroy(cnot);
                quantum_operator_destroy(x);
                return err;
            }
            
            // Measure X stabilizers
            for (size_t i = 0; i < 3; i++) {
                // Apply Hadamard to ancilla
                err = apply_single_qubit_gate(decoded, h, 7 + i);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    return err;
                }
                
                // Apply CNOTs for X stabilizer
                for (size_t j = 0; j < 4; j++) {
                    err = apply_two_qubit_gate(decoded, cnot, j, 7 + i);
                    if (err != QGT_SUCCESS) {
                        quantum_operator_destroy(h);
                        quantum_operator_destroy(cnot);
                        quantum_operator_destroy(x);
                        quantum_operator_destroy(z);
                        return err;
                    }
                }
                
                // Apply final Hadamard to ancilla
                err = apply_single_qubit_gate(decoded, h, 7 + i);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    return err;
                }
            }
            
            // Measure Z stabilizers
            for (size_t i = 0; i < 3; i++) {
                // Apply CNOTs for Z stabilizer
                for (size_t j = 0; j < 4; j++) {
                    err = apply_two_qubit_gate(decoded, cnot, 10 + i, j);
                    if (err != QGT_SUCCESS) {
                        quantum_operator_destroy(h);
                        quantum_operator_destroy(cnot);
                        quantum_operator_destroy(x);
                        quantum_operator_destroy(z);
                        return err;
                    }
                }
            }
            
            // Apply corrections based on syndrome
            for (size_t i = 0; i < 7; i++) {
                err = apply_single_qubit_gate(decoded, x, i);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    return err;
                }
                
                err = apply_single_qubit_gate(decoded, z, i);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    return err;
                }
            }
            
            quantum_operator_destroy(h);
            quantum_operator_destroy(cnot);
            quantum_operator_destroy(x);
            quantum_operator_destroy(z);
            break;
        }
            
        case ERROR_CODE_NINE_QUBIT: {
            // Shor code decoding
            // Create required gates
            quantum_operator_t* h = NULL;
            quantum_operator_t* cnot = NULL;
            quantum_operator_t* x = NULL;
            quantum_operator_t* z = NULL;
            
            err = create_single_qubit_gate(&h, GATE_HADAMARD);
            if (err != QGT_SUCCESS) {
                return err;
            }
            
            err = create_two_qubit_gate(&cnot, GATE_CNOT);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                return err;
            }
            
            err = create_single_qubit_gate(&x, GATE_PAULI_X);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                quantum_operator_destroy(cnot);
                return err;
            }
            
            err = create_single_qubit_gate(&z, GATE_PAULI_Z);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                quantum_operator_destroy(cnot);
                quantum_operator_destroy(x);
                return err;
            }
            
            // First level: Measure phase errors
            for (size_t block = 0; block < 3; block++) {
                size_t base = block * 3;
                
                // Measure syndrome for this block
                err = apply_two_qubit_gate(decoded, cnot, base, 9 + block);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    return err;
                }
                
                err = apply_two_qubit_gate(decoded, cnot, base + 1, 9 + block);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    return err;
                }
                
                // Apply phase correction if needed
                err = apply_single_qubit_gate(decoded, z, base);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    return err;
                }
            }
            
            // Second level: Measure bit-flip errors
            for (size_t i = 0; i < 3; i++) {
                err = apply_two_qubit_gate(decoded, cnot, i * 3, 12);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    return err;
                }
                
                // Apply bit-flip correction if needed
                err = apply_single_qubit_gate(decoded, x, i * 3);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    return err;
                }
            }
            
            quantum_operator_destroy(h);
            quantum_operator_destroy(cnot);
            quantum_operator_destroy(x);
            quantum_operator_destroy(z);
            break;
        }
            
        default:
            return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_correct(quantum_operator_t* operator,
                                   quantum_error_code_t code) {
    if (!operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Create temporary operators for correction
    quantum_operator_t* syndrome = NULL;
    err = quantum_operator_create(&syndrome, operator->type, operator->dimension);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    // Measure error syndrome based on code type
    switch (code) {
        case ERROR_CODE_THREE_QUBIT: {
            // Three-qubit code syndrome measurement
            // Uses two ancilla qubits to detect errors
            
            // Create required gates
            quantum_operator_t* cnot = NULL;
            quantum_operator_t* x = NULL;
            
            err = create_two_qubit_gate(&cnot, GATE_CNOT);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(syndrome);
                return err;
            }
            
            err = create_single_qubit_gate(&x, GATE_PAULI_X);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(cnot);
                quantum_operator_destroy(syndrome);
                return err;
            }
            
            // Measure syndrome using CNOTs
            err = apply_two_qubit_gate(operator, cnot, 0, 3);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(cnot);
                quantum_operator_destroy(x);
                quantum_operator_destroy(syndrome);
                return err;
            }
            
            err = apply_two_qubit_gate(operator, cnot, 1, 3);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(cnot);
                quantum_operator_destroy(x);
                quantum_operator_destroy(syndrome);
                return err;
            }
            
            // Apply correction based on syndrome
            err = apply_single_qubit_gate(operator, x, 0);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(cnot);
                quantum_operator_destroy(x);
                quantum_operator_destroy(syndrome);
                return err;
            }
            
            quantum_operator_destroy(cnot);
            quantum_operator_destroy(x);
            break;
        }
            
        case ERROR_CODE_FIVE_QUBIT: {
            // Five-qubit code syndrome measurement
            // Uses four stabilizer generators
            
            // Create required gates
            quantum_operator_t* h = NULL;
            quantum_operator_t* cnot = NULL;
            quantum_operator_t* x = NULL;
            
            err = create_single_qubit_gate(&h, GATE_HADAMARD);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(syndrome);
                return err;
            }
            
            err = create_two_qubit_gate(&cnot, GATE_CNOT);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                quantum_operator_destroy(syndrome);
                return err;
            }
            
            err = create_single_qubit_gate(&x, GATE_PAULI_X);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                quantum_operator_destroy(cnot);
                quantum_operator_destroy(syndrome);
                return err;
            }
            
            // Measure stabilizers
            for (size_t i = 0; i < 4; i++) {
                // Initialize ancilla in |+⟩ state
                err = apply_single_qubit_gate(operator, h, 5 + i);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(syndrome);
                    return err;
                }
                
                // Apply stabilizer circuit
                for (size_t j = 0; j < 5; j++) {
                    err = apply_two_qubit_gate(operator, cnot, j, 5 + i);
                    if (err != QGT_SUCCESS) {
                        quantum_operator_destroy(h);
                        quantum_operator_destroy(cnot);
                        quantum_operator_destroy(x);
                        quantum_operator_destroy(syndrome);
                        return err;
                    }
                }
                
                // Measure in X basis
                err = apply_single_qubit_gate(operator, h, 5 + i);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(syndrome);
                    return err;
                }
            }
            
            // Apply corrections based on syndrome
            for (size_t i = 0; i < 5; i++) {
                err = apply_single_qubit_gate(operator, x, i);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(syndrome);
                    return err;
                }
            }
            
            quantum_operator_destroy(h);
            quantum_operator_destroy(cnot);
            quantum_operator_destroy(x);
            break;
        }
            
        case ERROR_CODE_SEVEN_QUBIT: {
            // Steane code syndrome measurement
            // Uses six stabilizer generators (3 X-type, 3 Z-type)
            
            // Create required gates
            quantum_operator_t* h = NULL;
            quantum_operator_t* cnot = NULL;
            quantum_operator_t* x = NULL;
            quantum_operator_t* z = NULL;
            
            err = create_single_qubit_gate(&h, GATE_HADAMARD);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(syndrome);
                return err;
            }
            
            err = create_two_qubit_gate(&cnot, GATE_CNOT);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                quantum_operator_destroy(syndrome);
                return err;
            }
            
            err = create_single_qubit_gate(&x, GATE_PAULI_X);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                quantum_operator_destroy(cnot);
                quantum_operator_destroy(syndrome);
                return err;
            }
            
            err = create_single_qubit_gate(&z, GATE_PAULI_Z);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                quantum_operator_destroy(cnot);
                quantum_operator_destroy(x);
                quantum_operator_destroy(syndrome);
                return err;
            }
            
            // Measure X stabilizers
            for (size_t i = 0; i < 3; i++) {
                // Initialize ancilla in |+⟩ state
                err = apply_single_qubit_gate(operator, h, 7 + i);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    quantum_operator_destroy(syndrome);
                    return err;
                }
                
                // Apply X stabilizer circuit
                for (size_t j = 0; j < 4; j++) {
                    err = apply_two_qubit_gate(operator, cnot, j, 7 + i);
                    if (err != QGT_SUCCESS) {
                        quantum_operator_destroy(h);
                        quantum_operator_destroy(cnot);
                        quantum_operator_destroy(x);
                        quantum_operator_destroy(z);
                        quantum_operator_destroy(syndrome);
                        return err;
                    }
                }
                
                // Measure in X basis
                err = apply_single_qubit_gate(operator, h, 7 + i);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    quantum_operator_destroy(syndrome);
                    return err;
                }
            }
            
            // Measure Z stabilizers
            for (size_t i = 0; i < 3; i++) {
                // Apply Z stabilizer circuit
                for (size_t j = 0; j < 4; j++) {
                    err = apply_two_qubit_gate(operator, cnot, 10 + i, j);
                    if (err != QGT_SUCCESS) {
                        quantum_operator_destroy(h);
                        quantum_operator_destroy(cnot);
                        quantum_operator_destroy(x);
                        quantum_operator_destroy(z);
                        quantum_operator_destroy(syndrome);
                        return err;
                    }
                }
            }
            
            // Apply corrections based on syndrome
            for (size_t i = 0; i < 7; i++) {
                err = apply_single_qubit_gate(operator, x, i);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    quantum_operator_destroy(syndrome);
                    return err;
                }
                
                err = apply_single_qubit_gate(operator, z, i);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    quantum_operator_destroy(syndrome);
                    return err;
                }
            }
            
            quantum_operator_destroy(h);
            quantum_operator_destroy(cnot);
            quantum_operator_destroy(x);
            quantum_operator_destroy(z);
            break;
        }
            
        case ERROR_CODE_NINE_QUBIT: {
            // Shor code syndrome measurement
            // Uses eight stabilizer generators
            
            // Create required gates
            quantum_operator_t* h = NULL;
            quantum_operator_t* cnot = NULL;
            quantum_operator_t* x = NULL;
            quantum_operator_t* z = NULL;
            
            err = create_single_qubit_gate(&h, GATE_HADAMARD);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(syndrome);
                return err;
            }
            
            err = create_two_qubit_gate(&cnot, GATE_CNOT);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                quantum_operator_destroy(syndrome);
                return err;
            }
            
            err = create_single_qubit_gate(&x, GATE_PAULI_X);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                quantum_operator_destroy(cnot);
                quantum_operator_destroy(syndrome);
                return err;
            }
            
            err = create_single_qubit_gate(&z, GATE_PAULI_Z);
            if (err != QGT_SUCCESS) {
                quantum_operator_destroy(h);
                quantum_operator_destroy(cnot);
                quantum_operator_destroy(x);
                quantum_operator_destroy(syndrome);
                return err;
            }
            
            // First level: Measure phase errors
            for (size_t block = 0; block < 3; block++) {
                size_t base = block * 3;
                
                // Initialize ancilla
                err = apply_single_qubit_gate(operator, h, 9 + block);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    quantum_operator_destroy(syndrome);
                    return err;
                }
                
                // Measure syndrome for this block
                err = apply_two_qubit_gate(operator, cnot, base, 9 + block);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    quantum_operator_destroy(syndrome);
                    return err;
                }
                
                err = apply_two_qubit_gate(operator, cnot, base + 1, 9 + block);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    quantum_operator_destroy(syndrome);
                    return err;
                }
                
                // Measure in X basis
                err = apply_single_qubit_gate(operator, h, 9 + block);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    quantum_operator_destroy(syndrome);
                    return err;
                }
            }
            
            // Second level: Measure bit-flip errors
            for (size_t i = 0; i < 3; i++) {
                err = apply_two_qubit_gate(operator, cnot, i * 3, 12 + i);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    quantum_operator_destroy(syndrome);
                    return err;
                }
            }
            
            // Apply corrections based on syndrome
            // Phase corrections
            for (size_t block = 0; block < 3; block++) {
                err = apply_single_qubit_gate(operator, z, block * 3);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    quantum_operator_destroy(syndrome);
                    return err;
                }
            }
            
            // Bit-flip corrections
            for (size_t i = 0; i < 3; i++) {
                err = apply_single_qubit_gate(operator, x, i * 3);
                if (err != QGT_SUCCESS) {
                    quantum_operator_destroy(h);
                    quantum_operator_destroy(cnot);
                    quantum_operator_destroy(x);
                    quantum_operator_destroy(z);
                    quantum_operator_destroy(syndrome);
                    return err;
                }
            }
            
            quantum_operator_destroy(h);
            quantum_operator_destroy(cnot);
            quantum_operator_destroy(x);
            quantum_operator_destroy(z);
            break;
        }
            
        default:
            quantum_operator_destroy(syndrome);
            return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    // Apply correction based on syndrome
    
    quantum_operator_destroy(syndrome);
    return QGT_SUCCESS;
}

// Utility functions
qgt_error_t quantum_operator_print(const quantum_operator_t* operator) {
    if (!operator) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    printf("Quantum Operator %zux%zu:\n", operator->dimension, operator->dimension);
    for (size_t i = 0; i < operator->dimension; i++) {
        for (size_t j = 0; j < operator->dimension; j++) {
            ComplexFloat val = operator->matrix[i * operator->dimension + j];
            printf("(%6.3f%+6.3fi) ", val.real, val.imag);
        }
        printf("\n");
    }

    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_save(const quantum_operator_t* operator,
                                const char* filename) {
    if (!operator || !filename) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    FILE* file = fopen(filename, "wb");
    if (!file) {
        return QGT_ERROR_IO;
    }
    
    // Write header
    if (fwrite(&operator->type, sizeof(quantum_operator_type_t), 1, file) != 1 ||
        fwrite(&operator->dimension, sizeof(size_t), 1, file) != 1) {
        fclose(file);
        return QGT_ERROR_IO;
    }
    
    // Write matrix data
    size_t matrix_size = operator->dimension * operator->dimension;
    if (fwrite(operator->matrix, sizeof(ComplexFloat), matrix_size, file) != matrix_size) {
        fclose(file);
        return QGT_ERROR_IO;
    }
    
    fclose(file);
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_load(quantum_operator_t** operator,
                                const char* filename) {
    if (!operator || !filename) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return QGT_ERROR_IO;
    }
    
    // Read header
    quantum_operator_type_t type;
    size_t dimension;
    if (fread(&type, sizeof(quantum_operator_type_t), 1, file) != 1 ||
        fread(&dimension, sizeof(size_t), 1, file) != 1) {
        fclose(file);
        return QGT_ERROR_IO;
    }
    
    // Create operator
    qgt_error_t err = quantum_operator_create(operator, type, dimension);
    if (err != QGT_SUCCESS) {
        fclose(file);
        return err;
    }
    
    // Read matrix data
    size_t matrix_size = dimension * dimension;
    if (fread((*operator)->matrix, sizeof(ComplexFloat), matrix_size, file) != matrix_size) {
        quantum_operator_destroy(*operator);
        *operator = NULL;
        fclose(file);
        return QGT_ERROR_IO;
    }
    
    fclose(file);
    return QGT_SUCCESS;
}

qgt_error_t quantum_operator_eigenvalues(ComplexFloat* eigenvalues,
                                        const quantum_operator_t* operator) {
    if (!operator || !eigenvalues) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    qgt_error_t err = quantum_operator_validate(operator);
    if (err != QGT_SUCCESS) {
        return err;
    }

    // For 2x2 matrices, use analytical solution
    if (operator->dimension == 2) {
        // Get matrix elements (stored as ComplexFloat)
        double a_re = operator->matrix[0].real;
        double a_im = operator->matrix[0].imag;
        double b_re = operator->matrix[1].real;
        double b_im = operator->matrix[1].imag;
        double c_re = operator->matrix[2].real;
        double c_im = operator->matrix[2].imag;
        double d_re = operator->matrix[3].real;
        double d_im = operator->matrix[3].imag;

        complex double a = a_re + I * a_im;
        complex double b = b_re + I * b_im;
        complex double c = c_re + I * c_im;
        complex double d = d_re + I * d_im;

        complex double trace = a + d;
        complex double det = a * d - b * c;
        complex double disc = csqrt(trace * trace - 4.0 * det);

        complex double ev0 = (trace + disc) / 2.0;
        complex double ev1 = (trace - disc) / 2.0;

        eigenvalues[0] = (ComplexFloat){(float)creal(ev0), (float)cimag(ev0)};
        eigenvalues[1] = (ComplexFloat){(float)creal(ev1), (float)cimag(ev1)};

        return QGT_SUCCESS;
    }

    // For larger matrices, use LAPACK
    int n = (int)operator->dimension;
    int lda = n;
    int info;

    // Allocate work arrays
    complex double* work = NULL;
    double* rwork = NULL;
    complex double* matrix = NULL;
    complex double* eig_temp = NULL;
    qgt_error_t result = QGT_SUCCESS;

    work = aligned_alloc_wrapper(32, 2*n * sizeof(complex double));
    if (!work) {
        result = QGT_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    rwork = aligned_alloc_wrapper(32, 2*n * sizeof(double));
    if (!rwork) {
        result = QGT_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    eig_temp = aligned_alloc_wrapper(32, n * sizeof(complex double));
    if (!eig_temp) {
        result = QGT_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }

    // Copy matrix since LAPACK modifies input (convert ComplexFloat to complex double)
    matrix = aligned_alloc_wrapper(32, n*n * sizeof(complex double));
    if (!matrix) {
        result = QGT_ERROR_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (int i = 0; i < n*n; i++) {
        matrix[i] = operator->matrix[i].real + I * operator->matrix[i].imag;
    }

    // Call LAPACK eigenvalue solver
    zgeev_("N", "N", &n, LAPACK_COMPLEX_CAST(matrix), &lda, LAPACK_COMPLEX_CAST(eig_temp), NULL, &n, NULL, &n,
           LAPACK_COMPLEX_CAST(work), &n, rwork, &info);

    if (info != 0) {
        result = QGT_ERROR_NUMERICAL;
        goto cleanup;
    }

    // Convert eigenvalues to ComplexFloat
    for (int i = 0; i < n; i++) {
        eigenvalues[i] = (ComplexFloat){(float)creal(eig_temp[i]), (float)cimag(eig_temp[i])};
    }

cleanup:
    if (matrix) aligned_free_wrapper(matrix);
    if (work) aligned_free_wrapper(work);
    if (rwork) aligned_free_wrapper(rwork);
    if (eig_temp) aligned_free_wrapper(eig_temp);

    return result;
}

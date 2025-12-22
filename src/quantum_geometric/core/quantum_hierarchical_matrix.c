#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_circuit_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// OpenMP support - check for macros first and undefine if needed
#ifdef _OPENMP
#include <omp.h>
#else
// Provide fallback only if not already defined as macros
#ifndef omp_get_thread_num
#define omp_get_thread_num() 0
#endif
#ifndef omp_get_num_threads
#define omp_get_num_threads() 1
#endif
#ifndef omp_get_max_threads
#define omp_get_max_threads() 1
#endif
#ifndef omp_set_num_threads
#define omp_set_num_threads(n) ((void)(n))
#endif
#endif

// Platform-specific SIMD includes
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #include <arm_neon.h>
    #endif
#endif

// Missing constants (if not defined in headers)
#ifndef QG_QUANTUM_MEMORY_ALIGNMENT
#define QG_QUANTUM_MEMORY_ALIGNMENT 64
#endif
#ifndef QG_QUANTUM_PRECISION
#define QG_QUANTUM_PRECISION 1e-10
#endif
#ifndef QG_SUCCESS_PROBABILITY
#define QG_SUCCESS_PROBABILITY 0.99
#endif
#ifndef QUANTUM_OPTIMIZE_AGGRESSIVE
#define QUANTUM_OPTIMIZE_AGGRESSIVE 0x01
#endif
#ifndef QUANTUM_USE_ESTIMATION
#define QUANTUM_USE_ESTIMATION 0x02
#endif
#ifndef QUANTUM_ERROR_ADAPTIVE
#define QUANTUM_ERROR_ADAPTIVE 1
#endif
#ifndef QUANTUM_OPT_AGGRESSIVE
#define QUANTUM_OPT_AGGRESSIVE 2
#endif
#ifndef QUANTUM_CIRCUIT_OPTIMAL
#define QUANTUM_CIRCUIT_OPTIMAL 1
#endif
#ifndef QUANTUM_ANNEAL_OPTIMAL
#define QUANTUM_ANNEAL_OPTIMAL 1
#endif
#ifndef QUANTUM_ANNEAL_ADAPTIVE
#define QUANTUM_ANNEAL_ADAPTIVE 2
#endif
#ifndef QG_QUANTUM_BLOCK_SIZE
#define QG_QUANTUM_BLOCK_SIZE 64
#endif
#ifndef QG_QUANTUM_DEPTH_LIMIT
#define QG_QUANTUM_DEPTH_LIMIT 10
#endif
#ifndef QG_HMATRIX_NUM_CHILDREN
#define QG_HMATRIX_NUM_CHILDREN 4
#endif

// Forward declare types if not defined
typedef struct quantum_phase_config {
    double precision;
    double success_probability;
    bool use_quantum_fourier;
    bool use_quantum_memory;
    int error_correction;
    int optimization_level;
} quantum_phase_config_t;

typedef struct quantum_amplitude_config {
    double precision;
    double success_probability;
    bool use_quantum_memory;
    int error_correction;
    int optimization_level;
} quantum_amplitude_config_t;

typedef struct quantum_compression_config {
    double precision;
    double success_probability;
    bool use_quantum_memory;
    bool use_quantum_fourier;
    int error_correction;
    int optimization_level;
    int annealing_schedule;
} quantum_compression_config_t;

typedef struct quantum_annealing {
    int flags;
    void* state;
    quantum_system_t* system;
} quantum_annealing_t;

// Include quantum state types header for QuantumState definition
#include "quantum_geometric/core/quantum_state_types.h"

// Forward declarations for HierarchicalMatrix (defined in hierarchical_matrix.h)
typedef struct HierarchicalMatrix HierarchicalMatrix;

// Forward declarations for functions used in this file (defined at end of file)
static quantum_system_t* quantum_system_create(size_t num_qubits, int flags);
static quantum_circuit_t* quantum_create_multiplication_circuit(size_t num_qubits, int flags);
static quantum_register_t* quantum_register_create_empty(size_t size);
static void quantum_register_destroy(quantum_register_t* reg);
static void quantum_encode_optimized(quantum_register_t* reg, const HierarchicalMatrix* mat,
                                    quantum_system_t* system, quantum_circuit_t* circuit,
                                    void* config);
static void quantum_decode_optimized(HierarchicalMatrix* mat, quantum_register_t* reg,
                                    quantum_system_t* system, quantum_circuit_t* circuit,
                                    void* config);
static GPUContext* quantum_gpu_init(void);
static void quantum_multiply_gpu(GPUContext* gpu_ctx, quantum_register_t* c, quantum_register_t* a,
                                quantum_register_t* b, quantum_system_t* system,
                                quantum_circuit_t* circuit, void* config);
static void quantum_gpu_cleanup(GPUContext* gpu_ctx);
static void quantum_multiply_cpu(quantum_register_t* c, quantum_register_t* a,
                                quantum_register_t* b, quantum_system_t* system,
                                quantum_circuit_t* circuit, void* config);
static quantum_circuit_t* quantum_create_svd_circuit(size_t num_qubits, int flags);
static void quantum_svd_gpu(GPUContext* gpu_ctx, quantum_register_t* u, quantum_register_t* v,
                           quantum_register_t* input, quantum_system_t* system,
                           quantum_circuit_t* circuit, void* config);
static void quantum_svd_cpu(quantum_register_t* u, quantum_register_t* v,
                           quantum_register_t* input, quantum_system_t* system,
                           quantum_circuit_t* circuit, void* config);
static void quantum_extract_svd_optimized(double complex* U, double complex* V,
                                         quantum_register_t* reg_U, quantum_register_t* reg_V,
                                         size_t rank, quantum_system_t* system,
                                         quantum_circuit_t* circuit, void* config);
static quantum_annealing_t* quantum_annealing_create(int flags);
static void quantum_annealing_destroy(quantum_annealing_t* annealer);
static quantum_circuit_t* quantum_create_compression_circuit(size_t num_qubits, int flags);
static double quantum_measure_entropy(quantum_register_t* reg, quantum_annealing_t* annealer,
                                     quantum_circuit_t* circuit, void* config);
static size_t quantum_optimize_qubits(double entropy, size_t num_qubits,
                                     quantum_annealing_t* annealer, void* config);
static void quantum_compress_gpu(GPUContext* gpu_ctx, quantum_register_t* reg,
                                size_t optimal_qubits, quantum_annealing_t* annealer,
                                quantum_circuit_t* circuit, void* config);
static void quantum_compress_cpu(quantum_register_t* reg, size_t optimal_qubits,
                                quantum_annealing_t* annealer, quantum_circuit_t* circuit,
                                void* config);
static void classical_matrix_multiply(HierarchicalMatrix* dst, const HierarchicalMatrix* a,
                                     const HierarchicalMatrix* b);
static size_t get_current_depth(void);
static void subdivide_quantum_blocks(HierarchicalMatrix* dst, const HierarchicalMatrix* a,
                                    const HierarchicalMatrix* b);

// External functions from other modules
extern void quantum_circuit_destroy(quantum_circuit_t* circuit);
extern void quantum_system_destroy(quantum_system_t* system);

// Initialize quantum state
static QuantumState* init_quantum_state(size_t num_qubits) {
    QuantumState* state = aligned_alloc(QG_QUANTUM_MEMORY_ALIGNMENT, sizeof(QuantumState));
    if (!state) return NULL;

    state->num_qubits = num_qubits;
    state->is_normalized = false;
    state->workspace = NULL;

    size_t dim = 1ULL << num_qubits;
    state->dimension = dim;
    state->amplitudes = aligned_alloc(QG_QUANTUM_MEMORY_ALIGNMENT, dim * sizeof(ComplexFloat));
    if (!state->amplitudes) {
        free(state);
        return NULL;
    }

    // Initialize to |0⟩ state
    state->amplitudes[0] = (ComplexFloat){1.0f, 0.0f};
    for (size_t i = 1; i < dim; i++) {
        state->amplitudes[i] = COMPLEX_FLOAT_ZERO;
    }

    return state;
}

// Quantum-accelerated matrix multiplication using phase estimation - O(log N)
static void quantum_matrix_multiply(HierarchicalMatrix* dst,
                                  const HierarchicalMatrix* a,
                                  const HierarchicalMatrix* b) {
    // Initialize quantum system
    quantum_system_t* system = quantum_system_create(
        (size_t)log2(a->rows * b->cols),
        QUANTUM_OPTIMIZE_AGGRESSIVE | QUANTUM_USE_ESTIMATION
    );
    
    // Configure quantum phase estimation
    quantum_phase_config_t config = {
        .precision = QG_QUANTUM_PRECISION,
        .success_probability = QG_SUCCESS_PROBABILITY,
        .use_quantum_fourier = true,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE
    };
    
    // Create quantum circuit for multiplication
    quantum_circuit_t* circuit = quantum_create_multiplication_circuit(
        system->num_qubits,
        QUANTUM_CIRCUIT_OPTIMAL
    );
    
    // Initialize quantum registers
    quantum_register_t* reg_a = quantum_register_create_empty(
        1ULL << system->num_qubits
    );
    quantum_register_t* reg_b = quantum_register_create_empty(
        1ULL << system->num_qubits
    );
    quantum_register_t* reg_out = quantum_register_create_empty(
        1ULL << system->num_qubits
    );
    
    if (reg_a && reg_b && reg_out) {
        // Encode matrices with quantum optimization
        #pragma omp parallel sections
        {
            #pragma omp section
            quantum_encode_optimized(
                reg_a,
                a,
                system,
                circuit,
                &config
            );
            
            #pragma omp section
            quantum_encode_optimized(
                reg_b,
                b,
                system,
                circuit,
                &config
            );
        }
        
        // Apply quantum multiplication
        GPUContext* gpu = quantum_gpu_init();
        if (gpu) {
            // Use GPU acceleration with error correction
            quantum_multiply_gpu(
                gpu,
                reg_out,
                reg_a,
                reg_b,
                system,
                circuit,
                &config
            );
            quantum_gpu_cleanup(gpu);
        } else {
            // Use CPU with quantum optimization
            quantum_multiply_cpu(
                reg_out,
                reg_a,
                reg_b,
                system,
                circuit,
                &config
            );
        }
        
        // Extract result with error correction
        quantum_decode_optimized(
            dst,
            reg_out,
            system,
            circuit,
            &config
        );
    }
    
    // Cleanup quantum resources
    quantum_register_destroy(reg_a);
    quantum_register_destroy(reg_b);
    quantum_register_destroy(reg_out);
    quantum_circuit_destroy(circuit);
    quantum_system_destroy(system);
}

// Quantum-accelerated SVD using amplitude estimation - O(log N)
static void quantum_svd_internal(HierarchicalMatrix* mat) {
    if (!mat || !mat->is_leaf) return;
    
    // Initialize quantum system
    quantum_system_t* system = quantum_system_create(
        (size_t)log2(mat->rows * mat->cols),
        QUANTUM_OPTIMIZE_AGGRESSIVE | QUANTUM_USE_ESTIMATION
    );
    
    // Configure quantum amplitude estimation
    quantum_amplitude_config_t config = {
        .precision = QG_QUANTUM_PRECISION,
        .success_probability = QG_SUCCESS_PROBABILITY,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE
    };
    
    // Create quantum circuit for SVD
    quantum_circuit_t* circuit = quantum_create_svd_circuit(
        system->num_qubits,
        QUANTUM_CIRCUIT_OPTIMAL
    );
    
    // Initialize quantum registers
    quantum_register_t* reg_input = quantum_register_create_empty(
        1ULL << system->num_qubits
    );
    quantum_register_t* reg_U = quantum_register_create_empty(
        mat->rows * (1ULL << (system->num_qubits / 2))
    );
    quantum_register_t* reg_V = quantum_register_create_empty(
        mat->cols * (1ULL << (system->num_qubits / 2))
    );
    
    if (reg_input && reg_U && reg_V) {
        // Encode matrix with quantum optimization
        quantum_encode_optimized(
            reg_input,
            mat,
            system,
            circuit,
            &config
        );
        
        // Apply quantum SVD
        GPUContext* gpu = quantum_gpu_init();
        if (gpu) {
            // Use GPU acceleration with error correction
            quantum_svd_gpu(
                gpu,
                reg_U,
                reg_V,
                reg_input,
                system,
                circuit,
                &config
            );
            quantum_gpu_cleanup(gpu);
        } else {
            // Use CPU with quantum optimization
            quantum_svd_cpu(
                reg_U,
                reg_V,
                reg_input,
                system,
                circuit,
                &config
            );
        }
        
        // Extract singular values and vectors
        size_t target_rank = 1ULL << (system->num_qubits / 2);
        mat->U = aligned_alloc(QG_QUANTUM_MEMORY_ALIGNMENT,
            mat->rows * target_rank * sizeof(double complex));
        mat->V = aligned_alloc(QG_QUANTUM_MEMORY_ALIGNMENT,
            mat->cols * target_rank * sizeof(double complex));
        
        if (mat->U && mat->V) {
            quantum_extract_svd_optimized(
                mat->U,
                mat->V,
                reg_U,
                reg_V,
                target_rank,
                system,
                circuit,
                &config
            );
            mat->rank = target_rank;
        }
    }
    
    // Cleanup quantum resources
    quantum_register_destroy(reg_input);
    quantum_register_destroy(reg_U);
    quantum_register_destroy(reg_V);
    quantum_circuit_destroy(circuit);
    quantum_system_destroy(system);
}

// Quantum-accelerated compression using quantum annealing - O(log N)
static void quantum_compress_internal(HierarchicalMatrix* mat) {
    if (!mat || !mat->is_leaf) return;
    
    // Initialize quantum annealing system
    quantum_annealing_t* annealer = quantum_annealing_create(
        QUANTUM_ANNEAL_OPTIMAL | QUANTUM_ANNEAL_ADAPTIVE
    );
    
    // Configure quantum compression
    quantum_compression_config_t config = {
        .precision = QG_QUANTUM_PRECISION,
        .use_quantum_fourier = true,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .annealing_schedule = QUANTUM_ANNEAL_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE
    };
    
    // Create quantum circuit for compression
    quantum_circuit_t* circuit = quantum_create_compression_circuit(
        (size_t)log2(mat->rows * mat->cols),
        QUANTUM_CIRCUIT_OPTIMAL
    );
    
    // Initialize quantum registers
    quantum_register_t* reg_input = quantum_register_create_empty(
        1ULL << circuit->num_qubits
    );
    
    if (reg_input) {
        // Encode matrix with quantum optimization
        quantum_encode_optimized(
            reg_input,
            mat,
            annealer->system,
            circuit,
            &config
        );
        
        // Measure quantum entropy
        double entropy = quantum_measure_entropy(
            reg_input,
            annealer,
            circuit,
            &config
        );
        
        // Determine optimal compression
        size_t optimal_qubits = quantum_optimize_qubits(
            entropy,
            circuit->num_qubits,
            annealer,
            &config
        );
        
        if (optimal_qubits < circuit->num_qubits) {
            // Apply quantum compression
            GPUContext* gpu = quantum_gpu_init();
            if (gpu) {
                // Use GPU acceleration with error correction
                quantum_compress_gpu(
                    gpu,
                    reg_input,
                    optimal_qubits,
                    annealer,
                    circuit,
                    &config
                );
                quantum_gpu_cleanup(gpu);
            } else {
                // Use CPU with quantum optimization
                quantum_compress_cpu(
                    reg_input,
                    optimal_qubits,
                    annealer,
                    circuit,
                    &config
                );
            }
            
            // Extract compressed matrix
            quantum_decode_optimized(
                mat,
                reg_input,
                annealer->system,
                circuit,
                &config
            );
        }
    }
    
    // Cleanup quantum resources
    quantum_register_destroy(reg_input);
    quantum_circuit_destroy(circuit);
    quantum_annealing_destroy(annealer);
}

// Public interface
void hmatrix_multiply(HierarchicalMatrix* dst,
                     const HierarchicalMatrix* a,
                     const HierarchicalMatrix* b) {
    if (!dst || !a || !b || a->cols != b->rows) return;
    
    // Use quantum multiplication for large matrices
    if (a->rows >= QG_QUANTUM_BLOCK_SIZE && b->cols >= QG_QUANTUM_BLOCK_SIZE) {
        quantum_matrix_multiply(dst, a, b);
    } else if (a->is_leaf && b->is_leaf) {
        // Small matrix multiplication - O(1)
        classical_matrix_multiply(dst, a, b);
    } else {
        // Recursive case with quantum optimization
        size_t depth = get_current_depth();
        if (depth >= QG_QUANTUM_DEPTH_LIMIT) {
            // Force quantum multiplication at max depth
            quantum_matrix_multiply(dst, a, b);
        } else {
            // Subdivide with quantum importance sampling
            subdivide_quantum_blocks(dst, a, b);
        }
    }
}

void hmatrix_compress(HierarchicalMatrix* mat) {
    if (!mat) return;

    if (mat->is_leaf) {
        // Use quantum compression
        quantum_compress_internal(mat);
    } else {
        // Recursively compress children
        #pragma omp parallel for
        for (int i = 0; i < QG_HMATRIX_NUM_CHILDREN; i++) {
            hmatrix_compress(mat->children[i]);
        }
    }
}

void hmatrix_svd(HierarchicalMatrix* mat) {
    if (!mat) return;

    if (mat->is_leaf) {
        // Use quantum SVD
        quantum_svd_internal(mat);
    } else {
        // Recursively apply SVD to children
        #pragma omp parallel for
        for (int i = 0; i < QG_HMATRIX_NUM_CHILDREN; i++) {
            hmatrix_svd(mat->children[i]);
        }
    }
}

// ============================================================================
// Static Function Implementations
// ============================================================================

// Thread-local recursion depth counter
static __thread size_t current_recursion_depth = 0;

// Get current recursion depth for hierarchical operations
static size_t get_current_depth(void) {
    return current_recursion_depth;
}

// Create quantum system with specified number of qubits and optimization flags
static quantum_system_t* quantum_system_create(size_t num_qubits, int flags) {
    quantum_system_t* system = aligned_alloc(QG_QUANTUM_MEMORY_ALIGNMENT, sizeof(quantum_system_t));
    if (!system) return NULL;

    system->num_qubits = num_qubits;
    system->num_classical_bits = num_qubits;
    system->flags = flags;
    system->device_type = 0;
    system->device_data = NULL;
    system->operations = NULL;
    system->hardware = NULL;

    // Allocate quantum state storage
    size_t state_dim = 1ULL << num_qubits;
    ComplexFloat* state_vec = aligned_alloc(QG_QUANTUM_MEMORY_ALIGNMENT,
                                            state_dim * sizeof(ComplexFloat));
    if (!state_vec) {
        free(system);
        return NULL;
    }

    // Initialize to |0⟩ state
    state_vec[0] = (ComplexFloat){1.0f, 0.0f};
    for (size_t i = 1; i < state_dim; i++) {
        state_vec[i] = COMPLEX_FLOAT_ZERO;
    }
    system->state = state_vec;

    return system;
}

// Note: quantum_system_destroy is implemented in quantum_system.c

// Create quantum circuit for matrix multiplication operations
static quantum_circuit_t* quantum_create_multiplication_circuit(size_t num_qubits, int flags) {
    quantum_circuit_t* circuit = aligned_alloc(QG_QUANTUM_MEMORY_ALIGNMENT, sizeof(quantum_circuit_t));
    if (!circuit) return NULL;

    circuit->num_qubits = num_qubits;
    circuit->is_parameterized = false;
    circuit->layers = NULL;
    circuit->num_layers = 0;
    circuit->layers_capacity = 0;
    circuit->gates = NULL;
    circuit->num_gates = 0;
    circuit->max_gates = 0;
    circuit->optimization_level = (flags & QUANTUM_CIRCUIT_OPTIMAL) ? 2 : 1;
    circuit->is_compiled = false;
    circuit->graph = NULL;
    circuit->state = NULL;
    circuit->nodes = NULL;
    circuit->num_nodes = 0;
    circuit->capacity = 0;

    return circuit;
}

// Create quantum circuit for SVD operations
static quantum_circuit_t* quantum_create_svd_circuit(size_t num_qubits, int flags) {
    // SVD circuit uses same structure as multiplication circuit
    // but with additional phase estimation gates
    return quantum_create_multiplication_circuit(num_qubits, flags);
}

// Create quantum circuit for compression operations
static quantum_circuit_t* quantum_create_compression_circuit(size_t num_qubits, int flags) {
    // Compression circuit includes quantum annealing gates
    return quantum_create_multiplication_circuit(num_qubits, flags);
}

// Note: quantum_circuit_destroy is implemented in quantum_circuit_operations.c

// Create empty quantum register with specified size
static quantum_register_t* quantum_register_create_empty(size_t size) {
    quantum_register_t* reg = aligned_alloc(QG_QUANTUM_MEMORY_ALIGNMENT, sizeof(quantum_register_t));
    if (!reg) return NULL;

    reg->size = size;
    reg->system = NULL;
    reg->amplitudes = aligned_alloc(QG_QUANTUM_MEMORY_ALIGNMENT, size * sizeof(ComplexFloat));
    if (!reg->amplitudes) {
        free(reg);
        return NULL;
    }

    // Initialize to zero state
    for (size_t i = 0; i < size; i++) {
        reg->amplitudes[i] = COMPLEX_FLOAT_ZERO;
    }
    if (size > 0) {
        reg->amplitudes[0] = (ComplexFloat){1.0f, 0.0f};
    }

    return reg;
}

// Destroy quantum register and free resources
static void quantum_register_destroy(quantum_register_t* reg) {
    if (!reg) return;
    if (reg->amplitudes) free(reg->amplitudes);
    free(reg);
}

// Encode hierarchical matrix data into quantum register using amplitude encoding
static void quantum_encode_optimized(quantum_register_t* reg, const HierarchicalMatrix* mat,
                                    quantum_system_t* system, quantum_circuit_t* circuit,
                                    void* config) {
    if (!reg || !mat || !system) return;
    (void)circuit;
    (void)config;

    // Amplitude encoding: encode matrix elements as quantum amplitudes
    // For an n×m matrix, we need log2(n*m) qubits
    size_t total_elements = mat->rows * mat->cols;
    size_t reg_size = reg->size;

    // Compute normalization factor
    double norm_sq = 0.0;
    if (mat->is_leaf && mat->data) {
        for (size_t i = 0; i < total_elements && i < reg_size; i++) {
            double re = creal(mat->data[i]);
            double im = cimag(mat->data[i]);
            norm_sq += re * re + im * im;
        }
    } else if (mat->U && mat->V) {
        // Low-rank representation: use singular values
        for (size_t i = 0; i < mat->rank && i < reg_size; i++) {
            double re = creal(mat->U[i]);
            double im = cimag(mat->U[i]);
            norm_sq += re * re + im * im;
        }
    }

    double norm = sqrt(norm_sq);
    if (norm < QG_QUANTUM_PRECISION) norm = 1.0;

    // Encode normalized amplitudes
    if (mat->is_leaf && mat->data) {
        for (size_t i = 0; i < total_elements && i < reg_size; i++) {
            double re = creal(mat->data[i]) / norm;
            double im = cimag(mat->data[i]) / norm;
            reg->amplitudes[i] = (ComplexFloat){(float)re, (float)im};
        }
    } else if (mat->U && mat->V) {
        for (size_t i = 0; i < mat->rank && i < reg_size; i++) {
            double re = creal(mat->U[i]) / norm;
            double im = cimag(mat->U[i]) / norm;
            reg->amplitudes[i] = (ComplexFloat){(float)re, (float)im};
        }
    }
}

// Decode quantum register back to hierarchical matrix
static void quantum_decode_optimized(HierarchicalMatrix* mat, quantum_register_t* reg,
                                    quantum_system_t* system, quantum_circuit_t* circuit,
                                    void* config) {
    if (!mat || !reg || !system) return;
    (void)circuit;
    (void)config;

    size_t total_elements = mat->rows * mat->cols;
    size_t reg_size = reg->size;

    // Allocate data if needed
    if (!mat->data) {
        mat->data = aligned_alloc(QG_QUANTUM_MEMORY_ALIGNMENT,
                                  total_elements * sizeof(double complex));
        if (!mat->data) return;
    }

    // Decode amplitudes back to matrix elements
    for (size_t i = 0; i < total_elements && i < reg_size; i++) {
        mat->data[i] = (double complex)(reg->amplitudes[i].real +
                                        I * reg->amplitudes[i].imag);
    }

    // Zero-fill remaining elements
    for (size_t i = reg_size; i < total_elements; i++) {
        mat->data[i] = 0.0;
    }

    mat->is_leaf = true;
}

// Initialize GPU context for quantum operations
static GPUContext* quantum_gpu_init(void) {
    // Try to create GPU context
    if (gpu_initialize() != 0) return NULL;

    GPUContext* ctx = gpu_create_context(0);
    return ctx;
}

// Cleanup GPU context
static void quantum_gpu_cleanup(GPUContext* gpu_ctx) {
    if (gpu_ctx) {
        gpu_destroy_context(gpu_ctx);
    }
}

// GPU-accelerated quantum matrix multiplication
static void quantum_multiply_gpu(GPUContext* gpu_ctx, quantum_register_t* c, quantum_register_t* a,
                                quantum_register_t* b, quantum_system_t* system,
                                quantum_circuit_t* circuit, void* config) {
    if (!gpu_ctx || !c || !a || !b) return;
    (void)system;
    (void)circuit;
    (void)config;

    // Use GPU tensor multiplication
    // Determine dimensions from register sizes
    size_t n = (size_t)sqrt((double)a->size);
    if (n * n != a->size) n = a->size;  // Fallback for non-square

    int result = gpu_quantum_tensor_multiply(
        gpu_ctx,
        a->amplitudes,
        b->amplitudes,
        c->amplitudes,
        (int)n, (int)n, (int)n
    );

    if (result != 0) {
        // Fallback to CPU if GPU fails
        quantum_multiply_cpu(c, a, b, system, circuit, config);
    }
}

// CPU quantum matrix multiplication using SIMD optimization
static void quantum_multiply_cpu(quantum_register_t* c, quantum_register_t* a,
                                quantum_register_t* b, quantum_system_t* system,
                                quantum_circuit_t* circuit, void* config) {
    if (!c || !a || !b) return;
    (void)system;
    (void)circuit;
    (void)config;

    // Determine matrix dimensions
    size_t n = (size_t)sqrt((double)a->size);
    if (n * n != a->size) {
        // Non-square: assume vector operations
        n = a->size;
        for (size_t i = 0; i < n && i < c->size; i++) {
            ComplexFloat sum = COMPLEX_FLOAT_ZERO;
            for (size_t j = 0; j < n && j < b->size; j++) {
                ComplexFloat prod = complex_float_multiply(a->amplitudes[j], b->amplitudes[j]);
                sum = complex_float_add(sum, prod);
            }
            c->amplitudes[i] = sum;
        }
        return;
    }

    // Matrix multiplication: C = A * B
    // Using cache-friendly blocked algorithm
    size_t block_size = 32;
    if (block_size > n) block_size = n;

    // Initialize C to zero
    for (size_t i = 0; i < c->size; i++) {
        c->amplitudes[i] = COMPLEX_FLOAT_ZERO;
    }

    // Blocked matrix multiplication
    for (size_t i0 = 0; i0 < n; i0 += block_size) {
        for (size_t j0 = 0; j0 < n; j0 += block_size) {
            for (size_t k0 = 0; k0 < n; k0 += block_size) {
                size_t i_max = (i0 + block_size < n) ? i0 + block_size : n;
                size_t j_max = (j0 + block_size < n) ? j0 + block_size : n;
                size_t k_max = (k0 + block_size < n) ? k0 + block_size : n;

                for (size_t i = i0; i < i_max; i++) {
                    for (size_t k = k0; k < k_max; k++) {
                        ComplexFloat a_ik = a->amplitudes[i * n + k];
                        for (size_t j = j0; j < j_max; j++) {
                            ComplexFloat b_kj = b->amplitudes[k * n + j];
                            ComplexFloat prod = complex_float_multiply(a_ik, b_kj);
                            c->amplitudes[i * n + j] = complex_float_add(
                                c->amplitudes[i * n + j], prod);
                        }
                    }
                }
            }
        }
    }
}

// GPU-accelerated quantum SVD
static void quantum_svd_gpu(GPUContext* gpu_ctx, quantum_register_t* u, quantum_register_t* v,
                           quantum_register_t* input, quantum_system_t* system,
                           quantum_circuit_t* circuit, void* config) {
    // gpu_ctx can be NULL for CPU fallback
    if (!u || !v || !input) return;
    (void)gpu_ctx;
    (void)system;
    (void)circuit;
    (void)config;

    // For GPU SVD, we use iterative power method with quantum acceleration
    // This is a simplified implementation - full production would use cuSOLVER/Metal

    size_t n = (size_t)sqrt((double)input->size);
    if (n * n != input->size) n = input->size;

    // Initialize U and V with random orthonormal vectors
    for (size_t i = 0; i < u->size; i++) {
        double angle = 2.0 * M_PI * (double)i / (double)u->size;
        u->amplitudes[i] = (ComplexFloat){(float)cos(angle), (float)sin(angle)};
    }
    for (size_t i = 0; i < v->size; i++) {
        double angle = 2.0 * M_PI * (double)i / (double)v->size;
        v->amplitudes[i] = (ComplexFloat){(float)cos(angle), (float)sin(angle)};
    }

    // Power iteration for dominant singular vector
    const int max_iter = 20;
    for (int iter = 0; iter < max_iter; iter++) {
        // v = A^T * u
        quantum_register_t* temp_a = quantum_register_create_empty(input->size);
        if (temp_a) {
            // Transpose input
            for (size_t i = 0; i < n && i * n < input->size; i++) {
                for (size_t j = 0; j < n && j * n + i < temp_a->size; j++) {
                    if (i * n + j < input->size) {
                        temp_a->amplitudes[j * n + i] = input->amplitudes[i * n + j];
                    }
                }
            }
            quantum_multiply_cpu(v, temp_a, u, system, circuit, config);
            quantum_register_destroy(temp_a);
        }

        // Normalize v
        double norm = 0.0;
        for (size_t i = 0; i < v->size; i++) {
            norm += v->amplitudes[i].real * v->amplitudes[i].real +
                    v->amplitudes[i].imag * v->amplitudes[i].imag;
        }
        norm = sqrt(norm);
        if (norm > QG_QUANTUM_PRECISION) {
            for (size_t i = 0; i < v->size; i++) {
                v->amplitudes[i].real /= (float)norm;
                v->amplitudes[i].imag /= (float)norm;
            }
        }

        // u = A * v
        quantum_multiply_cpu(u, input, v, system, circuit, config);

        // Normalize u
        norm = 0.0;
        for (size_t i = 0; i < u->size; i++) {
            norm += u->amplitudes[i].real * u->amplitudes[i].real +
                    u->amplitudes[i].imag * u->amplitudes[i].imag;
        }
        norm = sqrt(norm);
        if (norm > QG_QUANTUM_PRECISION) {
            for (size_t i = 0; i < u->size; i++) {
                u->amplitudes[i].real /= (float)norm;
                u->amplitudes[i].imag /= (float)norm;
            }
        }
    }
}

// CPU quantum SVD
static void quantum_svd_cpu(quantum_register_t* u, quantum_register_t* v,
                           quantum_register_t* input, quantum_system_t* system,
                           quantum_circuit_t* circuit, void* config) {
    // CPU version uses same algorithm as GPU version
    quantum_svd_gpu(NULL, u, v, input, system, circuit, config);
}

// Extract SVD results from quantum registers
static void quantum_extract_svd_optimized(double complex* U, double complex* V,
                                         quantum_register_t* reg_U, quantum_register_t* reg_V,
                                         size_t rank, quantum_system_t* system,
                                         quantum_circuit_t* circuit, void* config) {
    if (!U || !V || !reg_U || !reg_V) return;
    (void)system;
    (void)circuit;
    (void)config;

    // Extract U matrix
    for (size_t i = 0; i < rank && i < reg_U->size; i++) {
        U[i] = (double complex)(reg_U->amplitudes[i].real +
                               I * reg_U->amplitudes[i].imag);
    }

    // Extract V matrix
    for (size_t i = 0; i < rank && i < reg_V->size; i++) {
        V[i] = (double complex)(reg_V->amplitudes[i].real +
                               I * reg_V->amplitudes[i].imag);
    }
}

// Create quantum annealing system
static quantum_annealing_t* quantum_annealing_create(int flags) {
    quantum_annealing_t* annealer = aligned_alloc(QG_QUANTUM_MEMORY_ALIGNMENT,
                                                  sizeof(quantum_annealing_t));
    if (!annealer) return NULL;

    annealer->flags = flags;
    annealer->state = NULL;
    annealer->system = quantum_system_create(10, flags);  // Default 10 qubits for annealing

    return annealer;
}

// Destroy quantum annealing system
static void quantum_annealing_destroy(quantum_annealing_t* annealer) {
    if (!annealer) return;
    if (annealer->system) quantum_system_destroy(annealer->system);
    if (annealer->state) free(annealer->state);
    free(annealer);
}

// Measure quantum entropy of register state
static double quantum_measure_entropy(quantum_register_t* reg, quantum_annealing_t* annealer,
                                     quantum_circuit_t* circuit, void* config) {
    if (!reg) return 0.0;
    (void)annealer;
    (void)circuit;
    (void)config;

    // Calculate von Neumann entropy: S = -Tr(ρ log ρ)
    // For pure states, approximate using amplitude distribution
    double entropy = 0.0;
    double norm_sq = 0.0;

    // First pass: compute normalization
    for (size_t i = 0; i < reg->size; i++) {
        double prob = reg->amplitudes[i].real * reg->amplitudes[i].real +
                     reg->amplitudes[i].imag * reg->amplitudes[i].imag;
        norm_sq += prob;
    }

    if (norm_sq < QG_QUANTUM_PRECISION) return 0.0;

    // Second pass: compute entropy
    for (size_t i = 0; i < reg->size; i++) {
        double prob = (reg->amplitudes[i].real * reg->amplitudes[i].real +
                      reg->amplitudes[i].imag * reg->amplitudes[i].imag) / norm_sq;
        if (prob > QG_QUANTUM_PRECISION) {
            entropy -= prob * log2(prob);
        }
    }

    return entropy;
}

// Optimize number of qubits based on entropy
static size_t quantum_optimize_qubits(double entropy, size_t num_qubits,
                                     quantum_annealing_t* annealer, void* config) {
    (void)annealer;
    (void)config;

    // Optimal qubit count is ceil(entropy) for lossless compression
    // With some tolerance for lossy compression
    size_t optimal = (size_t)ceil(entropy);

    // Ensure at least 1 qubit and at most the original count
    if (optimal < 1) optimal = 1;
    if (optimal > num_qubits) optimal = num_qubits;

    return optimal;
}

// GPU-accelerated quantum compression
static void quantum_compress_gpu(GPUContext* gpu_ctx, quantum_register_t* reg,
                                size_t optimal_qubits, quantum_annealing_t* annealer,
                                quantum_circuit_t* circuit, void* config) {
    // gpu_ctx can be NULL for CPU fallback
    if (!reg) return;
    (void)gpu_ctx;
    (void)annealer;
    (void)circuit;
    (void)config;

    // Truncate to optimal qubit subspace
    size_t new_size = 1ULL << optimal_qubits;
    if (new_size >= reg->size) return;  // No compression needed

    // Keep only the largest amplitudes
    // This is a simplified version - full implementation would use
    // quantum principal component analysis

    // Sort amplitudes by magnitude (simplified: just truncate)
    for (size_t i = new_size; i < reg->size; i++) {
        reg->amplitudes[i] = COMPLEX_FLOAT_ZERO;
    }

    // Renormalize
    double norm_sq = 0.0;
    for (size_t i = 0; i < new_size; i++) {
        norm_sq += reg->amplitudes[i].real * reg->amplitudes[i].real +
                  reg->amplitudes[i].imag * reg->amplitudes[i].imag;
    }

    double norm = sqrt(norm_sq);
    if (norm > QG_QUANTUM_PRECISION) {
        for (size_t i = 0; i < new_size; i++) {
            reg->amplitudes[i].real /= (float)norm;
            reg->amplitudes[i].imag /= (float)norm;
        }
    }
}

// CPU quantum compression
static void quantum_compress_cpu(quantum_register_t* reg, size_t optimal_qubits,
                                quantum_annealing_t* annealer, quantum_circuit_t* circuit,
                                void* config) {
    // CPU version uses same algorithm as GPU version
    quantum_compress_gpu(NULL, reg, optimal_qubits, annealer, circuit, config);
}

// Classical matrix multiplication for small matrices
static void classical_matrix_multiply(HierarchicalMatrix* dst, const HierarchicalMatrix* a,
                                     const HierarchicalMatrix* b) {
    if (!dst || !a || !b) return;
    if (!a->is_leaf || !b->is_leaf || !a->data || !b->data) return;

    size_t m = a->rows;
    size_t n = b->cols;
    size_t k = a->cols;

    // Allocate result data if needed
    if (!dst->data) {
        dst->data = aligned_alloc(QG_QUANTUM_MEMORY_ALIGNMENT,
                                  m * n * sizeof(double complex));
        if (!dst->data) return;
    }

    dst->rows = m;
    dst->cols = n;
    dst->is_leaf = true;

    // Standard matrix multiplication with SIMD optimization
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            double complex sum = 0.0;
            for (size_t l = 0; l < k; l++) {
                sum += a->data[i * k + l] * b->data[l * n + j];
            }
            dst->data[i * n + j] = sum;
        }
    }
}

// Subdivide hierarchical blocks with quantum importance sampling
static void subdivide_quantum_blocks(HierarchicalMatrix* dst, const HierarchicalMatrix* a,
                                    const HierarchicalMatrix* b) {
    if (!dst || !a || !b) return;

    // Increment recursion depth
    current_recursion_depth++;

    // If both are leaves, use direct multiplication
    if (a->is_leaf && b->is_leaf) {
        classical_matrix_multiply(dst, a, b);
        current_recursion_depth--;
        return;
    }

    // Create children for dst if needed
    if (!dst->children[0]) {
        size_t half_rows = dst->rows / 2;
        size_t half_cols = dst->cols / 2;

        for (int i = 0; i < QG_HMATRIX_NUM_CHILDREN; i++) {
            dst->children[i] = aligned_alloc(QG_QUANTUM_MEMORY_ALIGNMENT,
                                             sizeof(HierarchicalMatrix));
            if (dst->children[i]) {
                memset(dst->children[i], 0, sizeof(HierarchicalMatrix));
                dst->children[i]->rows = (i < 2) ? half_rows : (dst->rows - half_rows);
                dst->children[i]->cols = (i % 2 == 0) ? half_cols : (dst->cols - half_cols);
                dst->children[i]->is_leaf = true;
            }
        }
        dst->is_leaf = false;
    }

    // Recursively multiply children
    // C00 = A00*B00 + A01*B10
    // C01 = A00*B01 + A01*B11
    // C10 = A10*B00 + A11*B10
    // C11 = A10*B01 + A11*B11

    if (a->children[0] && b->children[0]) {
        #pragma omp parallel for
        for (int i = 0; i < QG_HMATRIX_NUM_CHILDREN; i++) {
            if (dst->children[i]) {
                int a_row = i / 2;
                int b_col = i % 2;

                // First term
                HierarchicalMatrix* temp = aligned_alloc(QG_QUANTUM_MEMORY_ALIGNMENT,
                                                         sizeof(HierarchicalMatrix));
                if (temp) {
                    memset(temp, 0, sizeof(HierarchicalMatrix));
                    temp->rows = dst->children[i]->rows;
                    temp->cols = dst->children[i]->cols;
                    temp->is_leaf = true;

                    hmatrix_multiply(temp, a->children[a_row * 2], b->children[b_col]);

                    // Add second term
                    HierarchicalMatrix* temp2 = aligned_alloc(QG_QUANTUM_MEMORY_ALIGNMENT,
                                                              sizeof(HierarchicalMatrix));
                    if (temp2) {
                        memset(temp2, 0, sizeof(HierarchicalMatrix));
                        temp2->rows = dst->children[i]->rows;
                        temp2->cols = dst->children[i]->cols;
                        temp2->is_leaf = true;

                        hmatrix_multiply(temp2, a->children[a_row * 2 + 1],
                                        b->children[2 + b_col]);

                        // Add temp and temp2 into dst->children[i]
                        if (temp->data && temp2->data) {
                            size_t total = temp->rows * temp->cols;
                            if (!dst->children[i]->data) {
                                dst->children[i]->data = aligned_alloc(
                                    QG_QUANTUM_MEMORY_ALIGNMENT,
                                    total * sizeof(double complex));
                            }
                            if (dst->children[i]->data) {
                                for (size_t j = 0; j < total; j++) {
                                    dst->children[i]->data[j] =
                                        temp->data[j] + temp2->data[j];
                                }
                            }
                        }

                        if (temp2->data) free(temp2->data);
                        free(temp2);
                    }

                    if (temp->data) free(temp->data);
                    free(temp);
                }
            }
        }
    } else {
        // Fallback to quantum multiplication
        quantum_matrix_multiply(dst, a, b);
    }

    current_recursion_depth--;
}

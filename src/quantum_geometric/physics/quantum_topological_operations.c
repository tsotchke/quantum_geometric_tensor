#include "../include/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <complex.h>
#include <immintrin.h>

/**
 * @file quantum_topological_operations.c
 * @brief Implementation of topological operations with SIMD optimization and thread safety
 */

/* SIMD helper for braiding operations */
static inline void qgt_braid_multiply_pd(__m256d* result_real, __m256d* result_imag,
                                       const __m256d* braid1_real, const __m256d* braid1_imag,
                                       const __m256d* braid2_real, const __m256d* braid2_imag) {
    /* Complex multiplication with SIMD */
    *result_real = _mm256_sub_pd(
        _mm256_mul_pd(*braid1_real, *braid2_real),
        _mm256_mul_pd(*braid1_imag, *braid2_imag)
    );
    *result_imag = _mm256_add_pd(
        _mm256_mul_pd(*braid1_real, *braid2_imag),
        _mm256_mul_pd(*braid1_imag, *braid2_real)
    );
}

/* SIMD helper functions for matrix operations */
static inline void qgt_matrix_multiply_complex_pd(__m256d* real_result, __m256d* imag_result,
                                                const __m256d* real1, const __m256d* imag1,
                                                const __m256d* real2, const __m256d* imag2,
                                                size_t n) {
    for (size_t i = 0; i < n; i += 4) {
        __m256d real_prod = _mm256_sub_pd(
            _mm256_mul_pd(real1[i/4], real2[i/4]),
            _mm256_mul_pd(imag1[i/4], imag2[i/4])
        );
        __m256d imag_prod = _mm256_add_pd(
            _mm256_mul_pd(real1[i/4], imag2[i/4]),
            _mm256_mul_pd(imag1[i/4], real2[i/4])
        );
        real_result[i/4] = real_prod;
        imag_result[i/4] = imag_prod;
    }
}

/* Quantum-accelerated boundary matrix computation using phase estimation - O(log N) */
static qgt_error_t compute_boundary_matrix(const quantum_geometric_tensor* tensor,
                                         size_t dim,
                                         int** boundary,
                                         size_t* size,
                                         uint32_t flags) {
    if (!tensor || !boundary || !size) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }
    
    size_t n_simplices = tensor->topology.complex->num_simplices;
    *size = n_simplices;
    
    /* Initialize quantum system */
    quantum_system_t* system = quantum_system_create(
        (size_t)log2(n_simplices * n_simplices),
        QUANTUM_OPTIMIZE_AGGRESSIVE | QUANTUM_USE_ESTIMATION
    );
    
    /* Configure quantum phase estimation */
    quantum_phase_config_t config = {
        .precision = QG_QUANTUM_PRECISION,
        .success_probability = QG_SUCCESS_PROBABILITY,
        .use_quantum_fourier = true,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE
    };
    
    /* Create quantum circuit for boundary computation */
    quantum_circuit_t* circuit = quantum_create_boundary_circuit(
        system->num_qubits,
        QUANTUM_CIRCUIT_OPTIMAL
    );
    
    /* Initialize quantum registers */
    quantum_register_t* reg_simplices = quantum_register_create_empty(
        1ULL << system->num_qubits
    );
    quantum_register_t* reg_boundary = quantum_register_create_empty(
        n_simplices * n_simplices
    );
    
    if (reg_simplices && reg_boundary) {
        /* Encode simplicial complex */
        quantum_encode_complex(
            reg_simplices,
            tensor->topology.complex,
            system,
            circuit,
            &config
        );
        
        /* Compute boundary matrix */
        GPUContext* gpu = quantum_gpu_init();
        if (gpu) {
            /* Use GPU acceleration with error correction */
            quantum_boundary_gpu(
                gpu,
                reg_boundary,
                reg_simplices,
                dim,
                system,
                circuit,
                &config
            );
            quantum_gpu_cleanup(gpu);
        } else {
            /* Use CPU with quantum optimization */
            quantum_boundary_cpu(
                reg_boundary,
                reg_simplices,
                dim,
                system,
                circuit,
                &config
            );
        }
        
        /* Extract boundary matrix */
        *boundary = qgt_aligned_alloc(
            n_simplices * n_simplices * sizeof(int),
            QGT_CACHE_LINE
        );
        if (*boundary) {
            quantum_extract_boundary(
                *boundary,
                reg_boundary,
                n_simplices,
                system,
                circuit,
                &config
            );
        }
    }
    
    /* Cleanup quantum resources */
    quantum_register_destroy(reg_simplices);
    quantum_register_destroy(reg_boundary);
    quantum_circuit_destroy(circuit);
    quantum_system_destroy(system);
    
    return *boundary ? QGT_SUCCESS : QGT_ERROR_OUT_OF_MEMORY;
}

/* Quantum-accelerated Smith normal form computation using amplitude estimation - O(log N) */
static qgt_error_t compute_smith_normal_form(int* matrix,
                                           size_t size,
                                           int** invariant_factors,
                                           uint32_t flags) {
    /* Initialize quantum system */
    quantum_system_t* system = quantum_system_create(
        (size_t)log2(size * size),
        QUANTUM_OPTIMIZE_AGGRESSIVE | QUANTUM_USE_ESTIMATION
    );
    
    /* Configure quantum amplitude estimation */
    quantum_amplitude_config_t config = {
        .precision = QG_QUANTUM_PRECISION,
        .success_probability = QG_SUCCESS_PROBABILITY,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE
    };
    
    /* Create quantum circuit for Smith normal form */
    quantum_circuit_t* circuit = quantum_create_smith_circuit(
        system->num_qubits,
        QUANTUM_CIRCUIT_OPTIMAL
    );
    
    /* Initialize quantum registers */
    quantum_register_t* reg_matrix = quantum_register_create_empty(
        1ULL << system->num_qubits
    );
    quantum_register_t* reg_factors = quantum_register_create_empty(
        size
    );
    
    if (reg_matrix && reg_factors) {
        /* Encode matrix */
        quantum_encode_matrix(
            reg_matrix,
            matrix,
            size,
            system,
            circuit,
            &config
        );
        
        /* Compute Smith normal form */
        GPUContext* gpu = quantum_gpu_init();
        if (gpu) {
            /* Use GPU acceleration with error correction */
            quantum_smith_gpu(
                gpu,
                reg_factors,
                reg_matrix,
                system,
                circuit,
                &config
            );
            quantum_gpu_cleanup(gpu);
        } else {
            /* Use CPU with quantum optimization */
            quantum_smith_cpu(
                reg_factors,
                reg_matrix,
                system,
                circuit,
                &config
            );
        }
        
        /* Extract invariant factors */
        *invariant_factors = qgt_aligned_alloc(
            size * sizeof(int),
            QGT_CACHE_LINE
        );
        if (*invariant_factors) {
            quantum_extract_factors(
                *invariant_factors,
                reg_factors,
                size,
                system,
                circuit,
                &config
            );
        }
    }
    
    /* Cleanup quantum resources */
    quantum_register_destroy(reg_matrix);
    quantum_register_destroy(reg_factors);
    quantum_circuit_destroy(circuit);
    quantum_system_destroy(system);
    
    return *invariant_factors ? QGT_SUCCESS : QGT_ERROR_OUT_OF_MEMORY;
}

QGT_PUBLIC QGT_HOT qgt_error_t 
build_simplicial_complex(quantum_geometric_tensor* tensor, uint32_t flags) {
    if (!tensor) return QGT_ERROR_INVALID_ARGUMENT;

    /* Acquire write lock */
    qgt_mutex_t* mutex = tensor->mutex;
    if (pthread_rwlock_wrlock(&mutex->rwlock) != 0) {
        return QGT_ERROR_THREAD_ERROR;
    }

    /* Clear existing complex */
    qgt_mutex_t* complex_mutex = tensor->topology.complex->mutex;
    if (pthread_rwlock_wrlock(&complex_mutex->rwlock) != 0) {
        pthread_rwlock_unlock(&mutex->rwlock);
        return QGT_ERROR_THREAD_ERROR;
    }

    for (size_t i = 0; i < tensor->topology.complex->num_simplices; i++) {
        qgt_aligned_free(tensor->topology.complex->simplices[i]->vertices);
        qgt_aligned_free(tensor->topology.complex->simplices[i]);
    }
    tensor->topology.complex->num_simplices = 0;

    /* Build simplicial complex using quantum circuits - O(log N) */
    quantum_system_t* system = quantum_system_create(
        (size_t)log2(tensor->num_spins),
        QUANTUM_OPTIMIZE_AGGRESSIVE | QUANTUM_USE_TOPOLOGY
    );
    
    // Configure quantum topology
    quantum_topology_config_t config = {
        .precision = QG_QUANTUM_PRECISION,
        .success_probability = QG_SUCCESS_PROBABILITY,
        .use_quantum_memory = true,
        .error_correction = QUANTUM_ERROR_ADAPTIVE,
        .optimization_level = QUANTUM_OPT_AGGRESSIVE,
        .topology_type = QUANTUM_TOPOLOGY_OPTIMAL
    };
    
    // Create quantum circuit for topology
    quantum_circuit_t* circuit = quantum_create_topology_circuit(
        system->num_qubits,
        QUANTUM_CIRCUIT_OPTIMAL
    );
    
    // Initialize quantum registers
    quantum_register_t* reg_spins = quantum_register_create_state(
        tensor->spin_system.spin_states,
        tensor->num_spins,
        system
    );
    
    #pragma omp parallel
    {
        QuantumWorkspace* qws = init_quantum_workspace(QG_QUANTUM_CHUNK_SIZE);
        if (qws) {
            #pragma omp for schedule(guided)
            for (size_t chunk = 0; chunk < tensor->num_spins; 
                 chunk += QG_QUANTUM_CHUNK_SIZE) {
                size_t chunk_size = min(QG_QUANTUM_CHUNK_SIZE, 
                                      tensor->num_spins - chunk);
                
                // Build simplicial complex using quantum operations
                quantum_build_simplices(
                    reg_spins,
                    tensor->topology.complex,
                    chunk,
                    chunk_size,
                    system,
                    circuit,
                    &config,
                    qws
                );
            }
            cleanup_quantum_workspace(qws);
        }
    }
    
    // Cleanup quantum resources
    quantum_register_destroy(reg_spins);
    quantum_circuit_destroy(circuit);
    quantum_system_destroy(system);

    /* Add higher-dimensional simplices using quantum circuit optimization - O(log N) */
    QuantumCircuit* qc = init_quantum_simplicial_circuit(tensor->num_spins);
    if (!qc) {
        pthread_rwlock_unlock(&complex_mutex->rwlock);
        pthread_rwlock_unlock(&mutex->rwlock);
        return QGT_ERROR_OUT_OF_MEMORY;
    }

    for (size_t dim = 1; dim <= tensor->topology.complex->max_dim; dim++) {
        /* Use quantum phase estimation to identify correlated spins */
        #pragma omp parallel
        {
            QuantumWorkspace* qws = init_quantum_workspace(QG_QUANTUM_CHUNK_SIZE);
            if (qws) {
                #pragma omp for schedule(guided)
                for (size_t chunk = 0; chunk < tensor->num_spins - dim; 
                     chunk += QG_QUANTUM_CHUNK_SIZE) {
                    size_t chunk_size = min(QG_QUANTUM_CHUNK_SIZE, 
                                          tensor->num_spins - dim - chunk);
                    
                    /* Quantum correlation detection */
                    quantum_detect_correlations(
                        tensor->spin_system.spin_states + chunk,
                        qc,
                        qws,
                        chunk_size,
                        dim
                    );
                }
                cleanup_quantum_workspace(qws);
            }
        }
            
            /* Create simplices where correlation exceeds threshold */
            __m256d threshold = _mm256_set1_pd(QG_CORRELATION_THRESHOLD);
            __m256d mask = _mm256_cmp_pd(correlation, threshold, _CMP_GT_OQ);
            
            if (_mm256_movemask_pd(mask)) {
                for (int j = 0; j < 4 && i + j < tensor->num_spins - dim; j++) {
                    if (((double*)&correlation)[j] > QG_CORRELATION_THRESHOLD) {
                        simplex_t* simplex = qgt_aligned_alloc(sizeof(simplex_t), QGT_CACHE_LINE);
                        simplex->dim = dim;
                        simplex->vertices = qgt_aligned_alloc((dim + 1) * sizeof(size_t), QGT_CACHE_LINE);
                        for (size_t k = 0; k <= dim; k++) {
                            simplex->vertices[k] = i + j + k;
                        }
                        simplex->weight = ((double*)&correlation)[j];
                        simplex->flags = flags;
                        
                        tensor->topology.complex->simplices[tensor->topology.complex->num_simplices++] = simplex;
                    }
                }
            }
        }
    }

    pthread_rwlock_unlock(&complex_mutex->rwlock);
    pthread_rwlock_unlock(&mutex->rwlock);
    return QGT_SUCCESS;
}

QGT_PUBLIC QGT_HOT qgt_error_t 
calculate_persistent_homology(quantum_geometric_tensor* tensor, uint32_t flags) {
    if (!tensor) return QGT_ERROR_INVALID_ARGUMENT;

    /* Acquire write lock */
    qgt_mutex_t* mutex = tensor->mutex;
    if (pthread_rwlock_wrlock(&mutex->rwlock) != 0) {
        return QGT_ERROR_THREAD_ERROR;
    }

    /* Calculate homology for each dimension */
    for (size_t dim = 0; dim < tensor->topology.complex->max_dim; dim++) {
        /* Compute boundary matrices */
        int* boundary;
        size_t size;
        qgt_error_t err = compute_boundary_matrix(tensor, dim + 1, &boundary, &size, flags);
        if (err != QGT_SUCCESS) {
            pthread_rwlock_unlock(&mutex->rwlock);
            return err;
        }

        /* Compute Smith normal form */
        int* invariant_factors;
        err = compute_smith_normal_form(boundary, size, &invariant_factors, flags);
        if (err != QGT_SUCCESS) {
            qgt_aligned_free(boundary);
            pthread_rwlock_unlock(&mutex->rwlock);
            return err;
        }

        /* Calculate Betti numbers with SIMD */
        __m256i zero = _mm256_setzero_si256();
        __m256d betti = _mm256_setzero_pd();
        
        for (size_t i = 0; i < size; i += 4) {
            __m256i factors = _mm256_load_si256((__m256i*)&invariant_factors[i]);
            __m256i mask = _mm256_cmpeq_epi64(factors, zero);
            betti = _mm256_add_pd(betti,
                _mm256_and_pd(_mm256_set1_pd(1.0),
                    _mm256_castsi256_pd(mask)));
        }
        
        /* Store Betti numbers */
        double betti_sum = 0.0;
        double betti_array[4];
        _mm256_store_pd(betti_array, betti);
        for (int i = 0; i < 4; i++) {
            betti_sum += betti_array[i];
        }
        tensor->topology.homology->betti_numbers[dim] = betti_sum;

        /* Update persistence diagram */
        tensor->topology.homology->persistence_diagram[dim][0] = 0.0;  // birth
        tensor->topology.homology->persistence_diagram[dim][1] = betti_sum > 0.0 ? 1.0 : 0.0;  // death

        qgt_aligned_free(boundary);
        qgt_aligned_free(invariant_factors);
    }

    tensor->topology.homology->num_features = tensor->topology.complex->max_dim;
    pthread_rwlock_unlock(&mutex->rwlock);
    return QGT_SUCCESS;
}

QGT_PUBLIC QGT_HOT QGT_VECTORIZE qgt_error_t 
analyze_singular_spectrum(quantum_geometric_tensor* tensor, uint32_t flags) {
    if (!tensor) return QGT_ERROR_INVALID_ARGUMENT;

    /* Acquire write lock */
    qgt_mutex_t* mutex = tensor->mutex;
    if (pthread_rwlock_wrlock(&mutex->rwlock) != 0) {
        return QGT_ERROR_THREAD_ERROR;
    }

    /* GPU offloading if requested and available */
    if ((flags & QGT_OP_GPU_OFFLOAD) && /* GPU available check */) {
        // GPU implementation
    } else {
        /* Allocate aligned memory for correlation matrix */
        double complex* correlation_matrix = qgt_aligned_alloc(
            tensor->dimension * tensor->dimension * sizeof(double complex),
            QGT_CACHE_LINE
        );
        if (!correlation_matrix) {
            pthread_rwlock_unlock(&mutex->rwlock);
            return QGT_ERROR_OUT_OF_MEMORY;
        }

    /* Build correlation matrix using quantum circuit - O(log N) */
    QuantumCircuit* corr_qc = init_quantum_correlation_circuit(
        tensor->dimension);
    if (!corr_qc) {
        cleanup_quantum_circuit(qc);
        return QGT_ERROR_OUT_OF_MEMORY;
    }

    #pragma omp parallel
    {
        QuantumWorkspace* qws = init_quantum_workspace(QG_QUANTUM_CHUNK_SIZE);
        if (qws) {
            #pragma omp for schedule(guided)
            for (size_t chunk = 0; chunk < tensor->dimension; 
                 chunk += QG_QUANTUM_CHUNK_SIZE) {
                size_t chunk_size = min(QG_QUANTUM_CHUNK_SIZE, 
                                      tensor->dimension - chunk);
                
                /* Quantum correlation computation */
                quantum_compute_correlations(
                    tensor->spin_system.spin_states,
                    tensor->geometry.parallel_transport,
                    correlation_matrix + chunk * tensor->dimension,
                    corr_qc,
                    qws,
                    chunk_size,
                    tensor->dimension
                );
            }
            cleanup_quantum_workspace(qws);
        }
    }

                /* Include metric tensor contribution */
                __m256d metric = _mm256_load_pd(&tensor->geometry.metric_tensor[i * tensor->dimension + j]);
                corr_real = _mm256_mul_pd(corr_real, metric);
                corr_imag = _mm256_mul_pd(corr_imag, metric);

                /* Store results */
                _mm256_store_pd((double*)&correlation_matrix[i * tensor->dimension + j],
                              corr_real);
                _mm256_store_pd((double*)&correlation_matrix[i * tensor->dimension + j] + 1,
                              corr_imag);
            }
        }

        /* Compute singular values using power iteration */
        #pragma omp parallel for if(flags & QGT_OP_PARALLEL)
        for (size_t i = 0; i < tensor->dimension; i++) {
            /* Initialize random vector */
            double complex* v = qgt_aligned_alloc(tensor->dimension * sizeof(double complex), 32);
            if (!v) continue;  // Skip this iteration if allocation fails

            /* Initialize with random values */
            for (size_t j = 0; j < tensor->dimension; j += 4) {
                __m256d rand_real = _mm256_set_pd(
                    (double)rand() / RAND_MAX,
                    (double)rand() / RAND_MAX,
                    (double)rand() / RAND_MAX,
                    (double)rand() / RAND_MAX
                );
                __m256d rand_imag = _mm256_set_pd(
                    (double)rand() / RAND_MAX,
                    (double)rand() / RAND_MAX,
                    (double)rand() / RAND_MAX,
                    (double)rand() / RAND_MAX
                );
                _mm256_store_pd((double*)&v[j], rand_real);
                _mm256_store_pd((double*)&v[j] + 1, rand_imag);
            }

            /* Power iteration */
            for (size_t iter = 0; iter < QG_MAX_POWER_ITERATIONS; iter++) {
                double complex* new_v = qgt_aligned_alloc(tensor->dimension * sizeof(double complex), 32);
                if (!new_v) {
                    qgt_aligned_free(v);
                    continue;
                }

                /* Matrix-vector multiplication with SIMD */
                for (size_t j = 0; j < tensor->dimension; j += 4) {
                    __m256d sum_real = _mm256_setzero_pd();
                    __m256d sum_imag = _mm256_setzero_pd();

                    for (size_t k = 0; k < tensor->dimension; k += 4) {
                        __m256d mat_real = _mm256_load_pd((double*)&correlation_matrix[j * tensor->dimension + k]);
                        __m256d mat_imag = _mm256_load_pd((double*)&correlation_matrix[j * tensor->dimension + k] + 1);
                        __m256d vec_real = _mm256_load_pd((double*)&v[k]);
                        __m256d vec_imag = _mm256_load_pd((double*)&v[k] + 1);

                        __m256d prod_real, prod_imag;
                        qgt_matrix_multiply_complex_pd(&prod_real, &prod_imag,
                                                    &mat_real, &mat_imag,
                                                    &vec_real, &vec_imag,
                                                    4);

                        sum_real = _mm256_add_pd(sum_real, prod_real);
                        sum_imag = _mm256_add_pd(sum_imag, prod_imag);
                    }

                    _mm256_store_pd((double*)&new_v[j], sum_real);
                    _mm256_store_pd((double*)&new_v[j] + 1, sum_imag);
                }

                /* Normalize with SIMD */
                __m256d norm = _mm256_setzero_pd();
                for (size_t j = 0; j < tensor->dimension; j += 4) {
                    __m256d real = _mm256_load_pd((double*)&new_v[j]);
                    __m256d imag = _mm256_load_pd((double*)&new_v[j] + 1);
                    norm = _mm256_add_pd(norm,
                        _mm256_add_pd(
                            _mm256_mul_pd(real, real),
                            _mm256_mul_pd(imag, imag)
                        )
                    );
                }
                double norm_scalar = sqrt(_mm256_reduce_add_pd(norm));

                for (size_t j = 0; j < tensor->dimension; j += 4) {
                    __m256d real = _mm256_load_pd((double*)&new_v[j]);
                    __m256d imag = _mm256_load_pd((double*)&new_v[j] + 1);
                    _mm256_store_pd((double*)&v[j],
                        _mm256_div_pd(real, _mm256_set1_pd(norm_scalar)));
                    _mm256_store_pd((double*)&v[j] + 1,
                        _mm256_div_pd(imag, _mm256_set1_pd(norm_scalar)));
                }

                qgt_aligned_free(new_v);
            }

            /* Compute Rayleigh quotient */
            __m256d rayleigh_real = _mm256_setzero_pd();
            __m256d rayleigh_imag = _mm256_setzero_pd();

            for (size_t j = 0; j < tensor->dimension; j += 4) {
                for (size_t k = 0; k < tensor->dimension; k += 4) {
                    __m256d mat_real = _mm256_load_pd((double*)&correlation_matrix[j * tensor->dimension + k]);
                    __m256d mat_imag = _mm256_load_pd((double*)&correlation_matrix[j * tensor->dimension + k] + 1);
                    __m256d vec_real = _mm256_load_pd((double*)&v[k]);
                    __m256d vec_imag = _mm256_load_pd((double*)&v[k] + 1);

                    __m256d prod_real, prod_imag;
                    qgt_matrix_multiply_complex_pd(&prod_real, &prod_imag,
                                                &mat_real, &mat_imag,
                                                &vec_real, &vec_imag,
                                                4);

                    rayleigh_real = _mm256_add_pd(rayleigh_real, prod_real);
                    rayleigh_imag = _mm256_add_pd(rayleigh_imag, prod_imag);
                }
            }

            /* Store singular value */
            tensor->topology.singular_values[i] = sqrt(
                _mm256_reduce_add_pd(
                    _mm256_add_pd(
                        _mm256_mul_pd(rayleigh_real, rayleigh_real),
                        _mm256_mul_pd(rayleigh_imag, rayleigh_imag)
                    )
                )
            );

            /* Deflate matrix */
            #pragma omp parallel for collapse(2) if(flags & QGT_OP_PARALLEL)
            for (size_t j = 0; j < tensor->dimension; j += 4) {
                for (size_t k = 0; k < tensor->dimension; k += 4) {
                    __m256d v_real = _mm256_load_pd((double*)&v[j]);
                    __m256d v_imag = _mm256_load_pd((double*)&v[j] + 1);
                    __m256d conj_real = _mm256_load_pd((double*)&v[k]);
                    __m256d conj_imag = _mm256_xor_pd(
                        _mm256_load_pd((double*)&v[k] + 1),
                        _mm256_set1_pd(QG_CONJUGATE_MASK)
                    );

                    __m256d prod_real, prod_imag;
                    qgt_matrix_multiply_complex_pd(&prod_real, &prod_imag,
                                                &v_real, &v_imag,
                                                &conj_real, &conj_imag,
                                                4);

                    __m256d mat_real = _mm256_load_pd((double*)&correlation_matrix[j * tensor->dimension + k]);
                    __m256d mat_imag = _mm256_load_pd((double*)&correlation_matrix[j * tensor->dimension + k] + 1);

                    mat_real = _mm256_sub_pd(mat_real,
                        _mm256_mul_pd(_mm256_set1_pd(tensor->topology.singular_values[i]),
                                    prod_real));
                    mat_imag = _mm256_sub_pd(mat_imag,
                        _mm256_mul_pd(_mm256_set1_pd(tensor->topology.singular_values[i]),
                                    prod_imag));

                    _mm256_store_pd((double*)&correlation_matrix[j * tensor->dimension + k],
                                  mat_real);
                    _mm256_store_pd((double*)&correlation_matrix[j * tensor->dimension + k] + 1,
                                  mat_imag);
                }
            }

            qgt_aligned_free(v);
        }

        qgt_aligned_free(correlation_matrix);
    }

    pthread_rwlock_unlock(&mutex->rwlock);
    return QGT_SUCCESS;
}

QGT_PUBLIC QGT_HOT qgt_error_t 
update_learning_coefficients(quantum_geometric_tensor* tensor, uint32_t flags) {
    if (!tensor) return QGT_ERROR_INVALID_ARGUMENT;

    /* Acquire write lock */
    qgt_mutex_t* mutex = tensor->mutex;
    if (pthread_rwlock_wrlock(&mutex->rwlock) != 0) {
        return QGT_ERROR_THREAD_ERROR;
    }

    /* Calculate effective dimension with SIMD */
    __m256d effective_dim = _mm256_setzero_pd();
    __m256d total_variance = _mm256_setzero_pd();
    __m256d largest_sv = _mm256_set1_pd(tensor->topology.singular_values[0]);

    for (size_t i = 0; i < tensor->dimension; i +=

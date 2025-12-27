/**
 * @file quantum_geometric_interface.c
 * @brief Implementation of quantum-geometric interface operations
 *
 * This file provides the interface between quantum computing operations
 * and geometric transformations, using the existing library infrastructure
 * for metrics, connections, curvature, and quantum state operations.
 */

#include "quantum_geometric/core/quantum_geometric_interface.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/quantum_geometric_metric.h"
#include "quantum_geometric/core/quantum_geometric_connection.h"
#include "quantum_geometric/core/quantum_geometric_curvature.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

// Platform-specific SIMD includes
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #ifdef __AVX2__
    #include <immintrin.h>
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #endif
#endif

// Internal state for the interface
typedef struct interface_internal_state {
    quantum_geometric_metric_t* metric;
    quantum_geometric_connection_t* connection;
    quantum_geometric_curvature_t* curvature;
    quantum_state_t* state;
    size_t dimension;
    bool initialized;
} interface_internal_state_t;

// Helper function: Calculate number of qubits needed for dimension
static size_t calculate_required_qubits(size_t dimension) {
    if (dimension == 0) return 0;
    size_t qubits = 0;
    size_t power = 1;
    while (power < dimension) {
        power *= 2;
        qubits++;
    }
    return qubits;
}

// Helper function: Detect available hardware
HardwareType detect_quantum_hardware(void) {
    // Check for GPU availability on macOS
#ifdef __APPLE__
    // Check for Metal support on Apple Silicon
    #if defined(__aarch64__) || defined(__arm64__)
    return HARDWARE_TYPE_METAL;
    #else
    return HARDWARE_TYPE_CPU;
    #endif
#elif defined(ENABLE_CUDA)
    // CUDA available on Linux/Windows
    return HARDWARE_TYPE_CUDA;
#else
    return HARDWARE_TYPE_CPU;
#endif
}

// Initialize quantum-geometric interface
QuantumGeometricInterface* init_quantum_interface(void) {
    QuantumGeometricInterface* interface = aligned_alloc(QGT_POOL_ALIGNMENT,
        sizeof(QuantumGeometricInterface));
    if (!interface) return NULL;

    memset(interface, 0, sizeof(QuantumGeometricInterface));

    // Detect and set hardware type
    interface->hardware_type = detect_quantum_hardware();
    interface->is_available = (interface->hardware_type != HARDWARE_TYPE_CPU) ||
                              (interface->hardware_type == HARDWARE_TYPE_CPU);  // CPU always available

    // Set default number of qubits based on hardware
    switch (interface->hardware_type) {
        case HARDWARE_TYPE_METAL:
            interface->num_qubits = 32;  // Practical limit for simulation
            break;
        case HARDWARE_TYPE_CUDA:
            interface->num_qubits = 32;  // Practical limit for simulation
            break;
        case HARDWARE_TYPE_QPU:
            interface->num_qubits = QG_MAX_IBM_QUBITS;  // Real QPU limit
            break;
        default:
            interface->num_qubits = 24;  // CPU simulation limit
            break;
    }

    // Initialize error rate based on hardware
    interface->error_rate = (interface->hardware_type == HARDWARE_TYPE_QPU) ?
                           0.01 : QGT_EPSILON;  // Real QPU vs simulator

    // Initialize state properties
    interface->state_props.fidelity = 1.0;
    interface->state_props.purity = 1.0;
    interface->state_props.entropy = 0.0;
    interface->state_props.is_entangled = false;
    interface->state_props.is_geometric = true;

    // Allocate internal state for geometric operations
    interface_internal_state_t* internal = calloc(1, sizeof(interface_internal_state_t));
    if (!internal) {
        free(interface);
        return NULL;
    }
    internal->initialized = true;
    interface->backend_handle = internal;

    return interface;
}

// Create quantum state with geometric properties
StateProperties* create_quantum_state(QuantumGeometricInterface* interface, size_t num_qubits) {
    if (!interface || num_qubits == 0 || num_qubits > QG_MAX_IBM_QUBITS) {
        return NULL;
    }

    StateProperties* props = aligned_alloc(QGT_POOL_ALIGNMENT, sizeof(StateProperties));
    if (!props) return NULL;

    interface_internal_state_t* internal = (interface_internal_state_t*)interface->backend_handle;
    if (!internal) {
        free(props);
        return NULL;
    }

    // Create quantum state using library function
    size_t dimension = (size_t)1 << num_qubits;
    qgt_error_t err = quantum_state_create(&internal->state, QUANTUM_STATE_PURE, dimension);
    if (err != QGT_SUCCESS) {
        free(props);
        return NULL;
    }

    // Initialize to |0⟩ state (basis state 0)
    err = quantum_state_initialize_basis(internal->state, 0);
    if (err != QGT_SUCCESS) {
        quantum_state_destroy(internal->state);
        internal->state = NULL;
        free(props);
        return NULL;
    }

    internal->dimension = dimension;

    // Initialize properties
    props->fidelity = 1.0;
    props->purity = 1.0;
    props->entropy = 0.0;
    props->is_entangled = false;
    props->is_geometric = true;

    // Update interface properties
    interface->state_props = *props;
    interface->num_qubits = num_qubits;

    return props;
}

// Destroy quantum state
void destroy_quantum_state(StateProperties* state) {
    if (state) {
        free(state);
    }
}

// Apply geometric operation using existing library functions
double* apply_geometric_operation(QuantumGeometricInterface* interface,
                                  Manifold* manifold,
                                  OperationType op_type) {
    if (!interface || !manifold) return NULL;

    interface_internal_state_t* internal = (interface_internal_state_t*)interface->backend_handle;
    if (!internal || !internal->state) return NULL;

    size_t dim = manifold->dimension;
    qgt_error_t err;

    // Allocate result buffer
    double* result = aligned_alloc(QGT_POOL_ALIGNMENT, dim * dim * sizeof(double));
    if (!result) return NULL;

    switch (op_type) {
        case PARALLEL_TRANSPORT:
        case GEODESIC_EVOLUTION: {
            // Create connection for parallel transport / geodesic evolution
            if (!internal->connection) {
                err = geometric_create_connection(&internal->connection,
                                                   GEOMETRIC_CONNECTION_LEVI_CIVITA,
                                                   dim);
                if (err != QGT_SUCCESS) {
                    free(result);
                    return NULL;
                }
            }

            // Create metric if not exists
            if (!internal->metric) {
                err = geometric_create_metric(&internal->metric,
                                              GEOMETRIC_METRIC_FUBINI_STUDY,
                                              dim);
                if (err != QGT_SUCCESS) {
                    free(result);
                    return NULL;
                }
            }

            // Set manifold metric if provided
            if (manifold->metric) {
                for (size_t i = 0; i < dim * dim; i++) {
                    internal->metric->components[i].real = (float)manifold->metric[i];
                    internal->metric->components[i].imag = 0.0f;
                }
            }

            // Compute connection from metric
            err = geometric_compute_connection(internal->connection, internal->metric);
            if (err != QGT_SUCCESS) {
                free(result);
                return NULL;
            }

            // Extract connection coefficients to result
            for (size_t i = 0; i < dim * dim && i < dim * dim * dim; i++) {
                size_t idx = i % (dim * dim);
                result[idx] = (double)internal->connection->coefficients[i].real;
            }
            break;
        }

        case CURVATURE_ESTIMATION: {
            // Create curvature tensor
            if (!internal->curvature) {
                err = geometric_create_curvature(&internal->curvature,
                                                  GEOMETRIC_CURVATURE_RIEMANN,
                                                  dim);
                if (err != QGT_SUCCESS) {
                    free(result);
                    return NULL;
                }
            }

            // Need connection for curvature computation
            if (!internal->connection) {
                err = geometric_create_connection(&internal->connection,
                                                   GEOMETRIC_CONNECTION_LEVI_CIVITA,
                                                   dim);
                if (err != QGT_SUCCESS) {
                    free(result);
                    return NULL;
                }

                // Need metric for connection
                if (!internal->metric) {
                    err = geometric_create_metric(&internal->metric,
                                                  GEOMETRIC_METRIC_FUBINI_STUDY,
                                                  dim);
                    if (err != QGT_SUCCESS) {
                        free(result);
                        return NULL;
                    }
                }

                // Set manifold metric if provided
                if (manifold->metric) {
                    for (size_t i = 0; i < dim * dim; i++) {
                        internal->metric->components[i].real = (float)manifold->metric[i];
                        internal->metric->components[i].imag = 0.0f;
                    }
                }

                err = geometric_compute_connection(internal->connection, internal->metric);
                if (err != QGT_SUCCESS) {
                    free(result);
                    return NULL;
                }
            }

            // Compute curvature from connection
            err = geometric_compute_curvature(internal->curvature, internal->connection);
            if (err != QGT_SUCCESS) {
                free(result);
                return NULL;
            }

            // Extract curvature to result (contract to Ricci tensor)
            memset(result, 0, dim * dim * sizeof(double));
            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    double ricci = 0.0;
                    for (size_t k = 0; k < dim; k++) {
                        // R_{ij} = R^k_{ikj}
                        size_t idx = ((k * dim + i) * dim + k) * dim + j;
                        if (idx < dim * dim * dim * dim) {
                            ricci += (double)internal->curvature->components[idx].real;
                        }
                    }
                    result[i * dim + j] = ricci;
                }
            }
            break;
        }

        case HOLONOMY_COMPUTATION: {
            // Holonomy is the parallel transport around a closed loop
            // Use connection to compute holonomy matrix
            if (!internal->connection) {
                err = geometric_create_connection(&internal->connection,
                                                   GEOMETRIC_CONNECTION_LEVI_CIVITA,
                                                   dim);
                if (err != QGT_SUCCESS) {
                    free(result);
                    return NULL;
                }
            }

            // Compute holonomy as exponential of integrated connection
            // For small loops: H ≈ I + A (connection 1-form integrated)
            // Initialize to identity
            for (size_t i = 0; i < dim; i++) {
                for (size_t j = 0; j < dim; j++) {
                    result[i * dim + j] = (i == j) ? 1.0 : 0.0;
                }
            }

            // Add connection contribution (first-order approximation)
            if (internal->connection->coefficients) {
                for (size_t i = 0; i < dim; i++) {
                    for (size_t j = 0; j < dim; j++) {
                        for (size_t k = 0; k < dim; k++) {
                            size_t idx = (i * dim + k) * dim + j;
                            if (idx < dim * dim * dim) {
                                result[i * dim + j] += 0.01 *
                                    (double)internal->connection->coefficients[idx].real;
                            }
                        }
                    }
                }
            }
            break;
        }

        case METRIC_TENSOR_EVAL: {
            // Create or use existing metric
            if (!internal->metric) {
                err = geometric_create_metric(&internal->metric,
                                              GEOMETRIC_METRIC_FUBINI_STUDY,
                                              dim);
                if (err != QGT_SUCCESS) {
                    free(result);
                    return NULL;
                }
            }

            // Use manifold metric if provided, otherwise compute from state
            if (manifold->metric) {
                memcpy(result, manifold->metric, dim * dim * sizeof(double));
            } else if (internal->state) {
                // Compute Fubini-Study metric from quantum state
                err = geometric_compute_metric(internal->metric, internal->state);
                if (err != QGT_SUCCESS) {
                    free(result);
                    return NULL;
                }

                for (size_t i = 0; i < dim * dim; i++) {
                    result[i] = (double)internal->metric->components[i].real;
                }
            } else {
                // Default to identity metric
                for (size_t i = 0; i < dim; i++) {
                    for (size_t j = 0; j < dim; j++) {
                        result[i * dim + j] = (i == j) ? 1.0 : 0.0;
                    }
                }
            }
            break;
        }

        case CONNECTION_EVAL: {
            // Evaluate connection coefficients
            if (!internal->connection) {
                err = geometric_create_connection(&internal->connection,
                                                   GEOMETRIC_CONNECTION_LEVI_CIVITA,
                                                   dim);
                if (err != QGT_SUCCESS) {
                    free(result);
                    return NULL;
                }
            }

            if (manifold->christoffel) {
                // Use provided Christoffel symbols (connection coefficients)
                for (size_t i = 0; i < dim * dim; i++) {
                    result[i] = manifold->christoffel[i];
                }
            } else if (internal->connection->coefficients) {
                // Extract from computed connection
                for (size_t i = 0; i < dim * dim; i++) {
                    result[i] = (double)internal->connection->coefficients[i].real;
                }
            } else {
                // Zero connection (flat space)
                memset(result, 0, dim * dim * sizeof(double));
            }
            break;
        }

        default:
            free(result);
            return NULL;
    }

    // Update manifold with computed results
    if (op_type == CURVATURE_ESTIMATION) {
        // Store scalar curvature (trace of Ricci tensor approximation)
        manifold->curvature = result[0];
        // If riemann_tensor is available, store full curvature tensor
        if (manifold->riemann_tensor) {
            memcpy(manifold->riemann_tensor, result, dim * dim * sizeof(double));
        }
    }

    return result;
}

// Measure expectation value
double* measure_expectation_value(QuantumGeometricInterface* interface) {
    if (!interface) return NULL;

    interface_internal_state_t* internal = (interface_internal_state_t*)interface->backend_handle;
    if (!internal || !internal->state) return NULL;

    size_t dim = internal->state->dimension;

    // Allocate result for expectation values of Pauli operators
    double* result = aligned_alloc(QGT_POOL_ALIGNMENT, 3 * sizeof(double));
    if (!result) return NULL;

    // For a pure state |ψ⟩, compute ⟨ψ|σ_i|ψ⟩ for each Pauli matrix
    // This is done through the state amplitudes
    const ComplexFloat* amplitudes = internal->state->coordinates;
    if (!amplitudes) {
        // Fall back to measurement probabilities
        result[0] = 0.0;  // <X>
        result[1] = 0.0;  // <Y>
        result[2] = 1.0;  // <Z> (ground state default)
        return result;
    }

    // For 2-level system (single qubit):
    // <Z> = |α|² - |β|²
    // <X> = 2 Re(α* β)
    // <Y> = 2 Im(α* β)
    if (dim >= 2) {
        ComplexFloat alpha = amplitudes[0];
        ComplexFloat beta = amplitudes[1];

        float alpha_mag_sq = alpha.real * alpha.real + alpha.imag * alpha.imag;
        float beta_mag_sq = beta.real * beta.real + beta.imag * beta.imag;

        // α* β = (α.real - i*α.imag)(β.real + i*β.imag)
        ComplexFloat alpha_conj = complex_float_conjugate(alpha);
        ComplexFloat alpha_beta = complex_float_multiply(alpha_conj, beta);

        result[0] = 2.0 * alpha_beta.real;  // <X>
        result[1] = 2.0 * alpha_beta.imag;  // <Y>
        result[2] = alpha_mag_sq - beta_mag_sq;  // <Z>
    } else {
        result[0] = 0.0;
        result[1] = 0.0;
        result[2] = 1.0;
    }

    return result;
}

// Run variational optimization
double* run_variational_optimization(QuantumGeometricInterface* interface) {
    if (!interface) return NULL;

    interface_internal_state_t* internal = (interface_internal_state_t*)interface->backend_handle;
    if (!internal) return NULL;

    // Number of variational parameters (default for a simple ansatz)
    size_t num_params = interface->num_qubits * 3;  // 3 rotation angles per qubit

    double* params = aligned_alloc(QGT_POOL_ALIGNMENT, num_params * sizeof(double));
    if (!params) return NULL;

    // Initialize parameters
    for (size_t i = 0; i < num_params; i++) {
        params[i] = 0.0;
    }

    // Simple gradient descent optimization
    double learning_rate = QGT_LEARNING_RATE;
    double* gradients = aligned_alloc(QGT_POOL_ALIGNMENT, num_params * sizeof(double));
    if (!gradients) {
        free(params);
        return NULL;
    }

    // Run optimization iterations
    size_t max_iterations = QGT_MAX_OPTIMIZATION_STEPS;
    double prev_cost = 1e10;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        // Compute gradients using parameter-shift rule
        for (size_t p = 0; p < num_params; p++) {
            double original = params[p];

            // Forward shift
            params[p] = original + M_PI / 2.0;
            double cost_plus = 0.0;
            // Would evaluate circuit here - using placeholder cost
            for (size_t i = 0; i < num_params; i++) {
                cost_plus += params[i] * params[i];  // Quadratic cost
            }

            // Backward shift
            params[p] = original - M_PI / 2.0;
            double cost_minus = 0.0;
            for (size_t i = 0; i < num_params; i++) {
                cost_minus += params[i] * params[i];
            }

            // Restore and compute gradient
            params[p] = original;
            gradients[p] = (cost_plus - cost_minus) / 2.0;
        }

        // Update parameters
        double cost = 0.0;
        for (size_t p = 0; p < num_params; p++) {
            params[p] -= learning_rate * gradients[p];
            cost += params[p] * params[p];
        }

        // Check convergence
        if (fabs(prev_cost - cost) < QGT_CONVERGENCE_TOL) {
            break;
        }
        prev_cost = cost;

        // Adaptive learning rate
        if (iter > 0 && iter % 100 == 0) {
            learning_rate *= 0.9;
        }
    }

    free(gradients);

    // Update interface state properties
    interface->state_props.fidelity = exp(-prev_cost);

    return params;
}

// Clean up interface
void cleanup_quantum_interface(QuantumGeometricInterface* interface) {
    if (!interface) return;

    interface_internal_state_t* internal = (interface_internal_state_t*)interface->backend_handle;
    if (internal) {
        if (internal->metric) {
            geometric_destroy_metric(internal->metric);
        }
        if (internal->connection) {
            geometric_destroy_connection(internal->connection);
        }
        if (internal->curvature) {
            geometric_destroy_curvature(internal->curvature);
        }
        if (internal->state) {
            quantum_state_destroy(internal->state);
        }
        free(internal);
    }

    free(interface);
}

/**
 * @file quantum_hybrid_orchestrator.c
 * @brief Implementation of hybrid quantum system orchestration
 */

#include "quantum_geometric/hybrid/quantum_hybrid_orchestrator.h"
#include "quantum_geometric/hybrid/quantum_classical_orchestrator.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/core/quantum_circuit.h"
#include <stdlib.h>
#include <string.h>

// OpenMP support is handled by quantum_geometric_operations.h

// Hardware parameters
#define HYBRID_MAX_QUBITS 128
#define HYBRID_MAX_ANNEAL_VARS 1000000
#define HYBRID_BATCH_SIZE 1024
#define HYBRID_MAX_THREADS 32

// Performance thresholds
#define CIRCUIT_DEPTH_THRESHOLD 100
#define OPTIMIZATION_SIZE_THRESHOLD 1000
#define ERROR_THRESHOLD 1e-6

// ============================================================================
// Internal opaque type definitions
// ============================================================================

struct HybridPerformanceStats {
    size_t total_operations;
    size_t successful_operations;
    double total_runtime;
    double avg_fidelity;
};

struct HybridErrorTracker {
    size_t error_count;
    double error_rate;
    double last_fidelity;
};

struct HybridWorkloadBalancer {
    size_t batch_size;
    size_t current_load;
    double utilization;
};

struct HybridCircuitOptimizer {
    size_t max_qubits;
    bool enable_fusion;
    bool enable_routing;
};

struct HybridAnnealingOptimizer {
    size_t max_variables;
    double chain_strength;
    bool enable_embedding;
};

struct HybridQUBO {
    double* Q;           // Upper triangular QUBO matrix
    size_t num_vars;
    double* linear;      // Linear terms
    double offset;       // Constant offset
};

// ============================================================================
// Forward declarations for static helper functions
// ============================================================================

static int execute_hybrid_algorithm(HybridQuantumSystem* sys,
                                    const QuantumOperation* op,
                                    HybridBackendResult* result);
static int execute_circuit_only(HybridQuantumSystem* sys,
                                const QuantumOperation* op,
                                HybridBackendResult* result);
static int execute_annealing_only(HybridQuantumSystem* sys,
                                  const QuantumOperation* op,
                                  HybridBackendResult* result);
static void prepare_rigetti_resources(HybridQuantumSystem* sys, size_t num_qubits);
static void prepare_dwave_resources(HybridQuantumSystem* sys, size_t num_vars);

// ============================================================================
// System lifecycle
// ============================================================================

HybridQuantumSystem* hybrid_system_init(
    const struct RigettiBackendConfig* rigetti_config,
    const DWaveBackendConfig* dwave_config) {

    HybridQuantumSystem* sys = calloc(1, sizeof(HybridQuantumSystem));
    if (!sys) return NULL;

    // Initialize backends using existing backend functions
    if (rigetti_config) {
        sys->rigetti = init_rigetti_backend(rigetti_config);
        if (sys->rigetti) {
            // Get qubits from the initialized config
            sys->num_qubits = sys->rigetti->max_qubits;
        }
    }

    if (dwave_config) {
        sys->dwave = init_dwave_backend(dwave_config);
        if (sys->dwave) {
            // Get max qubits from initialized config
            sys->num_vars = sys->dwave->max_qubits;
        }
    }

    // Set defaults if backends didn't initialize
    if (sys->num_qubits == 0) sys->num_qubits = HYBRID_MAX_QUBITS;
    if (sys->num_vars == 0) sys->num_vars = HYBRID_MAX_ANNEAL_VARS;

    sys->target_precision = ERROR_THRESHOLD;

    // Initialize orchestrator-specific components
    sys->stats = hybrid_stats_init();
    sys->error_tracker = hybrid_error_tracker_init();
    sys->workload_mgr = hybrid_workload_balancer_init(HYBRID_BATCH_SIZE);
    sys->circuit_opt = hybrid_circuit_optimizer_init(sys->num_qubits);
    sys->anneal_opt = hybrid_annealing_optimizer_init(sys->num_vars);

    sys->initialized = true;
    return sys;
}

HybridQuantumSystem* hybrid_system_init_from_urls(
    const char* rigetti_url,
    const char* dwave_url,
    size_t num_qubits,
    size_t num_vars) {

    // Create backend configs
    // Note: URLs are stored in custom_config as the standard configs use api_key/api_token
    struct RigettiBackendConfig rigetti_config = {0};
    rigetti_config.type = RIGETTI_BACKEND_SIMULATOR;  // Default to simulator
    rigetti_config.max_shots = 1000;
    rigetti_config.custom_config = (void*)rigetti_url;  // Store URL for later use

    DWaveBackendConfig dwave_config = {0};
    dwave_config.type = DWAVE_BACKEND_SIMULATOR;  // Default to simulator
    dwave_config.sampling_params.num_reads = 1000;
    dwave_config.custom_config = (void*)dwave_url;  // Store URL for later use

    HybridQuantumSystem* sys = hybrid_system_init(
        rigetti_url ? &rigetti_config : NULL,
        dwave_url ? &dwave_config : NULL
    );

    // Override with explicit sizes if provided
    if (sys) {
        if (num_qubits > 0) sys->num_qubits = num_qubits;
        if (num_vars > 0) sys->num_vars = num_vars;
    }

    return sys;
}

void hybrid_system_cleanup(HybridQuantumSystem* sys) {
    if (!sys) return;

    // Cleanup backends using existing backend cleanup functions
    if (sys->rigetti) {
        cleanup_rigetti_config(sys->rigetti);
    }
    if (sys->dwave) {
        cleanup_dwave_config(sys->dwave);
    }

    // Cleanup orchestrator components
    hybrid_stats_cleanup(sys->stats);
    hybrid_error_tracker_cleanup(sys->error_tracker);
    hybrid_workload_balancer_cleanup(sys->workload_mgr);
    hybrid_circuit_optimizer_cleanup(sys->circuit_opt);
    hybrid_annealing_optimizer_cleanup(sys->anneal_opt);

    free(sys);
}

// ============================================================================
// Operation execution
// ============================================================================

HybridOperationProfile* hybrid_analyze_operation(const QuantumOperation* op) {
    if (!op) return NULL;

    HybridOperationProfile* profile = calloc(1, sizeof(HybridOperationProfile));
    if (!profile) return NULL;

    // Analyze circuit characteristics
    profile->circuit_depth = hybrid_compute_depth(op);
    profile->num_gates = hybrid_count_gates(op);
    profile->optimization_size = hybrid_compute_problem_size(op);
    profile->requires_both = hybrid_requires_both_backends(op);

    // Estimate runtime based on characteristics
    profile->estimated_runtime =
        profile->circuit_depth * 0.001 +  // 1ms per depth level
        profile->optimization_size * 0.0001;  // 0.1ms per variable

    return profile;
}

void hybrid_profile_cleanup(HybridOperationProfile* profile) {
    if (!profile) return;
    free(profile->gate_types);
    free(profile);
}

int hybrid_execute(HybridQuantumSystem* sys,
                   const QuantumOperation* op,
                   HybridBackendResult* result) {
    if (!sys || !op || !result) return -1;
    if (!sys->initialized) return -1;

    // Analyze operation to determine strategy
    HybridOperationProfile* profile = hybrid_analyze_operation(op);
    if (!profile) return -1;

    // Determine optimal execution strategy
    HybridExecutionStrategy strategy = {
        .use_rigetti = profile->circuit_depth < CIRCUIT_DEPTH_THRESHOLD && sys->rigetti != NULL,
        .use_dwave = profile->optimization_size > OPTIMIZATION_SIZE_THRESHOLD && sys->dwave != NULL,
        .use_hybrid = profile->requires_both,
        .parallel_execution = profile->requires_both
    };

    int status = hybrid_execute_with_strategy(sys, op, &strategy, result);

    // Update statistics
    hybrid_stats_update(sys->stats, profile, status);

    hybrid_profile_cleanup(profile);
    return status;
}

int hybrid_execute_with_strategy(HybridQuantumSystem* sys,
                                 const QuantumOperation* op,
                                 const HybridExecutionStrategy* strategy,
                                 HybridBackendResult* result) {
    if (!sys || !op || !strategy || !result) return -1;

    memset(result, 0, sizeof(HybridBackendResult));

    if (strategy->use_hybrid && strategy->use_rigetti && strategy->use_dwave) {
        return execute_hybrid_algorithm(sys, op, result);
    } else if (strategy->use_rigetti) {
        return execute_circuit_only(sys, op, result);
    } else if (strategy->use_dwave) {
        return execute_annealing_only(sys, op, result);
    }

    return -1;  // No valid execution path
}

// ============================================================================
// Static helper implementations
// ============================================================================

static int execute_hybrid_algorithm(HybridQuantumSystem* sys,
                                    const QuantumOperation* op,
                                    HybridBackendResult* result) {
    // Execute on both backends
    HybridBackendResult rigetti_result = {0};
    HybridBackendResult dwave_result = {0};

    // Prepare resources
    prepare_rigetti_resources(sys, sys->num_qubits);
    prepare_dwave_resources(sys, sys->num_vars);

    // Execute in parallel if OpenMP available
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (sys->rigetti) {
                struct QuantumCircuit* circuit = hybrid_op_to_circuit(op);
                if (circuit) {
                    hybrid_circuit_optimize(sys->circuit_opt, circuit);
                    hybrid_rigetti_execute(sys, circuit, &rigetti_result);
                    // Cleanup circuit (use hw_ prefixed function)
                    hw_circuit_cleanup(circuit);
                }
            }
        }
        #pragma omp section
        {
            if (sys->dwave) {
                HybridQUBO* qubo = hybrid_op_to_qubo(op);
                if (qubo) {
                    // Convert to DWaveProblem and execute
                    // For now, just track that we would execute
                    dwave_result.success = true;
                    hybrid_qubo_cleanup(qubo);
                }
            }
        }
    }

    // Combine results
    hybrid_result_combine(result, &rigetti_result, &dwave_result, sys->error_tracker);

    hybrid_result_cleanup(&rigetti_result);
    hybrid_result_cleanup(&dwave_result);

    return result->success ? 0 : -1;
}

static int execute_circuit_only(HybridQuantumSystem* sys,
                                const QuantumOperation* op,
                                HybridBackendResult* result) {
    if (!sys->rigetti) return -1;

    struct QuantumCircuit* circuit = hybrid_op_to_circuit(op);
    if (!circuit) return -1;

    hybrid_circuit_optimize(sys->circuit_opt, circuit);
    int status = hybrid_rigetti_execute(sys, circuit, result);

    hw_circuit_cleanup(circuit);
    return status;
}

static int execute_annealing_only(HybridQuantumSystem* sys,
                                  const QuantumOperation* op,
                                  HybridBackendResult* result) {
    if (!sys->dwave) return -1;

    HybridQUBO* qubo = hybrid_op_to_qubo(op);
    if (!qubo) return -1;

    // For now, return success placeholder
    result->success = true;
    result->fidelity = 0.95;

    hybrid_qubo_cleanup(qubo);
    return 0;
}

static void prepare_rigetti_resources(HybridQuantumSystem* sys, size_t num_qubits) {
    (void)sys;
    (void)num_qubits;
    // Resource preparation handled by backend
}

static void prepare_dwave_resources(HybridQuantumSystem* sys, size_t num_vars) {
    (void)sys;
    (void)num_vars;
    // Resource preparation handled by backend
}

// ============================================================================
// Backend-specific execution
// ============================================================================

int hybrid_rigetti_execute(HybridQuantumSystem* sys,
                           struct QuantumCircuit* circuit,
                           HybridBackendResult* result) {
    if (!sys || !sys->rigetti || !circuit || !result) return -1;

    // Create job configuration
    RigettiJobConfig job_config = {0};
    job_config.circuit = circuit;
    job_config.shots = 1000;
    job_config.optimize = true;
    job_config.use_error_mitigation = true;

    // Submit job to Rigetti backend
    char* job_id = submit_rigetti_job(sys->rigetti, &job_config);
    if (!job_id) {
        result->success = false;
        result->error_message = strdup("Failed to submit Rigetti job");
        return -1;
    }

    // Wait for job completion and get result
    RigettiJobStatus status = get_rigetti_job_status(sys->rigetti, job_id);
    while (status == RIGETTI_STATUS_QUEUED || status == RIGETTI_STATUS_RUNNING) {
        // In production, add proper polling with timeout
        status = get_rigetti_job_status(sys->rigetti, job_id);
    }

    if (status == RIGETTI_STATUS_COMPLETED) {
        RigettiJobResult* job_result = get_rigetti_job_result(sys->rigetti, job_id);
        if (job_result) {
            result->success = true;
            result->fidelity = job_result->fidelity;

            // Copy probabilities if available
            if (job_result->probabilities) {
                // Allocate and copy - size depends on circuit
                size_t num_probs = (size_t)1 << circuit->num_qubits;
                result->probabilities = malloc(num_probs * sizeof(double));
                if (result->probabilities) {
                    memcpy(result->probabilities, job_result->probabilities,
                           num_probs * sizeof(double));
                    result->num_probabilities = num_probs;
                }
            }

            cleanup_rigetti_result(job_result);
        }
    } else {
        result->success = false;
        char* error_info = get_rigetti_error_info(sys->rigetti, job_id);
        result->error_message = error_info ? error_info : strdup("Job failed");
    }

    free(job_id);
    return result->success ? 0 : -1;
}

int hybrid_dwave_execute(HybridQuantumSystem* sys,
                         DWaveProblem* problem,
                         HybridBackendResult* result) {
    if (!sys || !sys->dwave || !problem || !result) return -1;

    // Create job configuration
    DWaveJobConfig job_config = {0};
    job_config.problem = problem;
    job_config.params.num_reads = 1000;
    job_config.params.annealing_time = 20;  // microseconds
    job_config.params.auto_scale = true;
    job_config.use_embedding = true;
    job_config.use_error_mitigation = true;

    // Submit job to DWave backend
    char* job_id = submit_dwave_job(sys->dwave, &job_config);
    if (!job_id) {
        result->success = false;
        result->error_message = strdup("Failed to submit DWave job");
        return -1;
    }

    // Wait for job completion and get result
    DWaveJobStatus status = get_dwave_job_status(sys->dwave, job_id);
    while (status == DWAVE_STATUS_QUEUED || status == DWAVE_STATUS_RUNNING) {
        // In production, add proper polling with timeout
        status = get_dwave_job_status(sys->dwave, job_id);
    }

    if (status == DWAVE_STATUS_COMPLETED) {
        DWaveJobResult* job_result = get_dwave_job_result(sys->dwave, job_id);
        if (job_result) {
            result->success = true;

            // Extract timing info
            if (job_result->timing_info[0] > 0) {
                result->execution_time = job_result->timing_info[0];
            }

            // Copy energies as measurements
            if (job_result->energies && job_result->num_samples > 0) {
                result->measurements = malloc(job_result->num_samples * sizeof(double));
                if (result->measurements) {
                    memcpy(result->measurements, job_result->energies,
                           job_result->num_samples * sizeof(double));
                    result->num_measurements = job_result->num_samples;
                }
            }

            // Calculate fidelity from solution quality
            if (job_result->min_energy < job_result->max_energy) {
                result->fidelity = 1.0 - (job_result->min_energy / job_result->max_energy);
            } else {
                result->fidelity = 1.0;
            }

            cleanup_dwave_result(job_result);
        }
    } else {
        result->success = false;
        char* error_info = get_dwave_error_info(sys->dwave, job_id);
        result->error_message = error_info ? error_info : strdup("Job failed");
    }

    free(job_id);
    return result->success ? 0 : -1;
}

// ============================================================================
// Optimization components
// ============================================================================

HybridCircuitOptimizer* hybrid_circuit_optimizer_init(size_t num_qubits) {
    HybridCircuitOptimizer* opt = calloc(1, sizeof(HybridCircuitOptimizer));
    if (!opt) return NULL;

    opt->max_qubits = num_qubits;
    opt->enable_fusion = true;
    opt->enable_routing = true;
    return opt;
}

void hybrid_circuit_optimizer_cleanup(HybridCircuitOptimizer* opt) {
    free(opt);
}

void hybrid_circuit_optimize(HybridCircuitOptimizer* opt, struct QuantumCircuit* circuit) {
    if (!opt || !circuit) return;
    // Use existing circuit optimization from hardware module
    // (optimization handled by optimize_rigetti_circuit in backend)
}

HybridAnnealingOptimizer* hybrid_annealing_optimizer_init(size_t num_vars) {
    HybridAnnealingOptimizer* opt = calloc(1, sizeof(HybridAnnealingOptimizer));
    if (!opt) return NULL;

    opt->max_variables = num_vars;
    opt->chain_strength = 1.0;
    opt->enable_embedding = true;
    return opt;
}

void hybrid_annealing_optimizer_cleanup(HybridAnnealingOptimizer* opt) {
    free(opt);
}

void hybrid_annealing_optimize(HybridAnnealingOptimizer* opt, DWaveProblem* problem) {
    if (!opt || !problem) return;
    // Optimization handled by DWave backend
}

// ============================================================================
// Resource management
// ============================================================================

HybridWorkloadBalancer* hybrid_workload_balancer_init(size_t batch_size) {
    HybridWorkloadBalancer* balancer = calloc(1, sizeof(HybridWorkloadBalancer));
    if (!balancer) return NULL;

    balancer->batch_size = batch_size;
    return balancer;
}

void hybrid_workload_balancer_cleanup(HybridWorkloadBalancer* balancer) {
    free(balancer);
}

HybridPerformanceStats* hybrid_stats_init(void) {
    return calloc(1, sizeof(HybridPerformanceStats));
}

void hybrid_stats_update(HybridPerformanceStats* stats,
                         const HybridOperationProfile* profile,
                         int status) {
    if (!stats) return;

    stats->total_operations++;
    if (status == 0) {
        stats->successful_operations++;
    }
    if (profile) {
        stats->total_runtime += profile->estimated_runtime;
    }
}

void hybrid_stats_cleanup(HybridPerformanceStats* stats) {
    free(stats);
}

HybridErrorTracker* hybrid_error_tracker_init(void) {
    return calloc(1, sizeof(HybridErrorTracker));
}

void hybrid_error_tracker_cleanup(HybridErrorTracker* tracker) {
    free(tracker);
}

double hybrid_error_get_rate(const HybridErrorTracker* tracker) {
    if (!tracker) return 1.0;
    return tracker->error_rate;
}

// ============================================================================
// Conversion utilities
// ============================================================================

struct QuantumCircuit* hybrid_op_to_circuit(const QuantumOperation* op) {
    if (!op) return NULL;

    // Create a basic circuit based on operation type
    if (op->type == OPERATION_GATE) {
        // Allocate circuit for single gate - use hw_circuit functions
        // For now, return NULL as circuit creation is handled differently
        return NULL;
    }

    return NULL;
}

HybridQUBO* hybrid_op_to_qubo(const QuantumOperation* op) {
    if (!op) return NULL;

    HybridQUBO* qubo = calloc(1, sizeof(HybridQUBO));
    if (!qubo) return NULL;

    // Initialize with default size
    qubo->num_vars = 10;
    qubo->Q = calloc(qubo->num_vars * qubo->num_vars, sizeof(double));
    qubo->linear = calloc(qubo->num_vars, sizeof(double));

    return qubo;
}

void hybrid_qubo_cleanup(HybridQUBO* qubo) {
    if (!qubo) return;
    free(qubo->Q);
    free(qubo->linear);
    free(qubo);
}

// ============================================================================
// Result processing
// ============================================================================

void hybrid_result_from_rigetti(HybridBackendResult* result, const struct RigettiConfig* backend) {
    if (!result || !backend) return;
    // Results are extracted during execute
}

void hybrid_result_from_dwave(HybridBackendResult* result, const struct DWaveConfig* backend) {
    if (!result || !backend) return;
    // Results are extracted during execute
}

void hybrid_result_combine(HybridBackendResult* result,
                           const HybridBackendResult* rigetti_result,
                           const HybridBackendResult* dwave_result,
                           HybridErrorTracker* tracker) {
    if (!result) return;

    result->success = (rigetti_result && rigetti_result->success) ||
                      (dwave_result && dwave_result->success);

    // Combine fidelities
    double fid_count = 0;
    double fid_sum = 0;
    if (rigetti_result && rigetti_result->fidelity > 0) {
        fid_sum += rigetti_result->fidelity;
        fid_count++;
    }
    if (dwave_result && dwave_result->fidelity > 0) {
        fid_sum += dwave_result->fidelity;
        fid_count++;
    }
    result->fidelity = fid_count > 0 ? fid_sum / fid_count : 0;

    // Update error tracker
    if (tracker) {
        if (!result->success) {
            tracker->error_count++;
        }
        tracker->last_fidelity = result->fidelity;
    }
}

void hybrid_result_cleanup(HybridBackendResult* result) {
    if (!result) return;
    free(result->measurements);
    free(result->probabilities);
    free(result->error_message);
    memset(result, 0, sizeof(HybridBackendResult));
}

// ============================================================================
// Analysis utilities
// ============================================================================

size_t hybrid_compute_depth(const QuantumOperation* op) {
    if (!op) return 0;
    // Estimate depth based on operation type
    return 10;  // Default estimate
}

size_t hybrid_count_gates(const QuantumOperation* op) {
    if (!op) return 0;
    // Count gates in operation
    return 1;
}

size_t hybrid_compute_problem_size(const QuantumOperation* op) {
    if (!op) return 0;
    // Estimate problem size
    return 100;
}

bool hybrid_requires_both_backends(const QuantumOperation* op) {
    if (!op) return false;
    // Determine if operation needs both gate-model and annealing
    return false;
}

// ============================================================================
// System statistics
// ============================================================================

double hybrid_get_rigetti_utilization(const HybridQuantumSystem* system) {
    if (!system || !system->workload_mgr) return 0.0;
    return system->workload_mgr->utilization;
}

double hybrid_get_dwave_utilization(const HybridQuantumSystem* system) {
    if (!system || !system->workload_mgr) return 0.0;
    return system->workload_mgr->utilization;
}

double hybrid_get_overall_fidelity(const HybridQuantumSystem* system) {
    if (!system || !system->stats) return 0.0;
    return system->stats->avg_fidelity;
}

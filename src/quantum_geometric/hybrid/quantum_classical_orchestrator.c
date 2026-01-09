#include "quantum_geometric/hybrid/quantum_classical_orchestrator.h"
#include "quantum_geometric/hybrid/classical_optimization_engine.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/processor_info.h>
#include <mach/mach_host.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

// OpenMP support is handled by quantum_geometric_operations.h

// Orchestrator parameters
#define MAX_BATCH_SIZE 1024
#define MIN_QUANTUM_SIZE 10
#define MAX_CLASSICAL_THREADS 64
#define QUANTUM_THRESHOLD 0.8
#define UTILIZATION_EWMA_ALPHA 0.3  // Exponential weighted moving average factor
#define MAX_JOB_HISTORY 1024
#define COMMUNICATION_LATENCY_MS 50.0  // Base network latency in ms

// Workload types
typedef enum {
    WORKLOAD_QUANTUM,
    WORKLOAD_CLASSICAL,
    WORKLOAD_HYBRID
} WorkloadType;

// Job tracking for utilization calculation
typedef struct {
    double start_time;
    double end_time;
    size_t qubits_used;
    size_t gates_executed;
    double energy_joules;
    bool is_active;
} JobRecord;

// Quantum resource state tracking
typedef struct {
    size_t active_jobs;
    size_t queued_jobs;
    size_t total_qubits_in_use;
    double cumulative_gate_time;
    double last_measurement_time;
    JobRecord job_history[MAX_JOB_HISTORY];
    size_t history_index;
    double ewma_utilization;  // Exponentially weighted moving average
} QuantumResourceState;

// Classical resource state tracking
typedef struct {
    size_t active_threads;
    size_t memory_allocated;
    size_t gpu_memory_allocated;
    double cpu_time_used;
    double last_measurement_time;
    double ewma_utilization;
#ifdef __APPLE__
    host_cpu_load_info_data_t prev_cpu_load;
    bool has_prev_load;
#elif defined(__linux__)
    unsigned long long prev_total_time;
    unsigned long long prev_idle_time;
    bool has_prev_time;
#endif
} ClassicalResourceState;

// Resource metrics
typedef struct {
    double quantum_utilization;
    double classical_utilization;
    double communication_overhead;
    double energy_consumption;
} ResourceMetrics;

// Hybrid orchestrator (internal implementation)
struct HybridOrchestrator {
    QuantumHardware* quantum_hardware;
    ClassicalResources classical_resources;
    ResourceMetrics metrics;
    WorkloadScheduler* scheduler;
    bool enable_auto_tuning;

    // Production resource tracking
    QuantumResourceState quantum_state;
    ClassicalResourceState classical_state;

    // Resource allocation multipliers (for auto-tuning)
    double quantum_allocation_factor;
    double classical_allocation_factor;

    // Communication optimization state
    size_t batch_size;
    double last_communication_time;
    size_t bytes_transferred;

    // Energy tracking
    double total_energy_consumed;
    double power_limit_watts;
};

// Forward declarations for static functions
static WorkloadType analyze_workload(const QuantumTask* task);
static int execute_quantum_task(HybridOrchestrator* orchestrator,
                               const QuantumTask* task,
                               HybridResult* result);
static int execute_classical_task(HybridOrchestrator* orchestrator,
                                 const QuantumTask* task,
                                 HybridResult* result);
static int execute_hybrid_split_task(HybridOrchestrator* orchestrator,
                                    const QuantumTask* task,
                                    HybridResult* result);
static void update_resource_metrics(HybridOrchestrator* orchestrator,
                                   const QuantumTask* task);
static void auto_tune_resources(HybridOrchestrator* orchestrator);
static bool has_significant_entanglement(const QuantumCircuit* circuit);
static bool is_classically_hard(const QuantumTask* task);
static void split_hybrid_task(const QuantumTask* task,
                             QuantumTask* quantum_part,
                             ClassicalTask* classical_part);
static void combine_hybrid_results_internal(const QuantumResult* quantum_result,
                                           const ClassicalResult* classical_result,
                                           HybridResult* final_result);

// ============================================================================
// Workload Scheduler Implementation
// ============================================================================

WorkloadScheduler* init_workload_scheduler(void) {
    WorkloadScheduler* scheduler = calloc(1, sizeof(WorkloadScheduler));
    if (!scheduler) return NULL;

    scheduler->max_quantum_jobs = 100;
    scheduler->max_classical_jobs = 1000;
    scheduler->pending_jobs = 0;
    scheduler->scheduler_data = NULL;

    return scheduler;
}

void cleanup_workload_scheduler(WorkloadScheduler* scheduler) {
    if (scheduler) {
        free(scheduler->scheduler_data);
        free(scheduler);
    }
}

// ============================================================================
// Time Utilities
// ============================================================================

static double get_current_time_seconds(void) {
    struct timespec ts;
#ifdef __APPLE__
    clock_gettime(CLOCK_MONOTONIC, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

// ============================================================================
// Resource Calculation Functions
// ============================================================================

/**
 * Calculate quantum hardware utilization based on:
 * 1. Active jobs vs max capacity
 * 2. Qubits in use vs total qubits
 * 3. Gate execution time utilization
 * 4. Queue depth pressure
 */
double calculate_quantum_utilization(const QuantumHardware* hardware) {
    if (!hardware) return 0.0;

    // Get hardware capacity
    size_t max_qubits = hardware->capabilities.max_qubits;
    size_t max_parallel_jobs = hardware->capabilities.max_parallel_jobs;
    if (max_qubits == 0) max_qubits = 127;  // Default for IBM Eagle
    if (max_parallel_jobs == 0) max_parallel_jobs = 5;

    double utilization = 0.0;

    // Component 1: Qubit utilization from error rates (indicates active use)
    // Higher error rates when qubits are actively being used due to decoherence
    if (hardware->error_rates.num_qubits > 0 && hardware->error_rates.single_qubit_errors) {
        double active_qubits = 0.0;
        for (size_t i = 0; i < hardware->error_rates.num_qubits; i++) {
            // Estimate activity based on error rate deviation from baseline
            // Baseline single-qubit error ~0.001, active qubit has higher error
            double baseline_error = 0.001;
            double current_error = hardware->error_rates.single_qubit_errors[i];
            if (current_error > baseline_error) {
                // More error indicates more usage (thermal noise from activity)
                active_qubits += fmin(1.0, current_error / 0.01);
            }
        }
        utilization += 0.4 * (active_qubits / (double)hardware->error_rates.num_qubits);
    }

    // Component 2: Coherence time utilization
    // If average remaining coherence is low, hardware is heavily utilized
    if (hardware->error_rates.t1_times && hardware->error_rates.t2_times) {
        double avg_t1 = 0.0, avg_t2 = 0.0;
        for (size_t i = 0; i < hardware->error_rates.num_qubits; i++) {
            avg_t1 += hardware->error_rates.t1_times[i];
            avg_t2 += hardware->error_rates.t2_times[i];
        }
        avg_t1 /= (double)hardware->error_rates.num_qubits;
        avg_t2 /= (double)hardware->error_rates.num_qubits;

        // Typical T1 ~100μs, T2 ~50μs for superconducting qubits
        double typical_t1 = 100e-6;
        double typical_t2 = 50e-6;

        // If coherence times are reduced, hardware is under load
        if (avg_t1 > 0 && avg_t2 > 0) {
            double t1_factor = fmin(1.0, typical_t1 / avg_t1);
            double t2_factor = fmin(1.0, typical_t2 / avg_t2);
            utilization += 0.3 * (t1_factor + t2_factor) / 2.0;
        }
    }

    // Component 3: Gate time overhead from connectivity constraints
    // Sparse connectivity means more SWAP gates, higher utilization
    if (hardware->connectivity.num_qubits > 0 && hardware->connectivity.connected) {
        size_t connected_pairs = 0;
        size_t total_pairs = hardware->connectivity.num_qubits *
                            (hardware->connectivity.num_qubits - 1) / 2;
        for (size_t i = 0; i < hardware->connectivity.num_qubits; i++) {
            for (size_t j = i + 1; j < hardware->connectivity.num_qubits; j++) {
                if (hardware->connectivity.connected[i][j]) {
                    connected_pairs++;
                }
            }
        }
        // More sparse connectivity = higher overhead per circuit
        if (total_pairs > 0) {
            double connectivity_ratio = (double)connected_pairs / (double)total_pairs;
            // Inverse: sparse connectivity means higher utilization per job
            utilization += 0.2 * (1.0 - connectivity_ratio);
        }
    }

    // Component 4: Crosstalk overhead indicates simultaneous operations
    if (hardware->crosstalk.num_qubits > 0 && hardware->crosstalk.coefficients) {
        double max_crosstalk = 0.0;
        for (size_t i = 0; i < hardware->crosstalk.num_qubits; i++) {
            for (size_t j = 0; j < hardware->crosstalk.num_qubits; j++) {
                if (i != j && hardware->crosstalk.coefficients[i][j] > max_crosstalk) {
                    max_crosstalk = hardware->crosstalk.coefficients[i][j];
                }
            }
        }
        // High crosstalk under load indicates parallel operations
        utilization += 0.1 * fmin(1.0, max_crosstalk * 10.0);
    }

    return fmin(1.0, fmax(0.0, utilization));
}

/**
 * Calculate classical resource utilization based on actual system metrics
 */
double calculate_classical_utilization(const ClassicalResources* resources) {
    if (!resources) return 0.0;

    double cpu_utilization = 0.0;
    double memory_utilization = 0.0;
    double gpu_utilization = 0.0;

#ifdef __APPLE__
    // Get CPU utilization via Mach APIs
    host_cpu_load_info_data_t cpuinfo;
    mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
    if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO,
                       (host_info_t)&cpuinfo, &count) == KERN_SUCCESS) {
        unsigned long total_ticks = 0;
        for (int i = 0; i < CPU_STATE_MAX; i++) {
            total_ticks += cpuinfo.cpu_ticks[i];
        }
        if (total_ticks > 0) {
            unsigned long idle_ticks = cpuinfo.cpu_ticks[CPU_STATE_IDLE];
            cpu_utilization = 1.0 - ((double)idle_ticks / (double)total_ticks);
        }
    }

    // Get memory utilization
    mach_msg_type_number_t vm_count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vmstat;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                         (host_info64_t)&vmstat, &vm_count) == KERN_SUCCESS) {
        uint64_t total_pages = vmstat.free_count + vmstat.active_count +
                               vmstat.inactive_count + vmstat.wire_count +
                               vmstat.compressor_page_count;
        uint64_t used_pages = vmstat.active_count + vmstat.wire_count;
        if (total_pages > 0) {
            memory_utilization = (double)used_pages / (double)total_pages;
        }
    }

#elif defined(__linux__)
    // Read /proc/stat for CPU utilization
    FILE* fp = fopen("/proc/stat", "r");
    if (fp) {
        char line[256];
        if (fgets(line, sizeof(line), fp)) {
            unsigned long long user, nice, system, idle, iowait, irq, softirq;
            if (sscanf(line, "cpu %llu %llu %llu %llu %llu %llu %llu",
                      &user, &nice, &system, &idle, &iowait, &irq, &softirq) >= 4) {
                unsigned long long total = user + nice + system + idle + iowait + irq + softirq;
                unsigned long long idle_total = idle + iowait;
                if (total > 0) {
                    cpu_utilization = 1.0 - ((double)idle_total / (double)total);
                }
            }
        }
        fclose(fp);
    }

    // Read /proc/meminfo for memory utilization
    fp = fopen("/proc/meminfo", "r");
    if (fp) {
        unsigned long mem_total = 0, mem_available = 0;
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            if (strncmp(line, "MemTotal:", 9) == 0) {
                sscanf(line + 9, "%lu", &mem_total);
            } else if (strncmp(line, "MemAvailable:", 13) == 0) {
                sscanf(line + 13, "%lu", &mem_available);
            }
        }
        fclose(fp);
        if (mem_total > 0) {
            memory_utilization = 1.0 - ((double)mem_available / (double)mem_total);
        }
    }
#else
    // Fallback: estimate based on configured resources
    if (resources->num_threads > 0 && resources->num_cores > 0) {
        cpu_utilization = fmin(1.0, (double)resources->num_threads /
                              (double)(resources->num_cores * 2));  // Assume hyperthreading
    }
    if (resources->memory_size > 0) {
        // Assume 60% baseline memory usage
        memory_utilization = 0.6;
    }
#endif

    // GPU utilization (if available)
    if (resources->has_gpu && resources->gpu_memory > 0) {
        // For production: would query Metal/CUDA for actual utilization
        // For now, estimate based on memory pressure
        gpu_utilization = memory_utilization * 0.8;  // GPU typically correlates with memory
    }

    // Weighted combination
    double weights[3] = {0.5, 0.35, 0.15};  // CPU, Memory, GPU
    if (!resources->has_gpu) {
        weights[0] = 0.6;
        weights[1] = 0.4;
        weights[2] = 0.0;
    }

    return fmin(1.0, fmax(0.0,
        weights[0] * cpu_utilization +
        weights[1] * memory_utilization +
        weights[2] * gpu_utilization));
}

/**
 * Calculate communication overhead based on task characteristics
 */
double calculate_communication_overhead(const QuantumTask* task) {
    if (!task) return 0.0;

    double overhead = 0.0;

    // Base network latency overhead
    overhead += 0.02;  // 2% baseline for any quantum-classical communication

    if (task->circuit) {
        // Measurement overhead: each measurement requires classical readout
        size_t num_measurements = 0;
        for (size_t i = 0; i < task->circuit->num_gates; i++) {
            if (task->circuit->gates[i].type == GATE_MEASURE) {
                num_measurements++;
            }
        }
        // Each measurement adds ~1% overhead due to classical communication
        overhead += 0.01 * (double)num_measurements;

        // Circuit depth impacts communication (deeper circuits need more synchronization)
        overhead += 0.005 * log2(1.0 + (double)task->circuit->depth);

        // Number of qubits affects data transfer volume
        overhead += 0.002 * (double)task->circuit->num_qubits;
    }

    // Parameter transfer overhead for variational algorithms
    if (task->num_parameters > 0) {
        // Each parameter requires bidirectional communication in VQE/QAOA loops
        double param_overhead = 0.001 * (double)task->num_parameters;
        overhead += param_overhead;
    }

    // Number of shots increases result data volume
    if (task->num_shots > 0) {
        // Log scale: 1000 shots adds ~2%, 10000 adds ~4%
        overhead += 0.02 * log10(1.0 + (double)task->num_shots) / 3.0;
    }

    // Hybrid algorithms have additional coordination overhead
    switch (task->algorithm_type) {
        case ALGORITHM_VQE:
        case ALGORITHM_QAOA:
            overhead *= 1.5;  // Iterative algorithms have more round-trips
            break;
        case ALGORITHM_QML:
            overhead *= 1.3;  // ML has batch data transfer
            break;
        default:
            break;
    }

    return fmin(0.5, overhead);  // Cap at 50% overhead
}

/**
 * Calculate total energy consumption based on utilization and hardware profiles
 */
double calculate_energy_consumption(const HybridOrchestrator* orchestrator) {
    if (!orchestrator) return 0.0;

    double quantum_power_watts = 0.0;
    double classical_power_watts = 0.0;

    // Quantum hardware power consumption
    // Dilution refrigerators: ~10-15 kW baseline, plus ~1W per active qubit operation
    if (orchestrator->quantum_hardware) {
        double baseline_power = 12000.0;  // 12 kW for dilution fridge
        double active_power_per_qubit = 1.0;
        size_t num_qubits = orchestrator->quantum_hardware->capabilities.max_qubits;
        if (num_qubits == 0) num_qubits = 127;

        // Scale with utilization
        quantum_power_watts = baseline_power +
                             active_power_per_qubit * (double)num_qubits *
                             orchestrator->metrics.quantum_utilization;
    }

    // Classical compute power consumption
    // Modern CPUs: ~100-300W, GPUs: ~200-500W
    double cpu_tdp = 250.0;  // Typical server CPU TDP
    double gpu_tdp = 350.0;  // Typical data center GPU TDP

    // CPU power scales with utilization (but never below ~30% of TDP when on)
    double cpu_power = cpu_tdp * (0.3 + 0.7 * orchestrator->metrics.classical_utilization);

    // Add per-core scaling
    size_t num_cores = orchestrator->classical_resources.num_cores;
    if (num_cores > 1) {
        cpu_power *= sqrt((double)num_cores);  // Sublinear scaling
    }

    classical_power_watts = cpu_power;

    // GPU power if present
    if (orchestrator->classical_resources.has_gpu) {
        double gpu_power = gpu_tdp * (0.1 + 0.9 * orchestrator->metrics.classical_utilization);
        classical_power_watts += gpu_power;
    }

    // Memory power: ~3W per 8GB DIMM
    size_t memory_gb = orchestrator->classical_resources.memory_size / (1024 * 1024 * 1024);
    classical_power_watts += 3.0 * (memory_gb / 8.0 + 1);

    // Total power in watts
    double total_power = quantum_power_watts + classical_power_watts;

    // Add communication infrastructure power (~5% overhead)
    total_power *= 1.05;

    // Add cooling overhead (PUE ~1.4 for efficient data centers)
    total_power *= 1.4;

    return total_power;
}

// ============================================================================
// Resource Adjustment Functions
// ============================================================================

/**
 * Increase classical resource allocation when quantum hardware is overloaded.
 * This shifts workload balance toward classical pre/post-processing.
 */
void increase_classical_allocation(HybridOrchestrator* orchestrator) {
    if (!orchestrator) return;

    // Increase classical allocation factor (used in workload splitting)
    double current_factor = orchestrator->classical_allocation_factor;
    double new_factor = fmin(2.0, current_factor * 1.1);  // Max 2x baseline

    if (new_factor != current_factor) {
        orchestrator->classical_allocation_factor = new_factor;

        // Correspondingly reduce quantum allocation
        orchestrator->quantum_allocation_factor =
            fmax(0.5, orchestrator->quantum_allocation_factor * 0.95);
    }

    // Request more threads if available
    if (orchestrator->classical_resources.num_threads <
        orchestrator->classical_resources.num_cores * 2) {
        orchestrator->classical_resources.num_threads++;
    }

    // Increase batch size for more efficient classical processing
    if (orchestrator->batch_size < MAX_BATCH_SIZE) {
        orchestrator->batch_size = (size_t)fmin(MAX_BATCH_SIZE,
                                                orchestrator->batch_size * 1.2);
    }
}

/**
 * Increase quantum resource allocation when classical resources are overloaded.
 * This shifts workload balance toward quantum processing.
 */
void increase_quantum_allocation(HybridOrchestrator* orchestrator) {
    if (!orchestrator) return;

    // Increase quantum allocation factor
    double current_factor = orchestrator->quantum_allocation_factor;
    double new_factor = fmin(2.0, current_factor * 1.1);  // Max 2x baseline

    if (new_factor != current_factor) {
        orchestrator->quantum_allocation_factor = new_factor;

        // Correspondingly reduce classical allocation
        orchestrator->classical_allocation_factor =
            fmax(0.5, orchestrator->classical_allocation_factor * 0.95);
    }

    // Reduce batch size for more quantum parallelism
    if (orchestrator->batch_size > 64) {
        orchestrator->batch_size = (size_t)fmax(64, orchestrator->batch_size * 0.9);
    }

    // Update scheduler to prioritize quantum jobs
    if (orchestrator->scheduler) {
        orchestrator->scheduler->max_quantum_jobs =
            (size_t)fmin(1000, orchestrator->scheduler->max_quantum_jobs * 1.1);
    }
}

/**
 * Optimize communication patterns between quantum and classical resources.
 * Reduces round-trip overhead through batching and pipelining.
 */
void optimize_communication_patterns(HybridOrchestrator* orchestrator) {
    if (!orchestrator) return;

    double current_overhead = orchestrator->metrics.communication_overhead;

    // Strategy 1: Increase batch size to amortize communication costs
    if (current_overhead > 0.15 && orchestrator->batch_size < MAX_BATCH_SIZE) {
        size_t new_batch_size = (size_t)fmin(MAX_BATCH_SIZE,
                                             orchestrator->batch_size * 1.5);
        orchestrator->batch_size = new_batch_size;
    }

    // Strategy 2: For iterative algorithms (VQE/QAOA), use local optimization
    // before quantum evaluation to reduce the number of circuit submissions

    // Strategy 3: Pipeline measurement results with next circuit preparation
    // This is handled by updating timing strategy

    double current_time = get_current_time_seconds();
    double time_since_last = current_time - orchestrator->last_communication_time;

    // If communications are happening too frequently, batch them
    if (time_since_last < 0.001) {  // Less than 1ms between communications
        // Increase minimum communication interval
        orchestrator->batch_size += 16;
    }

    // Strategy 4: Compress result data for large-scale experiments
    // (Would enable compression in actual implementation)

    // Update last communication time
    orchestrator->last_communication_time = current_time;
}

/**
 * Optimize energy usage while maintaining performance.
 * Implements dynamic voltage/frequency scaling concepts.
 */
void optimize_energy_usage(HybridOrchestrator* orchestrator) {
    if (!orchestrator) return;

    double current_energy = orchestrator->metrics.energy_consumption;
    double energy_budget = orchestrator->classical_resources.energy_budget;

    if (energy_budget <= 0) {
        // No budget specified, set reasonable default based on resources
        energy_budget = 5000.0;  // 5kW default
    }

    double budget_ratio = current_energy / energy_budget;

    if (budget_ratio > 1.0) {
        // Over budget: reduce resource usage

        // Strategy 1: Reduce thread count
        if (orchestrator->classical_resources.num_threads > 4) {
            orchestrator->classical_resources.num_threads =
                (size_t)(orchestrator->classical_resources.num_threads * 0.9);
        }

        // Strategy 2: Reduce batch size to lower memory bandwidth
        if (orchestrator->batch_size > 128) {
            orchestrator->batch_size = (size_t)(orchestrator->batch_size * 0.8);
        }

        // Strategy 3: Shift to quantum (typically more energy efficient per operation)
        orchestrator->quantum_allocation_factor *= 1.05;
        orchestrator->classical_allocation_factor *= 0.95;

        // Strategy 4: Reduce quantum utilization if fridge is the main consumer
        if (orchestrator->quantum_hardware) {
            // Only reduce if quantum is high relative to classical
            if (orchestrator->metrics.quantum_utilization > 0.8 &&
                orchestrator->metrics.classical_utilization < 0.5) {
                orchestrator->quantum_allocation_factor *= 0.95;
            }
        }

    } else if (budget_ratio < 0.5) {
        // Under-utilizing budget: can increase performance

        // Strategy 1: Increase parallelism
        if (orchestrator->classical_resources.num_threads <
            orchestrator->classical_resources.num_cores * 2) {
            orchestrator->classical_resources.num_threads++;
        }

        // Strategy 2: Increase batch size for throughput
        if (orchestrator->batch_size < MAX_BATCH_SIZE) {
            orchestrator->batch_size = (size_t)fmin(MAX_BATCH_SIZE,
                                                    orchestrator->batch_size * 1.2);
        }
    }

    // Track total energy consumed (for monitoring)
    double delta_time = get_current_time_seconds() - orchestrator->quantum_state.last_measurement_time;
    if (delta_time > 0 && delta_time < 1.0) {  // Reasonable measurement interval
        orchestrator->total_energy_consumed += current_energy * delta_time / 3600.0;  // Wh
    }
}

// ============================================================================
// Circuit Optimization
// ============================================================================

QuantumCircuit* optimize_for_hardware(const QuantumCircuit* circuit,
                                     const HardwareCapabilities* capabilities) {
    if (!circuit || !capabilities) return NULL;

    // Create a deep copy of the circuit
    QuantumCircuit* optimized = malloc(sizeof(QuantumCircuit));
    if (!optimized) return NULL;

    // Initialize basic fields
    optimized->num_qubits = circuit->num_qubits;
    optimized->num_classical_bits = circuit->num_classical_bits;
    optimized->depth = circuit->depth;
    optimized->max_circuit_depth = circuit->max_circuit_depth;
    optimized->num_gates = circuit->num_gates;
    optimized->capacity = circuit->num_gates > 0 ? circuit->num_gates : 16;
    optimized->optimization_data = NULL;
    optimized->metadata = NULL;

    // Deep copy gates array
    if (circuit->gates && circuit->num_gates > 0) {
        optimized->gates = malloc(optimized->capacity * sizeof(HardwareGate));
        if (!optimized->gates) {
            free(optimized);
            return NULL;
        }
        for (size_t i = 0; i < circuit->num_gates; i++) {
            optimized->gates[i] = circuit->gates[i];
            // Deep copy parameters if present
            if (circuit->gates[i].parameters && circuit->gates[i].num_params > 0) {
                optimized->gates[i].parameters = malloc(circuit->gates[i].num_params * sizeof(double));
                if (optimized->gates[i].parameters) {
                    memcpy(optimized->gates[i].parameters, circuit->gates[i].parameters,
                           circuit->gates[i].num_params * sizeof(double));
                }
            }
        }
    } else {
        optimized->gates = malloc(optimized->capacity * sizeof(HardwareGate));
        if (!optimized->gates) {
            free(optimized);
            return NULL;
        }
    }

    // Deep copy measured array
    if (circuit->measured && circuit->num_qubits > 0) {
        optimized->measured = malloc(circuit->num_qubits * sizeof(bool));
        if (optimized->measured) {
            memcpy(optimized->measured, circuit->measured, circuit->num_qubits * sizeof(bool));
        }
    } else {
        optimized->measured = calloc(circuit->num_qubits > 0 ? circuit->num_qubits : 1, sizeof(bool));
    }

    // Apply hardware-specific optimizations based on capabilities
    if (capabilities->max_qubits > 0 && optimized->num_qubits > capabilities->max_qubits) {
        // Circuit needs decomposition - this will be handled by the backend
    }

    // Gate set optimization: mark gates that may need native basis conversion
    for (size_t i = 0; i < optimized->num_gates; i++) {
        HardwareGate* gate = &optimized->gates[i];
        // SWAP gates may need decomposition to CNOTs based on connectivity
        if (gate->type == GATE_SWAP && capabilities->connectivity_size > 0) {
            // Hardware-specific decomposition handled by backend optimizer
        }
    }

    return optimized;
}

// Note: Renamed from cleanup_quantum_circuit to avoid conflict with quantum_circuit_operations.c
// This version works with QuantumCircuit (orchestrator type), not quantum_circuit_t
void cleanup_orchestrator_circuit(QuantumCircuit* circuit) {
    if (!circuit) return;

    // Free gates and their parameters
    if (circuit->gates) {
        for (size_t i = 0; i < circuit->num_gates; i++) {
            if (circuit->gates[i].parameters) {
                free(circuit->gates[i].parameters);
            }
        }
        free(circuit->gates);
    }

    // Free measured array
    if (circuit->measured) {
        free(circuit->measured);
    }

    // Free optimization data if present
    if (circuit->optimization_data) {
        free(circuit->optimization_data);
    }

    free(circuit);
}

// ============================================================================
// Hardware Submission
// ============================================================================

// Renamed to avoid conflict with quantum_hardware_abstraction.c (this is orchestrator-specific)
int submit_orchestrated_quantum_circuit(QuantumHardware* hardware,
                                        const QuantumCircuit* circuit,
                                        QuantumResult* result) {
    if (!hardware || !circuit || !result) return -1;

    // Create quantum program using the hardware abstraction API
    QuantumProgram* program = create_program((uint32_t)circuit->num_qubits,
                                              (uint32_t)circuit->num_classical_bits);
    if (!program) return -1;

    // Convert each gate in the circuit to a quantum operation
    for (size_t i = 0; i < circuit->num_gates; i++) {
        // Get pointer to the hardware gate in the circuit
        const HardwareGate* hw_gate = &circuit->gates[i];

        // Create quantum operation from hardware gate
        QuantumOperation op = {0};
        op.type = OPERATION_GATE;

        // Fill in the HardwareGate structure within the operation
        op.op.gate.type = hw_gate->type;
        op.op.gate.target = hw_gate->target;
        op.op.gate.control = hw_gate->control;
        op.op.gate.target1 = hw_gate->target1;
        op.op.gate.target2 = hw_gate->target2;
        op.op.gate.parameter = hw_gate->parameter;

        // Handle multi-parameter gates
        if (hw_gate->parameters && hw_gate->num_params > 0) {
            op.op.gate.parameters = malloc(hw_gate->num_params * sizeof(double));
            if (op.op.gate.parameters) {
                memcpy(op.op.gate.parameters, hw_gate->parameters,
                       hw_gate->num_params * sizeof(double));
                op.op.gate.num_params = hw_gate->num_params;
            }
        } else {
            op.op.gate.parameters = NULL;
            op.op.gate.num_params = 0;
        }

        // Add operation to program
        if (!add_operation(program, &op)) {
            // Free any allocated parameters on failure
            if (op.op.gate.parameters) {
                free(op.op.gate.parameters);
            }
            cleanup_program(program);
            return -1;
        }

        // Note: add_operation makes a copy, so we can free allocated parameters
        if (op.op.gate.parameters) {
            free(op.op.gate.parameters);
        }
    }

    // Add measurement operations for all qubits to get results
    for (uint32_t q = 0; q < (uint32_t)circuit->num_qubits; q++) {
        // Only add measurement if qubit is marked for measurement
        if (circuit->measured && circuit->measured[q]) {
            QuantumOperation measure_op = {0};
            measure_op.type = OPERATION_MEASURE;
            measure_op.op.measure.qubit = q;
            measure_op.op.measure.classical_bit = q;

            if (!add_operation(program, &measure_op)) {
                cleanup_program(program);
                return -1;
            }
        }
    }

    // Execute on hardware through the abstraction layer
    struct ExecutionResult* exec_result = execute_program(hardware, program);

    if (exec_result && exec_result->success) {
        result->fidelity = exec_result->fidelity;
        result->execution_time = exec_result->execution_time;

        // Copy probabilities if available
        if (exec_result->probabilities && exec_result->num_results > 0) {
            result->probabilities = malloc(exec_result->num_results * sizeof(double));
            if (result->probabilities) {
                memcpy(result->probabilities, exec_result->probabilities,
                       exec_result->num_results * sizeof(double));
                result->num_probabilities = exec_result->num_results;
            }
        }

        // Copy measurements as expectation values
        if (exec_result->measurements && exec_result->num_results > 0) {
            result->expectation_values = malloc(exec_result->num_results * sizeof(double));
            if (result->expectation_values) {
                memcpy(result->expectation_values, exec_result->measurements,
                       exec_result->num_results * sizeof(double));
                result->num_values = exec_result->num_results;
            }
        }

        cleanup_result(exec_result);
        cleanup_program(program);
        return 0;
    }

    // Execution failed - clean up and return error
    if (exec_result) {
        cleanup_result(exec_result);
    }
    cleanup_program(program);

    // Set default values on failure
    result->fidelity = 0.0;
    result->execution_time = 0.0;
    result->expectation_values = NULL;
    result->num_values = 0;
    result->probabilities = NULL;
    result->num_probabilities = 0;

    return -1;
}

// ============================================================================
// Classical Computation
// ============================================================================

int classical_computation(const QuantumTask* task,
                         const ClassicalConfig* config,
                         ClassicalResult* result) {
    if (!task || !config || !result) return -1;

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Initialize result
    result->output_data = NULL;
    result->output_size = 0;
    result->error_metric = 0.0;

    // For hybrid algorithms, classical computation operates on the task's parameters
    // VQE/QAOA: gradient computation, parameter updates
    // QML: data preprocessing, post-processing
    const double* parameters = task->parameters;
    size_t num_parameters = task->num_parameters;

    if (!parameters || num_parameters == 0) {
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        result->execution_time = (double)(end_time.tv_sec - start_time.tv_sec) +
                                 (double)(end_time.tv_nsec - start_time.tv_nsec) / 1e9;
        return 0;  // No parameters to process
    }

    // Allocate output buffer for processed parameters
    result->output_data = malloc(num_parameters * sizeof(double));
    if (!result->output_data) {
        return -1;
    }
    result->output_size = num_parameters;

    // Perform classical computation based on algorithm type
    double sum_squared_gradient = 0.0;

    switch (task->algorithm_type) {
        case ALGORITHM_VQE:
        case ALGORITHM_QAOA: {
            // For variational algorithms: compute parameter updates using SPSA
            // (Simultaneous Perturbation Stochastic Approximation)
            // This is efficient for high-dimensional optimization as it requires
            // only 2 function evaluations per iteration regardless of dimension

            // Default learning rate for SPSA optimization
            double learning_rate = 0.01;
            double perturbation_size = 0.1;  // c_k in SPSA, typically decreases over iterations

            // Use iteration counter based on current time for reproducibility
            static size_t iteration_counter = 0;
            size_t iteration = ++iteration_counter;

            // SPSA adaptive parameters: a_k = a / (A + k)^alpha, c_k = c / k^gamma
            double alpha = 0.602;  // Optimal for twice-differentiable functions
            double gamma = 0.101;
            double a_coeff = learning_rate * pow((double)(iteration + 100), alpha);
            double c_coeff = perturbation_size / pow((double)iteration, gamma);

            // Generate Bernoulli perturbation vector (+1 or -1 with equal probability)
            // Use deterministic seed based on iteration for reproducibility
            unsigned int seed = (unsigned int)(iteration * 31337 + (size_t)task->task_data);

            // Storage for perturbation vector
            double* delta = malloc(num_parameters * sizeof(double));
            if (!delta) {
                free(result->output_data);
                result->output_data = NULL;
                return -1;
            }

            // Generate perturbation direction
            for (size_t i = 0; i < num_parameters; i++) {
                seed = seed * 1103515245 + 12345;  // LCG for reproducibility
                delta[i] = ((seed >> 16) & 1) ? 1.0 : -1.0;
            }

            // Compute gradient estimates using parameter-shift rule approximation
            // For quantum circuits: ∂f/∂θ_i ≈ (f(θ + c*Δ) - f(θ - c*Δ)) / (2*c*Δ_i)
            // We approximate using finite differences with stored expectation values

            // Get expectation values from task_data if available
            double f_plus = 0.0, f_minus = 0.0;
            bool have_expectation = false;

            if (task->task_data) {
                // Try to use task_data as expectation values [f(θ+), f(θ-)]
                double* exp_vals = (double*)task->task_data;
                // Validate pointer is readable by checking alignment
                if (((uintptr_t)exp_vals & (sizeof(double) - 1)) == 0) {
                    f_plus = exp_vals[0];
                    f_minus = exp_vals[1];
                    have_expectation = (f_plus != 0.0 || f_minus != 0.0);
                }
            }

            if (!have_expectation) {
                // Fallback: estimate gradient from parameter curvature
                // This uses the natural gradient approximation
                for (size_t i = 0; i < num_parameters; i++) {
                    double param = parameters[i];
                    // Natural gradient for rotation gates: ∂E/∂θ ∝ -sin(θ) * <Z>
                    // Without <Z>, approximate using parameter value
                    double gradient = -sin(param) * cos(param * 0.5);
                    f_plus += gradient * delta[i] * c_coeff;
                }
                f_minus = -f_plus;  // Symmetric estimate
            }

            // Compute SPSA gradient estimate and update parameters
            double gradient_estimate = (f_plus - f_minus) / (2.0 * c_coeff);

            #pragma omp parallel for reduction(+:sum_squared_gradient) if(num_parameters > 100)
            for (size_t i = 0; i < num_parameters; i++) {
                double param = parameters[i];

                // SPSA gradient estimate for parameter i
                double gradient_i = gradient_estimate / delta[i];

                // Apply update with adaptive learning rate
                double updated_param = param - a_coeff * gradient_i;

                // Clamp to valid range [-π, π] for rotation angles
                while (updated_param > M_PI) updated_param -= 2.0 * M_PI;
                while (updated_param < -M_PI) updated_param += 2.0 * M_PI;

                ((double*)result->output_data)[i] = updated_param;
                sum_squared_gradient += gradient_i * gradient_i;
            }

            free(delta);

            // RMS gradient as error metric
            result->error_metric = sqrt(sum_squared_gradient / (double)num_parameters);
            break;
        }

        case ALGORITHM_QML: {
            // For quantum machine learning: data preprocessing/postprocessing
            #pragma omp parallel for if(num_parameters > 100)
            for (size_t i = 0; i < num_parameters; i++) {
                double param = parameters[i];

                // Normalize parameters for encoding
                double normalized = tanh(param);  // Map to [-1, 1]
                ((double*)result->output_data)[i] = normalized;
            }
            result->error_metric = 0.0;
            break;
        }

        case ALGORITHM_QUANTUM_CHEMISTRY: {
            // For chemistry: compute molecular integrals classically
            #pragma omp parallel for if(num_parameters > 100)
            for (size_t i = 0; i < num_parameters; i++) {
                // Transform parameters for chemistry ansatz
                double param = parameters[i];
                double transformed = param * exp(-param * param * 0.1);
                ((double*)result->output_data)[i] = transformed;
            }
            result->error_metric = 0.0;
            break;
        }

        case ALGORITHM_OPTIMIZATION:
        case ALGORITHM_SIMULATION:
        case ALGORITHM_GENERIC:
        default: {
            // Generic: pass through with optional scaling
            #pragma omp parallel for if(num_parameters > 100)
            for (size_t i = 0; i < num_parameters; i++) {
                ((double*)result->output_data)[i] = parameters[i];
            }
            result->error_metric = 0.0;
            break;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    result->execution_time = (double)(end_time.tv_sec - start_time.tv_sec) +
                             (double)(end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    return 0;
}

// ============================================================================
// Task Splitting Functions
// ============================================================================

void split_vqe_task(const QuantumTask* task,
                   QuantumTask* quantum_part,
                   ClassicalTask* classical_part) {
    if (!task || !quantum_part || !classical_part) return;

    // VQE: quantum circuit evaluation, classical optimization
    *quantum_part = *task;
    classical_part->algorithm_type = ALGORITHM_VQE;
    classical_part->input_data = task->parameters;
    classical_part->input_size = task->num_parameters * sizeof(double);
}

void split_qaoa_task(const QuantumTask* task,
                    QuantumTask* quantum_part,
                    ClassicalTask* classical_part) {
    if (!task || !quantum_part || !classical_part) return;

    // Similar to VQE
    *quantum_part = *task;
    classical_part->algorithm_type = ALGORITHM_QAOA;
    classical_part->input_data = task->parameters;
    classical_part->input_size = task->num_parameters * sizeof(double);
}

void split_qml_task(const QuantumTask* task,
                   QuantumTask* quantum_part,
                   ClassicalTask* classical_part) {
    if (!task || !quantum_part || !classical_part) return;

    *quantum_part = *task;
    classical_part->algorithm_type = ALGORITHM_QML;
    classical_part->input_data = NULL;
    classical_part->input_size = 0;
}

void split_generic_task(const QuantumTask* task,
                       QuantumTask* quantum_part,
                       ClassicalTask* classical_part) {
    if (!task || !quantum_part || !classical_part) return;

    *quantum_part = *task;
    classical_part->algorithm_type = ALGORITHM_GENERIC;
    classical_part->input_data = NULL;
    classical_part->input_size = 0;
}

// ============================================================================
// Result Combination Functions
// ============================================================================

void combine_vqe_results(const QuantumResult* quantum_result,
                        const ClassicalResult* classical_result,
                        HybridResult* final_result) {
    if (!quantum_result || !classical_result || !final_result) return;

    final_result->quantum_result = *quantum_result;
    final_result->classical_result = *classical_result;
    final_result->combined_fidelity = quantum_result->fidelity;
    final_result->total_time = quantum_result->execution_time + classical_result->execution_time;
}

void combine_qaoa_results(const QuantumResult* quantum_result,
                         const ClassicalResult* classical_result,
                         HybridResult* final_result) {
    combine_vqe_results(quantum_result, classical_result, final_result);
}

void combine_qml_results(const QuantumResult* quantum_result,
                        const ClassicalResult* classical_result,
                        HybridResult* final_result) {
    combine_vqe_results(quantum_result, classical_result, final_result);
}

void combine_generic_results(const QuantumResult* quantum_result,
                            const ClassicalResult* classical_result,
                            HybridResult* final_result) {
    combine_vqe_results(quantum_result, classical_result, final_result);
}

// ============================================================================
// Optimization
// ============================================================================

int orchestrator_optimize_parameters(OptimizationContext* optimizer,
                                     double (*objective_fn)(const double*, double*, void*),
                                     void* objective_data,
                                     HybridOrchestrator* orchestrator) {
    if (!optimizer || !objective_fn || !orchestrator) return -1;

    // Use the classical optimizer to run the optimization
    OptimizationObjective obj = {
        .function = objective_fn,
        .data = objective_data
    };

    return optimize_parameters(optimizer, obj, objective_data);
}

// ============================================================================
// Core Orchestrator Implementation
// ============================================================================

// Initialize hybrid orchestrator
HybridOrchestrator* init_hybrid_orchestrator(QuantumHardware* quantum_hw,
                                           const ClassicalResources* classical) {
    HybridOrchestrator* orchestrator = calloc(1, sizeof(HybridOrchestrator));
    if (!orchestrator) return NULL;

    orchestrator->quantum_hardware = quantum_hw;
    orchestrator->classical_resources = *classical;

    // Initialize metrics
    orchestrator->metrics.quantum_utilization = 0.0;
    orchestrator->metrics.classical_utilization = 0.0;
    orchestrator->metrics.communication_overhead = 0.0;
    orchestrator->metrics.energy_consumption = 0.0;

    // Initialize scheduler
    orchestrator->scheduler = init_workload_scheduler();
    orchestrator->enable_auto_tuning = true;

    // Initialize quantum resource state
    double current_time = get_current_time_seconds();
    orchestrator->quantum_state.active_jobs = 0;
    orchestrator->quantum_state.queued_jobs = 0;
    orchestrator->quantum_state.total_qubits_in_use = 0;
    orchestrator->quantum_state.cumulative_gate_time = 0.0;
    orchestrator->quantum_state.last_measurement_time = current_time;
    orchestrator->quantum_state.history_index = 0;
    orchestrator->quantum_state.ewma_utilization = 0.0;
    memset(orchestrator->quantum_state.job_history, 0, sizeof(orchestrator->quantum_state.job_history));

    // Initialize classical resource state
    orchestrator->classical_state.active_threads = 0;
    orchestrator->classical_state.memory_allocated = 0;
    orchestrator->classical_state.gpu_memory_allocated = 0;
    orchestrator->classical_state.cpu_time_used = 0.0;
    orchestrator->classical_state.last_measurement_time = current_time;
    orchestrator->classical_state.ewma_utilization = 0.0;
#ifdef __APPLE__
    orchestrator->classical_state.has_prev_load = false;
#elif defined(__linux__)
    orchestrator->classical_state.has_prev_time = false;
#endif

    // Initialize allocation factors (1.0 = baseline)
    orchestrator->quantum_allocation_factor = 1.0;
    orchestrator->classical_allocation_factor = 1.0;

    // Initialize communication optimization state
    orchestrator->batch_size = 256;  // Default batch size
    orchestrator->last_communication_time = current_time;
    orchestrator->bytes_transferred = 0;

    // Initialize energy tracking
    orchestrator->total_energy_consumed = 0.0;
    orchestrator->power_limit_watts = classical->energy_budget > 0 ?
                                      classical->energy_budget : 5000.0;  // 5kW default

    return orchestrator;
}

// Analyze workload characteristics
static WorkloadType analyze_workload(const QuantumTask* task) {
    if (!task) return WORKLOAD_CLASSICAL;

    // Analyze quantum advantage potential
    double quantum_score = 0.0;

    // Check circuit depth and width
    if (task->circuit && task->circuit->num_qubits >= MIN_QUANTUM_SIZE) {
        quantum_score += 0.4;  // Circuit size suitable for quantum
    }

    // Check entanglement
    if (has_significant_entanglement(task->circuit)) {
        quantum_score += 0.3;  // High entanglement favors quantum
    }

    // Check classical difficulty
    if (is_classically_hard(task)) {
        quantum_score += 0.3;  // Classical hardness favors quantum
    }

    // Determine workload type
    if (quantum_score >= QUANTUM_THRESHOLD) {
        return WORKLOAD_QUANTUM;
    } else if (quantum_score <= 1.0 - QUANTUM_THRESHOLD) {
        return WORKLOAD_CLASSICAL;
    } else {
        return WORKLOAD_HYBRID;
    }
}

// Execute hybrid quantum-classical task
int execute_hybrid_task(HybridOrchestrator* orchestrator,
                       const QuantumTask* task,
                       HybridResult* result) {
    if (!orchestrator || !task || !result) return -1;

    // Analyze workload
    WorkloadType type = analyze_workload(task);

    // Update metrics
    update_resource_metrics(orchestrator, task);

    // Execute based on workload type
    switch (type) {
        case WORKLOAD_QUANTUM:
            return execute_quantum_task(orchestrator, task, result);

        case WORKLOAD_CLASSICAL:
            return execute_classical_task(orchestrator, task, result);

        case WORKLOAD_HYBRID:
            return execute_hybrid_split_task(orchestrator, task, result);

        default:
            return -1;
    }
}

// Execute quantum portion
static int execute_quantum_task(HybridOrchestrator* orchestrator,
                              const QuantumTask* task,
                              HybridResult* result) {
    // Optimize circuit for quantum hardware
    QuantumCircuit* optimized = optimize_for_hardware(
        task->circuit,
        &orchestrator->quantum_hardware->capabilities);

    if (!optimized) return -1;

    // Submit to quantum hardware
    int status = submit_orchestrated_quantum_circuit(
        orchestrator->quantum_hardware,
        optimized,
        &result->quantum_result);

    cleanup_orchestrator_circuit(optimized);
    return status;
}

// Execute classical portion
static int execute_classical_task(HybridOrchestrator* orchestrator,
                                const QuantumTask* task,
                                HybridResult* result) {
    // Set up classical computation
    ClassicalConfig config = {
        .num_threads = orchestrator->classical_resources.num_cores,
        .use_gpu = orchestrator->classical_resources.has_gpu,
        .memory_limit = orchestrator->classical_resources.memory_size
    };

    // Execute classical algorithm
    return classical_computation(task, &config, &result->classical_result);
}

// Execute hybrid split task
static int execute_hybrid_split_task(HybridOrchestrator* orchestrator,
                                   const QuantumTask* task,
                                   HybridResult* result) {
    // Split task into quantum and classical parts
    QuantumTask quantum_part;
    ClassicalTask classical_part;
    split_hybrid_task(task, &quantum_part, &classical_part);

    // Execute quantum part
    int quantum_status = execute_quantum_task(
        orchestrator,
        &quantum_part,
        result);

    if (quantum_status != 0) return quantum_status;

    // Execute classical part
    int classical_status = execute_classical_task(
        orchestrator,
        &quantum_part,
        result);

    if (classical_status != 0) return classical_status;

    // Combine results
    combine_hybrid_results_internal(&result->quantum_result,
                                   &result->classical_result,
                                   result);

    return 0;
}

// Update resource metrics
static void update_resource_metrics(HybridOrchestrator* orchestrator,
                                  const QuantumTask* task) {
    // Update quantum utilization
    orchestrator->metrics.quantum_utilization =
        calculate_quantum_utilization(orchestrator->quantum_hardware);

    // Update classical utilization
    orchestrator->metrics.classical_utilization =
        calculate_classical_utilization(&orchestrator->classical_resources);

    // Update communication overhead
    orchestrator->metrics.communication_overhead =
        calculate_communication_overhead(task);

    // Update energy consumption
    orchestrator->metrics.energy_consumption =
        calculate_energy_consumption(orchestrator);

    // Auto-tune if enabled
    if (orchestrator->enable_auto_tuning) {
        auto_tune_resources(orchestrator);
    }
}

// Auto-tune resource allocation
static void auto_tune_resources(HybridOrchestrator* orchestrator) {
    // Adjust quantum/classical split based on metrics
    if (orchestrator->metrics.quantum_utilization > 0.9) {
        // Quantum hardware overloaded, shift more to classical
        increase_classical_allocation(orchestrator);
    } else if (orchestrator->metrics.classical_utilization > 0.9) {
        // Classical hardware overloaded, shift more to quantum
        increase_quantum_allocation(orchestrator);
    }

    // Optimize communication patterns
    if (orchestrator->metrics.communication_overhead > 0.2) {
        optimize_communication_patterns(orchestrator);
    }

    // Energy optimization
    if (orchestrator->metrics.energy_consumption >
        orchestrator->classical_resources.energy_budget) {
        optimize_energy_usage(orchestrator);
    }
}

// Helper functions

static bool has_significant_entanglement(const QuantumCircuit* circuit) {
    if (!circuit) return false;

    size_t entangling_gates = 0;
    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (circuit->gates[i].type == GATE_CNOT ||
            circuit->gates[i].type == GATE_CZ) {
            entangling_gates++;
        }
    }

    return (double)entangling_gates / circuit->num_gates > 0.1;
}

static bool is_classically_hard(const QuantumTask* task) {
    if (!task) return false;

    // Check for quantum advantage indicators
    if (task->circuit && task->circuit->num_qubits > 50) {
        return true;  // Large quantum circuits
    }

    if (task->algorithm_type == ALGORITHM_QUANTUM_CHEMISTRY ||
        task->algorithm_type == ALGORITHM_OPTIMIZATION) {
        return true;  // Known quantum advantage domains
    }

    return false;
}

static void split_hybrid_task(const QuantumTask* task,
                            QuantumTask* quantum_part,
                            ClassicalTask* classical_part) {
    // Split based on algorithm type
    switch (task->algorithm_type) {
        case ALGORITHM_VQE:
            split_vqe_task(task, quantum_part, classical_part);
            break;
        case ALGORITHM_QAOA:
            split_qaoa_task(task, quantum_part, classical_part);
            break;
        case ALGORITHM_QML:
            split_qml_task(task, quantum_part, classical_part);
            break;
        default:
            // Default to even split
            split_generic_task(task, quantum_part, classical_part);
            break;
    }
}

static void combine_hybrid_results_internal(const QuantumResult* quantum_result,
                                           const ClassicalResult* classical_result,
                                           HybridResult* final_result) {
    // Combine based on algorithm type
    switch (final_result->algorithm_type) {
        case ALGORITHM_VQE:
            combine_vqe_results(quantum_result, classical_result, final_result);
            break;
        case ALGORITHM_QAOA:
            combine_qaoa_results(quantum_result, classical_result, final_result);
            break;
        case ALGORITHM_QML:
            combine_qml_results(quantum_result, classical_result, final_result);
            break;
        default:
            // Default combination strategy
            combine_generic_results(quantum_result, classical_result, final_result);
            break;
    }
}

// Clean up orchestrator
void cleanup_hybrid_orchestrator(HybridOrchestrator* orchestrator) {
    if (!orchestrator) return;

    if (orchestrator->scheduler) {
        cleanup_workload_scheduler(orchestrator->scheduler);
    }

    free(orchestrator);
}

// Clean up hardware quantum circuit
void hw_circuit_cleanup(QuantumCircuit* circuit) {
    if (!circuit) return;

    // Free gates array
    if (circuit->gates) {
        free(circuit->gates);
        circuit->gates = NULL;
    }

    // Free measured array
    if (circuit->measured) {
        free(circuit->measured);
        circuit->measured = NULL;
    }

    // Free optimization data
    if (circuit->optimization_data) {
        free(circuit->optimization_data);
        circuit->optimization_data = NULL;
    }

    // Free metadata
    if (circuit->metadata) {
        free(circuit->metadata);
        circuit->metadata = NULL;
    }

    // Reset counts
    circuit->num_gates = 0;
    circuit->capacity = 0;
    circuit->num_qubits = 0;
    circuit->num_classical_bits = 0;
    circuit->depth = 0;

    // Free the circuit struct itself
    free(circuit);
}

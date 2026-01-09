#include "quantum_geometric/hybrid/quantum_hardware_scheduler.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include <stdlib.h>
#include <string.h>

// OpenMP support is handled by quantum_geometric_operations.h

// Hardware parameters
#define MAX_QUANTUM_SYSTEMS 8
#define MAX_CONCURRENT_OPS 32
#define BATCH_SIZE 1024
#define CACHE_LINE 64

// Network parameters
#define MIN_BANDWIDTH 100  // MB/s
#define MAX_LATENCY 10    // ms
#define BUFFER_SIZE (1ULL << 20)  // 1MB

// Scheduling thresholds
#define MIN_BATCH_SIZE 10
#define MAX_SYNC_POINTS 5
#define OVERLAP_THRESHOLD 0.1

// SystemStatus is defined in the header

// OperationRequirements is forward declared in header
typedef struct OperationRequirements {
    // Backend-specific resource requirements
    size_t ibm_qubits;        // IBM Quantum qubits needed
    size_t rigetti_qubits;    // Rigetti qubits needed
    size_t dwave_vars;        // DWave variables needed
    size_t memory_needed;

    // IBM-specific requirements
    size_t ibm_circuit_depth;         // Expected circuit depth
    double ibm_error_budget;          // Maximum error tolerance
    uint32_t ibm_optimization_level;  // Required optimization level
    bool ibm_feedback_required;       // Needs fast feedback

    // Timing constraints
    double max_latency;
    double target_runtime;

    // Dependencies
    size_t num_dependencies;
    QuantumOperation** dependencies;
} OperationRequirements;

// ExecutionSchedule is defined in the header

// Forward declarations for static functions
static ExecutionSchedule* create_schedule(HardwareScheduler* scheduler,
                                         const QuantumOperation* op,
                                         const SystemStatus* status,
                                         const OperationRequirements* requirements);
static bool allocate_quantum_resources(ExecutionSchedule* schedule,
                                       const SystemStatus* status,
                                       const OperationRequirements* requirements);
static bool schedule_data_transfers(ExecutionSchedule* schedule,
                                    const QuantumOperation* op,
                                    const SystemStatus* status);
static void add_sync_points(ExecutionSchedule* schedule,
                            const QuantumOperation* op,
                            const OperationRequirements* requirements);
static bool validate_schedule(const ExecutionSchedule* schedule,
                              const SystemStatus* status);
static void cleanup_scheduling(SystemStatus* status,
                               OperationRequirements* requirements,
                               ExecutionSchedule* schedule);

// Initialize hardware scheduler
HardwareScheduler* init_hardware_scheduler(
    const HardwareSchedulerConfig* config) {

    HardwareScheduler* scheduler = malloc(sizeof(HardwareScheduler));
    if (!scheduler) return NULL;

    // Initialize system monitoring
    scheduler->status = init_system_status();
    if (config->enable_monitoring) {
        scheduler->monitor = init_performance_monitor();
    } else {
        scheduler->monitor = NULL;
    }

    // Initialize resource management
    scheduler->resource_mgr = init_resource_manager(
        config->max_memory,
        config->max_threads
    );

    // Initialize network management
    scheduler->network_mgr = init_network_manager(
        config->bandwidth,
        config->latency
    );

    // Initialize scheduling components
    scheduler->operation_queue = init_operation_queue(MAX_CONCURRENT_OPS);
    scheduler->execution_log = init_execution_log();

    return scheduler;
}

// Schedule quantum operation
int schedule_quantum_operation(
    HardwareScheduler* scheduler,
    const QuantumOperation* op,
    ExecutionSchedule* schedule) {
    
    if (!scheduler || !op || !schedule) return -1;
    
    // Get current system status
    SystemStatus* status = get_system_status(scheduler);
    if (!status) return -1;
    
    // Analyze operation requirements
    OperationRequirements* requirements = analyze_requirements(op);
    if (!requirements) {
        free(status);
        return -1;
    }
    
    // Check resource availability
    if (!check_resources(status, requirements)) {
        cleanup_scheduling(status, requirements, NULL);
        return -1;
    }
    
    // Create execution schedule
    ExecutionSchedule* exec_schedule = create_schedule(
        scheduler,
        op,
        status,
        requirements
    );
    
    if (!exec_schedule) {
        cleanup_scheduling(status, requirements, NULL);
        return -1;
    }
    
    // Validate schedule
    if (!validate_schedule(exec_schedule, status)) {
        cleanup_scheduling(status, requirements, exec_schedule);
        return -1;
    }
    
    // Update system status
    update_system_status(scheduler, exec_schedule);
    
    // Copy schedule to output
    memcpy(schedule, exec_schedule, sizeof(ExecutionSchedule));
    
    // Cleanup
    cleanup_scheduling(status, requirements, exec_schedule);
    
    return 0;
}

// Create execution schedule
static ExecutionSchedule* create_schedule(
    HardwareScheduler* scheduler,
    const QuantumOperation* op,
    const SystemStatus* status,
    const OperationRequirements* requirements) {
    
    ExecutionSchedule* schedule = malloc(sizeof(ExecutionSchedule));
    if (!schedule) return NULL;
    
    // Schedule start time
    schedule->start_time = get_next_start_time(scheduler);
    
    // Allocate quantum resources
    if (!allocate_quantum_resources(
        schedule,
        status,
        requirements)) {
        free(schedule);
        return NULL;
    }
    
    // Schedule data transfers
    if (!schedule_data_transfers(
        schedule,
        op,
        status)) {
        free(schedule);
        return NULL;
    }
    
    // Add synchronization points
    add_sync_points(
        schedule,
        op,
        requirements
    );
    
    // Calculate end time
    schedule->end_time = calculate_end_time(schedule);
    
    return schedule;
}

// Allocate quantum resources
static bool allocate_quantum_resources(
    ExecutionSchedule* schedule,
    const SystemStatus* status,
    const OperationRequirements* requirements) {

    // IBM is primary backend - allocate first
    if (requirements->ibm_qubits > 0) {
        if (!status->ibm_available ||
            status->ibm_utilization > 0.8) {
            return false;
        }
        if (requirements->ibm_qubits > status->ibm_max_qubits) {
            return false;
        }
        schedule->ibm_qubits = requirements->ibm_qubits;

        // Set IBM-specific optimization parameters
        schedule->ibm_circuit_depth = requirements->ibm_circuit_depth;
        schedule->ibm_error_budget = requirements->ibm_error_budget;
        schedule->ibm_optimization_level = requirements->ibm_optimization_level;
        schedule->ibm_feedback_enabled = requirements->ibm_feedback_required;

        // Estimate fidelity based on error rates and circuit depth
        double gate_fidelity = 1.0 - status->ibm_gate_error_rate;
        double depth_fidelity = pow(gate_fidelity, (double)requirements->ibm_circuit_depth);
        double readout_fidelity = 1.0 - status->ibm_readout_error_rate;
        schedule->ibm_estimated_fidelity = depth_fidelity * readout_fidelity;
    }

    // Check Rigetti availability (fallback for gate-based)
    if (requirements->rigetti_qubits > 0) {
        if (!status->rigetti_available ||
            status->rigetti_utilization > 0.8) {
            return false;
        }
        schedule->rigetti_qubits = requirements->rigetti_qubits;
    }

    // Check DWave availability (annealing)
    if (requirements->dwave_vars > 0) {
        if (!status->dwave_available ||
            status->dwave_utilization > 0.8) {
            return false;
        }
        schedule->dwave_vars = requirements->dwave_vars;
    }

    // Allocate workspace
    schedule->workspace_size = requirements->memory_needed;

    return true;
}

// Schedule data transfers
static bool schedule_data_transfers(
    ExecutionSchedule* schedule,
    const QuantumOperation* op,
    const SystemStatus* status) {
    
    // Calculate total data transfer
    schedule->data_transfer = calculate_data_transfer(op);
    
    // Calculate required bandwidth
    schedule->bandwidth_needed = calculate_bandwidth_needed(
        schedule->data_transfer,
        schedule->end_time - schedule->start_time
    );
    
    // Check network capacity
    if (schedule->bandwidth_needed > status->bandwidth) {
        return false;
    }
    
    // Check latency constraints
    double transfer_time = schedule->data_transfer / status->bandwidth;
    if (transfer_time > MAX_LATENCY) {
        return false;
    }
    
    return true;
}

// Add synchronization points
static void add_sync_points(
    ExecutionSchedule* schedule,
    const QuantumOperation* op,
    const OperationRequirements* requirements) {
    
    // Add initial sync point
    schedule->sync_points[0] = schedule->start_time;
    
    // Add sync points for dependencies
    size_t sync_count = 1;
    for (size_t i = 0; i < requirements->num_dependencies; i++) {
        if (sync_count >= MAX_SYNC_POINTS) break;
        
        double sync_time = calculate_sync_time(
            schedule,
            requirements->dependencies[i]
        );
        
        schedule->sync_points[sync_count++] = sync_time;
    }
    
    // Add final sync point
    if (sync_count < MAX_SYNC_POINTS) {
        schedule->sync_points[sync_count] = schedule->end_time;
    }
}

// Validate execution schedule
static bool validate_schedule(
    const ExecutionSchedule* schedule,
    const SystemStatus* status) {
    
    // Check timing constraints
    if (schedule->end_time - schedule->start_time > MAX_LATENCY) {
        return false;
    }
    
    // Check resource constraints
    if (schedule->workspace_size > status->available_memory) {
        return false;
    }
    
    // Check network constraints
    if (schedule->bandwidth_needed > status->bandwidth) {
        return false;
    }
    
    // Validate sync points
    for (size_t i = 1; i < MAX_SYNC_POINTS; i++) {
        if (schedule->sync_points[i] <= schedule->sync_points[i-1]) {
            return false;
        }
    }
    
    return true;
}

// Clean up scheduling resources
static void cleanup_scheduling(
    SystemStatus* status,
    OperationRequirements* requirements,
    ExecutionSchedule* schedule) {

    if (status) free(status);
    if (requirements) free(requirements);
    if (schedule) free(schedule);
}

// ============================================================================
// Helper Function Implementations
// ============================================================================

// Initialize system status
struct SystemStatus* init_system_status(void) {
    SystemStatus* status = calloc(1, sizeof(SystemStatus));
    if (!status) return NULL;

    // Initialize with default available state - IBM is primary backend
    status->ibm_available = true;
    status->rigetti_available = true;
    status->dwave_available = true;
    status->ibm_utilization = 0.0;
    status->rigetti_utilization = 0.0;
    status->dwave_utilization = 0.0;

    // IBM-specific capabilities (default to IBM Brisbane-class backend)
    status->ibm_max_qubits = 127;
    status->ibm_gate_error_rate = 0.001;      // 0.1% typical gate error
    status->ibm_readout_error_rate = 0.01;    // 1% typical readout error
    status->ibm_optimization_level = 2;        // Default optimization level

    // Network defaults
    status->bandwidth = 1000.0;  // 1 GB/s
    status->latency = 1.0;       // 1ms
    status->buffer_usage = 0;

    // Resource defaults
    status->available_memory = 16ULL * 1024 * 1024 * 1024;  // 16GB
    status->available_threads = 8;
    status->cpu_utilization = 0.0;

    return (struct SystemStatus*)status;
}

// Get system status from scheduler
struct SystemStatus* get_system_status(HardwareScheduler* scheduler) {
    if (!scheduler || !scheduler->status) {
        return init_system_status();
    }

    // Return a copy of current status
    SystemStatus* copy = malloc(sizeof(SystemStatus));
    if (!copy) return NULL;

    memcpy(copy, scheduler->status, sizeof(SystemStatus));
    return (struct SystemStatus*)copy;
}

// Update system status after scheduling
void update_system_status(HardwareScheduler* scheduler, const ExecutionSchedule* schedule) {
    if (!scheduler || !scheduler->status || !schedule) return;

    SystemStatus* status = (SystemStatus*)scheduler->status;

    // Update utilization based on allocated resources
    // IBM is primary - update first
    if (schedule->ibm_qubits > 0) {
        // IBM utilization based on qubit allocation relative to max
        double qubit_fraction = (double)schedule->ibm_qubits / (double)status->ibm_max_qubits;
        status->ibm_utilization += qubit_fraction * 0.5;
        if (status->ibm_utilization > 1.0) {
            status->ibm_utilization = 1.0;
        }
    }

    if (schedule->rigetti_qubits > 0) {
        status->rigetti_utilization += 0.1;
        if (status->rigetti_utilization > 1.0) {
            status->rigetti_utilization = 1.0;
        }
    }

    if (schedule->dwave_vars > 0) {
        status->dwave_utilization += 0.1;
        if (status->dwave_utilization > 1.0) {
            status->dwave_utilization = 1.0;
        }
    }

    // Update memory usage
    if (schedule->workspace_size > 0 && status->available_memory >= schedule->workspace_size) {
        status->available_memory -= schedule->workspace_size;
    }

    // Update buffer usage
    status->buffer_usage += schedule->data_transfer;
}

// Initialize resource manager
struct ResourceManager* init_resource_manager(size_t max_memory, int max_threads) {
    typedef struct ResourceManagerImpl {
        size_t max_memory;
        size_t used_memory;
        int max_threads;
        int used_threads;
        bool* thread_allocated;
    } ResourceManagerImpl;

    ResourceManagerImpl* mgr = calloc(1, sizeof(ResourceManagerImpl));
    if (!mgr) return NULL;

    mgr->max_memory = max_memory;
    mgr->used_memory = 0;
    mgr->max_threads = max_threads;
    mgr->used_threads = 0;
    mgr->thread_allocated = calloc(max_threads > 0 ? max_threads : 1, sizeof(bool));

    return (struct ResourceManager*)mgr;
}

// Initialize network manager
struct NetworkManager* init_network_manager(double bandwidth, double latency) {
    typedef struct NetworkManagerImpl {
        double max_bandwidth;
        double used_bandwidth;
        double latency;
        size_t buffer_size;
        size_t buffer_used;
    } NetworkManagerImpl;

    NetworkManagerImpl* mgr = calloc(1, sizeof(NetworkManagerImpl));
    if (!mgr) return NULL;

    mgr->max_bandwidth = bandwidth;
    mgr->used_bandwidth = 0.0;
    mgr->latency = latency;
    mgr->buffer_size = BUFFER_SIZE;
    mgr->buffer_used = 0;

    return (struct NetworkManager*)mgr;
}

// Initialize operation queue
struct OperationQueue* init_operation_queue(size_t max_ops) {
    typedef struct OperationQueueImpl {
        QuantumOperation** operations;
        size_t count;
        size_t capacity;
        size_t head;
        size_t tail;
    } OperationQueueImpl;

    OperationQueueImpl* queue = calloc(1, sizeof(OperationQueueImpl));
    if (!queue) return NULL;

    queue->capacity = max_ops > 0 ? max_ops : MAX_CONCURRENT_OPS;
    queue->operations = calloc(queue->capacity, sizeof(QuantumOperation*));
    queue->count = 0;
    queue->head = 0;
    queue->tail = 0;

    if (!queue->operations) {
        free(queue);
        return NULL;
    }

    return (struct OperationQueue*)queue;
}

// Initialize execution log
struct ExecutionLog* init_execution_log(void) {
    typedef struct ExecutionLogImpl {
        ExecutionSchedule* entries;
        size_t count;
        size_t capacity;
        double total_runtime;
        size_t total_operations;
    } ExecutionLogImpl;

    ExecutionLogImpl* log = calloc(1, sizeof(ExecutionLogImpl));
    if (!log) return NULL;

    log->capacity = 1024;
    log->entries = calloc(log->capacity, sizeof(ExecutionSchedule));
    log->count = 0;
    log->total_runtime = 0.0;
    log->total_operations = 0;

    if (!log->entries) {
        free(log);
        return NULL;
    }

    return (struct ExecutionLog*)log;
}

// Analyze operation requirements
struct OperationRequirements* analyze_requirements(const QuantumOperation* op) {
    if (!op) return NULL;

    OperationRequirements* req = calloc(1, sizeof(OperationRequirements));
    if (!req) return NULL;

    // Analyze based on operation type
    // IBM is our primary gate-based backend
    switch (op->type) {
        case OPERATION_GATE:
            // Gate operations - IBM is primary, Rigetti as alternative
            // Determine qubit count from gate structure
            if (op->op.gate.control != op->op.gate.target) {
                // Two-qubit gate
                req->ibm_qubits = 2;
                req->rigetti_qubits = 2;
            } else {
                // Single-qubit gate
                req->ibm_qubits = 1;
                req->rigetti_qubits = 1;
            }
            req->dwave_vars = 0;
            req->memory_needed = 1024;  // 1KB per gate

            // IBM-specific requirements for gates
            req->ibm_circuit_depth = 1;
            req->ibm_error_budget = 0.01;  // 1% error budget per gate
            req->ibm_optimization_level = 1;  // Basic optimization
            req->ibm_feedback_required = false;
            break;

        case OPERATION_ANNEAL:
            // Annealing needs DWave variables - not IBM
            req->ibm_qubits = 0;
            req->rigetti_qubits = 0;
            req->dwave_vars = op->op.anneal.schedule_points > 0 ? op->op.anneal.schedule_points : 100;
            req->memory_needed = req->dwave_vars * sizeof(double) * 2;

            // No IBM requirements for annealing
            req->ibm_circuit_depth = 0;
            req->ibm_error_budget = 0.0;
            req->ibm_optimization_level = 0;
            req->ibm_feedback_required = false;
            break;

        case OPERATION_BARRIER:
            // Barriers synchronize qubits - IBM supports this
            req->ibm_qubits = op->op.barrier.num_qubits;
            req->rigetti_qubits = op->op.barrier.num_qubits;
            req->dwave_vars = 0;
            req->memory_needed = op->op.barrier.num_qubits * 16;

            req->ibm_circuit_depth = 1;
            req->ibm_error_budget = 0.001;  // Barriers have minimal error
            req->ibm_optimization_level = 0;
            req->ibm_feedback_required = false;
            break;

        case OPERATION_MEASURE:
            // Measurement - IBM has excellent measurement support
            req->ibm_qubits = 1;
            req->rigetti_qubits = 1;
            req->dwave_vars = 0;
            req->memory_needed = 64;

            req->ibm_circuit_depth = 1;
            req->ibm_error_budget = 0.02;  // Readout typically ~1-2% error
            req->ibm_optimization_level = 2;  // Higher optimization for measurement
            req->ibm_feedback_required = false;  // Unless conditional
            break;

        case OPERATION_RESET:
            // Reset operation - IBM supports mid-circuit reset
            req->ibm_qubits = 1;
            req->rigetti_qubits = 1;
            req->dwave_vars = 0;
            req->memory_needed = 64;

            req->ibm_circuit_depth = 1;
            req->ibm_error_budget = 0.005;
            req->ibm_optimization_level = 1;
            req->ibm_feedback_required = true;  // Reset may need feedback
            break;

        case OPERATION_CUSTOM:
        default:
            // Custom operations - assume gate-like behavior on IBM
            req->ibm_qubits = 2;
            req->rigetti_qubits = 2;
            req->dwave_vars = 0;
            req->memory_needed = 1024;

            req->ibm_circuit_depth = 1;
            req->ibm_error_budget = 0.01;
            req->ibm_optimization_level = 2;
            req->ibm_feedback_required = false;
            break;
    }

    // Timing constraints
    req->max_latency = MAX_LATENCY;
    req->target_runtime = 1.0;  // 1 second default

    // No dependencies for single operations
    req->num_dependencies = 0;
    req->dependencies = NULL;

    return (struct OperationRequirements*)req;
}

// Check resource availability
bool check_resources(const struct SystemStatus* status_ptr,
                     const struct OperationRequirements* requirements_ptr) {
    const SystemStatus* status = (const SystemStatus*)status_ptr;
    const OperationRequirements* requirements = (const OperationRequirements*)requirements_ptr;

    if (!status || !requirements) return false;

    // Check IBM availability first (primary backend)
    if (requirements->ibm_qubits > 0) {
        if (!status->ibm_available) {
            return false;
        }
        if (status->ibm_utilization > 0.9) {
            return false;
        }
        // Check qubit count against IBM backend capacity
        if (requirements->ibm_qubits > status->ibm_max_qubits) {
            return false;
        }
    }

    // Check Rigetti availability (alternative gate-based backend)
    if (requirements->rigetti_qubits > 0 && requirements->ibm_qubits == 0) {
        // Only check Rigetti if IBM is not being used
        if (!status->rigetti_available || status->rigetti_utilization > 0.9) {
            return false;
        }
    }

    // Check DWave availability (annealing backend)
    if (requirements->dwave_vars > 0) {
        if (!status->dwave_available || status->dwave_utilization > 0.9) {
            return false;
        }
    }

    // Check memory
    if (requirements->memory_needed > status->available_memory) {
        return false;
    }

    return true;
}

// Get next available start time
double get_next_start_time(HardwareScheduler* scheduler) {
    if (!scheduler || !scheduler->execution_log) {
        return 0.0;
    }

    typedef struct ExecutionLogImpl {
        ExecutionSchedule* entries;
        size_t count;
        size_t capacity;
        double total_runtime;
        size_t total_operations;
    } ExecutionLogImpl;

    ExecutionLogImpl* log = (ExecutionLogImpl*)scheduler->execution_log;

    // Find the latest end time from previous operations
    double latest_end = 0.0;
    for (size_t i = 0; i < log->count; i++) {
        if (log->entries[i].end_time > latest_end) {
            latest_end = log->entries[i].end_time;
        }
    }

    return latest_end;
}

// Calculate data transfer size for operation
double calculate_data_transfer(const QuantumOperation* op) {
    if (!op) return 0.0;

    size_t data_size = 0;

    switch (op->type) {
        case OPERATION_GATE:
            // Gate matrix + qubit indices
            data_size = 64 + 16;  // 8x8 bytes for 2-qubit gate + indices
            break;

        case OPERATION_ANNEAL:
            // Schedule data
            data_size = op->op.anneal.schedule_points * sizeof(double);
            break;

        case OPERATION_BARRIER:
            // Qubit indices
            data_size = op->op.barrier.num_qubits * sizeof(uint32_t);
            break;

        case OPERATION_MEASURE:
        case OPERATION_RESET:
            data_size = sizeof(uint32_t);  // Single qubit index
            break;

        default:
            data_size = 64;
            break;
    }

    return (double)data_size;
}

// Calculate bandwidth needed for transfer
double calculate_bandwidth_needed(size_t data_transfer, double duration) {
    if (duration <= 0.0) return 0.0;

    // Bytes per second
    return (double)data_transfer / duration;
}

// Calculate synchronization time for dependent operation
double calculate_sync_time(const ExecutionSchedule* schedule, const QuantumOperation* op) {
    if (!schedule || !op) return 0.0;

    // Calculate sync time based on operation type
    double base_sync = schedule->start_time;
    double duration = schedule->end_time - schedule->start_time;

    // Add sync overhead based on operation complexity
    switch (op->type) {
        case OPERATION_ANNEAL:
            // Annealing needs more sync time
            return base_sync + duration * 0.5;

        case OPERATION_BARRIER:
            // Barrier is a sync point itself
            return base_sync + duration * 0.1;

        default:
            return base_sync + duration * 0.25;
    }
}

// Calculate end time for schedule
double calculate_end_time(const ExecutionSchedule* schedule) {
    if (!schedule) return 0.0;

    // Base duration from allocated resources
    double base_duration = 0.001;  // 1ms minimum

    // Add time for IBM quantum operations (primary backend)
    if (schedule->ibm_qubits > 0) {
        // IBM gate time: ~100ns per single-qubit gate, ~300ns per two-qubit gate
        // Circuit depth determines total gate time
        double gate_time_ns = 200.0;  // Average gate time
        double circuit_time_ns = gate_time_ns * (double)schedule->ibm_circuit_depth;

        // Add measurement time (~1us per qubit)
        double measure_time_ns = 1000.0 * (double)schedule->ibm_qubits;

        // Add feedback overhead if enabled (~1us)
        double feedback_time_ns = schedule->ibm_feedback_enabled ? 1000.0 : 0.0;

        // Convert to seconds
        base_duration += (circuit_time_ns + measure_time_ns + feedback_time_ns) / 1e9;

        // Add optimization overhead based on level
        base_duration += 0.001 * schedule->ibm_optimization_level;
    }

    // Add time for Rigetti operations
    if (schedule->rigetti_qubits > 0) {
        base_duration += 0.0001 * schedule->rigetti_qubits;  // 100us per qubit
    }

    // Add time for DWave annealing
    if (schedule->dwave_vars > 0) {
        base_duration += 0.00002;  // 20us for annealing (fixed)
    }

    // Add data transfer time
    if (schedule->bandwidth_needed > 0 && schedule->data_transfer > 0) {
        base_duration += schedule->data_transfer / schedule->bandwidth_needed;
    }

    return schedule->start_time + base_duration;
}

// Cleanup hardware scheduler
void cleanup_hardware_scheduler(HardwareScheduler* scheduler) {
    if (!scheduler) return;

    // Cleanup system status
    free(scheduler->status);

    // Cleanup performance monitor
    if (scheduler->monitor) {
        // Assume cleanup function exists
        // cleanup_performance_monitor(scheduler->monitor);
        free(scheduler->monitor);
    }

    // Cleanup resource manager
    if (scheduler->resource_mgr) {
        typedef struct ResourceManagerImpl {
            size_t max_memory;
            size_t used_memory;
            int max_threads;
            int used_threads;
            bool* thread_allocated;
        } ResourceManagerImpl;
        ResourceManagerImpl* mgr = (ResourceManagerImpl*)scheduler->resource_mgr;
        free(mgr->thread_allocated);
        free(mgr);
    }

    // Cleanup network manager
    free(scheduler->network_mgr);

    // Cleanup operation queue
    if (scheduler->operation_queue) {
        typedef struct OperationQueueImpl {
            QuantumOperation** operations;
            size_t count;
            size_t capacity;
            size_t head;
            size_t tail;
        } OperationQueueImpl;
        OperationQueueImpl* queue = (OperationQueueImpl*)scheduler->operation_queue;
        free(queue->operations);
        free(queue);
    }

    // Cleanup execution log
    if (scheduler->execution_log) {
        typedef struct ExecutionLogImpl {
            ExecutionSchedule* entries;
            size_t count;
            size_t capacity;
            double total_runtime;
            size_t total_operations;
        } ExecutionLogImpl;
        ExecutionLogImpl* log = (ExecutionLogImpl*)scheduler->execution_log;
        free(log->entries);
        free(log);
    }

    free(scheduler);
}

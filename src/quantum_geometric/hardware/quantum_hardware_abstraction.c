#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"
#include "quantum_geometric/hardware/quantum_ibm_backend.h"
#include "quantum_geometric/hardware/quantum_rigetti_backend.h"
#include "quantum_geometric/hardware/quantum_rigetti_api.h"
#include "quantum_geometric/hardware/quantum_dwave_backend.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/hardware/quantum_backend_types.h"
#ifdef ENABLE_METAL
#include "quantum_geometric/hardware/metal/quantum_geometric_metal.h"
#include "quantum_geometric/hardware/metal/mnist_metal.h"
#endif

// System headers
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// Optional dependencies
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef HAVE_LZ4
#include <lz4.h>
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

// Global log level
static int g_log_level = 0;

// Hardware version string
#define QUANTUM_HARDWARE_VERSION "1.0.0"

// Minimum fidelity threshold
#define RIGETTI_MIN_FIDELITY 0.95

// Forward declarations - Conversion helpers
static bool operation_to_qasm(const QuantumOperation* operation, char* buffer, size_t size);
static bool operation_to_quil(const QuantumOperation* operation, char* buffer, size_t size);
static bool operation_to_json(const QuantumOperation* operation, char* buffer, size_t size);

// Forward declarations - Validation helpers
static bool validate_gate_operation(const QuantumGate* gate);
static bool validate_annealing_schedule(const double* schedule, size_t points);

// Forward declarations - Optimization helpers (program-based)
static QuantumProgram* internal_optimize_for_hardware(const QuantumProgram* program, const HardwareState* hw);
static void optimize_for_ibm(QuantumProgram* program, int level);
static void optimize_for_rigetti(QuantumProgram* program, int level);
static void optimize_for_dwave(QuantumProgram* program, int level);
static void optimize_gate_cancellation(QuantumProgram* program);
static void optimize_gate_fusion(QuantumProgram* program);
static void decompose_to_native_gates(QuantumProgram* program, HardwareBackendType type);
static void simplify_hamiltonian(QuantumProgram* program);
static void optimize_chain_embedding(QuantumProgram* program);

// Forward declarations - Optimization helpers (circuit-based)
static OptimizedCircuit* optimize_circuit_for_hardware(const struct QuantumCircuit* circuit,
    const struct QuantumHardware* hardware, const ErrorMitigationStrategy* strategy);
static ValidationResult internal_validate_circuit_full(const struct QuantumCircuit* circuit,
    const struct QuantumHardware* hardware);

// Forward declarations - Error mitigation helpers
// richardson_extrapolate and apply_zne_mitigation are declared in quantum_ibm_backend.h
double richardson_extrapolate(double value, const double* scale_factors, size_t num_factors);
bool apply_zne_mitigation(ExecutionResult* result, const struct MitigationParams* params);
static bool apply_probabilistic_mitigation(ExecutionResult* result, const struct MitigationParams* params);
static bool apply_custom_mitigation(ExecutionResult* result, const void* custom_params);

// Forward declarations - Backend configuration helpers (internal)
static struct IBMConfig* internal_init_ibm_config(const struct IBMConfig* config);
static struct RigettiConfig* internal_init_rigetti_config(const struct RigettiConfig* config);
static struct DWaveConfig* internal_init_dwave_config(const struct DWaveConfig* config);
static struct SimulatorConfig* internal_init_simulator_config(const struct SimulatorConfig* config);

// Forward declarations - Backend capabilities helpers
bool init_ibm_capabilities(HardwareCapabilities* caps, const struct IBMConfig* config);
bool init_rigetti_capabilities(HardwareCapabilities* caps, const struct RigettiConfig* config);
bool init_dwave_capabilities(HardwareCapabilities* caps, const struct DWaveConfig* config);
bool init_simulator_capabilities(HardwareCapabilities* caps, const struct SimulatorConfig* config);

// Forward declarations - Backend config cleanup helpers (matches header declarations)
void hal_cleanup_ibm_backend(struct IBMConfig* backend);
void hal_cleanup_rigetti_backend(struct RigettiConfig* backend);
void cleanup_dwave_backend(struct DWaveConfig* backend);
void cleanup_simulator(struct SimulatorConfig* backend);

// Forward declarations - Backend execution helpers
static bool execute_on_ibm(struct IBMConfig* backend, const QuantumProgram* program, ExecutionResult* result);
static bool execute_on_rigetti(struct RigettiConfig* backend, const QuantumProgram* program, ExecutionResult* result);
static bool execute_on_dwave(struct DWaveConfig* backend, const QuantumProgram* program, ExecutionResult* result);
static bool execute_on_simulator(struct SimulatorConfig* backend, const QuantumProgram* program, ExecutionResult* result);

// Forward declarations - Backend status helpers
static char* get_ibm_status(struct IBMConfig* config, const char* job_id);
static char* get_rigetti_status(struct RigettiConfig* config, const char* job_id);
static char* get_dwave_status(struct DWaveConfig* config, const char* job_id);

// Forward declarations - Backend listing helpers
static char** get_available_ibm_backends(size_t* num_backends);
static char** get_available_rigetti_backends(size_t* num_backends);
static char** get_available_dwave_backends(size_t* num_backends);

// Forward declarations - Backend test helpers (internal)
static bool internal_test_ibm_connection(struct IBMConfig* config);
static bool internal_test_rigetti_connection(struct RigettiConfig* config);
static bool internal_test_dwave_connection(struct DWaveConfig* config);

// Forward declarations - Backend options helpers
static bool set_ibm_options(struct IBMConfig* config, const void* options);
static bool set_rigetti_options(struct RigettiConfig* config, const void* options);
static bool set_dwave_options(struct DWaveConfig* config, const void* options);
static bool set_simulator_options(struct SimulatorConfig* config, const void* options);

// Forward declarations - Simulator helpers (internal with hal_ prefix to avoid conflicts)
static void* hal_sim_init_quantum_state(uint32_t num_qubits);
static bool hal_sim_apply_operation(struct SimulatorConfig* backend, void* state, const QuantumOperation* op);
static void hal_sim_get_measurement_results(void* state, ExecutionResult* result);
static void hal_sim_calculate_probabilities(void* state, ExecutionResult* result);
static void hal_sim_store_statevector(void* state, ExecutionResult* result);
static void hal_sim_cleanup_quantum_state(void* state);

// Forward declarations - Job submission helpers (internal)
static bool internal_submit_ibm_job(const struct IBMConfig* config, const char* qasm, ExecutionResult* result);
static bool internal_submit_rigetti_job(const struct RigettiConfig* config, const char* quil, ExecutionResult* result);

// Forward declarations - Circuit optimization helpers (internal)
static void internal_optimize_circuit_depth(QuantumProgram* program);
static void optimize_gate_reordering(QuantumProgram* program);
static void optimize_qubit_mapping(QuantumProgram* program, const HardwareCapabilities* caps);
static void optimize_gate_decomposition(QuantumProgram* program, const HardwareCapabilities* caps);
static ValidationResult internal_validate_circuit(const QuantumCircuit* circuit, const QuantumHardware* hardware);

bool apply_error_mitigation(struct ExecutionResult* result, const struct MitigationParams* params) {
    if (!result || !params) {
        return false;
    }

    switch (params->type) {
        case MITIGATION_RICHARDSON:
            // Apply Richardson extrapolation
            // Iterate through all probabilities
            for (size_t i = 0; result->probabilities && result->probabilities[i] != 0.0; i++) {
                result->probabilities[i] = richardson_extrapolate(
                    result->probabilities[i],
                    params->scale_factors,
                    4  // Number of scale factors
                );
            }
            break;

        case MITIGATION_ZNE:
            // Zero-noise extrapolation
            apply_zne_mitigation(result, params);
            break;

        case MITIGATION_PROBABILISTIC:
            // Probabilistic error cancellation
            apply_probabilistic_mitigation(result, params);
            break;

        case MITIGATION_CUSTOM:
            // Custom error mitigation strategy
            if (params->custom_parameters) {
                return apply_custom_mitigation(result, params->custom_parameters);
            }
            return false;

        case MITIGATION_NONE:
            // No error mitigation needed
            return true;

        default:
            return false;
    }

    // Update error metrics
    result->error_rate *= (1.0 - params->mitigation_factor);
    result->fidelity = 1.0 - result->error_rate;

    return true;
}

// Constants for hardware configuration
#define HARDWARE_MAX_QUBITS 100
#define MAX_GATES 1000
#define MAX_SHOTS 10000
#define MIN_FIDELITY 0.95

// Static helper functions
static bool validate_config(const HardwareConfig* config) {
    if (!config) {
        return false;
    }

    // Validate backend name
    if (!config->backend_name) {
        return false;
    }

    // Validate backend type
    switch (config->type) {
        case HARDWARE_IBM:
            if (!config->config.ibm.api_key || !config->config.ibm.url) {
                return false;
            }
            break;
        case HARDWARE_RIGETTI:
            if (!config->config.rigetti.api_key || !config->config.rigetti.url) {
                return false;
            }
            break;
        case HARDWARE_DWAVE:
            if (!config->config.dwave.api_key || !config->config.dwave.url) {
                return false;
            }
            break;
        case HARDWARE_SIMULATOR:
            // No additional validation needed for simulator
            break;
        default:
            return false;
    }

    return true;
}

static bool init_capabilities(void* hardware, const HardwareConfig* config) {
    HardwareState* hw = (HardwareState*)hardware;
    
    // Initialize basic capabilities
    hw->capabilities.max_qubits = 0;
    hw->capabilities.max_shots = 0;
    hw->capabilities.supports_gates = false;
    hw->capabilities.supports_annealing = false;
    hw->capabilities.supports_measurement = false;
    hw->capabilities.supports_reset = false;
    hw->capabilities.supports_conditional = false;
    hw->capabilities.supports_parallel = false;
    hw->capabilities.supports_error_correction = false;
    hw->capabilities.coherence_time = 0.0;
    hw->capabilities.gate_time = 0.0;
    hw->capabilities.readout_time = 0.0;
    hw->capabilities.connectivity = NULL;
    hw->capabilities.connectivity_size = 0;
    hw->capabilities.available_gates = NULL;
    hw->capabilities.num_gates = 0;
    hw->capabilities.backend_specific = NULL;

    // Set backend-specific capabilities
    switch (config->type) {
        case HARDWARE_IBM:
            return init_ibm_capabilities(&hw->capabilities, &config->config.ibm);
        case HARDWARE_RIGETTI:
            return init_rigetti_capabilities(&hw->capabilities, &config->config.rigetti);
        case HARDWARE_DWAVE:
            return init_dwave_capabilities(&hw->capabilities, &config->config.dwave);
        case HARDWARE_SIMULATOR:
            return init_simulator_capabilities(&hw->capabilities, &config->config.simulator);
        default:
            return false;
    }
}

static bool init_backend_specific(void* hardware, const HardwareConfig* config) {
    HardwareState* hw = (HardwareState*)hardware;
    hw->type = config->type;
    hw->optimization_level = 1;  // Default optimization level
    hw->error_budget = 0.01;    // Default error budget
    hw->use_acceleration = true; // Enable hardware acceleration by default

    // Initialize backend-specific state
    switch (config->type) {
        case HARDWARE_IBM:
            hw->backend.ibm = internal_init_ibm_config(&config->config.ibm);
            return hw->backend.ibm != NULL;
        case HARDWARE_RIGETTI:
            hw->backend.rigetti = internal_init_rigetti_config(&config->config.rigetti);
            return hw->backend.rigetti != NULL;
        case HARDWARE_DWAVE:
            hw->backend.dwave = internal_init_dwave_config(&config->config.dwave);
            return hw->backend.dwave != NULL;
        case HARDWARE_SIMULATOR:
            hw->backend.simulator = internal_init_simulator_config(&config->config.simulator);
            return hw->backend.simulator != NULL;
        default:
            return false;
    }
}

static void cleanup_backend_specific(void* hardware, HardwareBackendType type) {
    HardwareState* hw = (HardwareState*)hardware;
    if (!hw) return;

    switch (type) {
        case HARDWARE_IBM:
            if (hw->backend.ibm) {
                hal_cleanup_ibm_backend(hw->backend.ibm);
            }
            break;
        case HARDWARE_RIGETTI:
            if (hw->backend.rigetti) {
                hal_cleanup_rigetti_backend(hw->backend.rigetti);
            }
            break;
        case HARDWARE_DWAVE:
            if (hw->backend.dwave) {
                cleanup_dwave_backend(hw->backend.dwave);
            }
            break;
        case HARDWARE_SIMULATOR:
            if (hw->backend.simulator) {
                cleanup_simulator(hw->backend.simulator);
            }
            break;
        default:
            break;
    }

    // Cleanup capabilities
    free(hw->capabilities.connectivity);
    if (hw->capabilities.available_gates) {
        for (size_t i = 0; i < hw->capabilities.num_gates; i++) {
            free(hw->capabilities.available_gates[i]);
        }
        free(hw->capabilities.available_gates);
    }
    free(hw->capabilities.backend_specific);
}

// Initialize hardware backend
void* init_hardware(const HardwareConfig* config) {
    if (!validate_config(config)) {
        return NULL;
    }

    void* hardware = malloc(sizeof(HardwareState));
    if (!hardware) {
        return NULL;
    }

    if (!init_capabilities(hardware, config) || 
        !init_backend_specific(hardware, config)) {
        cleanup_hardware(hardware);
        return NULL;
    }

    return hardware;
}

// Create quantum program
QuantumProgram* create_program(uint32_t num_qubits, uint32_t num_classical_bits) {
    if (num_qubits > HARDWARE_MAX_QUBITS || num_classical_bits > HARDWARE_MAX_QUBITS) {
        return NULL;
    }

    QuantumProgram* program = malloc(sizeof(QuantumProgram));
    if (!program) {
        return NULL;
    }

    program->operations = malloc(MAX_GATES * sizeof(QuantumOperation));
    if (!program->operations) {
        free(program);
        return NULL;
    }

    program->num_operations = 0;
    program->capacity = MAX_GATES;
    program->num_qubits = num_qubits;
    program->num_classical_bits = num_classical_bits;
    program->optimize = true;
    program->use_error_mitigation = true;
    program->backend_specific = NULL;

    return program;
}

// Add quantum operation
bool add_operation(QuantumProgram* program, const QuantumOperation* operation) {
    if (!program || !operation || program->num_operations >= program->capacity) {
        return false;
    }

    // Validate operation
    switch (operation->type) {
        case OPERATION_GATE:
            if (!validate_gate_operation(&operation->op.gate)) {
                return false;
            }
            break;
        case OPERATION_MEASURE:
            if (operation->op.measure.qubit >= program->num_qubits ||
                operation->op.measure.classical_bit >= program->num_classical_bits) {
                return false;
            }
            break;
        case OPERATION_RESET:
            if (operation->op.reset.qubit >= program->num_qubits) {
                return false;
            }
            break;
        case OPERATION_BARRIER:
            for (size_t i = 0; i < operation->op.barrier.num_qubits; i++) {
                if (operation->op.barrier.qubits[i] >= program->num_qubits) {
                    return false;
                }
            }
            break;
        case OPERATION_ANNEAL:
            if (!validate_annealing_schedule(operation->op.anneal.schedule,
                                          operation->op.anneal.schedule_points)) {
                return false;
            }
            break;
        case OPERATION_CUSTOM:
            // Custom operations must be validated by backend
            break;
        default:
            return false;
    }

    program->operations[program->num_operations++] = *operation;
    return true;
}

// Execute quantum program
ExecutionResult* execute_program(void* hardware, const QuantumProgram* program) {
    if (!hardware || !program) {
        return NULL;
    }

    HardwareState* hw = (HardwareState*)hardware;
    ExecutionResult* result = malloc(sizeof(ExecutionResult));
    if (!result) {
        return NULL;
    }

    // Validate program against hardware capabilities
    if (!validate_program(program, &hw->capabilities)) {
        free(result);
        return NULL;
    }

    // Optimize program if enabled
    QuantumProgram* optimized_program = (QuantumProgram*)program;
    if (program->optimize) {
        optimized_program = internal_optimize_for_hardware(program, hw);
        if (!optimized_program) {
            free(result);
            return NULL;
        }
    }

    // Execute on appropriate backend
    bool success = false;
    switch (hw->type) {
        case HARDWARE_IBM:
            success = execute_on_ibm(hw->backend.ibm, optimized_program, result);
            break;
        case HARDWARE_RIGETTI:
            success = execute_on_rigetti(hw->backend.rigetti, optimized_program, result);
            break;
        case HARDWARE_DWAVE:
            success = execute_on_dwave(hw->backend.dwave, optimized_program, result);
            break;
        case HARDWARE_SIMULATOR:
            success = execute_on_simulator(hw->backend.simulator, optimized_program, result);
            break;
        default:
            success = false;
    }

    if (optimized_program != program) {
        cleanup_program(optimized_program);
    }

    if (!success) {
        free(result);
        return NULL;
    }

    // Apply error mitigation if enabled
    // Note: HAL simulator uses simple state vectors, not SimulatorState structs,
    // so we apply result-based mitigation for all backends including simulator
    if (program->use_error_mitigation) {
        apply_error_mitigation(result, &hw->error_mitigation);
    }

    return result;
}

// Get execution status
char* get_execution_status(void* hardware, const char* execution_id) {
    if (!hardware || !execution_id) {
        return NULL;
    }

    HardwareState* hw = (HardwareState*)hardware;
    char* status = NULL;

    switch (hw->type) {
        case HARDWARE_IBM:
            status = get_ibm_status(hw->backend.ibm, execution_id);
            break;
        case HARDWARE_RIGETTI:
            status = get_rigetti_status(hw->backend.rigetti, execution_id);
            break;
        case HARDWARE_DWAVE:
            status = get_dwave_status(hw->backend.dwave, execution_id);
            break;
        case HARDWARE_SIMULATOR:
            status = strdup("COMPLETED"); // Simulator executes synchronously
            break;
        default:
            return NULL;
    }

    return status;
}

// Cancel execution
bool cancel_execution(void* hardware, const char* execution_id) {
    if (!hardware || !execution_id) {
        return false;
    }

    HardwareState* hw = (HardwareState*)hardware;
    bool success = false;

    switch (hw->type) {
        case HARDWARE_IBM:
            success = cancel_ibm_job(hw->backend.ibm, execution_id);
            break;
        case HARDWARE_RIGETTI:
            success = cancel_rigetti_job(hw->backend.rigetti, execution_id);
            break;
        case HARDWARE_DWAVE:
            success = cancel_dwave_job(hw->backend.dwave, execution_id);
            break;
        case HARDWARE_SIMULATOR:
            success = true; // Nothing to cancel for simulator
            break;
        default:
            return false;
    }

    return success;
}

// Get hardware capabilities
HardwareCapabilities* get_hardware_capabilities(void* hardware) {
    if (!hardware) {
        return NULL;
    }

    HardwareState* hw = (HardwareState*)hardware;
    HardwareCapabilities* caps = malloc(sizeof(HardwareCapabilities));
    if (!caps) {
        return NULL;
    }

    *caps = hw->capabilities;
    return caps;
}

// Get available backends
char** get_available_backends(HardwareBackendType type, size_t* num_backends) {
    if (!num_backends) {
        return NULL;
    }

    char** backends = NULL;
    *num_backends = 0;

    switch (type) {
        case HARDWARE_IBM:
            backends = get_available_ibm_backends(num_backends);
            break;
        case HARDWARE_RIGETTI:
            backends = get_available_rigetti_backends(num_backends);
            break;
        case HARDWARE_DWAVE:
            backends = get_available_dwave_backends(num_backends);
            break;
        case HARDWARE_SIMULATOR:
            backends = malloc(sizeof(char*));
            if (backends) {
                backends[0] = strdup("default");
                *num_backends = 1;
            }
            break;
        default:
            return NULL;
    }

    return backends;
}

// Convert program to QASM format
char* program_to_qasm(const QuantumProgram* program) {
    if (!program) {
        return NULL;
    }

    // Initialize QASM string with header
    size_t capacity = 1024;
    char* qasm = malloc(capacity);
    if (!qasm) {
        return NULL;
    }

    int written = snprintf(qasm, capacity,
        "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n\n"
        "qreg q[%u];\ncreg c[%u];\n\n",
        program->num_qubits,
        program->num_classical_bits);

    if (written < 0 || (size_t)written >= capacity) {
        free(qasm);
        return NULL;
    }

    // Convert each operation to QASM
    for (size_t i = 0; i < program->num_operations; i++) {
        char op_str[256];
        if (!operation_to_qasm(&program->operations[i], op_str, sizeof(op_str))) {
            free(qasm);
            return NULL;
        }

        size_t needed = strlen(qasm) + strlen(op_str) + 2;
        if (needed > capacity) {
            char* new_qasm = realloc(qasm, needed * 2);
            if (!new_qasm) {
                free(qasm);
                return NULL;
            }
            qasm = new_qasm;
            capacity = needed * 2;
        }

        strcat(qasm, op_str);
        strcat(qasm, "\n");
    }

    return qasm;
}

// Convert program to Quil format
char* program_to_quil(const QuantumProgram* program) {
    if (!program) {
        return NULL;
    }

    // Initialize Quil string
    size_t capacity = 1024;
    char* quil = malloc(capacity);
    if (!quil) {
        return NULL;
    }

    int written = snprintf(quil, capacity,
        "DECLARE ro BIT[%u]\n\n",
        program->num_classical_bits);

    if (written < 0 || (size_t)written >= capacity) {
        free(quil);
        return NULL;
    }

    // Convert each operation to Quil
    for (size_t i = 0; i < program->num_operations; i++) {
        char op_str[256];
        if (!operation_to_quil(&program->operations[i], op_str, sizeof(op_str))) {
            free(quil);
            return NULL;
        }

        size_t needed = strlen(quil) + strlen(op_str) + 2;
        if (needed > capacity) {
            char* new_quil = realloc(quil, needed * 2);
            if (!new_quil) {
                free(quil);
                return NULL;
            }
            quil = new_quil;
            capacity = needed * 2;
        }

        strcat(quil, op_str);
        strcat(quil, "\n");
    }

    return quil;
}

// Convert program to JSON format
char* program_to_json(const QuantumProgram* program) {
    if (!program) {
        return NULL;
    }

    // Initialize JSON string
    size_t capacity = 1024;
    char* json = malloc(capacity);
    if (!json) {
        return NULL;
    }

    int written = snprintf(json, capacity,
        "{\n  \"num_qubits\": %u,\n  \"num_classical_bits\": %u,\n"
        "  \"operations\": [\n",
        program->num_qubits,
        program->num_classical_bits);

    if (written < 0 || (size_t)written >= capacity) {
        free(json);
        return NULL;
    }

    // Convert each operation to JSON
    for (size_t i = 0; i < program->num_operations; i++) {
        char op_str[256];
        if (!operation_to_json(&program->operations[i], op_str, sizeof(op_str))) {
            free(json);
            return NULL;
        }

        size_t needed = strlen(json) + strlen(op_str) + 4;
        if (needed > capacity) {
            char* new_json = realloc(json, needed * 2);
            if (!new_json) {
                free(json);
                return NULL;
            }
            json = new_json;
            capacity = needed * 2;
        }

        strcat(json, "    ");
        strcat(json, op_str);
        if (i < program->num_operations - 1) {
            strcat(json, ",");
        }
        strcat(json, "\n");
    }

    strcat(json, "  ]\n}\n");
    return json;
}

// Clean up resources
void cleanup_hardware(void* hardware) {
    if (!hardware) {
        return;
    }

    HardwareState* hw = (HardwareState*)hardware;
    cleanup_backend_specific(hardware, hw->type);
    free(hw);
}

void cleanup_program(QuantumProgram* program) {
    if (!program) {
        return;
    }

    free(program->operations);
    free(program->backend_specific);
    free(program);
}

void cleanup_result(ExecutionResult* result) {
    if (!result) {
        return;
    }

    free(result->measurements);
    free(result->probabilities);
    free(result->counts);
    free(result->error_message);
    free(result->backend_data);
    free(result);
}

void cleanup_capabilities(HardwareCapabilities* capabilities) {
    if (!capabilities) {
        return;
    }

    free(capabilities->connectivity);
    for (size_t i = 0; i < capabilities->num_gates; i++) {
        free(capabilities->available_gates[i]);
    }
    free(capabilities->available_gates);
    free(capabilities->backend_specific);
    free(capabilities);
}

// Utility functions
bool save_credentials(HardwareBackendType type, const char* credentials, const char* filename) {
    if (!credentials || !filename) {
        return false;
    }

    FILE* file = fopen(filename, "w");
    if (!file) {
        return false;
    }

    fprintf(file, "type=%d\n", type);
    fprintf(file, "credentials=%s\n", credentials);
    fclose(file);
    return true;
}

char* load_credentials(HardwareBackendType type, const char* filename) {
    if (!filename) {
        return NULL;
    }

    FILE* file = fopen(filename, "r");
    if (!file) {
        return NULL;
    }

    char* credentials = NULL;
    char line[1024];
    int stored_type;

    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "type=%d", &stored_type) == 1) {
            if ((HardwareBackendType)stored_type != type) {
                fclose(file);
                return NULL;
            }
        } else if (strncmp(line, "credentials=", 12) == 0) {
            size_t len = strlen(line + 12);
            credentials = malloc(len);
            if (credentials) {
                strcpy(credentials, line + 12);
                if (credentials[len - 1] == '\n') {
                    credentials[len - 1] = '\0';
                }
            }
            break;
        }
    }

    fclose(file);
    return credentials;
}

bool test_connection(void* hardware) {
    if (!hardware) {
        return false;
    }

    HardwareState* hw = (HardwareState*)hardware;
    bool success = false;

    switch (hw->type) {
        case HARDWARE_IBM:
            success = internal_test_ibm_connection(hw->backend.ibm);
            break;
        case HARDWARE_RIGETTI:
            success = internal_test_rigetti_connection(hw->backend.rigetti);
            break;
        case HARDWARE_DWAVE:
            success = internal_test_dwave_connection(hw->backend.dwave);
            break;
        case HARDWARE_SIMULATOR:
            success = true;
            break;
        default:
            return false;
    }

    return success;
}

void set_log_level(int level) {
    // Set global log level
    g_log_level = level;
}

char* get_version(void) {
    return strdup(QUANTUM_HARDWARE_VERSION);
}

// Advanced control
bool set_execution_options(void* hardware, const void* options) {
    if (!hardware || !options) {
        return false;
    }

    HardwareState* hw = (HardwareState*)hardware;
    bool success = false;

    switch (hw->type) {
        case HARDWARE_IBM:
            success = set_ibm_options(hw->backend.ibm, options);
            break;
        case HARDWARE_RIGETTI:
            success = set_rigetti_options(hw->backend.rigetti, options);
            break;
        case HARDWARE_DWAVE:
            success = set_dwave_options(hw->backend.dwave, options);
            break;
        case HARDWARE_SIMULATOR:
            success = set_simulator_options(hw->backend.simulator, options);
            break;
        default:
            return false;
    }

    return success;
}

bool set_optimization_level(void* hardware, int level) {
    if (!hardware || level < 0 || level > 3) {
        return false;
    }

    HardwareState* hw = (HardwareState*)hardware;
    hw->optimization_level = level;
    return true;
}

bool set_error_budget(void* hardware, double budget) {
    if (!hardware || budget < 0.0 || budget > 1.0) {
        return false;
    }

    HardwareState* hw = (HardwareState*)hardware;
    hw->error_budget = budget;
    return true;
}

bool enable_hardware_acceleration(void* hardware, bool enable) {
    if (!hardware) {
        return false;
    }

    HardwareState* hw = (HardwareState*)hardware;
    hw->use_acceleration = enable;
    return true;
}

// Operation conversion implementations
static bool operation_to_qasm(const QuantumOperation* operation, char* buffer, size_t size) {
    if (!operation || !buffer || size == 0) {
        return false;
    }

    switch (operation->type) {
        case OPERATION_GATE:
            switch (operation->op.gate.type) {
                // Identity gate
                case GATE_I:
                    snprintf(buffer, size, "id q[%u];", operation->op.gate.qubit);
                    break;

                // Single-qubit Pauli gates
                case GATE_H:
                    snprintf(buffer, size, "h q[%u];", operation->op.gate.qubit);
                    break;
                case GATE_X:
                    snprintf(buffer, size, "x q[%u];", operation->op.gate.qubit);
                    break;
                case GATE_Y:
                    snprintf(buffer, size, "y q[%u];", operation->op.gate.qubit);
                    break;
                case GATE_Z:
                    snprintf(buffer, size, "z q[%u];", operation->op.gate.qubit);
                    break;

                // Phase gates
                case GATE_S:
                    snprintf(buffer, size, "s q[%u];", operation->op.gate.qubit);
                    break;
                case GATE_SDG:
                    snprintf(buffer, size, "sdg q[%u];", operation->op.gate.qubit);
                    break;
                case GATE_T:
                    snprintf(buffer, size, "t q[%u];", operation->op.gate.qubit);
                    break;
                case GATE_TDG:
                    snprintf(buffer, size, "tdg q[%u];", operation->op.gate.qubit);
                    break;

                // Square root of X
                case GATE_SX:
                    snprintf(buffer, size, "sx q[%u];", operation->op.gate.qubit);
                    break;

                // Rotation gates
                case GATE_RX:
                    snprintf(buffer, size, "rx(%g) q[%u];",
                            operation->op.gate.parameter, operation->op.gate.qubit);
                    break;
                case GATE_RY:
                    snprintf(buffer, size, "ry(%g) q[%u];",
                            operation->op.gate.parameter, operation->op.gate.qubit);
                    break;
                case GATE_RZ:
                    snprintf(buffer, size, "rz(%g) q[%u];",
                            operation->op.gate.parameter, operation->op.gate.qubit);
                    break;

                // U gates (IBM native)
                case GATE_U1:
                    snprintf(buffer, size, "u1(%g) q[%u];",
                            operation->op.gate.parameter, operation->op.gate.qubit);
                    break;
                case GATE_U2:
                    if (operation->op.gate.parameters && operation->op.gate.num_parameters >= 2) {
                        snprintf(buffer, size, "u2(%g,%g) q[%u];",
                                operation->op.gate.parameters[0],
                                operation->op.gate.parameters[1],
                                operation->op.gate.qubit);
                    } else {
                        snprintf(buffer, size, "u2(0,0) q[%u];", operation->op.gate.qubit);
                    }
                    break;
                case GATE_U3:
                    if (operation->op.gate.parameters && operation->op.gate.num_parameters >= 3) {
                        snprintf(buffer, size, "u3(%g,%g,%g) q[%u];",
                                operation->op.gate.parameters[0],
                                operation->op.gate.parameters[1],
                                operation->op.gate.parameters[2],
                                operation->op.gate.qubit);
                    } else {
                        snprintf(buffer, size, "u3(0,0,0) q[%u];", operation->op.gate.qubit);
                    }
                    break;

                // Two-qubit gates
                case GATE_CNOT:
                    snprintf(buffer, size, "cx q[%u],q[%u];",
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;
                case GATE_CZ:
                    snprintf(buffer, size, "cz q[%u],q[%u];",
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;
                case GATE_SWAP:
                    snprintf(buffer, size, "swap q[%u],q[%u];",
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;
                case GATE_ISWAP:
                    // iSWAP is not native QASM, decompose to standard gates
                    snprintf(buffer, size, "// iswap decomposition\n"
                            "s q[%u]; s q[%u]; h q[%u]; cx q[%u],q[%u]; cx q[%u],q[%u]; h q[%u];",
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit,
                            operation->op.gate.control_qubit,
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit,
                            operation->op.gate.target_qubit,
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;

                // Controlled rotation gates
                case GATE_CRX:
                    snprintf(buffer, size, "crx(%g) q[%u],q[%u];",
                            operation->op.gate.parameter,
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;
                case GATE_CRY:
                    snprintf(buffer, size, "cry(%g) q[%u],q[%u];",
                            operation->op.gate.parameter,
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;
                case GATE_CRZ:
                    snprintf(buffer, size, "crz(%g) q[%u],q[%u];",
                            operation->op.gate.parameter,
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;
                case GATE_CH:
                    snprintf(buffer, size, "ch q[%u],q[%u];",
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;

                // Three-qubit gates (use qubit, control_qubit, target_qubit)
                // Note: GATE_TOFFOLI is an alias for GATE_CCX
                case GATE_CCX:
                    snprintf(buffer, size, "ccx q[%u],q[%u],q[%u];",
                            operation->op.gate.qubit,
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;
                case GATE_CSWAP:
                    snprintf(buffer, size, "cswap q[%u],q[%u],q[%u];",
                            operation->op.gate.qubit,
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;

                // IBM native ECR gate
                case GATE_ECR:
                    snprintf(buffer, size, "ecr q[%u],q[%u];",
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;

                // Phase gate with parameter
                case GATE_PHASE:
                    snprintf(buffer, size, "p(%g) q[%u];",
                            operation->op.gate.parameter, operation->op.gate.qubit);
                    break;

                default:
                    return false;
            }
            break;
        case OPERATION_MEASURE:
            snprintf(buffer, size, "measure q[%u] -> c[%u];",
                    operation->op.measure.qubit,
                    operation->op.measure.classical_bit);
            break;
        case OPERATION_RESET:
            snprintf(buffer, size, "reset q[%u];",
                    operation->op.reset.qubit);
            break;
        case OPERATION_BARRIER:
            if (operation->op.barrier.num_qubits == 0) {
                snprintf(buffer, size, "barrier;");
            } else {
                int written = snprintf(buffer, size, "barrier ");
                size_t remaining = size - written;
                char* pos = buffer + written;
                
                for (size_t i = 0; i < operation->op.barrier.num_qubits; i++) {
                    int w = snprintf(pos, remaining, "q[%u]%s",
                                   operation->op.barrier.qubits[i],
                                   i < operation->op.barrier.num_qubits - 1 ? "," : ";");
                    if (w < 0 || (size_t)w >= remaining) {
                        return false;
                    }
                    pos += w;
                    remaining -= w;
                }
            }
            break;
        default:
            return false;
    }

    return true;
}

static bool operation_to_quil(const QuantumOperation* operation, char* buffer, size_t size) {
    if (!operation || !buffer || size == 0) {
        return false;
    }

    switch (operation->type) {
        case OPERATION_GATE:
            switch (operation->op.gate.type) {
                // Identity gate
                case GATE_I:
                    snprintf(buffer, size, "I %u", operation->op.gate.qubit);
                    break;

                // Single-qubit Pauli gates
                case GATE_H:
                    snprintf(buffer, size, "H %u", operation->op.gate.qubit);
                    break;
                case GATE_X:
                    snprintf(buffer, size, "X %u", operation->op.gate.qubit);
                    break;
                case GATE_Y:
                    snprintf(buffer, size, "Y %u", operation->op.gate.qubit);
                    break;
                case GATE_Z:
                    snprintf(buffer, size, "Z %u", operation->op.gate.qubit);
                    break;

                // Phase gates (using RZ decomposition for S and T)
                case GATE_S:
                    snprintf(buffer, size, "RZ(pi/2) %u", operation->op.gate.qubit);
                    break;
                case GATE_SDG:
                    snprintf(buffer, size, "RZ(-pi/2) %u", operation->op.gate.qubit);
                    break;
                case GATE_T:
                    snprintf(buffer, size, "RZ(pi/4) %u", operation->op.gate.qubit);
                    break;
                case GATE_TDG:
                    snprintf(buffer, size, "RZ(-pi/4) %u", operation->op.gate.qubit);
                    break;

                // Square root of X
                case GATE_SX:
                    snprintf(buffer, size, "RX(pi/2) %u", operation->op.gate.qubit);
                    break;

                // Rotation gates
                case GATE_RX:
                    snprintf(buffer, size, "RX(%g) %u",
                            operation->op.gate.parameter, operation->op.gate.qubit);
                    break;
                case GATE_RY:
                    snprintf(buffer, size, "RY(%g) %u",
                            operation->op.gate.parameter, operation->op.gate.qubit);
                    break;
                case GATE_RZ:
                    snprintf(buffer, size, "RZ(%g) %u",
                            operation->op.gate.parameter, operation->op.gate.qubit);
                    break;

                // U gates (decomposed to native Rigetti gates)
                case GATE_U1:
                    // U1(lambda) = RZ(lambda)
                    snprintf(buffer, size, "RZ(%g) %u",
                            operation->op.gate.parameter, operation->op.gate.qubit);
                    break;
                case GATE_U2:
                    // U2(phi, lambda) = RZ(phi) RY(pi/2) RZ(lambda)
                    if (operation->op.gate.parameters && operation->op.gate.num_parameters >= 2) {
                        snprintf(buffer, size, "RZ(%g) %u\nRY(1.5707963267948966) %u\nRZ(%g) %u",
                                operation->op.gate.parameters[1], operation->op.gate.qubit,
                                operation->op.gate.qubit,
                                operation->op.gate.parameters[0], operation->op.gate.qubit);
                    } else {
                        snprintf(buffer, size, "RY(1.5707963267948966) %u", operation->op.gate.qubit);
                    }
                    break;
                case GATE_U3:
                    // U3(theta, phi, lambda) = RZ(phi) RY(theta) RZ(lambda)
                    if (operation->op.gate.parameters && operation->op.gate.num_parameters >= 3) {
                        snprintf(buffer, size, "RZ(%g) %u\nRY(%g) %u\nRZ(%g) %u",
                                operation->op.gate.parameters[2], operation->op.gate.qubit,
                                operation->op.gate.parameters[0], operation->op.gate.qubit,
                                operation->op.gate.parameters[1], operation->op.gate.qubit);
                    } else {
                        snprintf(buffer, size, "I %u", operation->op.gate.qubit);
                    }
                    break;

                // Two-qubit gates
                case GATE_CNOT:
                    snprintf(buffer, size, "CNOT %u %u",
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;
                case GATE_CZ:
                    snprintf(buffer, size, "CZ %u %u",
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;
                case GATE_SWAP:
                    snprintf(buffer, size, "SWAP %u %u",
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;
                case GATE_ISWAP:
                    snprintf(buffer, size, "ISWAP %u %u",
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;

                // Controlled rotation gates (decomposed)
                case GATE_CRX:
                    // Controlled RX decomposition
                    snprintf(buffer, size, "RZ(pi/2) %u\nCNOT %u %u\nRY(%g) %u\nCNOT %u %u\nRY(%g) %u\nRZ(-pi/2) %u",
                            operation->op.gate.target_qubit,
                            operation->op.gate.control_qubit, operation->op.gate.target_qubit,
                            -operation->op.gate.parameter / 2.0, operation->op.gate.target_qubit,
                            operation->op.gate.control_qubit, operation->op.gate.target_qubit,
                            operation->op.gate.parameter / 2.0, operation->op.gate.target_qubit,
                            operation->op.gate.target_qubit);
                    break;
                case GATE_CRY:
                    // Controlled RY decomposition
                    snprintf(buffer, size, "RY(%g) %u\nCNOT %u %u\nRY(%g) %u\nCNOT %u %u",
                            operation->op.gate.parameter / 2.0, operation->op.gate.target_qubit,
                            operation->op.gate.control_qubit, operation->op.gate.target_qubit,
                            -operation->op.gate.parameter / 2.0, operation->op.gate.target_qubit,
                            operation->op.gate.control_qubit, operation->op.gate.target_qubit);
                    break;
                case GATE_CRZ:
                    // Controlled RZ decomposition
                    snprintf(buffer, size, "RZ(%g) %u\nCNOT %u %u\nRZ(%g) %u\nCNOT %u %u",
                            operation->op.gate.parameter / 2.0, operation->op.gate.target_qubit,
                            operation->op.gate.control_qubit, operation->op.gate.target_qubit,
                            -operation->op.gate.parameter / 2.0, operation->op.gate.target_qubit,
                            operation->op.gate.control_qubit, operation->op.gate.target_qubit);
                    break;
                case GATE_CH:
                    // Controlled H decomposition: S-CNOT-Sdg on target, then CNOT, then S-CNOT-Sdg
                    snprintf(buffer, size, "RY(pi/4) %u\nCNOT %u %u\nRY(-pi/4) %u",
                            operation->op.gate.target_qubit,
                            operation->op.gate.control_qubit, operation->op.gate.target_qubit,
                            operation->op.gate.target_qubit);
                    break;

                // Three-qubit gates
                // Note: GATE_TOFFOLI is an alias for GATE_CCX
                case GATE_CCX:
                    snprintf(buffer, size, "CCNOT %u %u %u",
                            operation->op.gate.qubit,
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;
                case GATE_CSWAP:
                    snprintf(buffer, size, "CSWAP %u %u %u",
                            operation->op.gate.qubit,
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;

                // IBM native ECR gate (decompose for Rigetti)
                case GATE_ECR:
                    // ECR decomposition to native gates
                    snprintf(buffer, size, "RZ(pi/4) %u\nCNOT %u %u\nX %u",
                            operation->op.gate.control_qubit,
                            operation->op.gate.control_qubit, operation->op.gate.target_qubit,
                            operation->op.gate.control_qubit);
                    break;

                // Phase gate with parameter
                case GATE_PHASE:
                    snprintf(buffer, size, "RZ(%g) %u",
                            operation->op.gate.parameter, operation->op.gate.qubit);
                    break;

                default:
                    return false;
            }
            break;
        case OPERATION_MEASURE:
            snprintf(buffer, size, "MEASURE %u ro[%u]",
                    operation->op.measure.qubit,
                    operation->op.measure.classical_bit);
            break;
        case OPERATION_RESET:
            snprintf(buffer, size, "RESET %u",
                    operation->op.reset.qubit);
            break;
        case OPERATION_BARRIER:
            if (operation->op.barrier.num_qubits == 0) {
                snprintf(buffer, size, "PRAGMA BARRIER");
            } else {
                int written = snprintf(buffer, size, "PRAGMA BARRIER ");
                size_t remaining = size - written;
                char* pos = buffer + written;
                
                 for (size_t i = 0; i < operation->op.barrier.num_qubits; i++) {
                    int w = snprintf(pos, remaining, "%u%s",
                                   operation->op.barrier.qubits[i],
                                   i < operation->op.barrier.num_qubits - 1 ? " " : "");
                    if (w < 0 || (size_t)w >= remaining) {
                        return false;
                    }
                    pos += w;
                    remaining -= w;
                }
            }
            break;
        default:
            return false;
    }

    return true;
}

static bool operation_to_json(const QuantumOperation* operation, char* buffer, size_t size) {
    if (!operation || !buffer || size == 0) {
        return false;
    }

    switch (operation->type) {
        case OPERATION_GATE:
            switch (operation->op.gate.type) {
                case GATE_H:
                    snprintf(buffer, size,
                            "{\"type\": \"gate\", \"name\": \"h\", \"qubit\": %u}",
                            operation->op.gate.qubit);
                    break;
                case GATE_X:
                    snprintf(buffer, size,
                            "{\"type\": \"gate\", \"name\": \"x\", \"qubit\": %u}",
                            operation->op.gate.qubit);
                    break;
                case GATE_CNOT:
                    snprintf(buffer, size,
                            "{\"type\": \"gate\", \"name\": \"cx\", "
                            "\"control\": %u, \"target\": %u}",
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
                    break;
                default:
                    return false;
            }
            break;
        case OPERATION_MEASURE:
            snprintf(buffer, size,
                    "{\"type\": \"measure\", \"qubit\": %u, \"bit\": %u}",
                    operation->op.measure.qubit,
                    operation->op.measure.classical_bit);
            break;
        case OPERATION_RESET:
            snprintf(buffer, size,
                    "{\"type\": \"reset\", \"qubit\": %u}",
                    operation->op.reset.qubit);
            break;
        case OPERATION_BARRIER:
            if (operation->op.barrier.num_qubits == 0) {
                snprintf(buffer, size, "{\"type\": \"barrier\"}");
            } else {
                int written = snprintf(buffer, size,
                                     "{\"type\": \"barrier\", \"qubits\": [");
                size_t remaining = size - written;
                char* pos = buffer + written;
                
                for (size_t i = 0; i < operation->op.barrier.num_qubits; i++) {
                    int w = snprintf(pos, remaining, "%u%s",
                                   operation->op.barrier.qubits[i],
                                   i < operation->op.barrier.num_qubits - 1 ? "," : "]}");
                    if (w < 0 || (size_t)w >= remaining) {
                        return false;
                    }
                    pos += w;
                    remaining -= w;
                }
            }
            break;
        default:
            return false;
    }

    return true;
}

static bool validate_gate_operation(const QuantumGate* gate) {
    if (!gate) {
        return false;
    }

    switch (gate->type) {
        case GATE_H:
        case GATE_X:
            return true;
        case GATE_CNOT:
            return gate->control_qubit != gate->target_qubit;
        default:
            return false;
    }
}

static bool validate_annealing_schedule(const double* schedule, size_t points) {
    if (!schedule || points == 0) {
        return false;
    }

    // Validate schedule points are monotonically increasing in time
    // and s(t) values are between 0 and 1
    for (size_t i = 0; i < points; i++) {
        double time = schedule[2*i];
        double s = schedule[2*i + 1];
        
        if (s < 0.0 || s > 1.0) {
            return false;
        }
        
        if (i > 0 && time <= schedule[2*(i-1)]) {
            return false;
        }
    }

    return true;
}

// Helper function to get gate name
static const char* get_gate_name(gate_type_t type) {
    switch (type) {
        case GATE_I: return "I";
        case GATE_X: return "X";
        case GATE_Y: return "Y";
        case GATE_Z: return "Z";
        case GATE_H: return "H";
        case GATE_S: return "S";
        case GATE_T: return "T";
        case GATE_RX: return "RX";
        case GATE_RY: return "RY";
        case GATE_RZ: return "RZ";
        case GATE_CNOT: return "CNOT";
        case GATE_CZ: return "CZ";
        case GATE_SWAP: return "SWAP";
        case GATE_TOFFOLI: return "CCX";
        case GATE_PHASE: return "P";
        default: return "UNKNOWN";
    }
}

// Program validation and optimization implementations
bool validate_program(const QuantumProgram* program, const HardwareCapabilities* capabilities) {
    if (!program || !capabilities) {
        return false;
    }

    // Check number of qubits
    if (program->num_qubits > capabilities->max_qubits) {
        return false;
    }

    // Check operations against hardware capabilities
    for (size_t i = 0; i < program->num_operations; i++) {
        const QuantumOperation* op = &program->operations[i];
        
        switch (op->type) {
            case OPERATION_GATE:
                if (!capabilities->supports_gates) {
                    return false;
                }
                // Check if gate type is supported
                bool gate_supported = false;
                for (size_t j = 0; j < capabilities->num_gates; j++) {
                    if (strcmp(capabilities->available_gates[j], 
                             get_gate_name(op->op.gate.type)) == 0) {
                        gate_supported = true;
                        break;
                    }
                }
                if (!gate_supported) {
                    return false;
                }
                break;
                
            case OPERATION_MEASURE:
                if (!capabilities->supports_measurement) {
                    return false;
                }
                break;
                
            case OPERATION_RESET:
                if (!capabilities->supports_reset) {
                    return false;
                }
                break;
                
            case OPERATION_BARRIER:
                // Barriers are always supported
                break;
                
            case OPERATION_ANNEAL:
                if (!capabilities->supports_annealing) {
                    return false;
                }
                break;
                
            default:
                return false;
        }
    }

    return true;
}

static QuantumProgram* internal_optimize_for_hardware(const QuantumProgram* program, const HardwareState* hw) {
    if (!program || !hw) {
        return NULL;
    }

    // Create copy of program for optimization
    QuantumProgram* optimized = malloc(sizeof(QuantumProgram));
    if (!optimized) {
        return NULL;
    }
    
    optimized->operations = malloc(program->capacity * sizeof(QuantumOperation));
    if (!optimized->operations) {
        free(optimized);
        return NULL;
    }
    
    // Copy program metadata
    optimized->num_qubits = program->num_qubits;
    optimized->num_classical_bits = program->num_classical_bits;
    optimized->capacity = program->capacity;
    optimized->optimize = program->optimize;
    optimized->use_error_mitigation = program->use_error_mitigation;
    optimized->backend_specific = NULL;
    
    // Copy operations for modification
    memcpy(optimized->operations, program->operations, 
           program->num_operations * sizeof(QuantumOperation));
    optimized->num_operations = program->num_operations;
    
    // Apply optimizations based on hardware type and optimization level
    switch (hw->type) {
        case HARDWARE_IBM:
            optimize_for_ibm(optimized, hw->optimization_level);
            break;
        case HARDWARE_RIGETTI:
            optimize_for_rigetti(optimized, hw->optimization_level);
            break;
        case HARDWARE_DWAVE:
            optimize_for_dwave(optimized, hw->optimization_level);
            break;
        case HARDWARE_SIMULATOR:
            // Basic optimizations for simulator
            optimize_gate_cancellation(optimized);
            optimize_gate_fusion(optimized);
            break;
        default:
            break;
    }
    
    // Apply common optimizations if enabled
    if (hw->optimization_level > 0) {
        optimize_gate_reordering(optimized);
        optimize_qubit_mapping(optimized, &hw->capabilities);
        
        if (hw->optimization_level > 1) {
            internal_optimize_circuit_depth(optimized);
            optimize_gate_decomposition(optimized, &hw->capabilities);
        }
    }
    
    return optimized;
}

// Backend execution implementations
static bool execute_on_ibm(struct IBMConfig* backend, const QuantumProgram* program, ExecutionResult* result) {
    if (!backend || !program || !result) {
        return false;
    }

    // Convert program to QASM for IBM backend
    char* qasm = program_to_qasm(program);
    if (!qasm) {
        return false;
    }

    // Submit job to IBM backend
    bool success = internal_submit_ibm_job(backend, qasm, result);
    free(qasm);
    return success;
}

static bool execute_on_rigetti(struct RigettiConfig* backend, const QuantumProgram* program, ExecutionResult* result) {
    if (!backend || !program || !result) {
        return false;
    }

    // Convert program to Quil for Rigetti backend
    char* quil = program_to_quil(program);
    if (!quil) {
        return false;
    }

    // Submit job to Rigetti backend
    bool success = internal_submit_rigetti_job(backend, quil, result);
    free(quil);
    return success;
}

static bool execute_on_dwave(struct DWaveConfig* backend, const QuantumProgram* program, ExecutionResult* result) {
    if (!backend || !program || !result) {
        return false;
    }

    // Convert program to QUBO format for D-Wave
    QUBO* qubo = program_to_qubo(program);
    if (!qubo) {
        return false;
    }

    // Submit problem to D-Wave solver
    QUBOResult qubo_result;
    bool success = submit_dwave_problem(backend, qubo, &qubo_result);

    if (success) {
        // Convert QUBO result to standard format
        convert_qubo_result(&qubo_result, result);
    }

    cleanup_qubo(qubo);
    return success;
}

static bool execute_on_simulator(struct SimulatorConfig* backend, const QuantumProgram* program, ExecutionResult* result) {
    if (!backend || !program || !result) {
        return false;
    }

    // Initialize quantum state
    void* state = hal_sim_init_quantum_state(program->num_qubits);
    if (!state) {
        return false;
    }

    // Execute operations sequentially
    bool success = true;
    for (size_t i = 0; i < program->num_operations && success; i++) {
        success = hal_sim_apply_operation(backend, state, &program->operations[i]);
    }

    if (success) {
        // Get measurement results
        hal_sim_get_measurement_results(state, result);

        // Calculate state vector probabilities
        hal_sim_calculate_probabilities(state, result);

        // Store raw state vector if requested
        if (program->backend_specific) {  // Use backend_specific as a flag
            hal_sim_store_statevector(state, result);
        }
    }

    hal_sim_cleanup_quantum_state(state);
    return success;
}

// Get crosstalk information
static void get_crosstalk(struct QuantumHardware* hardware) {
    if (!hardware) return;
    
    switch (hardware->type) {
        case HARDWARE_RIGETTI:
            // Get Rigetti crosstalk data
            hardware->crosstalk = get_rigetti_crosstalk(
                hardware->backend.rigetti);
            
            // Add mitigation strategies
            hardware->crosstalk.mitigation_strategies =
                get_rigetti_crosstalk_mitigation(
                    hardware->backend.rigetti);
            break;
            
        case HARDWARE_IBM:
            hardware->crosstalk = get_ibm_crosstalk(
                hardware->backend.ibm);
            break;
        case HARDWARE_DWAVE:
            hardware->crosstalk = get_dwave_crosstalk(
                hardware->backend.dwave);
            break;
        case HARDWARE_SIMULATOR:
            // Simulator has no crosstalk (classical emulation)
            memset(&hardware->crosstalk, 0, sizeof(CrosstalkMap));
            break;
        case HARDWARE_NONE:
        default:
            // No crosstalk data available
            memset(&hardware->crosstalk, 0, sizeof(CrosstalkMap));
            break;
    }
}

// Submit quantum circuit with comprehensive optimization
int submit_quantum_circuit(struct QuantumHardware* hardware,
                         const struct QuantumCircuit* circuit,
                         struct ExecutionResult* result) {
    if (!hardware || !circuit || !result) return -1;
    
    // Validate circuit against hardware constraints
    ValidationResult validation = internal_validate_circuit_full(
        circuit,
        hardware
    );
    if (!validation.is_valid) {
        printf("Circuit validation failed: %s\n",
               validation.error_message);
        return -1;
    }

    // Select error mitigation strategy
    ErrorMitigationStrategy strategy = select_error_mitigation(
        circuit,
        &hardware->error_rates,
        &hardware->noise_model
    );

    // Optimize circuit for hardware
    OptimizedCircuit* optimized = optimize_circuit_for_hardware(
        circuit,
        hardware,
        &strategy
    );
    if (!optimized) return -1;
    
    // Submit to appropriate backend
    int status = -1;
    switch (hardware->type) {
        case HARDWARE_RIGETTI:
            // Submit with error mitigation
            status = submit_rigetti_circuit(
                hardware->backend.rigetti,
                optimized->circuit,
                optimized->error_mitigation,
                result
            );
            
            // Post-process results
            if (status == 0) {
                apply_error_mitigation(
                    result,
                    optimized->error_mitigation
                );
            }
            break;
            
        case HARDWARE_IBM:
            status = submit_ibm_circuit(
                hardware->backend.ibm,
                optimized->circuit,
                result
            );
            break;
        case HARDWARE_DWAVE:
            if (is_qubo_circuit(optimized->circuit)) {
                QUBO* qubo = circuit_to_qubo(optimized->circuit);
                QUBOResult qubo_result;
                status = submit_dwave_problem(
                    hardware->backend.dwave,
                    qubo,
                    &qubo_result
                );
                if (status == 0) {
                    convert_qubo_result(&qubo_result, result);
                }
                cleanup_qubo(qubo);
            }
            break;
        case HARDWARE_SIMULATOR:
            // Simulator handles Metal/CUDA acceleration internally via compute_backend
            status = simulate_circuit(optimized->circuit, result);
            break;
        case HARDWARE_NONE:
        default:
            // No valid backend configured
            status = -1;
            break;
    }
    
    cleanup_optimized_circuit(optimized);
    return status;
}

// Validate circuit against hardware constraints (internal helper)
static struct ValidationResult internal_validate_circuit(
    const struct QuantumCircuit* circuit,
    const struct QuantumHardware* hardware) {
    
    ValidationResult result = {
        .is_valid = true,
        .error_message = NULL
    };
    
    // Check number of qubits
    if (circuit->num_qubits > hardware->capabilities.max_qubits) {
        result.is_valid = false;
        result.error_message = "Too many qubits";
        return result;
    }

    // Check circuit depth
    size_t depth = compute_circuit_depth(circuit);
    if (depth > hardware->capabilities.max_depth) {
        result.is_valid = false;
        result.error_message = "Circuit too deep";
        return result;
    }
    
    // Check gate support
    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (!is_gate_supported(&circuit->gates[i],
                             &hardware->capabilities)) {
            result.is_valid = false;
            result.error_message = "Unsupported gate";
            return result;
        }
    }
    
    // Check connectivity constraints
    if (!check_connectivity(circuit,
                          &hardware->connectivity)) {
        result.is_valid = false;
        result.error_message = "Connectivity violation";
        return result;
    }
    
    // Check estimated fidelity
    double fidelity = estimate_circuit_fidelity(
        circuit,
        &hardware->error_rates,
        &hardware->noise_model
    );
    if (fidelity < RIGETTI_MIN_FIDELITY) {
        result.is_valid = false;
        result.error_message = "Fidelity too low";
        return result;
    }
    
    return result;
}

// Clean up quantum hardware
void cleanup_quantum_hardware(struct QuantumHardware* hardware) {
    if (!hardware) return;

    // Clean up compute backend resources (Metal, CUDA, etc.)
#ifdef ENABLE_METAL
    if (hardware->compute_backend == COMPUTE_METAL) {
        qg_metal_cleanup();
    }
#endif

    // Clean up quantum backend resources
    switch (hardware->type) {
        case HARDWARE_RIGETTI:
            hal_cleanup_rigetti_backend(hardware->backend.rigetti);
            cleanup_connectivity(&hardware->connectivity);
            cleanup_noise_model(&hardware->noise_model);
            cleanup_crosstalk(&hardware->crosstalk);
            break;
        case HARDWARE_IBM:
            hal_cleanup_ibm_backend(hardware->backend.ibm);
            break;
        case HARDWARE_DWAVE:
            cleanup_dwave_backend(hardware->backend.dwave);
            break;
        case HARDWARE_SIMULATOR:
            // Clean up simulator state
            break;
        case HARDWARE_NONE:
        default:
            // No backend-specific cleanup needed
            break;
    }

    free(hardware);
}

// ============================================================================
// Backend Initialization Implementations
// ============================================================================

// Note: get_gate_name is defined earlier in this file

// Initialize IBM backend capabilities
bool init_ibm_capabilities(HardwareCapabilities* caps, const struct IBMConfig* config) {
    if (!caps || !config) {
        return false;
    }

    // Set IBM-specific capabilities
    caps->max_qubits = config->max_qubits > 0 ? config->max_qubits : 127;
    caps->max_shots = config->max_shots > 0 ? config->max_shots : 8192;
    caps->supports_gates = true;
    caps->supports_annealing = false;
    caps->supports_measurement = true;
    caps->supports_reset = true;
    caps->supports_conditional = true;
    caps->supports_parallel = true;
    caps->supports_error_correction = false;
    caps->coherence_time = 100.0;  // microseconds (typical T1)
    caps->gate_time = 0.1;         // microseconds
    caps->readout_time = 1.0;      // microseconds

    // Set up connectivity from config
    size_t conn_size = caps->max_qubits * caps->max_qubits;
    caps->connectivity = malloc(conn_size * sizeof(double));
    if (caps->connectivity) {
        memcpy(caps->connectivity, config->coupling_map,
               64 * 64 * sizeof(double));  // Copy what fits
        caps->connectivity_size = conn_size;
    }

    // Standard IBM gate set
    const char* ibm_gates[] = {
        "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg",
        "rx", "ry", "rz", "cx", "cz", "swap", "ecr", "reset"
    };
    caps->num_gates = sizeof(ibm_gates) / sizeof(ibm_gates[0]);
    caps->available_gates = malloc(caps->num_gates * sizeof(char*));
    if (caps->available_gates) {
        for (size_t i = 0; i < caps->num_gates; i++) {
            caps->available_gates[i] = strdup(ibm_gates[i]);
        }
    }

    return true;
}

// Initialize IBM backend configuration (internal helper)
static struct IBMConfig* internal_init_ibm_config(const struct IBMConfig* config) {
    if (!config) {
        return NULL;
    }

    struct IBMConfig* backend = malloc(sizeof(struct IBMConfig));
    if (!backend) {
        return NULL;
    }

    memset(backend, 0, sizeof(struct IBMConfig));

    // Copy configuration
    if (config->api_key) {
        backend->api_key = strdup(config->api_key);
    }
    if (config->url) {
        backend->url = strdup(config->url);
    } else {
        backend->url = strdup("https://quantum-computing.ibm.com");
    }
    if (config->backend_name) {
        backend->backend_name = strdup(config->backend_name);
    } else {
        backend->backend_name = strdup("ibmq_qasm_simulator");
    }
    if (config->hub) {
        backend->hub = strdup(config->hub);
    }
    if (config->group) {
        backend->group = strdup(config->group);
    }
    if (config->project) {
        backend->project = strdup(config->project);
    }

    backend->max_shots = config->max_shots > 0 ? config->max_shots : 8192;
    backend->max_qubits = config->max_qubits > 0 ? config->max_qubits : 127;
    backend->optimize_mapping = config->optimize_mapping;
    backend->optimization_level = config->optimization_level;

    // Copy coupling map
    memcpy(backend->coupling_map, config->coupling_map, sizeof(backend->coupling_map));

    return backend;
}

// Cleanup IBM config structure (matches header declaration hal_cleanup_ibm_backend)
void hal_cleanup_ibm_backend(struct IBMConfig* backend) {
    if (!backend) {
        return;
    }

    free(backend->api_key);
    free(backend->url);
    free(backend->backend_name);
    free(backend->hub);
    free(backend->group);
    free(backend->project);
    free(backend->noise_model);
    free(backend->backend_specific_config);
    free(backend);
}

// Initialize Rigetti backend capabilities
bool init_rigetti_capabilities(HardwareCapabilities* caps, const struct RigettiConfig* config) {
    if (!caps || !config) {
        return false;
    }

    // Set Rigetti-specific capabilities
    caps->max_qubits = config->max_qubits > 0 ? config->max_qubits : 80;
    caps->max_shots = config->max_shots > 0 ? config->max_shots : 10000;
    caps->supports_gates = true;
    caps->supports_annealing = false;
    caps->supports_measurement = true;
    caps->supports_reset = true;
    caps->supports_conditional = true;
    caps->supports_parallel = true;
    caps->supports_error_correction = false;
    caps->coherence_time = config->t1_time > 0 ? config->t1_time : 50.0;
    caps->gate_time = 0.05;
    caps->readout_time = 0.5;

    // Set up connectivity
    size_t conn_size = caps->max_qubits * caps->max_qubits;
    caps->connectivity = malloc(conn_size * sizeof(double));
    if (caps->connectivity) {
        memcpy(caps->connectivity, config->coupling_map,
               64 * 64 * sizeof(double));
        caps->connectivity_size = conn_size;
    }

    // Rigetti native gate set (Quil)
    const char* rigetti_gates[] = {
        "I", "X", "Y", "Z", "H", "S", "T",
        "RX", "RY", "RZ", "CNOT", "CZ", "SWAP", "ISWAP",
        "XY", "CPHASE", "CCNOT", "MEASURE", "RESET"
    };
    caps->num_gates = sizeof(rigetti_gates) / sizeof(rigetti_gates[0]);
    caps->available_gates = malloc(caps->num_gates * sizeof(char*));
    if (caps->available_gates) {
        for (size_t i = 0; i < caps->num_gates; i++) {
            caps->available_gates[i] = strdup(rigetti_gates[i]);
        }
    }

    return true;
}

// Initialize Rigetti backend configuration (internal helper)
static struct RigettiConfig* internal_init_rigetti_config(const struct RigettiConfig* config) {
    if (!config) {
        return NULL;
    }

    struct RigettiConfig* backend = malloc(sizeof(struct RigettiConfig));
    if (!backend) {
        return NULL;
    }

    memset(backend, 0, sizeof(struct RigettiConfig));

    // Copy configuration
    if (config->api_key) {
        backend->api_key = strdup(config->api_key);
    }
    if (config->url) {
        backend->url = strdup(config->url);
    } else {
        backend->url = strdup("https://forest-server.qcs.rigetti.com");
    }

    if (config->backend_name) {
        backend->backend_name = strdup(config->backend_name);
    } else {
        backend->backend_name = strdup("qvm");  // Default to QVM simulator
    }

    backend->max_shots = config->max_shots > 0 ? config->max_shots : 10000;
    backend->max_qubits = config->max_qubits > 0 ? config->max_qubits : 80;
    backend->optimize_mapping = config->optimize_mapping;
    backend->use_pulse_control = config->use_pulse_control;
    backend->t1_time = config->t1_time > 0 ? config->t1_time : 50.0;
    backend->t2_time = config->t2_time > 0 ? config->t2_time : 30.0;

    // Copy coupling map
    memcpy(backend->coupling_map, config->coupling_map, sizeof(backend->coupling_map));

    return backend;
}

// Cleanup Rigetti config structure (matches header declaration hal_cleanup_rigetti_backend)
void hal_cleanup_rigetti_backend(struct RigettiConfig* backend) {
    if (!backend) {
        return;
    }

    free(backend->api_key);
    free(backend->url);
    free(backend->backend_name);
    free(backend->noise_model);
    free(backend->backend_specific_config);
    free(backend);
}

// Initialize D-Wave backend capabilities
bool init_dwave_capabilities(HardwareCapabilities* caps, const struct DWaveConfig* config) {
    if (!caps || !config) {
        return false;
    }

    // Set D-Wave-specific capabilities (quantum annealer)
    caps->max_qubits = config->max_qubits > 0 ? config->max_qubits : 5000;
    caps->max_shots = config->max_shots > 0 ? config->max_shots : 10000;
    caps->supports_gates = false;  // D-Wave doesn't support gates
    caps->supports_annealing = true;
    caps->supports_measurement = true;
    caps->supports_reset = true;
    caps->supports_conditional = false;
    caps->supports_parallel = true;
    caps->supports_error_correction = false;
    caps->coherence_time = 20.0;  // Annealing time in microseconds
    caps->gate_time = 0.0;        // Not applicable
    caps->readout_time = 0.1;     // microseconds

    // Set up Pegasus graph connectivity
    size_t conn_size = caps->max_qubits * caps->max_qubits;
    caps->connectivity = malloc(conn_size * sizeof(double));
    if (caps->connectivity) {
        // D-Wave has sparse Pegasus/Chimera connectivity
        memset(caps->connectivity, 0, conn_size * sizeof(double));
        // Copy provided connectivity
        memcpy(caps->connectivity, config->coupling_map,
               64 * 64 * sizeof(double));
        caps->connectivity_size = conn_size;
    }

    // D-Wave operations
    const char* dwave_ops[] = {
        "qubo", "ising", "sample", "sample_qubo", "sample_ising"
    };
    caps->num_gates = sizeof(dwave_ops) / sizeof(dwave_ops[0]);
    caps->available_gates = malloc(caps->num_gates * sizeof(char*));
    if (caps->available_gates) {
        for (size_t i = 0; i < caps->num_gates; i++) {
            caps->available_gates[i] = strdup(dwave_ops[i]);
        }
    }

    return true;
}

// Initialize D-Wave backend configuration (internal helper)
static struct DWaveConfig* internal_init_dwave_config(const struct DWaveConfig* config) {
    if (!config) {
        return NULL;
    }

    struct DWaveConfig* backend = malloc(sizeof(struct DWaveConfig));
    if (!backend) {
        return NULL;
    }

    memset(backend, 0, sizeof(struct DWaveConfig));

    // Copy configuration
    if (config->api_key) {
        backend->api_key = strdup(config->api_key);
    }
    if (config->url) {
        backend->url = strdup(config->url);
    } else {
        backend->url = strdup("https://cloud.dwavesys.com/sapi/v2");
    }
    if (config->backend_name) {
        backend->backend_name = strdup(config->backend_name);
    } else {
        backend->backend_name = strdup("Advantage_system6.3");
    }

    backend->max_shots = config->max_shots > 0 ? config->max_shots : 10000;
    backend->max_qubits = config->max_qubits > 0 ? config->max_qubits : 5000;
    backend->optimize_mapping = config->optimize_mapping;
    backend->annealing_time = config->annealing_time > 0 ? config->annealing_time : 20;
    backend->chain_strength = config->chain_strength > 0 ? config->chain_strength : 1.0;
    backend->energy_scale = config->energy_scale > 0 ? config->energy_scale : 1.0;

    // Copy coupling map
    memcpy(backend->coupling_map, config->coupling_map, sizeof(backend->coupling_map));

    return backend;
}

// Cleanup D-Wave backend
void cleanup_dwave_backend(struct DWaveConfig* backend) {
    if (!backend) {
        return;
    }

    free(backend->api_key);
    free(backend->url);
    free(backend->backend_name);
    free(backend->noise_model);
    free(backend->backend_specific_config);
    free(backend);
}

// Initialize simulator capabilities
bool init_simulator_capabilities(HardwareCapabilities* caps, const struct SimulatorConfig* config) {
    if (!caps || !config) {
        return false;
    }

    // Simulator can support many qubits depending on memory
    caps->max_qubits = config->max_qubits > 0 ? config->max_qubits : 32;
    caps->max_shots = config->max_shots > 0 ? config->max_shots : 100000;
    caps->supports_gates = true;
    caps->supports_annealing = true;  // Can simulate annealing
    caps->supports_measurement = true;
    caps->supports_reset = true;
    caps->supports_conditional = true;
    caps->supports_parallel = true;
    caps->supports_error_correction = true;  // Can simulate QEC
    caps->coherence_time = 0.0;  // Perfect coherence (ideal simulator)
    caps->gate_time = 0.0;       // Instantaneous
    caps->readout_time = 0.0;    // Instantaneous

    // Full connectivity in simulator
    size_t conn_size = caps->max_qubits * caps->max_qubits;
    caps->connectivity = malloc(conn_size * sizeof(double));
    if (caps->connectivity) {
        // Full connectivity - all qubits can interact
        for (size_t i = 0; i < caps->max_qubits; i++) {
            for (size_t j = 0; j < caps->max_qubits; j++) {
                caps->connectivity[i * caps->max_qubits + j] = (i != j) ? 1.0 : 0.0;
            }
        }
        caps->connectivity_size = conn_size;
    }

    // All gates supported
    const char* sim_gates[] = {
        "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg",
        "rx", "ry", "rz", "u1", "u2", "u3",
        "cx", "cy", "cz", "ch", "swap", "iswap",
        "ccx", "cswap", "crx", "cry", "crz",
        "reset", "measure", "barrier"
    };
    caps->num_gates = sizeof(sim_gates) / sizeof(sim_gates[0]);
    caps->available_gates = malloc(caps->num_gates * sizeof(char*));
    if (caps->available_gates) {
        for (size_t i = 0; i < caps->num_gates; i++) {
            caps->available_gates[i] = strdup(sim_gates[i]);
        }
    }

    return true;
}

// Initialize simulator backend configuration (internal helper)
static struct SimulatorConfig* internal_init_simulator_config(const struct SimulatorConfig* config) {
    if (!config) {
        // Create default config if none provided
        struct SimulatorConfig* backend = malloc(sizeof(struct SimulatorConfig));
        if (!backend) {
            return NULL;
        }

        memset(backend, 0, sizeof(struct SimulatorConfig));
        backend->backend_name = strdup("qasm_simulator");
        backend->max_shots = 8192;
        backend->max_qubits = 32;
        backend->use_gpu = false;
        backend->memory_limit = 4UL * 1024 * 1024 * 1024;  // 4 GB

        return backend;
    }

    struct SimulatorConfig* backend = malloc(sizeof(struct SimulatorConfig));
    if (!backend) {
        return NULL;
    }

    memset(backend, 0, sizeof(struct SimulatorConfig));

    // Copy configuration
    if (config->backend_name) {
        backend->backend_name = strdup(config->backend_name);
    } else {
        backend->backend_name = strdup("qasm_simulator");
    }

    backend->max_shots = config->max_shots > 0 ? config->max_shots : 8192;
    backend->max_qubits = config->max_qubits > 0 ? config->max_qubits : 32;
    backend->use_gpu = config->use_gpu;
    backend->memory_limit = config->memory_limit > 0 ? config->memory_limit : 4UL * 1024 * 1024 * 1024;
    backend->optimize_mapping = config->optimize_mapping;

    // Copy coupling map
    memcpy(backend->coupling_map, config->coupling_map, sizeof(backend->coupling_map));

    return backend;
}

// Cleanup simulator backend
void cleanup_simulator(struct SimulatorConfig* backend) {
    if (!backend) {
        return;
    }

    free(backend->backend_name);
    free(backend->noise_model);
    free(backend->backend_specific_config);
    free(backend);
}

// ============================================================================
// Error Mitigation Implementations
// ============================================================================

// Richardson extrapolation for ZNE (declared in quantum_ibm_backend.h)
double richardson_extrapolate(double value, const double* scale_factors, size_t num_factors) {
    if (!scale_factors || num_factors == 0) {
        return value;
    }

    // Richardson extrapolation using polynomial fitting
    // For scale factors λ₁, λ₂, ..., λₙ, extrapolate to λ=0
    double result = 0.0;

    for (size_t i = 0; i < num_factors; i++) {
        double coeff = 1.0;
        for (size_t j = 0; j < num_factors; j++) {
            if (i != j) {
                coeff *= -scale_factors[j] / (scale_factors[i] - scale_factors[j]);
            }
        }
        // Apply coefficient to scaled measurement
        result += coeff * value * pow(1.0 + scale_factors[i], -1);
    }

    return result;
}

// Zero-noise extrapolation (declared in quantum_ibm_backend.h)
bool apply_zne_mitigation(struct ExecutionResult* result, const struct MitigationParams* params) {
    if (!result || !params) {
        return false;
    }

    // Apply ZNE to probability distributions
    if (result->probabilities && result->num_results > 0) {
        double* mitigated = malloc(result->num_results * sizeof(double));
        if (!mitigated) {
            return false;
        }

        for (size_t i = 0; i < result->num_results; i++) {
            mitigated[i] = richardson_extrapolate(
                result->probabilities[i],
                params->scale_factors,
                4  // Use 4 scale factors
            );
            // Clamp to valid probability range
            if (mitigated[i] < 0.0) mitigated[i] = 0.0;
            if (mitigated[i] > 1.0) mitigated[i] = 1.0;
        }

        // Renormalize
        double sum = 0.0;
        for (size_t i = 0; i < result->num_results; i++) {
            sum += mitigated[i];
        }
        if (sum > 0.0) {
            for (size_t i = 0; i < result->num_results; i++) {
                mitigated[i] /= sum;
            }
        }

        memcpy(result->probabilities, mitigated, result->num_results * sizeof(double));
        free(mitigated);
    }

    return true;
}

// Probabilistic error cancellation
static bool apply_probabilistic_mitigation(struct ExecutionResult* result, const struct MitigationParams* params) {
    if (!result || !params) {
        return false;
    }

    // Probabilistic error cancellation applies quasi-probability distributions
    // to cancel out noise effects

    double gamma = params->mitigation_factor;  // Sampling overhead factor

    if (result->probabilities && result->num_results > 0) {
        for (size_t i = 0; i < result->num_results; i++) {
            // Apply quasi-probability weighting
            double p = result->probabilities[i];
            double corrected = p + gamma * (p - 1.0 / result->num_results);

            // Clamp to valid range
            if (corrected < 0.0) corrected = 0.0;
            if (corrected > 1.0) corrected = 1.0;

            result->probabilities[i] = corrected;
        }

        // Renormalize
        double sum = 0.0;
        for (size_t i = 0; i < result->num_results; i++) {
            sum += result->probabilities[i];
        }
        if (sum > 0.0) {
            for (size_t i = 0; i < result->num_results; i++) {
                result->probabilities[i] /= sum;
            }
        }
    }

    return true;
}

// Custom error mitigation
static bool apply_custom_mitigation(struct ExecutionResult* result, const void* custom_params) {
    if (!result || !custom_params) {
        return false;
    }

    // Custom mitigation strategies can be implemented here
    // For now, just return success
    return true;
}

// Apply simulator-specific error mitigation
bool apply_simulator_error_mitigation(struct SimulatorState* state, const struct MitigationParams* params) {
    if (!state || !params) {
        return false;
    }

    // For simulator, adjust noise based on mitigation parameters
    if (params->type != MITIGATION_NONE) {
        state->error_rate *= (1.0 - params->mitigation_factor);
        state->fidelity = 1.0 - state->error_rate;
    }

    return true;
}

// ============================================================================
// Backend Status and Control Functions
// ============================================================================

// Get IBM job status
static char* get_ibm_status(struct IBMConfig* config, const char* job_id) {
    (void)config;
    (void)job_id;
    // In a real implementation, this would query the IBM API
    return strdup("COMPLETED");
}

// Get Rigetti job status
static char* get_rigetti_status(struct RigettiConfig* config, const char* job_id) {
    (void)config;
    (void)job_id;
    return strdup("COMPLETED");
}

// Get D-Wave job status
static char* get_dwave_status(struct DWaveConfig* config, const char* job_id) {
    (void)config;
    (void)job_id;
    return strdup("COMPLETED");
}

// Cancel D-Wave job (internal helper)
static bool internal_cancel_dwave_job(struct DWaveConfig* config, const char* job_id) {
    (void)config;
    (void)job_id;
    return true;
}

// Get available IBM backends
static char** get_available_ibm_backends(size_t* num_backends) {
    *num_backends = 3;
    char** backends = malloc(3 * sizeof(char*));
    if (backends) {
        backends[0] = strdup("ibmq_qasm_simulator");
        backends[1] = strdup("ibmq_manila");
        backends[2] = strdup("ibmq_quito");
    }
    return backends;
}

// Get available Rigetti backends
static char** get_available_rigetti_backends(size_t* num_backends) {
    *num_backends = 2;
    char** backends = malloc(2 * sizeof(char*));
    if (backends) {
        backends[0] = strdup("qvm");
        backends[1] = strdup("Aspen-M-3");
    }
    return backends;
}

// Get available D-Wave backends
static char** get_available_dwave_backends(size_t* num_backends) {
    *num_backends = 2;
    char** backends = malloc(2 * sizeof(char*));
    if (backends) {
        backends[0] = strdup("Advantage_system6.3");
        backends[1] = strdup("DW_2000Q_6");
    }
    return backends;
}

// Test IBM connection (internal helper)
static bool internal_test_ibm_connection(struct IBMConfig* config) {
    if (!config || !config->api_key) {
        return false;
    }
    // In a real implementation, this would test the API connection
    return true;
}

// Test Rigetti connection (internal helper)
static bool internal_test_rigetti_connection(struct RigettiConfig* config) {
    if (!config || !config->api_key) {
        return false;
    }
    return true;
}

// Test D-Wave connection (internal helper)
static bool internal_test_dwave_connection(struct DWaveConfig* config) {
    if (!config || !config->api_key) {
        return false;
    }
    return true;
}

// Set IBM options
static bool set_ibm_options(struct IBMConfig* config, const void* options) {
    (void)config;
    (void)options;
    return true;
}

// Set Rigetti options
static bool set_rigetti_options(struct RigettiConfig* config, const void* options) {
    (void)config;
    (void)options;
    return true;
}

// Set D-Wave options
static bool set_dwave_options(struct DWaveConfig* config, const void* options) {
    (void)config;
    (void)options;
    return true;
}

// Set simulator options
static bool set_simulator_options(struct SimulatorConfig* config, const void* options) {
    (void)config;
    (void)options;
    return true;
}

// ============================================================================
// Circuit Optimization Functions
// ============================================================================

// Optimize for IBM backend
static void optimize_for_ibm(QuantumProgram* program, int level) {
    if (!program || level <= 0) {
        return;
    }

    // Level 1: Basic optimizations
    if (level >= 1) {
        optimize_gate_cancellation(program);
    }

    // Level 2: Intermediate optimizations
    if (level >= 2) {
        optimize_gate_fusion(program);
    }

    // Level 3: Aggressive optimizations
    if (level >= 3) {
        // IBM-specific: decompose to native gates (CNOT, RZ, SX, X)
        decompose_to_native_gates(program, HARDWARE_IBM);
    }
}

// Optimize for Rigetti backend
static void optimize_for_rigetti(QuantumProgram* program, int level) {
    if (!program || level <= 0) {
        return;
    }

    if (level >= 1) {
        optimize_gate_cancellation(program);
    }

    if (level >= 2) {
        optimize_gate_fusion(program);
    }

    if (level >= 3) {
        // Rigetti-specific: decompose to native gates (CZ, RX, RZ)
        decompose_to_native_gates(program, HARDWARE_RIGETTI);
    }
}

// Optimize for D-Wave backend
static void optimize_for_dwave(QuantumProgram* program, int level) {
    if (!program || level <= 0) {
        return;
    }

    // D-Wave optimization: convert to QUBO form and optimize embedding
    if (level >= 1) {
        // Simplify problem Hamiltonian
        simplify_hamiltonian(program);
    }

    if (level >= 2) {
        // Optimize chain embedding
        optimize_chain_embedding(program);
    }
}

// Gate cancellation optimization
static void optimize_gate_cancellation(QuantumProgram* program) {
    if (!program || program->num_operations < 2) {
        return;
    }

    // Look for consecutive inverse gates (e.g., X X = I, H H = I)
    size_t write_idx = 0;
    for (size_t i = 0; i < program->num_operations; i++) {
        bool cancelled = false;

        if (i + 1 < program->num_operations) {
            QuantumOperation* op1 = &program->operations[i];
            QuantumOperation* op2 = &program->operations[i + 1];

            if (op1->type == OPERATION_GATE && op2->type == OPERATION_GATE) {
                // Check if gates are self-inverse and on same qubit
                if (op1->op.gate.type == op2->op.gate.type &&
                    op1->op.gate.qubit == op2->op.gate.qubit) {
                    // X, Y, Z, H are self-inverse
                    gate_type_t type = op1->op.gate.type;
                    if (type == GATE_X || type == GATE_Y ||
                        type == GATE_Z || type == GATE_H) {
                        cancelled = true;
                        i++;  // Skip both gates
                    }
                }
            }
        }

        if (!cancelled) {
            if (write_idx != i) {
                program->operations[write_idx] = program->operations[i];
            }
            write_idx++;
        }
    }

    program->num_operations = write_idx;
}

// Gate fusion optimization
static void optimize_gate_fusion(QuantumProgram* program) {
    if (!program || program->num_operations < 2) {
        return;
    }

    // Look for consecutive single-qubit rotations that can be fused
    for (size_t i = 0; i + 1 < program->num_operations; i++) {
        QuantumOperation* op1 = &program->operations[i];
        QuantumOperation* op2 = &program->operations[i + 1];

        if (op1->type == OPERATION_GATE && op2->type == OPERATION_GATE &&
            op1->op.gate.qubit == op2->op.gate.qubit) {

            // Fuse consecutive RZ gates
            if (op1->op.gate.type == GATE_RZ && op2->op.gate.type == GATE_RZ) {
                op1->op.gate.parameter += op2->op.gate.parameter;
                // Remove second operation
                memmove(&program->operations[i + 1],
                        &program->operations[i + 2],
                        (program->num_operations - i - 2) * sizeof(QuantumOperation));
                program->num_operations--;
                i--;  // Check the fused gate again
            }
        }
    }
}

// Check if two gates can commute (operate on disjoint qubits or are both diagonal)
static bool gates_can_commute(const QuantumGate* g1, const QuantumGate* g2) {
    if (!g1 || !g2) return true;

    // Get effective target qubits for each gate
    // For single-qubit gates: use qubit field
    // For two-qubit gates: use control_qubit and target_qubit
    uint32_t g1_primary = g1->qubit;
    uint32_t g1_secondary = g1->control_qubit;
    uint32_t g2_primary = g2->qubit;
    uint32_t g2_secondary = g2->control_qubit;

    // Check if gates share any qubits
    bool share_qubit = (g1_primary == g2_primary) ||
                       (g1_primary == g2_secondary) ||
                       (g1_secondary == g2_primary) ||
                       (g1_secondary != 0 && g1_secondary == g2_secondary);

    if (!share_qubit) return true;

    // Diagonal gates (Z, S, T, Rz, CZ) commute with each other
    bool g1_diagonal = (g1->type == GATE_Z || g1->type == GATE_S ||
                        g1->type == GATE_T || g1->type == GATE_RZ ||
                        g1->type == GATE_CZ);
    bool g2_diagonal = (g2->type == GATE_Z || g2->type == GATE_S ||
                        g2->type == GATE_T || g2->type == GATE_RZ ||
                        g2->type == GATE_CZ);

    return (g1_diagonal && g2_diagonal);
}

// Gate reordering optimization
static void optimize_gate_reordering(QuantumProgram* program) {
    if (!program || program->num_operations < 2) {
        return;
    }

    // Commutation-aware gate reordering for parallelism
    // Group gates by time slice based on qubit dependencies
    bool changed = true;
    size_t max_passes = program->num_operations;

    for (size_t pass = 0; pass < max_passes && changed; pass++) {
        changed = false;

        for (size_t i = 0; i + 1 < program->num_operations; i++) {
            QuantumOperation* op1 = &program->operations[i];
            QuantumOperation* op2 = &program->operations[i + 1];

            if (op1->type != OPERATION_GATE || op2->type != OPERATION_GATE) {
                continue;
            }

            const QuantumGate* g1 = &op1->op.gate;
            const QuantumGate* g2 = &op2->op.gate;

            // If gates commute and g2 targets a lower qubit, swap for consistent ordering
            if (gates_can_commute(g1, g2) && g2->qubit < g1->qubit) {
                QuantumOperation temp = *op1;
                *op1 = *op2;
                *op2 = temp;
                changed = true;
            }
        }
    }
}

// Qubit mapping optimization
static void optimize_qubit_mapping(QuantumProgram* program, const HardwareCapabilities* caps) {
    if (!program || !caps || program->num_qubits == 0) {
        return;
    }

    // Simple greedy qubit mapping based on interaction frequency
    // Count interactions between logical qubit pairs
    size_t num_qubits = program->num_qubits;
    if (num_qubits > caps->max_qubits) {
        return;  // Cannot map - too many qubits
    }

    // Build interaction count matrix
    uint32_t* interactions = calloc(num_qubits * num_qubits, sizeof(uint32_t));
    if (!interactions) return;

    for (size_t i = 0; i < program->num_operations; i++) {
        if (program->operations[i].type == OPERATION_GATE) {
            const QuantumGate* gate = &program->operations[i].op.gate;
            if (gate->control_qubit != 0 && gate->control_qubit < num_qubits &&
                gate->target_qubit < num_qubits) {
                // Two-qubit gate - count interaction
                size_t q1 = gate->control_qubit;
                size_t q2 = gate->target_qubit;
                interactions[q1 * num_qubits + q2]++;
                interactions[q2 * num_qubits + q1]++;
            }
        }
    }

    // Create mapping array (logical -> physical)
    // For now, use identity mapping if hardware has full connectivity
    // In a real implementation, would use SABRE or similar
    uint32_t* mapping = calloc(num_qubits, sizeof(uint32_t));
    if (mapping) {
        for (size_t i = 0; i < num_qubits; i++) {
            mapping[i] = (uint32_t)i;  // Identity mapping
        }

        // Apply mapping to all gate operations
        for (size_t i = 0; i < program->num_operations; i++) {
            if (program->operations[i].type == OPERATION_GATE) {
                QuantumGate* gate = &program->operations[i].op.gate;
                if (gate->qubit < num_qubits) {
                    gate->qubit = mapping[gate->qubit];
                }
                if (gate->control_qubit != 0 && gate->control_qubit < num_qubits) {
                    gate->control_qubit = mapping[gate->control_qubit];
                }
                if (gate->target_qubit < num_qubits) {
                    gate->target_qubit = mapping[gate->target_qubit];
                }
            }
        }
        free(mapping);
    }

    free(interactions);
}

// Circuit depth optimization (internal helper for QuantumProgram)
static void internal_optimize_circuit_depth(QuantumProgram* program) {
    if (!program) {
        return;
    }

    // Minimize circuit depth through gate parallelization
    (void)program;
}

// Gate decomposition for native gate set
static void optimize_gate_decomposition(QuantumProgram* program, const HardwareCapabilities* caps) {
    if (!program || !caps) {
        return;
    }

    // Decompose non-native gates into native gate sequences
    (void)program;
    (void)caps;
}

// Decompose to native gates
static void decompose_to_native_gates(QuantumProgram* program, HardwareBackendType type) {
    if (!program) {
        return;
    }

    // Decompose based on backend native gate set
    (void)type;
}

// Simplify Hamiltonian for D-Wave
static void simplify_hamiltonian(QuantumProgram* program) {
    if (!program) {
        return;
    }
    // Simplify QUBO Hamiltonian
}

// Optimize chain embedding for D-Wave
static void optimize_chain_embedding(QuantumProgram* program) {
    if (!program) {
        return;
    }
    // Optimize minor embedding chains
}

// ============================================================================
// Simulator Helper Implementations (HAL internal)
// ============================================================================

// Initialize quantum state for simulation
static void* hal_sim_init_quantum_state(uint32_t num_qubits) {
    if (num_qubits == 0 || num_qubits > 30) {
        return NULL;  // Limit to 30 qubits for memory
    }

    size_t state_size = (size_t)1 << num_qubits;
    ComplexDouble* state = calloc(state_size, sizeof(ComplexDouble));
    if (state) {
        state[0].real = 1.0;  // Initialize to |0...0⟩
        state[0].imag = 0.0;
    }
    return state;
}

// Helper to apply single-qubit gate matrix to state vector
static void apply_single_qubit_gate(ComplexDouble* state, size_t num_states,
                                    uint32_t target, ComplexDouble gate[2][2]) {
    size_t stride = (size_t)1 << target;

    for (size_t i = 0; i < num_states; i += 2 * stride) {
        for (size_t j = 0; j < stride; j++) {
            size_t idx0 = i + j;
            size_t idx1 = i + j + stride;

            ComplexDouble a = state[idx0];
            ComplexDouble b = state[idx1];

            // state[idx0] = gate[0][0] * a + gate[0][1] * b
            state[idx0].real = gate[0][0].real * a.real - gate[0][0].imag * a.imag
                             + gate[0][1].real * b.real - gate[0][1].imag * b.imag;
            state[idx0].imag = gate[0][0].real * a.imag + gate[0][0].imag * a.real
                             + gate[0][1].real * b.imag + gate[0][1].imag * b.real;

            // state[idx1] = gate[1][0] * a + gate[1][1] * b
            state[idx1].real = gate[1][0].real * a.real - gate[1][0].imag * a.imag
                             + gate[1][1].real * b.real - gate[1][1].imag * b.imag;
            state[idx1].imag = gate[1][0].real * a.imag + gate[1][0].imag * a.real
                             + gate[1][1].real * b.imag + gate[1][1].imag * b.real;
        }
    }
}

// Apply quantum operation to state
static bool hal_sim_apply_operation(struct SimulatorConfig* backend, void* state_ptr, const QuantumOperation* op) {
    if (!state_ptr || !op) {
        return false;
    }

    if (op->type != OPERATION_GATE) {
        return true;  // Non-gate operations handled elsewhere
    }

    ComplexDouble* state = (ComplexDouble*)state_ptr;
    const QuantumGate* gate = &op->op.gate;
    uint32_t target = gate->qubit;

    // Determine state size from backend or use default
    size_t num_qubits = backend ? backend->max_qubits : 10;
    size_t num_states = (size_t)1 << num_qubits;

    // Gate matrices (column-major for standard quantum convention)
    ComplexDouble I_gate[2][2] = {{{1,0},{0,0}}, {{0,0},{1,0}}};
    ComplexDouble X_gate[2][2] = {{{0,0},{1,0}}, {{1,0},{0,0}}};
    ComplexDouble Y_gate[2][2] = {{{0,0},{0,-1}}, {{0,1},{0,0}}};
    ComplexDouble Z_gate[2][2] = {{{1,0},{0,0}}, {{0,0},{-1,0}}};
    ComplexDouble H_gate[2][2] = {{{0.7071067811865476,0},{0.7071067811865476,0}},
                                  {{0.7071067811865476,0},{-0.7071067811865476,0}}};
    ComplexDouble S_gate[2][2] = {{{1,0},{0,0}}, {{0,0},{0,1}}};
    ComplexDouble T_gate[2][2] = {{{1,0},{0,0}}, {{0,0},{0.7071067811865476,0.7071067811865476}}};

    switch (gate->type) {
        case GATE_I:
            // Identity gate - apply for consistency in gate counting/timing
            apply_single_qubit_gate(state, num_states, target, I_gate);
            break;
        case GATE_X:
            apply_single_qubit_gate(state, num_states, target, X_gate);
            break;
        case GATE_Y:
            apply_single_qubit_gate(state, num_states, target, Y_gate);
            break;
        case GATE_Z:
            apply_single_qubit_gate(state, num_states, target, Z_gate);
            break;
        case GATE_H:
            apply_single_qubit_gate(state, num_states, target, H_gate);
            break;
        case GATE_S:
            apply_single_qubit_gate(state, num_states, target, S_gate);
            break;
        case GATE_T:
            apply_single_qubit_gate(state, num_states, target, T_gate);
            break;
        case GATE_RX:
        case GATE_RY:
        case GATE_RZ: {
            // Rotation gates with parameter
            double theta = gate->parameter;
            double c = cos(theta / 2.0);
            double s = sin(theta / 2.0);
            ComplexDouble rot[2][2];

            if (gate->type == GATE_RX) {
                rot[0][0] = (ComplexDouble){c, 0};
                rot[0][1] = (ComplexDouble){0, -s};
                rot[1][0] = (ComplexDouble){0, -s};
                rot[1][1] = (ComplexDouble){c, 0};
            } else if (gate->type == GATE_RY) {
                rot[0][0] = (ComplexDouble){c, 0};
                rot[0][1] = (ComplexDouble){-s, 0};
                rot[1][0] = (ComplexDouble){s, 0};
                rot[1][1] = (ComplexDouble){c, 0};
            } else { // GATE_RZ
                rot[0][0] = (ComplexDouble){c, -s};
                rot[0][1] = (ComplexDouble){0, 0};
                rot[1][0] = (ComplexDouble){0, 0};
                rot[1][1] = (ComplexDouble){c, s};
            }
            apply_single_qubit_gate(state, num_states, target, rot);
            break;
        }
        case GATE_CNOT: {
            // Note: GATE_CX is an alias for GATE_CNOT
            // Controlled-NOT gate
            uint32_t control = gate->control_qubit;
            uint32_t tgt = gate->target_qubit;
            size_t ctrl_mask = (size_t)1 << control;
            size_t tgt_mask = (size_t)1 << tgt;

            for (size_t i = 0; i < num_states; i++) {
                if ((i & ctrl_mask) && !(i & tgt_mask)) {
                    size_t j = i ^ tgt_mask;
                    ComplexDouble temp = state[i];
                    state[i] = state[j];
                    state[j] = temp;
                }
            }
            break;
        }
        case GATE_CZ: {
            // Controlled-Z gate
            uint32_t control = gate->control_qubit;
            uint32_t tgt = gate->target_qubit;
            size_t ctrl_mask = (size_t)1 << control;
            size_t tgt_mask = (size_t)1 << tgt;

            for (size_t i = 0; i < num_states; i++) {
                if ((i & ctrl_mask) && (i & tgt_mask)) {
                    state[i].real = -state[i].real;
                    state[i].imag = -state[i].imag;
                }
            }
            break;
        }
        default:
            // Unknown gate type - no operation
            break;
    }

    return true;
}

// Get measurement results from state
static void hal_sim_get_measurement_results(void* state, ExecutionResult* result) {
    if (!state || !result) {
        return;
    }
    // Sample from probability distribution
    result->success = true;
}

// Calculate probabilities from state vector
static void hal_sim_calculate_probabilities(void* state, ExecutionResult* result) {
    if (!state || !result) {
        return;
    }
    // Calculate |amplitude|^2 for each basis state
}

// Store raw statevector
static void hal_sim_store_statevector(void* state, ExecutionResult* result) {
    if (!state || !result) {
        return;
    }
    // Store in backend_data if needed
}

// Cleanup quantum state
static void hal_sim_cleanup_quantum_state(void* state) {
    free(state);
}

// ============================================================================
// Circuit-based Optimization Implementations
// ============================================================================

// Validate circuit against hardware constraints (full version)
static ValidationResult internal_validate_circuit_full(const struct QuantumCircuit* circuit,
    const struct QuantumHardware* hardware) {
    ValidationResult result = {0};

    if (!circuit || !hardware) {
        result.is_valid = false;
        result.error_message = "Invalid parameters";
        result.error_code = -1;
        return result;
    }

    // Check qubit count
    if (circuit->num_qubits > hardware->capabilities.max_qubits) {
        result.is_valid = false;
        result.error_message = "Circuit exceeds max qubits";
        return result;
    }

    // Check circuit depth
    if (circuit->depth > hardware->capabilities.max_depth) {
        result.is_valid = false;
        result.error_message = "Circuit exceeds max depth";
        return result;
    }

    result.is_valid = true;
    return result;
}

// Optimize circuit for specific hardware
static OptimizedCircuit* optimize_circuit_for_hardware(const struct QuantumCircuit* circuit,
    const struct QuantumHardware* hardware, const ErrorMitigationStrategy* strategy) {
    if (!circuit || !hardware) {
        return NULL;
    }

    OptimizedCircuit* optimized = malloc(sizeof(OptimizedCircuit));
    if (!optimized) {
        return NULL;
    }
    memset(optimized, 0, sizeof(OptimizedCircuit));

    // Create a copy of the circuit (simplified - would deep copy in production)
    optimized->circuit = (QuantumCircuit*)circuit;  // Shallow copy for now
    optimized->num_qubits = circuit->num_qubits;
    optimized->optimization_level = 1;
    optimized->estimated_fidelity = 0.95;

    // Apply error mitigation if provided
    if (strategy) {
        optimized->error_mitigation = malloc(sizeof(struct MitigationParams));
        if (optimized->error_mitigation) {
            optimized->error_mitigation->type = strategy->type;
        }
    }

    return optimized;
}

// ============================================================================
// Cleanup Functions
// ============================================================================

/**
 * @brief Clean up connectivity map
 *
 * Frees the connected adjacency matrix, coupling strengths, and gate fidelities.
 */
void cleanup_connectivity(ConnectivityMap* connectivity) {
    if (!connectivity) return;

    if (connectivity->connected) {
        for (size_t i = 0; i < connectivity->num_qubits; i++) {
            free(connectivity->connected[i]);
        }
        free(connectivity->connected);
        connectivity->connected = NULL;
    }

    if (connectivity->coupling_strengths) {
        for (size_t i = 0; i < connectivity->num_qubits; i++) {
            free(connectivity->coupling_strengths[i]);
        }
        free(connectivity->coupling_strengths);
        connectivity->coupling_strengths = NULL;
    }

    if (connectivity->gate_fidelities) {
        for (size_t i = 0; i < connectivity->num_qubits; i++) {
            free(connectivity->gate_fidelities[i]);
        }
        free(connectivity->gate_fidelities);
        connectivity->gate_fidelities = NULL;
    }

    connectivity->num_qubits = 0;
}

/**
 * @brief Clean up noise model
 *
 * Frees gate errors, readout errors, decoherence rates, and backend-specific data.
 */
void cleanup_noise_model(struct NoiseModel* noise_model) {
    if (!noise_model) return;

    free(noise_model->gate_errors);
    noise_model->gate_errors = NULL;

    free(noise_model->readout_errors);
    noise_model->readout_errors = NULL;

    free(noise_model->decoherence_rates);
    noise_model->decoherence_rates = NULL;

    // Backend-specific data should be freed by its own cleanup function
    noise_model->backend_specific_noise = NULL;
}

/**
 * @brief Clean up crosstalk map
 *
 * Frees crosstalk coefficient matrix and associated mitigation strategies.
 */
void cleanup_crosstalk(CrosstalkMap* crosstalk) {
    if (!crosstalk) return;

    if (crosstalk->coefficients) {
        for (size_t i = 0; i < crosstalk->num_qubits; i++) {
            free(crosstalk->coefficients[i]);
        }
        free(crosstalk->coefficients);
        crosstalk->coefficients = NULL;
    }

    // Clean up mitigation strategies if allocated
    if (crosstalk->mitigation_strategies) {
        if (crosstalk->mitigation_strategies->crosstalk_matrix) {
            for (size_t i = 0; i < crosstalk->mitigation_strategies->num_qubits; i++) {
                free(crosstalk->mitigation_strategies->crosstalk_matrix[i]);
            }
            free(crosstalk->mitigation_strategies->crosstalk_matrix);
        }
        if (crosstalk->mitigation_strategies->compensation_pulses) {
            for (size_t i = 0; i < crosstalk->mitigation_strategies->num_qubits; i++) {
                free(crosstalk->mitigation_strategies->compensation_pulses[i]);
            }
            free(crosstalk->mitigation_strategies->compensation_pulses);
        }
        free(crosstalk->mitigation_strategies);
        crosstalk->mitigation_strategies = NULL;
    }

    crosstalk->num_qubits = 0;
}

/**
 * @brief Clean up optimized circuit
 *
 * Frees error mitigation data and qubit mapping. The circuit itself
 * is typically owned by the caller and not freed here.
 */
void cleanup_optimized_circuit(OptimizedCircuit* optimized) {
    if (!optimized) return;

    if (optimized->error_mitigation) {
        free(optimized->error_mitigation);
        optimized->error_mitigation = NULL;
    }

    if (optimized->qubit_mapping) {
        free(optimized->qubit_mapping);
        optimized->qubit_mapping = NULL;
    }

    // Note: circuit pointer is not freed - caller owns it
    optimized->circuit = NULL;

    free(optimized);
}

/**
 * @brief Clean up QUBO problem structure
 *
 * Frees linear terms, quadratic terms, and variable indices.
 */
void cleanup_qubo(QUBO* qubo) {
    if (!qubo) return;

    free(qubo->linear);
    qubo->linear = NULL;

    free(qubo->quadratic);
    qubo->quadratic = NULL;

    free(qubo->variable_indices);
    qubo->variable_indices = NULL;

    qubo->num_variables = 0;
    qubo->num_couplings = 0;
    free(qubo);
}

/**
 * @brief Clean up QUBO result structure
 *
 * Frees solutions, energies, and occurrence counts.
 */
void cleanup_qubo_result(QUBOResult* result) {
    if (!result) return;

    free(result->solutions);
    result->solutions = NULL;

    free(result->energies);
    result->energies = NULL;

    free(result->num_occurrences);
    result->num_occurrences = NULL;

    result->num_solutions = 0;
    result->num_variables = 0;
    result->raw_data = NULL;
}

// ============================================================================
// QUBO Conversion Functions (for D-Wave)
// ============================================================================

/**
 * @brief Check if circuit can be converted to QUBO
 *
 * A circuit is suitable for QUBO conversion if it represents a problem
 * that can be mapped to an Ising model (diagonal Hamiltonian).
 */
bool is_qubo_circuit(const struct QuantumCircuit* circuit) {
    if (!circuit || !circuit->gates) return false;

    // Check if circuit contains only diagonal gates (Z, CZ, RZ) and measurements
    // These are gates that can be represented in an Ising/QUBO formulation
    for (size_t i = 0; i < circuit->num_gates; i++) {
        HardwareGate* gate = &circuit->gates[i];
        switch (gate->type) {
            case GATE_TYPE_Z:
            case GATE_TYPE_CZ:
            case GATE_TYPE_RZ:
            case GATE_TYPE_I:
                // Identity gate is compatible with QUBO
                continue;
            case GATE_TYPE_H:
            case GATE_TYPE_X:
            case GATE_TYPE_Y:
            case GATE_TYPE_RX:
            case GATE_TYPE_RY:
            case GATE_TYPE_CNOT:
            case GATE_TYPE_SWAP:
            default:
                // Non-diagonal gates cannot be directly represented in QUBO
                return false;
        }
    }
    return true;
}

/**
 * @brief Convert circuit to QUBO problem for D-Wave
 *
 * Extracts the Ising coefficients from a quantum circuit containing
 * diagonal gates (Z, CZ, RZ) and converts them to QUBO format.
 */
QUBO* circuit_to_qubo(const struct QuantumCircuit* circuit) {
    if (!circuit) return NULL;

    QUBO* qubo = calloc(1, sizeof(QUBO));
    if (!qubo) return NULL;

    qubo->num_variables = circuit->num_qubits;

    // Allocate linear terms (bias)
    qubo->linear = calloc(qubo->num_variables, sizeof(double));
    if (!qubo->linear) {
        free(qubo);
        return NULL;
    }

    // Count quadratic terms first to allocate proper size
    size_t max_couplings = qubo->num_variables * (qubo->num_variables - 1) / 2;
    qubo->quadratic = calloc(max_couplings, sizeof(double));
    qubo->variable_indices = calloc(max_couplings * 2, sizeof(uint32_t));
    if (!qubo->quadratic || !qubo->variable_indices) {
        cleanup_qubo(qubo);
        return NULL;
    }

    // Extract Ising coefficients from circuit gates
    size_t coupling_idx = 0;
    for (size_t i = 0; i < circuit->num_gates; i++) {
        HardwareGate* gate = &circuit->gates[i];

        switch (gate->type) {
            case GATE_TYPE_Z:
                // Z gate contributes to linear term
                if (gate->target < qubo->num_variables) {
                    double coeff = (gate->parameter != 0) ? gate->parameter : 1.0;
                    qubo->linear[gate->target] += coeff;
                }
                break;

            case GATE_TYPE_CZ:
                // CZ contributes to quadratic term
                if (gate->control < qubo->num_variables &&
                    gate->target < qubo->num_variables &&
                    coupling_idx < max_couplings) {
                    qubo->variable_indices[coupling_idx * 2] = (uint32_t)gate->control;
                    qubo->variable_indices[coupling_idx * 2 + 1] = (uint32_t)gate->target;
                    qubo->quadratic[coupling_idx] = 1.0;
                    coupling_idx++;
                }
                break;

            case GATE_TYPE_RZ:
                // RZ contributes to linear term with angle/pi coefficient
                if (gate->target < qubo->num_variables) {
                    qubo->linear[gate->target] += gate->parameter / M_PI;
                }
                break;

            default:
                break;
        }
    }

    qubo->num_couplings = coupling_idx;
    return qubo;
}

/**
 * @brief Convert quantum program to QUBO
 *
 * Converts a QuantumProgram's operations to QUBO format for D-Wave execution.
 */
QUBO* program_to_qubo(const struct QuantumProgram* program) {
    if (!program) return NULL;

    QUBO* qubo = calloc(1, sizeof(QUBO));
    if (!qubo) return NULL;

    qubo->num_variables = program->num_qubits;

    // Allocate linear terms
    qubo->linear = calloc(qubo->num_variables, sizeof(double));
    if (!qubo->linear) {
        free(qubo);
        return NULL;
    }

    // Allocate quadratic terms (max possible)
    size_t max_couplings = qubo->num_variables * (qubo->num_variables - 1) / 2;
    qubo->quadratic = calloc(max_couplings, sizeof(double));
    qubo->variable_indices = calloc(max_couplings * 2, sizeof(uint32_t));
    if (!qubo->quadratic || !qubo->variable_indices) {
        cleanup_qubo(qubo);
        return NULL;
    }

    // Convert program operations to QUBO coefficients
    size_t coupling_idx = 0;
    for (size_t i = 0; i < program->num_operations; i++) {
        const QuantumOperation* op = &program->operations[i];

        // Only process gate operations
        if (op->type != OPERATION_GATE) continue;

        switch (op->op.gate.type) {
            case GATE_Z:
                if (op->op.gate.target_qubit < qubo->num_variables) {
                    qubo->linear[op->op.gate.target_qubit] += 1.0;
                }
                break;

            case GATE_CZ:
                if (op->op.gate.control_qubit < qubo->num_variables &&
                    op->op.gate.target_qubit < qubo->num_variables &&
                    coupling_idx < max_couplings) {
                    qubo->variable_indices[coupling_idx * 2] = (uint32_t)op->op.gate.control_qubit;
                    qubo->variable_indices[coupling_idx * 2 + 1] = (uint32_t)op->op.gate.target_qubit;
                    qubo->quadratic[coupling_idx] = 1.0;
                    coupling_idx++;
                }
                break;

            case GATE_RZ:
                if (op->op.gate.target_qubit < qubo->num_variables) {
                    qubo->linear[op->op.gate.target_qubit] += op->op.gate.parameter / M_PI;
                }
                break;

            default:
                break;
        }
    }

    qubo->num_couplings = coupling_idx;
    return qubo;
}

/**
 * @brief Convert QUBO result back to execution result
 *
 * Converts D-Wave QUBO solution format to the unified ExecutionResult format.
 */
void convert_qubo_result(const QUBOResult* qubo_result, struct ExecutionResult* result) {
    if (!qubo_result || !result) return;

    result->success = (qubo_result->num_solutions > 0);
    result->num_results = qubo_result->num_solutions;

    if (qubo_result->num_solutions > 0 && qubo_result->solutions) {
        // Convert best solution to measurements
        result->measurements = calloc(qubo_result->num_variables, sizeof(double));
        if (result->measurements) {
            // Solutions are stored as flat array: [sol0_var0, sol0_var1, ..., sol1_var0, ...]
            for (size_t i = 0; i < qubo_result->num_variables; i++) {
                result->measurements[i] = (double)qubo_result->solutions[i];
            }
        }

        // Store energy as inverse fidelity metric (lower energy = higher fidelity)
        if (qubo_result->energies) {
            result->fidelity = 1.0 / (1.0 + fabs(qubo_result->energies[0]));
        }

        // Store timing information
        result->execution_time = qubo_result->timing_total;
    }
}

// ============================================================================
// Hardware Gate Functions
// ============================================================================

/**
 * @brief Insert gate at position in circuit
 */
void hw_insert_gate(QuantumCircuit* circuit, size_t index, HardwareGate gate) {
    if (!circuit || index > circuit->num_gates) return;

    // Grow capacity if needed
    if (circuit->num_gates >= circuit->capacity) {
        size_t new_capacity = circuit->capacity * 2;
        if (new_capacity < 16) new_capacity = 16;

        HardwareGate* new_gates = realloc(circuit->gates, new_capacity * sizeof(HardwareGate));
        if (!new_gates) return;

        circuit->gates = new_gates;
        circuit->capacity = new_capacity;
    }

    // Shift gates after index
    for (size_t i = circuit->num_gates; i > index; i--) {
        circuit->gates[i] = circuit->gates[i - 1];
    }

    circuit->gates[index] = gate;
    circuit->num_gates++;
}

/**
 * @brief Replace gate at position in circuit
 */
void hw_replace_gate(QuantumCircuit* circuit, size_t index, HardwareGate gate) {
    if (!circuit || index >= circuit->num_gates) return;
    circuit->gates[index] = gate;
}

/**
 * @brief Remove gate at position from circuit
 */
void hw_remove_gate(QuantumCircuit* circuit, size_t index) {
    if (!circuit || index >= circuit->num_gates) return;

    // Shift gates after index
    for (size_t i = index; i < circuit->num_gates - 1; i++) {
        circuit->gates[i] = circuit->gates[i + 1];
    }

    circuit->num_gates--;
}

/**
 * @brief Add gate to end of circuit
 */
void hw_add_gate(QuantumCircuit* circuit, HardwareGate gate) {
    if (!circuit) return;
    hw_insert_gate(circuit, circuit->num_gates, gate);
}

// ============================================================================
// Backend Submission Helpers
// ============================================================================

/**
 * @brief Internal IBM job submission
 *
 * Submits a QASM program to IBM Quantum via the Qiskit Runtime API.
 * Uses the backend_specific_config field which contains the IBM API handle.
 */
static bool internal_submit_ibm_job(const struct IBMConfig* config, const char* qasm,
                                    ExecutionResult* result) {
    if (!config || !qasm || !result) return false;

    // Initialize result
    result->success = false;
    result->error_message = NULL;

    // Get the IBM API handle from backend_specific_config
    void* api_handle = config->backend_specific_config;
    if (!api_handle) {
        // If no API handle, try to initialize one with the API key
        if (config->api_key) {
            api_handle = ibm_api_init(config->api_key);
            if (api_handle && config->backend_name) {
                ibm_api_connect_backend(api_handle, config->backend_name);
            }
        }

        if (!api_handle) {
            result->error_message = strdup("IBM API handle not initialized and no API key available");
            return false;
        }
    }

    // Submit job to IBM Quantum
    char* job_id = ibm_api_submit_job(api_handle, qasm);
    if (!job_id) {
        result->error_message = strdup("Failed to submit job to IBM Quantum");
        return false;
    }

    // Poll for job completion with timeout
    const int MAX_POLL_ATTEMPTS = 600;  // 10 minutes at 1 second intervals
    const int POLL_INTERVAL_MS = 1000;
    IBMJobStatus status = IBM_STATUS_QUEUED;

    // Track execution time
    struct timespec exec_start, exec_end;
    clock_gettime(CLOCK_MONOTONIC, &exec_start);

    for (int attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt++) {
        status = ibm_api_get_job_status(api_handle, job_id);

        if (status == IBM_STATUS_COMPLETED) {
            break;
        } else if (status == IBM_STATUS_ERROR || status == IBM_STATUS_CANCELLED) {
            char* error = ibm_api_get_job_error(api_handle, job_id);
            result->error_message = error ? error : strdup("Job failed or cancelled");
            free(job_id);
            return false;
        }

        // Sleep for poll interval
        struct timespec sleep_time = {
            .tv_sec = POLL_INTERVAL_MS / 1000,
            .tv_nsec = (POLL_INTERVAL_MS % 1000) * 1000000
        };
        nanosleep(&sleep_time, NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &exec_end);

    if (status != IBM_STATUS_COMPLETED) {
        result->error_message = strdup("Job timed out waiting for completion");
        free(job_id);
        return false;
    }

    // Get job results
    IBMJobResult* ibm_result = ibm_api_get_job_result(api_handle, job_id);
    if (!ibm_result) {
        result->error_message = strdup("Failed to retrieve job results");
        free(job_id);
        return false;
    }

    // Convert IBM results to ExecutionResult format
    result->success = true;
    result->num_results = ibm_result->num_counts;

    // Calculate execution time from our timing
    result->execution_time = (exec_end.tv_sec - exec_start.tv_sec) +
                             (exec_end.tv_nsec - exec_start.tv_nsec) * 1e-9;

    // Copy probabilities if available
    if (ibm_result->probabilities && ibm_result->num_counts > 0) {
        result->probabilities = malloc(ibm_result->num_counts * sizeof(double));
        if (result->probabilities) {
            memcpy(result->probabilities, ibm_result->probabilities,
                   ibm_result->num_counts * sizeof(double));
        }
    }

    // Copy measurement counts if available
    if (ibm_result->counts && ibm_result->num_counts > 0) {
        result->counts = malloc(ibm_result->num_counts * sizeof(uint64_t));
        if (result->counts) {
            memcpy(result->counts, ibm_result->counts,
                   ibm_result->num_counts * sizeof(uint64_t));
        }
    }

    // Use fidelity and error_rate from the result
    result->fidelity = ibm_result->fidelity > 0 ? ibm_result->fidelity :
                       (1.0 - (ibm_result->error_rate > 0 ? ibm_result->error_rate : 0.01));

    // Clean up
    free(job_id);
    cleanup_ibm_result(ibm_result);

    return true;
}

/**
 * @brief Internal Rigetti job submission
 *
 * Submits a Quil program string directly to the Rigetti backend.
 * Uses the QCS API to execute the program and retrieve results.
 */
static bool internal_submit_rigetti_job(const struct RigettiConfig* config, const char* quil,
                                        ExecutionResult* result) {
    if (!config || !quil || !result) return false;

    // Get the internal backend state which contains the QCS handle
    // The backend_specific_config points to RigettiInternalBackend
    if (!config->backend_specific_config) {
        result->success = false;
        result->error_message = strdup("Rigetti backend not initialized");
        return false;
    }

    // Access the QCS handle from the internal backend structure
    // RigettiInternalBackend has qcs_handle at a known offset
    typedef struct {
        char* api_key;
        char* device_name;
        size_t num_qubits;
        bool is_simulator;
        void* curl;
        void* json_config;
        char error_buffer[256];  // CURL_ERROR_SIZE
        char padding[128];       // ErrorMitigationConfig
        rigetti_qcs_handle_t* qcs_handle;
    } RigettiInternalBackendRef;

    RigettiInternalBackendRef* internal = (RigettiInternalBackendRef*)config->backend_specific_config;
    rigetti_qcs_handle_t* qcs_handle = internal->qcs_handle;

    if (!qcs_handle) {
        result->success = false;
        result->error_message = strdup("QCS handle not available");
        return false;
    }

    // Set up execution options
    qcs_execution_options_t exec_options = {
        .target = QCS_TARGET_QPU,
        .qpu_name = config->backend_name,
        .shots = config->max_shots > 0 ? config->max_shots : 1024,
        .use_quilc = true,
        .use_parametric = false,
        .use_active_reset = true,
        .timeout_seconds = 60
    };

    // If backend name suggests simulator, use QVM
    if (config->backend_name && strstr(config->backend_name, "qvm")) {
        exec_options.target = QCS_TARGET_QVM;
    }

    // Submit the Quil program and wait for results
    qcs_job_result_t qcs_result = {0};
    bool exec_success = qcs_execute_program(qcs_handle, quil, &exec_options, &qcs_result);

    if (!exec_success) {
        result->success = false;
        const char* err = qcs_get_last_error(qcs_handle);
        result->error_message = strdup(err ? err : "Quil program execution failed");
        return false;
    }

    // Convert QCS result to ExecutionResult
    result->success = (qcs_result.status == QCS_JOB_COMPLETED);
    result->num_shots = qcs_result.num_shots;

    // Copy measurement counts
    if (qcs_result.num_outcomes > 0 && qcs_result.counts) {
        result->num_results = qcs_result.num_outcomes;
        result->counts = calloc(qcs_result.num_outcomes, sizeof(uint64_t));
        if (result->counts) {
            memcpy(result->counts, qcs_result.counts, qcs_result.num_outcomes * sizeof(uint64_t));
        }
    }

    // Copy probabilities if available
    if (qcs_result.probabilities && qcs_result.num_outcomes > 0) {
        result->probabilities = calloc(qcs_result.num_outcomes, sizeof(double));
        if (result->probabilities) {
            memcpy(result->probabilities, qcs_result.probabilities, qcs_result.num_outcomes * sizeof(double));
        }
    }

    // Set execution time
    result->execution_time = qcs_result.execution_time;

    // Estimate fidelity from success (no direct fidelity in result)
    result->fidelity = result->success ? 1.0 : 0.0;
    result->error_rate = 0.0;

    // Free QCS result resources
    qcs_free_result(&qcs_result);

    return true;
}

/**
 * @brief Convert QUBO to DWaveProblem format
 *
 * Transforms the unified QUBO structure to D-Wave's internal problem format.
 */
static DWaveProblem* qubo_to_dwave_internal(const QUBO* qubo) {
    if (!qubo) return NULL;

    size_t num_interactions = qubo->num_couplings > 0 ? qubo->num_couplings :
                              qubo->num_variables * (qubo->num_variables - 1) / 2;

    DWaveProblem* dwave = create_dwave_problem(qubo->num_variables, num_interactions);
    if (!dwave) return NULL;

    // Copy linear terms (bias values)
    if (qubo->linear) {
        for (size_t i = 0; i < qubo->num_variables; i++) {
            if (fabs(qubo->linear[i]) > 1e-10) {
                add_linear_term(dwave, i, qubo->linear[i]);
            }
        }
    }

    // Copy quadratic terms using variable indices
    if (qubo->quadratic && qubo->variable_indices) {
        for (size_t k = 0; k < qubo->num_couplings; k++) {
            size_t i = qubo->variable_indices[k * 2];
            size_t j = qubo->variable_indices[k * 2 + 1];
            double coeff = qubo->quadratic[k];
            if (fabs(coeff) > 1e-10) {
                add_quadratic_term(dwave, i, j, coeff);
            }
        }
    }

    dwave->offset = qubo->offset;
    return dwave;
}

/**
 * @brief Submit D-Wave problem
 *
 * Submits a QUBO problem to the D-Wave quantum annealer,
 * polls for completion, and returns the optimization results.
 */
bool submit_dwave_problem(struct DWaveConfig* config, QUBO* qubo, QUBOResult* result) {
    if (!qubo || !config || !result) return false;

    memset(result, 0, sizeof(QUBOResult));
    result->num_variables = qubo->num_variables;

    // Convert QUBO to D-Wave internal format
    DWaveProblem* dwave_problem = qubo_to_dwave_internal(qubo);
    if (!dwave_problem) {
        return false;
    }

    // Configure job with D-Wave sampling parameters
    DWaveJobConfig job_config = {
        .problem = dwave_problem,
        .params = {
            .num_reads = 1000,
            .annealing_time = 20,
            .chain_strength = 0.0,
            .programming_thermalization = 0.0,
            .auto_scale = true,
            .reduce_intersample_correlation = true,
            .readout_thermalization = NULL,
            .custom_params = NULL
        },
        .use_embedding = true,
        .use_error_mitigation = true
    };

    // Submit job to D-Wave
    char* job_id = submit_dwave_job(config, &job_config);
    if (!job_id) {
        cleanup_dwave_problem(dwave_problem);
        return false;
    }

    // Poll for completion with timeout
    const int max_poll_attempts = 120;
    DWaveJobStatus status = DWAVE_STATUS_QUEUED;

    for (int attempt = 0; attempt < max_poll_attempts; attempt++) {
        status = get_dwave_job_status(config, job_id);

        if (status == DWAVE_STATUS_COMPLETED) {
            break;
        } else if (status == DWAVE_STATUS_ERROR || status == DWAVE_STATUS_CANCELLED) {
            free(job_id);
            cleanup_dwave_problem(dwave_problem);
            return false;
        }

        struct timespec ts = {1, 0};
        nanosleep(&ts, NULL);
    }

    if (status != DWAVE_STATUS_COMPLETED) {
        cancel_dwave_job(config, job_id);
        free(job_id);
        cleanup_dwave_problem(dwave_problem);
        return false;
    }

    // Get results from D-Wave
    DWaveJobResult* dwave_result = get_dwave_job_result(config, job_id);
    free(job_id);
    cleanup_dwave_problem(dwave_problem);

    if (!dwave_result) {
        return false;
    }

    // Convert D-Wave result to unified QUBOResult format
    result->num_solutions = dwave_result->num_samples;

    if (dwave_result->num_samples > 0) {
        // Allocate flat solutions array [num_solutions * num_variables]
        size_t total_vars = dwave_result->num_samples * qubo->num_variables;
        result->solutions = calloc(total_vars, sizeof(int));
        result->energies = calloc(dwave_result->num_samples, sizeof(double));
        result->num_occurrences = calloc(dwave_result->num_samples, sizeof(int));

        if (!result->solutions || !result->energies) {
            cleanup_dwave_result(dwave_result);
            return false;
        }

        // Copy solutions, converting Ising (-1,+1) to binary (0,1)
        for (size_t i = 0; i < dwave_result->num_samples; i++) {
            if (dwave_result->samples[i].variables) {
                for (size_t j = 0; j < qubo->num_variables; j++) {
                    int ising_val = dwave_result->samples[i].variables[j];
                    result->solutions[i * qubo->num_variables + j] = (ising_val + 1) / 2;
                }
            }
            result->energies[i] = dwave_result->samples[i].energy;
            if (result->num_occurrences) {
                result->num_occurrences[i] = (int)dwave_result->samples[i].occurrence;
            }
        }

        // Store timing information from timing_info array
        // timing_info[0] = total time, timing_info[1] = QPU access time
        result->timing_total = dwave_result->timing_info[0];
        result->timing_sampling = dwave_result->timing_info[1] > 0 ?
                                  dwave_result->timing_info[1] :
                                  dwave_result->timing_info[0] * 0.8;
    }

    cleanup_dwave_result(dwave_result);
    return true;
}

/**
 * @brief Simulate circuit locally
 *
 * Uses statevector simulation for small circuits to compute
 * measurement probabilities and generate simulated shot counts.
 */
int simulate_circuit(struct QuantumCircuit* circuit, struct ExecutionResult* result) {
    if (!circuit || !result) return -1;

    // Use the quantum simulator
    size_t num_states = 1UL << circuit->num_qubits;
    if (num_states > 65536) num_states = 65536;  // Limit for memory

    // Allocate statevector (complex amplitudes)
    double* state_real = calloc(num_states, sizeof(double));
    double* state_imag = calloc(num_states, sizeof(double));
    result->probabilities = calloc(num_states, sizeof(double));
    result->counts = calloc(num_states, sizeof(uint64_t));

    if (!state_real || !state_imag || !result->probabilities || !result->counts) {
        free(state_real);
        free(state_imag);
        free(result->probabilities);
        free(result->counts);
        result->probabilities = NULL;
        result->counts = NULL;
        return -2;  // Memory allocation error
    }

    // Initialize to |0> state
    state_real[0] = 1.0;
    state_imag[0] = 0.0;

    // Apply gates using statevector simulation
    for (size_t i = 0; i < circuit->num_gates; i++) {
        HardwareGate* gate = &circuit->gates[i];
        uint32_t target = gate->target;
        uint32_t control = gate->control;

        // Apply gate based on type
        switch (gate->type) {
            case GATE_TYPE_X: {
                // Pauli X: swap amplitudes
                size_t mask = 1UL << target;
                for (size_t j = 0; j < num_states; j++) {
                    if ((j & mask) == 0) {
                        size_t k = j | mask;
                        double tr = state_real[j], ti = state_imag[j];
                        state_real[j] = state_real[k];
                        state_imag[j] = state_imag[k];
                        state_real[k] = tr;
                        state_imag[k] = ti;
                    }
                }
                break;
            }
            case GATE_TYPE_Y: {
                // Pauli Y: swap with phase
                size_t mask = 1UL << target;
                for (size_t j = 0; j < num_states; j++) {
                    if ((j & mask) == 0) {
                        size_t k = j | mask;
                        double tr = state_real[j], ti = state_imag[j];
                        // |0> -> i|1>, |1> -> -i|0>
                        state_real[j] = state_imag[k];
                        state_imag[j] = -state_real[k];
                        state_real[k] = -ti;
                        state_imag[k] = tr;
                    }
                }
                break;
            }
            case GATE_TYPE_Z: {
                // Pauli Z: phase flip |1>
                size_t mask = 1UL << target;
                for (size_t j = 0; j < num_states; j++) {
                    if (j & mask) {
                        state_real[j] = -state_real[j];
                        state_imag[j] = -state_imag[j];
                    }
                }
                break;
            }
            case GATE_TYPE_H: {
                // Hadamard
                double inv_sqrt2 = 1.0 / sqrt(2.0);
                size_t mask = 1UL << target;
                for (size_t j = 0; j < num_states; j++) {
                    if ((j & mask) == 0) {
                        size_t k = j | mask;
                        double ar = state_real[j], ai = state_imag[j];
                        double br = state_real[k], bi = state_imag[k];
                        state_real[j] = (ar + br) * inv_sqrt2;
                        state_imag[j] = (ai + bi) * inv_sqrt2;
                        state_real[k] = (ar - br) * inv_sqrt2;
                        state_imag[k] = (ai - bi) * inv_sqrt2;
                    }
                }
                break;
            }
            case GATE_TYPE_RZ: {
                // RZ rotation
                double angle = gate->parameter / 2.0;
                double cos_a = cos(angle), sin_a = sin(angle);
                size_t mask = 1UL << target;
                for (size_t j = 0; j < num_states; j++) {
                    double r = state_real[j], im = state_imag[j];
                    if (j & mask) {
                        // |1> gets e^(i*angle/2)
                        state_real[j] = r * cos_a - im * sin_a;
                        state_imag[j] = r * sin_a + im * cos_a;
                    } else {
                        // |0> gets e^(-i*angle/2)
                        state_real[j] = r * cos_a + im * sin_a;
                        state_imag[j] = -r * sin_a + im * cos_a;
                    }
                }
                break;
            }
            case GATE_TYPE_CNOT: {
                // CNOT: flip target if control is 1
                size_t cmask = 1UL << control;
                size_t tmask = 1UL << target;
                for (size_t j = 0; j < num_states; j++) {
                    if ((j & cmask) && !(j & tmask)) {
                        size_t k = j | tmask;
                        double tr = state_real[j], ti = state_imag[j];
                        state_real[j] = state_real[k];
                        state_imag[j] = state_imag[k];
                        state_real[k] = tr;
                        state_imag[k] = ti;
                    }
                }
                break;
            }
            case GATE_TYPE_CZ: {
                // CZ: phase flip if both qubits are 1
                size_t cmask = 1UL << control;
                size_t tmask = 1UL << target;
                for (size_t j = 0; j < num_states; j++) {
                    if ((j & cmask) && (j & tmask)) {
                        state_real[j] = -state_real[j];
                        state_imag[j] = -state_imag[j];
                    }
                }
                break;
            }
            default:
                // Identity or unsupported gate - no operation
                break;
        }
    }

    // Compute probabilities from amplitudes
    for (size_t i = 0; i < num_states; i++) {
        result->probabilities[i] = state_real[i] * state_real[i] +
                                   state_imag[i] * state_imag[i];
    }

    // Generate shot counts from probabilities
    result->num_shots = 1024;
    for (size_t i = 0; i < num_states; i++) {
        result->counts[i] = (uint64_t)(result->probabilities[i] * result->num_shots + 0.5);
    }

    // Clean up temporary state
    free(state_real);
    free(state_imag);

    result->success = true;
    result->num_results = num_states;
    result->fidelity = 1.0;  // Perfect simulation
    result->error_rate = 0.0;

    return 0;  // Success
}

/**
 * @brief Select appropriate error mitigation strategy
 *
 * Analyzes the circuit characteristics, error rates, and noise model
 * to determine the optimal error mitigation strategy.
 */
ErrorMitigationStrategy select_error_mitigation(const struct QuantumCircuit* circuit,
                                                const ErrorRates* rates,
                                                const struct NoiseModel* noise) {
    ErrorMitigationStrategy strategy = {0};
    strategy.type = MITIGATION_NONE;

    if (!circuit) return strategy;

    // Compute average error from error rates or noise model
    double avg_error = 0.0;
    size_t count = 0;

    if (rates && rates->single_qubit_errors && rates->num_qubits > 0) {
        for (size_t i = 0; i < rates->num_qubits; i++) {
            avg_error += rates->single_qubit_errors[i];
            count++;
        }
        if (rates->two_qubit_errors) {
            for (size_t i = 0; i < rates->num_qubits; i++) {
                avg_error += rates->two_qubit_errors[i];
                count++;
            }
        }
    } else if (noise && noise->gate_errors) {
        // Estimate from noise model - assume first few entries are gate errors
        for (size_t i = 0; i < 10 && noise->gate_errors[i] > 0; i++) {
            avg_error += noise->gate_errors[i];
            count++;
        }
    }

    if (count > 0) {
        avg_error /= count;
    } else {
        avg_error = 0.01;  // Default assumption
    }

    size_t depth = circuit->depth;

    // Select mitigation type based on circuit depth and error rates
    if (avg_error < 0.001) {
        strategy.type = MITIGATION_NONE;
    } else if (depth < 10) {
        strategy.type = MITIGATION_PROBABILISTIC;
    } else if (depth < 50) {
        strategy.type = MITIGATION_ZNE;

        // Set up ZNE scale factors
        strategy.num_amplification = 3;
        strategy.noise_amplification = calloc(3, sizeof(double));
        if (strategy.noise_amplification) {
            strategy.noise_amplification[0] = 1.0;
            strategy.noise_amplification[1] = 2.0;
            strategy.noise_amplification[2] = 3.0;
        }
    } else {
        strategy.type = MITIGATION_RICHARDSON;

        // Set up Richardson extrapolation factors
        strategy.num_amplification = 4;
        strategy.noise_amplification = calloc(4, sizeof(double));
        if (strategy.noise_amplification) {
            strategy.noise_amplification[0] = 1.0;
            strategy.noise_amplification[1] = 1.5;
            strategy.noise_amplification[2] = 2.0;
            strategy.noise_amplification[3] = 3.0;
        }
    }

    return strategy;
}

#ifdef ENABLE_METAL
/**
 * @brief Get Metal GPU context
 */
void* get_metal_context(void) {
    // Return the Metal device context
    // This would be implemented in the Metal backend
    return NULL;  // Placeholder
}
#else
void* get_metal_context(void) {
    return NULL;
}
#endif

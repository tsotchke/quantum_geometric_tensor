#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"
#include "quantum_geometric/hardware/quantum_ibm_backend.h"
#include "quantum_geometric/hardware/quantum_rigetti_backend.h"
#include "quantum_geometric/hardware/quantum_dwave_backend.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
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

// Forward declarations
static bool operation_to_qasm(const QuantumOperation* operation, char* buffer, size_t size);
static bool operation_to_quil(const QuantumOperation* operation, char* buffer, size_t size);
static bool operation_to_json(const QuantumOperation* operation, char* buffer, size_t size);
static bool validate_gate_operation(const QuantumGate* gate);
static bool validate_annealing_schedule(const double* schedule, size_t points);
static QuantumProgram* optimize_for_hardware(const QuantumProgram* program, const struct HardwareState* hw);
static double richardson_extrapolate(double value, const double* scale_factors, size_t num_factors);
static bool apply_zne_mitigation(struct ExecutionResult* result, const struct MitigationParams* params);
static bool apply_probabilistic_mitigation(struct ExecutionResult* result, const struct MitigationParams* params);
static bool apply_custom_mitigation(struct ExecutionResult* result, const void* custom_params);

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
            hw->backend.ibm = init_ibm_backend(&config->config.ibm);
            return hw->backend.ibm != NULL;
        case HARDWARE_RIGETTI:
            hw->backend.rigetti = init_rigetti_backend(&config->config.rigetti);
            return hw->backend.rigetti != NULL;
        case HARDWARE_DWAVE:
            hw->backend.dwave = init_dwave_backend(&config->config.dwave);
            return hw->backend.dwave != NULL;
        case HARDWARE_SIMULATOR:
            hw->backend.simulator = init_simulator(&config->config.simulator);
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
                cleanup_ibm_backend(hw->backend.ibm);
            }
            break;
        case HARDWARE_RIGETTI:
            if (hw->backend.rigetti) {
                cleanup_rigetti_backend(hw->backend.rigetti);
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
    QuantumProgram* optimized_program = program;
    if (program->optimize) {
        optimized_program = optimize_for_hardware(program, hw);
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
    if (program->use_error_mitigation) {
        if (hw->type == HARDWARE_SIMULATOR) {
            apply_simulator_error_mitigation(hw->backend.simulator, &hw->error_mitigation);
        } else {
            apply_error_mitigation(result, &hw->error_mitigation);
        }
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

    free(result->probabilities);
    free(result->counts);
    free(result->error_message);
    free(result->raw_data);
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
            if (stored_type != type) {
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
            success = test_ibm_connection(hw->backend.ibm);
            break;
        case HARDWARE_RIGETTI:
            success = test_rigetti_connection(hw->backend.rigetti);
            break;
        case HARDWARE_DWAVE:
            success = test_dwave_connection(hw->backend.dwave);
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
                case GATE_H:
                    snprintf(buffer, size, "h q[%u];", operation->op.gate.qubit);
                    break;
                case GATE_X:
                    snprintf(buffer, size, "x q[%u];", operation->op.gate.qubit);
                    break;
                case GATE_CNOT:
                    snprintf(buffer, size, "cx q[%u],q[%u];", 
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
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
                case GATE_H:
                    snprintf(buffer, size, "H %u", operation->op.gate.qubit);
                    break;
                case GATE_X:
                    snprintf(buffer, size, "X %u", operation->op.gate.qubit);
                    break;
                case GATE_CNOT:
                    snprintf(buffer, size, "CNOT %u %u",
                            operation->op.gate.control_qubit,
                            operation->op.gate.target_qubit);
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

// Program validation and optimization implementations
static bool validate_program(const struct QuantumProgram* program, const struct HardwareCapabilities* capabilities) {
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

static struct QuantumProgram* optimize_for_hardware(const struct QuantumProgram* program, const struct HardwareState* hw) {
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
            optimize_circuit_depth(optimized);
            optimize_gate_decomposition(optimized, &hw->capabilities);
        }
    }
    
    return optimized;
}

// Backend execution implementations
static bool execute_on_ibm(struct IBMQuantumBackend* backend, const struct QuantumProgram* program, struct ExecutionResult* result) {
    if (!backend || !program || !result) {
        return false;
    }

    // Convert program to QASM for IBM backend
    char* qasm = program_to_qasm(program);
    if (!qasm) {
        return false;
    }

    // Submit job to IBM backend
    bool success = submit_ibm_job(backend, qasm, result);
    free(qasm);
    return success;
}

static bool execute_on_rigetti(struct RigettiBackend* backend, const struct QuantumProgram* program, struct ExecutionResult* result) {
    if (!backend || !program || !result) {
        return false;
    }

    // Convert program to Quil for Rigetti backend
    char* quil = program_to_quil(program);
    if (!quil) {
        return false;
    }

    // Submit job to Rigetti backend
    bool success = submit_rigetti_job(backend, quil, result);
    free(quil);
    return success;
}

static bool execute_on_dwave(struct DWaveBackend* backend, const struct QuantumProgram* program, struct ExecutionResult* result) {
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

static bool execute_on_simulator(struct SimulatorBackend* backend, const struct QuantumProgram* program, struct ExecutionResult* result) {
    if (!backend || !program || !result) {
        return false;
    }

    // Initialize quantum state
    QuantumState* state = init_quantum_state(program->num_qubits);
    if (!state) {
        return false;
    }

    // Execute operations sequentially
    bool success = true;
    for (size_t i = 0; i < program->num_operations && success; i++) {
        success = apply_operation(backend, state, &program->operations[i]);
    }

    if (success) {
        // Get measurement results
        get_measurement_results(state, result);
        
        // Calculate state vector probabilities
        calculate_probabilities(state, result);
        
        // Store raw state vector if requested
        if (program->store_statevector) {
            store_statevector(state, result);
        }
    }

    cleanup_quantum_state(state);
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
        case HARDWARE_METAL:
#ifdef ENABLE_METAL
            // Metal has no crosstalk
            memset(&hardware->crosstalk, 0, sizeof(CrosstalkMap));
#endif
            break;
        case HARDWARE_SIMULATOR:
            // Set no crosstalk
            memset(&hardware->crosstalk, 0,
                  sizeof(CrosstalkMap));
            break;
    }
}

// Submit quantum circuit with comprehensive optimization
int submit_quantum_circuit(struct QuantumHardware* hardware,
                         const struct QuantumCircuit* circuit,
                         struct QuantumResult* result) {
    if (!hardware || !circuit || !result) return -1;
    
    // Validate circuit against hardware constraints
    ValidationResult validation = validate_circuit(
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
    OptimizedCircuit* optimized = optimize_for_hardware(
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
                    optimized->error_mitigation,
                    &hardware->error_rates
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
        case HARDWARE_METAL:
#ifdef ENABLE_METAL
            // Submit to Metal backend
            status = 0; // Metal operations are handled through specific Metal functions
#endif
            break;
        case HARDWARE_SIMULATOR:
            status = simulate_circuit(optimized->circuit,
                                   result);
            break;
    }
    
    cleanup_optimized_circuit(optimized);
    return status;
}

// Validate circuit against hardware constraints
static struct ValidationResult validate_circuit(
    const struct QuantumCircuit* circuit,
    const struct QuantumHardware* hardware) {
    
    ValidationResult result = {
        .is_valid = true,
        .error_message = NULL
    };
    
    // Check number of qubits
    if (circuit->num_qubits > hardware->capabilities.num_qubits) {
        result.is_valid = false;
        result.error_message = "Too many qubits";
        return result;
    }
    
    // Check circuit depth
    size_t depth = compute_circuit_depth(circuit);
    if (depth > hardware->capabilities.max_circuit_depth) {
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
    
    switch (hardware->type) {
        case HARDWARE_RIGETTI:
            cleanup_rigetti_backend(hardware->backend.rigetti);
            cleanup_connectivity(&hardware->connectivity);
            cleanup_noise_model(&hardware->noise_model);
            cleanup_crosstalk(&hardware->crosstalk);
            break;
            
        case HARDWARE_IBM:
            cleanup_ibm_backend(hardware->backend.ibm);
            break;
        case HARDWARE_DWAVE:
            cleanup_dwave_backend(hardware->backend.dwave);
            break;
        case HARDWARE_METAL:
#ifdef ENABLE_METAL
            qg_metal_cleanup();
#endif
            break;
        case HARDWARE_SIMULATOR:
            // Clean up simulator
            break;
    }
    
    free(hardware);
}

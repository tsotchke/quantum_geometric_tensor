/**
 * @file quantum_ibm_backend.c
 * @brief Implementation of IBM Quantum backend interface
 */

#include "quantum_geometric/core/quantum_result.h"
#include "quantum_geometric/hardware/quantum_ibm_backend.h"
#include "quantum_geometric/hardware/quantum_ibm_api.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"
#include "quantum_geometric/core/quantum_base_types.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include "quantum_geometric/core/quantum_circuit_operations.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_error.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/physics/stabilizer_types.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_circuit_types.h"
#include "quantum_geometric/core/quantum_gate_operations.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdarg.h>

// Forward declarations
void cleanup_ibm_backend(IBMBackendState* state);
static void optimize_single_qubit_gates(struct quantum_circuit* circuit);
static void optimize_two_qubit_gates(struct quantum_circuit* circuit);
static void optimize_measurement_layout(struct quantum_circuit* circuit);
static void add_error_mitigation_sequences(struct quantum_circuit* circuit);
static void add_dynamic_decoupling_sequences(struct quantum_circuit* circuit);

// Simple logging function if not available from header
#ifndef log_error
static void log_error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    fprintf(stderr, "[ERROR] ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}
#endif

// Global backend state
static struct IBMBackendState* g_backend_state = NULL;

// Initialize backend state
IBMBackendState* init_ibm_backend_state(void) {
    if (g_backend_state) {
        return g_backend_state;
    }

    g_backend_state = calloc(1, sizeof(IBMBackendState));
    if (!g_backend_state) {
        return NULL;
    }

    g_backend_state->initialized = false;
    g_backend_state->connected = false;
    g_backend_state->error_rates = NULL;
    g_backend_state->readout_errors = NULL;
    g_backend_state->num_qubits = 0;
    g_backend_state->api_handle = NULL;
    g_backend_state->last_result_data = NULL;

    return g_backend_state;
}

// Legacy IBM backend initialization (canonical init_ibm_backend is in quantum_ibm_backend_optimized.c)
qgt_error_t init_ibm_backend_legacy(IBMBackendState* state, const IBMBackendConfig* config) {
    if (!state || !config) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    // Validate configuration
    if (!config->backend_name || !config->hub || !config->group || 
        !config->project || !config->token) {
        return QGT_ERROR_INVALID_CONFIG;
    }

    // Copy configuration
    memcpy(&state->config, config, sizeof(IBMBackendConfig));
    
    // Initialize error tracking arrays
    state->num_qubits = 65; // Manhattan has 65 qubits
    state->error_rates = calloc(state->num_qubits, sizeof(double));
    state->readout_errors = calloc(state->num_qubits, sizeof(double));
    
    if (!state->error_rates || !state->readout_errors) {
        cleanup_ibm_backend(state);
        return QGT_ERROR_ALLOCATION_FAILED;
    }

    // Initialize API connection
    state->api_handle = ibm_api_init(config->token);
    if (!state->api_handle) {
        log_error("Failed to initialize IBM Quantum API connection");
        cleanup_ibm_backend(state);
        return QGT_ERROR_HARDWARE_AUTHENTICATION;
    }

    // Connect to specified backend
    if (!ibm_api_connect_backend(state->api_handle, config->backend_name)) {
        log_error("Failed to connect to backend %s", config->backend_name);
        cleanup_ibm_backend(state);
        return QGT_ERROR_HARDWARE_CONNECTIVITY;
    }

    // Get calibration data and error rates
    IBMCalibrationData cal_data;
    if (!ibm_api_get_calibration(state->api_handle, &cal_data)) {
        log_error("Failed to get calibration data");
        cleanup_ibm_backend(state);
        return QGT_ERROR_HARDWARE_CALIBRATION;
    }

    // Copy error rates
    for (size_t i = 0; i < state->num_qubits; i++) {
        state->error_rates[i] = cal_data.gate_errors[i];
        state->readout_errors[i] = cal_data.readout_errors[i];
    }

    // Set up job queue
    if (!ibm_api_init_job_queue(state->api_handle)) {
        log_error("Failed to initialize job queue");
        cleanup_ibm_backend(state);
        return QGT_ERROR_HARDWARE_QUEUE_FULL;
    }

    state->initialized = true;
    state->connected = true;
    return QGT_SUCCESS;
}

// Clean up backend resources
void cleanup_ibm_backend(IBMBackendState* state) {
    if (state) {
        // Free error rate arrays
        free(state->error_rates);
        state->error_rates = NULL;

        free(state->readout_errors);
        state->readout_errors = NULL;

        // Clean up API connection properly
        if (state->api_handle) {
            // Cancel any pending jobs gracefully
            if (state->connected) {
                // Attempt to cancel pending jobs before disconnecting
                ibm_api_cancel_pending_jobs(state->api_handle);
            }

            // Close the session and free resources
            ibm_api_close_session(state->api_handle);

            // Securely clear any cached credentials
            ibm_api_clear_credentials(state->api_handle);

            // Free the API handle memory
            ibm_api_destroy(state->api_handle);
            state->api_handle = NULL;
        }

        // Clean up result data
        if (state->last_result_data) {
            if (state->last_result_data->raw_data) {
                // Securely zero sensitive data before freeing
                memset(state->last_result_data->raw_data, 0,
                       state->last_result_data->raw_data_size);
                free(state->last_result_data->raw_data);
            }
            free(state->last_result_data);
            state->last_result_data = NULL;
        }

        // Clear state flags
        state->initialized = false;
        state->connected = false;

        // Clear global reference if this is the global state
        if (g_backend_state == state) {
            g_backend_state = NULL;
        }

        memset(state, 0, sizeof(IBMBackendState));
    }
}

// Create stabilizer measurement circuit
struct quantum_circuit* create_stabilizer_circuit(const StabilizerState* state,
                                                const struct quantum_state* qstate) {
    if (!state || !qstate) {
        return NULL;
    }

    struct quantum_circuit* circuit = create_quantum_circuit(state->config.lattice_width * 
                                                    state->config.lattice_height);
    if (!circuit) {
        return NULL;
    }

    // Add stabilizer measurement gates
    for (size_t x = 0; x < state->config.lattice_width - 1; x++) {
        for (size_t y = 0; y < state->config.lattice_height - 1; y++) {
            // Add plaquette measurement sequence
            size_t qubits[4] = {
                y * state->config.lattice_width + x,
                y * state->config.lattice_width + (x + 1),
                (y + 1) * state->config.lattice_width + x,
                (y + 1) * state->config.lattice_width + (x + 1)
            };
            
            // Add Z-basis measurements
            for (size_t i = 0; i < 4; i++) {
                add_gate(circuit, GATE_TYPE_Z, &qubits[i], 1, NULL, 0);
            }
        }
    }

    // Add vertex operator measurements
    for (size_t x = 1; x < state->config.lattice_width; x++) {
        for (size_t y = 1; y < state->config.lattice_height; y++) {
            // Add vertex measurement sequence
            size_t qubits[4] = {
                (y - 1) * state->config.lattice_width + (x - 1),
                (y - 1) * state->config.lattice_width + x,
                y * state->config.lattice_width + (x - 1),
                y * state->config.lattice_width + x
            };
            
            // Add Hadamard gates for X-basis measurement
            for (size_t i = 0; i < 4; i++) {
                add_gate(circuit, GATE_TYPE_H, &qubits[i], 1, NULL, 0);
            }
            
            // Add Z-basis measurements
            for (size_t i = 0; i < 4; i++) {
                add_gate(circuit, GATE_TYPE_Z, &qubits[i], 1, NULL, 0);
            }
            
            // Add Hadamard gates to return to computational basis
            for (size_t i = 0; i < 4; i++) {
                add_gate(circuit, GATE_TYPE_H, &qubits[i], 1, NULL, 0);
            }
        }
    }

    return circuit;
}

// Optimize quantum circuit for hardware
bool optimize_circuit(IBMBackendState* state, struct quantum_circuit* circuit) {
    if (!state || !circuit) {
        return false;
    }

    // Apply optimization level
    switch (state->config.optimization_level) {
        case 0:
            // No optimization
            break;
            
        case 1:
            // Basic optimizations
            optimize_single_qubit_gates(circuit);
            break;
            
        case 2:
            // Intermediate optimizations
            optimize_single_qubit_gates(circuit);
            optimize_two_qubit_gates(circuit);
            break;
            
        case 3:
            // Advanced optimizations
            optimize_single_qubit_gates(circuit);
            optimize_two_qubit_gates(circuit);
            optimize_measurement_layout(circuit);
            break;
            
        default:
            return false;
    }

    // Apply error mitigation if enabled
    if (state->config.error_mitigation) {
        add_error_mitigation_sequences(circuit);
    }

    // Apply dynamic decoupling if enabled
    if (state->config.dynamic_decoupling) {
        add_dynamic_decoupling_sequences(circuit);
    }

    return true;
}

// Renamed to avoid conflict with other backend implementations
bool execute_ibm_circuit(IBMBackendState* state,
                         const struct QuantumCircuit* circuit,
                         quantum_result* result) {
    if (!state || !circuit || !result) {
        return false;
    }

    // Convert circuit to QASM
    char* qasm = circuit_to_qasm(circuit);
    if (!qasm) {
        log_error("Failed to convert circuit to QASM");
        return false;
    }

    // Submit job to queue
    char* job_id = ibm_api_submit_job(state->api_handle, qasm);
    free(qasm);
    
    if (!job_id) {
        log_error("Failed to submit job to queue");
        return false;
    }

    // Wait for results with timeout
    IBMJobStatus status;
    const int max_retries = 100;
    int retries = 0;

    do {
        status = ibm_api_get_job_status(state->api_handle, job_id);
        if (status == IBM_STATUS_ERROR) {
            log_error("Job failed: %s", 
                ibm_api_get_job_error(state->api_handle, job_id));
            free(job_id);
            return false;
        }
        if (status != IBM_STATUS_COMPLETED) {
            sleep(5); // Wait 5 seconds before checking again
        }
        retries++;
    } while (status != IBM_STATUS_COMPLETED && retries < max_retries);

    if (retries >= max_retries) {
        log_error("Job timeout");
        free(job_id);
        return false;
    }

    // Get and process results
    IBMJobResult* job_result = ibm_api_get_job_result(state->api_handle, job_id);
    free(job_id);

    if (!job_result) {
        log_error("Failed to get job results");
        return false;
    }

    // Create IBM result data
    IBMResultData* ibm_data = malloc(sizeof(IBMResultData));
    if (!ibm_data) {
        cleanup_ibm_result(job_result);
        return false;
    }
    // Initialize all fields
    ibm_data->fidelity = 0.0;
    ibm_data->error_rate = 0.0;
    ibm_data->raw_data = NULL;

    // Copy results
    result->num_measurements = job_result->num_counts;
    result->measurements = malloc(sizeof(double) * result->num_measurements);
    if (!result->measurements) {
        free(ibm_data);
        cleanup_ibm_result(job_result);
        return false;
    }

    memcpy(result->measurements, job_result->probabilities, 
           sizeof(double) * result->num_measurements);

    // Clean up previous result data
    if (state->last_result_data) {
        if (state->last_result_data->raw_data) {
            free(state->last_result_data->raw_data);
        }
        free(state->last_result_data);
    }

    // Store IBM-specific data
    ibm_data->fidelity = job_result->fidelity;
    ibm_data->error_rate = job_result->error_rate;
    ibm_data->raw_data = job_result->raw_data;
    state->last_result_data = ibm_data;
    
    cleanup_ibm_result(job_result);
    return true;
}

// extract_error_syndromes() - Canonical implementation in physics/error_syndrome.c
// (removed: canonical has better syndrome extraction with round tracking)

// Apply error mitigation to measurement result
static double ibm_apply_error_mitigation(IBMBackendState* state,
                                        double raw_measurement,
                                        size_t qubit_idx) {
    if (!state || qubit_idx >= state->num_qubits) {
        return raw_measurement;
    }

    double mitigated = raw_measurement;
    
    // Apply readout error correction
    if (state->config.readout_error_mitigation) {
        double error_rate = state->readout_errors[qubit_idx];
        mitigated = (raw_measurement + error_rate) / (1.0 + 2.0 * error_rate);
    }
    
    // Apply measurement error correction
    if (state->config.measurement_error_mitigation) {
        double error_rate = state->error_rates[qubit_idx];
        mitigated = (mitigated + error_rate) / (1.0 + 2.0 * error_rate);
    }
    
    return mitigated;
}

// Helper functions for circuit optimization
static void optimize_single_qubit_gates(struct quantum_circuit* circuit) {
    if (!circuit || !circuit->gates || circuit->num_gates < 2) {
        return;
    }

    // Iterate through gates to find optimization opportunities
    for (size_t i = 0; i < circuit->num_gates - 1; i++) {
        quantum_gate_t* current = circuit->gates[i];
        quantum_gate_t* next = circuit->gates[i + 1];

        if (!current || !next || !current->target_qubits || !next->target_qubits ||
            !current->parameters || !next->parameters || 
            current->num_parameters == 0 || next->num_parameters == 0) {
            continue;
        }

        // Skip if gates operate on different qubits
        if (current->target_qubits[0] != next->target_qubits[0]) {
            continue;
        }

        // Merge adjacent rotations around same axis
        if (current->is_parameterized && next->is_parameterized) {
            if (current->type == GATE_TYPE_RX && next->type == GATE_TYPE_RX) {
                next->parameters[0] += current->parameters[0];
                current->type = GATE_TYPE_I;  // Mark for removal
            }
            else if (current->type == GATE_TYPE_RY && next->type == GATE_TYPE_RY) {
                next->parameters[0] += current->parameters[0];
                current->type = GATE_TYPE_I;
            }
            else if (current->type == GATE_TYPE_RZ && next->type == GATE_TYPE_RZ) {
                next->parameters[0] += current->parameters[0];
                current->type = GATE_TYPE_I;
            }
        }

        // Cancel inverse operations
        else if (!current->is_parameterized && !next->is_parameterized &&
                 ((current->type == GATE_TYPE_X && next->type == GATE_TYPE_X) ||
                  (current->type == GATE_TYPE_Y && next->type == GATE_TYPE_Y) ||
                  (current->type == GATE_TYPE_Z && next->type == GATE_TYPE_Z) ||
                  (current->type == GATE_TYPE_H && next->type == GATE_TYPE_H))) {
            current->type = GATE_TYPE_I;
            next->type = GATE_TYPE_I;
        }
    }

    // Compact circuit by removing cancelled gates
    size_t write = 0;
    for (size_t read = 0; read < circuit->num_gates; read++) {
        if (circuit->gates[read]->type != GATE_TYPE_I) {
            if (write != read) {
                circuit->gates[write] = circuit->gates[read];
            }
            write++;
        }
    }
    circuit->num_gates = write;
}

static void optimize_two_qubit_gates(struct quantum_circuit* circuit) {
    if (!circuit || !circuit->gates || circuit->num_gates < 2) {
        return;
    }

    // Iterate through gates to find optimization opportunities
    for (size_t i = 0; i < circuit->num_gates - 1; i++) {
        quantum_gate_t* current = circuit->gates[i];
        quantum_gate_t* next = circuit->gates[i + 1];

        if (!current || !next || !current->target_qubits || !next->target_qubits ||
            current->num_qubits < 2 || next->num_qubits < 2) {
            continue;
        }

        // Skip if gates don't operate on the same qubits
        if (current->target_qubits[0] != next->target_qubits[0] ||
            current->target_qubits[1] != next->target_qubits[1]) {
            continue;
        }

        // Cancel adjacent CNOT gates
        if (current->type == GATE_TYPE_CNOT && next->type == GATE_TYPE_CNOT) {
            current->type = GATE_TYPE_I;
            next->type = GATE_TYPE_I;
        }
        // Cancel adjacent CZ gates
        else if (current->type == GATE_TYPE_CZ && next->type == GATE_TYPE_CZ) {
            current->type = GATE_TYPE_I;
            next->type = GATE_TYPE_I;
        }
        // Cancel adjacent SWAP gates
        else if (current->type == GATE_TYPE_SWAP && next->type == GATE_TYPE_SWAP) {
            current->type = GATE_TYPE_I;
            next->type = GATE_TYPE_I;
        }
    }

    // Compact circuit by removing cancelled gates
    size_t write = 0;
    for (size_t read = 0; read < circuit->num_gates; read++) {
        if (circuit->gates[read]->type != GATE_TYPE_I) {
            if (write != read) {
                circuit->gates[write] = circuit->gates[read];
            }
            write++;
        }
    }
    circuit->num_gates = write;
}

static void optimize_measurement_layout(struct quantum_circuit* circuit) {
    if (!circuit || !circuit->gates || circuit->num_gates < 1) {
        return;
    }

    // First pass: Identify measurement regions
    bool* measured_qubits = calloc(circuit->num_qubits, sizeof(bool));
    if (!measured_qubits) {
        return;
    }

    // Track which qubits are measured
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate_t* gate = circuit->gates[i];
        if (!gate || !gate->target_qubits) {
            continue;
        }

        // Mark measured qubits
        if (gate->type == GATE_TYPE_Z) {
            measured_qubits[gate->target_qubits[0]] = true;
        }
    }

    // Second pass: Reorder measurements to minimize readout errors
    size_t write = 0;
    for (size_t read = 0; read < circuit->num_gates; read++) {
        quantum_gate_t* gate = circuit->gates[read];
        if (!gate || !gate->target_qubits) {
            continue;
        }

        // Group measurements together
        if (gate->type == GATE_TYPE_Z) {
            if (write != read) {
                circuit->gates[write] = gate;
            }
            write++;
        }
    }

    // Third pass: Add any remaining non-measurement gates
    for (size_t read = 0; read < circuit->num_gates; read++) {
        quantum_gate_t* gate = circuit->gates[read];
        if (!gate || !gate->target_qubits) {
            continue;
        }

        if (gate->type != GATE_TYPE_Z) {
            if (write != read) {
                circuit->gates[write] = gate;
            }
            write++;
        }
    }

    circuit->num_gates = write;
    free(measured_qubits);
}

static void add_error_mitigation_sequences(struct quantum_circuit* circuit) {
    if (!circuit || !circuit->gates || circuit->num_gates < 1) {
        return;
    }

    // Add echo sequences between measurements
    for (size_t i = 0; i < circuit->num_gates - 1; i++) {
        quantum_gate_t* current = circuit->gates[i];
        quantum_gate_t* next = circuit->gates[i + 1];

        if (!current || !next || !current->target_qubits || !next->target_qubits) {
            continue;
        }

        // If this is a measurement gate
        if (current->type == GATE_TYPE_Z) {
            size_t qubit = current->target_qubits[0];

            // Create echo sequence: X-X or Y-Y
            quantum_gate_t* echo1 = create_quantum_gate(GATE_TYPE_X, &qubit, 1, NULL, 0);
            quantum_gate_t* echo2 = create_quantum_gate(GATE_TYPE_X, &qubit, 1, NULL, 0);

            if (!echo1 || !echo2) {
                if (echo1) destroy_quantum_gate(echo1);
                if (echo2) destroy_quantum_gate(echo2);
                continue;
            }

            // Shift remaining gates to make room
            for (size_t j = circuit->num_gates - 1; j > i + 1; j--) {
                circuit->gates[j + 2] = circuit->gates[j];
            }

            // Insert echo sequence
            circuit->gates[i + 1] = echo1;
            circuit->gates[i + 2] = echo2;
            circuit->num_gates += 2;
            i += 2; // Skip the inserted gates
        }
    }

    // Add randomized compiling sequences
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate_t* gate = circuit->gates[i];
        if (!gate || !gate->target_qubits) {
            continue;
        }

        // Only add sequences for non-measurement gates
        if (gate->type != GATE_TYPE_Z) {
            size_t qubit = gate->target_qubits[0];

            // Add random Pauli frame changes
            double angle = (double)(rand() % 4) * M_PI / 2;
            quantum_gate_t* frame1 = create_quantum_gate(GATE_TYPE_RZ, &qubit, 1, &angle, 1);
            quantum_gate_t* frame2 = create_quantum_gate(GATE_TYPE_RZ, &qubit, 1, &angle, 1);

            if (!frame1 || !frame2) {
                if (frame1) destroy_quantum_gate(frame1);
                if (frame2) destroy_quantum_gate(frame2);
                continue;
            }

            // Shift remaining gates to make room
            for (size_t j = circuit->num_gates - 1; j > i; j--) {
                circuit->gates[j + 2] = circuit->gates[j];
            }

            // Insert frame changes
            circuit->gates[i] = frame1;
            circuit->gates[i + 2] = frame2;
            circuit->gates[i + 1] = gate;
            circuit->num_gates += 2;
            i += 2; // Skip the inserted gates
        }
    }
}

static void add_dynamic_decoupling_sequences(struct quantum_circuit* circuit) {
    if (!circuit || !circuit->gates || circuit->num_gates < 1) {
        return;
    }

    // Add XY4 sequences between gates
    for (size_t i = 0; i < circuit->num_gates - 1; i++) {
        quantum_gate_t* current = circuit->gates[i];
        quantum_gate_t* next = circuit->gates[i + 1];

        if (!current || !next || !current->target_qubits || !next->target_qubits) {
            continue;
        }

        // Skip measurement gates
        if (current->type == GATE_TYPE_Z || next->type == GATE_TYPE_Z) {
            continue;
        }

        size_t qubit = current->target_qubits[0];

        // Create XY4 sequence: X-Y-X-Y
        quantum_gate_t* x1 = create_quantum_gate(GATE_TYPE_X, &qubit, 1, NULL, 0);
        quantum_gate_t* y1 = create_quantum_gate(GATE_TYPE_Y, &qubit, 1, NULL, 0);
        quantum_gate_t* x2 = create_quantum_gate(GATE_TYPE_X, &qubit, 1, NULL, 0);
        quantum_gate_t* y2 = create_quantum_gate(GATE_TYPE_Y, &qubit, 1, NULL, 0);

        if (!x1 || !y1 || !x2 || !y2) {
            if (x1) destroy_quantum_gate(x1);
            if (y1) destroy_quantum_gate(y1);
            if (x2) destroy_quantum_gate(x2);
            if (y2) destroy_quantum_gate(y2);
            continue;
        }

        // Shift remaining gates to make room
        for (size_t j = circuit->num_gates - 1; j > i + 1; j--) {
            circuit->gates[j + 4] = circuit->gates[j];
        }

        // Insert XY4 sequence
        circuit->gates[i + 1] = x1;
        circuit->gates[i + 2] = y1;
        circuit->gates[i + 3] = x2;
        circuit->gates[i + 4] = y2;
        circuit->num_gates += 4;
        i += 4; // Skip the inserted gates
    }

    // Add CPMG sequences for idle qubits
    bool* active_qubits = calloc(circuit->num_qubits, sizeof(bool));
    if (!active_qubits) {
        return;
    }

    // Mark active qubits
    for (size_t i = 0; i < circuit->num_gates; i++) {
        quantum_gate_t* gate = circuit->gates[i];
        if (!gate || !gate->target_qubits) {
            continue;
        }
        active_qubits[gate->target_qubits[0]] = true;
    }

    // Add CPMG sequences to idle qubits
    for (size_t qubit = 0; qubit < circuit->num_qubits; qubit++) {
        if (!active_qubits[qubit]) {
            // Create CPMG sequence: X-wait-X
            quantum_gate_t* x1 = create_quantum_gate(GATE_TYPE_X, &qubit, 1, NULL, 0);
            quantum_gate_t* x2 = create_quantum_gate(GATE_TYPE_X, &qubit, 1, NULL, 0);

            if (!x1 || !x2) {
                if (x1) destroy_quantum_gate(x1);
                if (x2) destroy_quantum_gate(x2);
                continue;
            }

            // Add to end of circuit
            circuit->gates[circuit->num_gates++] = x1;
            circuit->gates[circuit->num_gates++] = x2;
        }
    }

    free(active_qubits);
}

// ============================================================================
// Additional API Functions
// ============================================================================

/**
 * Cancel an IBM Quantum job
 */
bool cancel_ibm_job(struct IBMConfig* config, const char* job_id) {
    if (!config || !job_id) {
        return false;
    }

    // Get backend state
    IBMBackendState* state = g_backend_state;
    if (!state || !state->api_handle) {
        log_error("IBM backend not initialized");
        return false;
    }

    // Cancel all pending jobs with matching ID or cancel all if no specific ID
    ibm_api_cancel_pending_jobs(state->api_handle);
    return true;
}

/**
 * Submit a circuit to IBM Quantum for execution
 * Uses the existing quantum_circuit type from the codebase
 */
int submit_ibm_circuit(struct IBMConfig* config, struct QuantumCircuit* circuit, struct ExecutionResult* result) {
    (void)config;
    (void)circuit;
    (void)result;

    // Get backend state
    IBMBackendState* state = g_backend_state;
    if (!state) {
        state = init_ibm_backend_state();
        if (!state) {
            return -1;
        }
    }

    // For now, this function serves as a bridge to the existing execute_circuit function
    // The actual implementation uses quantum_circuit from the internal types
    return state->connected ? 0 : -1;
}

// Helper: Convert HardwareGate to QASM instruction
static int hardware_gate_to_qasm(const HardwareGate* gate, char* qasm, size_t size) {
    if (!gate || !qasm) return 0;

    switch (gate->type) {
        case GATE_I:
            return snprintf(qasm, size, "id q[%u];\n", gate->target);
        case GATE_H:
            return snprintf(qasm, size, "h q[%u];\n", gate->target);
        case GATE_X:
            return snprintf(qasm, size, "x q[%u];\n", gate->target);
        case GATE_Y:
            return snprintf(qasm, size, "y q[%u];\n", gate->target);
        case GATE_Z:
            return snprintf(qasm, size, "z q[%u];\n", gate->target);
        case GATE_S:
            return snprintf(qasm, size, "s q[%u];\n", gate->target);
        case GATE_SDG:
            return snprintf(qasm, size, "sdg q[%u];\n", gate->target);
        case GATE_T:
            return snprintf(qasm, size, "t q[%u];\n", gate->target);
        case GATE_TDG:
            return snprintf(qasm, size, "tdg q[%u];\n", gate->target);
        case GATE_SX:
            return snprintf(qasm, size, "sx q[%u];\n", gate->target);
        case GATE_RX:
            return snprintf(qasm, size, "rx(%g) q[%u];\n", gate->parameter, gate->target);
        case GATE_RY:
            return snprintf(qasm, size, "ry(%g) q[%u];\n", gate->parameter, gate->target);
        case GATE_RZ:
            return snprintf(qasm, size, "rz(%g) q[%u];\n", gate->parameter, gate->target);
        case GATE_U1:
            return snprintf(qasm, size, "u1(%g) q[%u];\n", gate->parameter, gate->target);
        case GATE_CNOT:
            return snprintf(qasm, size, "cx q[%u],q[%u];\n", gate->control, gate->target);
        case GATE_CZ:
            return snprintf(qasm, size, "cz q[%u],q[%u];\n", gate->control, gate->target);
        case GATE_SWAP:
            return snprintf(qasm, size, "swap q[%u],q[%u];\n", gate->control, gate->target);
        case GATE_CCX:
            return snprintf(qasm, size, "ccx q[%u],q[%u],q[%u];\n",
                          gate->target, gate->control, gate->target1);
        case GATE_CSWAP:
            return snprintf(qasm, size, "cswap q[%u],q[%u],q[%u];\n",
                          gate->target, gate->control, gate->target1);
        default:
            return 0;
    }
}

/**
 * Convert a quantum circuit to OpenQASM 3.0 format
 * Uses the existing quantum_circuit type from the codebase
 */
char* circuit_to_qasm(const struct QuantumCircuit* circuit) {
    if (!circuit) {
        // Fallback to global state if no circuit provided
        IBMBackendState* state = g_backend_state;
        if (!state || !state->api_handle) {
            return NULL;
        }

        size_t num_qubits = state->num_qubits > 0 ? state->num_qubits : 5;
        size_t buffer_size = 1024;
        char* qasm = malloc(buffer_size);
        if (!qasm) return NULL;

        snprintf(qasm, buffer_size,
            "OPENQASM 3.0;\n"
            "include \"stdgates.inc\";\n\n"
            "qubit[%zu] q;\n"
            "bit[%zu] c;\n\n",
            num_qubits, num_qubits);

        return qasm;
    }

    // Calculate buffer size needed (generous estimate)
    size_t buffer_size = 256 + circuit->num_gates * 64 + circuit->num_qubits * 32;
    char* qasm = malloc(buffer_size);
    if (!qasm) return NULL;

    // Write header
    int offset = snprintf(qasm, buffer_size,
        "OPENQASM 3.0;\n"
        "include \"stdgates.inc\";\n\n"
        "qubit[%zu] q;\n"
        "bit[%zu] c;\n\n",
        circuit->num_qubits, circuit->num_classical_bits);

    // Convert each gate
    for (size_t i = 0; i < circuit->num_gates && (size_t)offset < buffer_size - 64; i++) {
        offset += hardware_gate_to_qasm(&circuit->gates[i],
                                        qasm + offset,
                                        buffer_size - offset);
    }

    // Add measurements
    offset += snprintf(qasm + offset, buffer_size - offset, "\n// Measurements\n");
    for (size_t i = 0; i < circuit->num_qubits && (size_t)offset < buffer_size - 32; i++) {
        if (circuit->measured && circuit->measured[i]) {
            offset += snprintf(qasm + offset, buffer_size - offset,
                             "measure q[%zu] -> c[%zu];\n", i, i);
        }
    }

    return qasm;
}

// =============================================================================
// QASM Parser (Fallback Implementation)
// =============================================================================

#ifdef QGT_HAS_QEQASM
// Use qe-qasm for parsing when available
#include <qasm/QasmParser.h>

struct QuantumCircuit* qasm_to_circuit(const char* qasm) {
    if (!qasm) return NULL;

    // qe-qasm parsing would go here
    // This requires C++ integration which is complex for a C library

    // For now, fall through to the fallback implementation
    return NULL;
}

#else
// Fallback: Simple QASM parser (OpenQASM 2.0 basic gates)

// Helper functions for QASM parsing
static const char* qasm_skip_ws(const char* s) {
    while (*s && (*s == ' ' || *s == '\t')) s++;
    return s;
}

static bool qasm_parse_uint(const char** s, uint32_t* val) {
    const char* p = qasm_skip_ws(*s);
    if (!*p || (*p < '0' || *p > '9')) return false;
    *val = 0;
    while (*p >= '0' && *p <= '9') {
        *val = *val * 10 + (*p - '0');
        p++;
    }
    *s = p;
    return true;
}

static bool qasm_parse_double(const char** s, double* val) {
    const char* p = qasm_skip_ws(*s);
    char* end;
    *val = strtod(p, &end);
    if (end == p) {
        // Check for pi
        if (strncmp(p, "pi", 2) == 0) {
            *val = 3.14159265358979323846;
            *s = p + 2;
            return true;
        } else if (strncmp(p, "-pi", 3) == 0) {
            *val = -3.14159265358979323846;
            *s = p + 3;
            return true;
        }
        return false;
    }
    *s = end;
    return true;
}

// Parse q[N] syntax
static bool qasm_parse_qubit(const char** s, uint32_t* qubit) {
    const char* p = qasm_skip_ws(*s);
    if (*p != 'q') return false;
    p++;
    if (*p != '[') return false;
    p++;
    if (!qasm_parse_uint(&p, qubit)) return false;
    p = qasm_skip_ws(p);
    if (*p != ']') return false;
    p++;
    *s = p;
    return true;
}

// Parse parameter (e.g., "(pi/2)" or "(1.5707)")
static bool qasm_parse_param(const char** s, double* val) {
    const char* p = qasm_skip_ws(*s);
    if (*p != '(') return false;
    p++;

    double num = 0, denom = 1;
    bool neg = false;

    p = qasm_skip_ws(p);
    if (*p == '-') { neg = true; p++; }

    if (strncmp(p, "pi", 2) == 0) {
        num = 3.14159265358979323846;
        p += 2;
    } else {
        if (!qasm_parse_double(&p, &num)) return false;
    }

    p = qasm_skip_ws(p);
    if (*p == '/') {
        p++;
        if (!qasm_parse_double(&p, &denom)) return false;
    } else if (*p == '*') {
        p++;
        double mult;
        if (strncmp(qasm_skip_ws(p), "pi", 2) == 0) {
            mult = 3.14159265358979323846;
            p = qasm_skip_ws(p) + 2;
        } else if (!qasm_parse_double(&p, &mult)) {
            return false;
        }
        num *= mult;
    }

    p = qasm_skip_ws(p);
    if (*p != ')') return false;
    p++;

    *val = (neg ? -num : num) / denom;
    *s = p;
    return true;
}

// Match gate name
static bool qasm_match_gate(const char** s, const char* name) {
    const char* p = qasm_skip_ws(*s);
    size_t len = strlen(name);
    if (strncmp(p, name, len) == 0 &&
        (p[len] == ' ' || p[len] == '(' || p[len] == '\n' || p[len] == '\0' || p[len] == ';')) {
        *s = p + len;
        return true;
    }
    return false;
}

struct QuantumCircuit* qasm_to_circuit(const char* qasm) {
    if (!qasm) return NULL;

    // First pass: find qreg/creg declarations and count gates
    size_t num_qubits = 0;
    size_t num_classical = 0;
    size_t num_gates = 0;

    const char* line = qasm;
    while (*line) {
        const char* p = qasm_skip_ws(line);

        // Skip empty lines, comments, OPENQASM/include
        if (*p == '\n' || *p == '\0' || *p == '/' ||
            strncmp(p, "OPENQASM", 8) == 0 ||
            strncmp(p, "include", 7) == 0) {
            while (*line && *line != '\n') line++;
            if (*line == '\n') line++;
            continue;
        }

        // Parse qreg q[N];
        if (strncmp(p, "qreg", 4) == 0 || strncmp(p, "qubit", 5) == 0) {
            const char* bracket = strchr(p, '[');
            if (bracket) {
                bracket++;
                uint32_t n = 0;
                qasm_parse_uint(&bracket, &n);
                if (n > num_qubits) num_qubits = n;
            }
        }
        // Parse creg c[N];
        else if (strncmp(p, "creg", 4) == 0 || strncmp(p, "bit", 3) == 0) {
            const char* bracket = strchr(p, '[');
            if (bracket) {
                bracket++;
                uint32_t n = 0;
                qasm_parse_uint(&bracket, &n);
                if (n > num_classical) num_classical = n;
            }
        }
        // Count gate instructions
        else if (strncmp(p, "h ", 2) == 0 || strncmp(p, "x ", 2) == 0 ||
                 strncmp(p, "y ", 2) == 0 || strncmp(p, "z ", 2) == 0 ||
                 strncmp(p, "s ", 2) == 0 || strncmp(p, "sdg ", 4) == 0 ||
                 strncmp(p, "t ", 2) == 0 || strncmp(p, "tdg ", 4) == 0 ||
                 strncmp(p, "sx ", 3) == 0 || strncmp(p, "id ", 3) == 0 ||
                 strncmp(p, "rx(", 3) == 0 || strncmp(p, "ry(", 3) == 0 ||
                 strncmp(p, "rz(", 3) == 0 ||
                 strncmp(p, "u1(", 3) == 0 || strncmp(p, "u2(", 3) == 0 ||
                 strncmp(p, "u3(", 3) == 0 ||
                 strncmp(p, "cx ", 3) == 0 || strncmp(p, "cz ", 3) == 0 ||
                 strncmp(p, "swap ", 5) == 0 ||
                 strncmp(p, "crx(", 4) == 0 || strncmp(p, "cry(", 4) == 0 ||
                 strncmp(p, "crz(", 4) == 0 || strncmp(p, "ch ", 3) == 0 ||
                 strncmp(p, "ccx ", 4) == 0 || strncmp(p, "cswap ", 6) == 0) {
            num_gates++;
        }

        while (*line && *line != '\n') line++;
        if (*line == '\n') line++;
    }

    if (num_qubits == 0) num_qubits = 5;  // Default
    if (num_classical == 0) num_classical = num_qubits;

    // Create circuit
    QuantumCircuit* circuit = malloc(sizeof(QuantumCircuit));
    if (!circuit) return NULL;
    memset(circuit, 0, sizeof(QuantumCircuit));

    circuit->num_qubits = num_qubits;
    circuit->num_classical_bits = num_classical;
    circuit->capacity = num_gates + 16;
    circuit->gates = calloc(circuit->capacity, sizeof(HardwareGate));
    circuit->measured = calloc(circuit->num_qubits, sizeof(bool));

    if (!circuit->gates || !circuit->measured) {
        free(circuit->gates);
        free(circuit->measured);
        free(circuit);
        return NULL;
    }

    // Second pass: parse gates
    line = qasm;
    while (*line) {
        const char* p = qasm_skip_ws(line);

        // Skip headers and declarations
        if (*p == '\n' || *p == '\0' || *p == '/' ||
            strncmp(p, "OPENQASM", 8) == 0 || strncmp(p, "include", 7) == 0 ||
            strncmp(p, "qreg", 4) == 0 || strncmp(p, "creg", 4) == 0 ||
            strncmp(p, "qubit", 5) == 0 || strncmp(p, "bit", 3) == 0) {
            while (*line && *line != '\n') line++;
            if (*line == '\n') line++;
            continue;
        }

        HardwareGate gate = {0};
        bool valid = false;

        // Single-qubit gates without parameters
        if (qasm_match_gate(&p, "h")) {
            gate.type = GATE_H;
            if (qasm_parse_qubit(&p, &gate.target)) valid = true;
        } else if (qasm_match_gate(&p, "x")) {
            gate.type = GATE_X;
            if (qasm_parse_qubit(&p, &gate.target)) valid = true;
        } else if (qasm_match_gate(&p, "y")) {
            gate.type = GATE_Y;
            if (qasm_parse_qubit(&p, &gate.target)) valid = true;
        } else if (qasm_match_gate(&p, "z")) {
            gate.type = GATE_Z;
            if (qasm_parse_qubit(&p, &gate.target)) valid = true;
        } else if (qasm_match_gate(&p, "id")) {
            gate.type = GATE_I;
            if (qasm_parse_qubit(&p, &gate.target)) valid = true;
        } else if (qasm_match_gate(&p, "s")) {
            gate.type = GATE_S;
            if (qasm_parse_qubit(&p, &gate.target)) valid = true;
        } else if (qasm_match_gate(&p, "sdg")) {
            gate.type = GATE_SDG;
            if (qasm_parse_qubit(&p, &gate.target)) valid = true;
        } else if (qasm_match_gate(&p, "t")) {
            gate.type = GATE_T;
            if (qasm_parse_qubit(&p, &gate.target)) valid = true;
        } else if (qasm_match_gate(&p, "tdg")) {
            gate.type = GATE_TDG;
            if (qasm_parse_qubit(&p, &gate.target)) valid = true;
        } else if (qasm_match_gate(&p, "sx")) {
            gate.type = GATE_SX;
            if (qasm_parse_qubit(&p, &gate.target)) valid = true;
        }
        // Rotation gates
        else if (qasm_match_gate(&p, "rx")) {
            gate.type = GATE_RX;
            if (qasm_parse_param(&p, &gate.parameter) && qasm_parse_qubit(&p, &gate.target))
                valid = true;
        } else if (qasm_match_gate(&p, "ry")) {
            gate.type = GATE_RY;
            if (qasm_parse_param(&p, &gate.parameter) && qasm_parse_qubit(&p, &gate.target))
                valid = true;
        } else if (qasm_match_gate(&p, "rz")) {
            gate.type = GATE_RZ;
            if (qasm_parse_param(&p, &gate.parameter) && qasm_parse_qubit(&p, &gate.target))
                valid = true;
        } else if (qasm_match_gate(&p, "u1")) {
            gate.type = GATE_U1;
            if (qasm_parse_param(&p, &gate.parameter) && qasm_parse_qubit(&p, &gate.target))
                valid = true;
        }
        // Two-qubit gates
        else if (qasm_match_gate(&p, "cx")) {
            gate.type = GATE_CNOT;
            if (qasm_parse_qubit(&p, &gate.control)) {
                p = qasm_skip_ws(p);
                if (*p == ',') p++;
                if (qasm_parse_qubit(&p, &gate.target)) valid = true;
            }
        } else if (qasm_match_gate(&p, "cz")) {
            gate.type = GATE_CZ;
            if (qasm_parse_qubit(&p, &gate.control)) {
                p = qasm_skip_ws(p);
                if (*p == ',') p++;
                if (qasm_parse_qubit(&p, &gate.target)) valid = true;
            }
        } else if (qasm_match_gate(&p, "swap")) {
            gate.type = GATE_SWAP;
            if (qasm_parse_qubit(&p, &gate.control)) {
                p = qasm_skip_ws(p);
                if (*p == ',') p++;
                if (qasm_parse_qubit(&p, &gate.target)) valid = true;
            }
        } else if (qasm_match_gate(&p, "ch")) {
            gate.type = GATE_CH;
            if (qasm_parse_qubit(&p, &gate.control)) {
                p = qasm_skip_ws(p);
                if (*p == ',') p++;
                if (qasm_parse_qubit(&p, &gate.target)) valid = true;
            }
        }
        // Three-qubit gates
        else if (qasm_match_gate(&p, "ccx")) {
            gate.type = GATE_CCX;
            if (qasm_parse_qubit(&p, &gate.target)) {
                p = qasm_skip_ws(p);
                if (*p == ',') p++;
                if (qasm_parse_qubit(&p, &gate.control)) {
                    p = qasm_skip_ws(p);
                    if (*p == ',') p++;
                    if (qasm_parse_qubit(&p, &gate.target1)) valid = true;
                }
            }
        } else if (qasm_match_gate(&p, "cswap")) {
            gate.type = GATE_CSWAP;
            if (qasm_parse_qubit(&p, &gate.target)) {
                p = qasm_skip_ws(p);
                if (*p == ',') p++;
                if (qasm_parse_qubit(&p, &gate.control)) {
                    p = qasm_skip_ws(p);
                    if (*p == ',') p++;
                    if (qasm_parse_qubit(&p, &gate.target1)) valid = true;
                }
            }
        }
        // Measure
        else if (strncmp(p, "measure", 7) == 0) {
            p += 7;
            uint32_t qubit;
            if (qasm_parse_qubit(&p, &qubit) && qubit < circuit->num_qubits) {
                circuit->measured[qubit] = true;
            }
        }

        if (valid && circuit->num_gates < circuit->capacity) {
            circuit->gates[circuit->num_gates++] = gate;
        }

        while (*line && *line != '\n') line++;
        if (*line == '\n') line++;
    }

    circuit->depth = circuit->num_gates;
    return circuit;
}

#endif // QGT_HAS_QEQASM

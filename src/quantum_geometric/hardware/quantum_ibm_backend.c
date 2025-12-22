/**
 * @file quantum_ibm_backend.c
 * @brief Implementation of IBM Quantum backend interface
 */

#include "quantum_geometric/core/quantum_result.h"
#include "quantum_geometric/hardware/quantum_ibm_backend.h"
#include "quantum_geometric/hardware/quantum_ibm_api.h"
#include "quantum_geometric/hardware/quantum_hardware_types.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include "quantum_geometric/core/quantum_circuit_operations.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_error.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/physics/stabilizer_types.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_circuit_types.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

// Initialize IBM backend
qgt_error_t init_ibm_backend(IBMBackendState* state, const IBMBackendConfig* config) {
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

// Execute quantum circuit
bool execute_circuit(IBMBackendState* state,
                    const struct quantum_circuit* circuit,
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

// Extract error syndromes from measurement results
size_t extract_error_syndromes(const quantum_result* result,
                              const SyndromeConfig* config,
                              MatchingGraph* graph) {
    if (!result || !config || !graph) {
        return 0;
    }

    size_t num_syndromes = 0;
    
    // Process measurement results
    for (size_t i = 0; i < result->num_measurements; i++) {
        if (fabs(result->measurements[i] + 1.0) < 1e-6) {
            // Negative measurement indicates error syndrome
            if (i < graph->num_vertices) {
                graph->vertices[i].weight = 1.0;
                num_syndromes++;
            }
        }
    }

    return num_syndromes;
}

// Apply error mitigation to measurement result
double apply_error_mitigation(IBMBackendState* state,
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

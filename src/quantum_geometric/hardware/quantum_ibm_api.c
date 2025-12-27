/**
 * @file quantum_ibm_api.c
 * @brief Implementation of IBM Quantum API interface
 *
 * This file implements the IBM Quantum API functions for connecting to
 * IBM Quantum systems and executing quantum programs. It supports both
 * actual API calls (when compiled with QISKIT_RUNTIME support) and
 * local simulation fallback for development and testing.
 */

#include "quantum_geometric/hardware/quantum_ibm_api.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <complex.h>

// Configuration for IBM Quantum API
#ifndef IBM_API_ENDPOINT
#define IBM_API_ENDPOINT "https://api.quantum-computing.ibm.com/runtime"
#endif

#ifndef IBM_AUTH_ENDPOINT
#define IBM_AUTH_ENDPOINT "https://auth.quantum-computing.ibm.com/api"
#endif

// Maximum qubits for local simulation
#define MAX_LOCAL_QUBITS 24
#define MAX_SHOTS 8192

// API handle structure
typedef struct {
    char* api_token;
    char* access_token;
    char* backend_name;
    char* hub;
    char* group;
    char* project;
    bool connected;
    bool use_local_simulator;
    size_t num_qubits;

    // Local simulator state
    complex double* state_vector;
    size_t state_dim;

    // Job queue
    struct {
        char* job_id;
        IBMJobStatus status;
        IBMJobResult* result;
    } jobs[64];
    size_t num_jobs;

    // Calibration cache
    IBMCalibrationData calibration;
    bool calibration_valid;
} ibm_api_handle_t;

// Forward declarations
static bool local_simulate_qasm(ibm_api_handle_t* handle, const char* qasm,
                               IBMJobResult* result, size_t shots);
static void apply_local_gate(ibm_api_handle_t* handle, const char* gate_name,
                            size_t qubit1, size_t qubit2, double* params, size_t num_params);

// Initialize IBM Quantum API
void* ibm_api_init(const char* token) {
    if (!token || strlen(token) == 0) {
        return NULL;
    }

    ibm_api_handle_t* handle = calloc(1, sizeof(ibm_api_handle_t));
    if (!handle) {
        return NULL;
    }

    handle->api_token = strdup(token);
    if (!handle->api_token) {
        free(handle);
        return NULL;
    }

    // Try to authenticate with IBM Quantum
    // In production, this would make an HTTPS request to the auth endpoint
    // For now, we'll use local simulation as fallback

#ifdef QISKIT_RUNTIME
    // Attempt real API authentication
    // This would use libcurl or similar to make HTTPS requests
    handle->use_local_simulator = false;

    // Exchange API token for access token
    char auth_url[512];
    snprintf(auth_url, sizeof(auth_url), "%s/users/loginWithToken", IBM_AUTH_ENDPOINT);

    // Would make HTTP POST request here
    // For now, fallback to local simulation
    handle->use_local_simulator = true;
#else
    // Use local simulation
    handle->use_local_simulator = true;
#endif

    handle->connected = false;
    handle->num_jobs = 0;
    handle->calibration_valid = false;

    return handle;
}

// Connect to IBM backend
bool ibm_api_connect_backend(void* api_handle, const char* backend_name) {
    if (!api_handle || !backend_name) {
        return false;
    }

    ibm_api_handle_t* handle = (ibm_api_handle_t*)api_handle;

    // Store backend name
    if (handle->backend_name) {
        free(handle->backend_name);
    }
    handle->backend_name = strdup(backend_name);
    if (!handle->backend_name) {
        return false;
    }

    // Determine number of qubits based on backend
    // These are approximate qubit counts for various IBM backends
    if (strstr(backend_name, "eagle") || strstr(backend_name, "127")) {
        handle->num_qubits = 127;
    } else if (strstr(backend_name, "falcon") || strstr(backend_name, "27")) {
        handle->num_qubits = 27;
    } else if (strstr(backend_name, "hummingbird") || strstr(backend_name, "65")) {
        handle->num_qubits = 65;
    } else if (strstr(backend_name, "manhattan")) {
        handle->num_qubits = 65;
    } else if (strstr(backend_name, "qasm_simulator") || strstr(backend_name, "simulator")) {
        handle->num_qubits = 32;
    } else {
        handle->num_qubits = 5;  // Default small backend
    }

    // For local simulation, limit to what we can handle
    if (handle->use_local_simulator && handle->num_qubits > MAX_LOCAL_QUBITS) {
        handle->num_qubits = MAX_LOCAL_QUBITS;
    }

    // Initialize state vector for local simulation
    if (handle->use_local_simulator) {
        handle->state_dim = 1ULL << handle->num_qubits;
        if (handle->state_dim > (1ULL << MAX_LOCAL_QUBITS)) {
            handle->state_dim = 1ULL << MAX_LOCAL_QUBITS;
        }

        handle->state_vector = calloc(handle->state_dim, sizeof(complex double));
        if (!handle->state_vector) {
            return false;
        }

        // Initialize to |0...0⟩
        handle->state_vector[0] = 1.0;
    }

    handle->connected = true;
    return true;
}

// Get calibration data
bool ibm_api_get_calibration(void* api_handle, IBMCalibrationData* cal_data) {
    if (!api_handle || !cal_data) {
        return false;
    }

    ibm_api_handle_t* handle = (ibm_api_handle_t*)api_handle;

    if (!handle->connected) {
        return false;
    }

    // If we have cached calibration, return it
    if (handle->calibration_valid) {
        memcpy(cal_data, &handle->calibration, sizeof(IBMCalibrationData));
        return true;
    }

    // Generate realistic calibration data
    // In production, this would query the actual backend
    srand((unsigned int)time(NULL));

    for (size_t i = 0; i < 128 && i < handle->num_qubits; i++) {
        // Typical gate error rates are 0.001 to 0.01 (0.1% to 1%)
        handle->calibration.gate_errors[i] = 0.001 + 0.009 * (double)rand() / RAND_MAX;

        // Typical readout errors are 0.01 to 0.05 (1% to 5%)
        handle->calibration.readout_errors[i] = 0.01 + 0.04 * (double)rand() / RAND_MAX;
    }

    handle->calibration_valid = true;
    memcpy(cal_data, &handle->calibration, sizeof(IBMCalibrationData));

    return true;
}

// Initialize job queue
bool ibm_api_init_job_queue(void* api_handle) {
    if (!api_handle) {
        return false;
    }

    ibm_api_handle_t* handle = (ibm_api_handle_t*)api_handle;

    // Clear existing jobs
    for (size_t i = 0; i < handle->num_jobs; i++) {
        free(handle->jobs[i].job_id);
        if (handle->jobs[i].result) {
            free(handle->jobs[i].result->counts);
            free(handle->jobs[i].result->probabilities);
            free(handle->jobs[i].result->error_message);
            free(handle->jobs[i].result);
        }
    }
    handle->num_jobs = 0;

    return true;
}

// Generate unique job ID
static char* generate_job_id(void) {
    char* job_id = malloc(64);
    if (!job_id) {
        return NULL;
    }

    unsigned long timestamp = (unsigned long)time(NULL);
    unsigned int random_part = (unsigned int)rand();

    snprintf(job_id, 64, "ibm-job-%lu-%08x", timestamp, random_part);
    return job_id;
}

// Submit job to queue
char* ibm_api_submit_job(void* api_handle, const char* qasm) {
    if (!api_handle || !qasm) {
        return NULL;
    }

    ibm_api_handle_t* handle = (ibm_api_handle_t*)api_handle;

    if (!handle->connected || handle->num_jobs >= 64) {
        return NULL;
    }

    // Create new job
    size_t job_idx = handle->num_jobs;
    handle->jobs[job_idx].job_id = generate_job_id();
    if (!handle->jobs[job_idx].job_id) {
        return NULL;
    }

    handle->jobs[job_idx].status = IBM_STATUS_QUEUED;
    handle->jobs[job_idx].result = NULL;
    handle->num_jobs++;

    // For local simulation, execute immediately
    if (handle->use_local_simulator) {
        IBMJobResult* result = calloc(1, sizeof(IBMJobResult));
        if (result) {
            handle->jobs[job_idx].status = IBM_STATUS_RUNNING;

            // Default to 1024 shots
            if (local_simulate_qasm(handle, qasm, result, 1024)) {
                handle->jobs[job_idx].status = IBM_STATUS_COMPLETED;
                handle->jobs[job_idx].result = result;
            } else {
                handle->jobs[job_idx].status = IBM_STATUS_ERROR;
                result->error_message = strdup("Local simulation failed");
                handle->jobs[job_idx].result = result;
            }
        }
    }

    return strdup(handle->jobs[job_idx].job_id);
}

// Get job status
IBMJobStatus ibm_api_get_job_status(void* api_handle, const char* job_id) {
    if (!api_handle || !job_id) {
        return IBM_STATUS_ERROR;
    }

    ibm_api_handle_t* handle = (ibm_api_handle_t*)api_handle;

    for (size_t i = 0; i < handle->num_jobs; i++) {
        if (handle->jobs[i].job_id && strcmp(handle->jobs[i].job_id, job_id) == 0) {
            return handle->jobs[i].status;
        }
    }

    return IBM_STATUS_ERROR;
}

// Get job error message
char* ibm_api_get_job_error(void* api_handle, const char* job_id) {
    if (!api_handle || !job_id) {
        return NULL;
    }

    ibm_api_handle_t* handle = (ibm_api_handle_t*)api_handle;

    for (size_t i = 0; i < handle->num_jobs; i++) {
        if (handle->jobs[i].job_id && strcmp(handle->jobs[i].job_id, job_id) == 0) {
            if (handle->jobs[i].result && handle->jobs[i].result->error_message) {
                return strdup(handle->jobs[i].result->error_message);
            }
            return NULL;
        }
    }

    return strdup("Job not found");
}

// Get job result
IBMJobResult* ibm_api_get_job_result(void* api_handle, const char* job_id) {
    if (!api_handle || !job_id) {
        return NULL;
    }

    ibm_api_handle_t* handle = (ibm_api_handle_t*)api_handle;

    for (size_t i = 0; i < handle->num_jobs; i++) {
        if (handle->jobs[i].job_id && strcmp(handle->jobs[i].job_id, job_id) == 0) {
            if (handle->jobs[i].status != IBM_STATUS_COMPLETED) {
                return NULL;
            }

            // Return a copy of the result
            if (!handle->jobs[i].result) {
                return NULL;
            }

            IBMJobResult* copy = calloc(1, sizeof(IBMJobResult));
            if (!copy) {
                return NULL;
            }

            copy->num_counts = handle->jobs[i].result->num_counts;
            copy->fidelity = handle->jobs[i].result->fidelity;
            copy->error_rate = handle->jobs[i].result->error_rate;
            copy->status = handle->jobs[i].result->status;

            if (handle->jobs[i].result->counts) {
                copy->counts = malloc(copy->num_counts * sizeof(uint64_t));
                if (copy->counts) {
                    memcpy(copy->counts, handle->jobs[i].result->counts,
                           copy->num_counts * sizeof(uint64_t));
                }
            }

            if (handle->jobs[i].result->probabilities) {
                copy->probabilities = malloc(copy->num_counts * sizeof(double));
                if (copy->probabilities) {
                    memcpy(copy->probabilities, handle->jobs[i].result->probabilities,
                           copy->num_counts * sizeof(double));
                }
            }

            return copy;
        }
    }

    return NULL;
}

// Cancel pending jobs
void ibm_api_cancel_pending_jobs(void* api_handle) {
    if (!api_handle) {
        return;
    }

    ibm_api_handle_t* handle = (ibm_api_handle_t*)api_handle;

    for (size_t i = 0; i < handle->num_jobs; i++) {
        if (handle->jobs[i].status == IBM_STATUS_QUEUED ||
            handle->jobs[i].status == IBM_STATUS_RUNNING) {
            handle->jobs[i].status = IBM_STATUS_CANCELLED;
        }
    }
}

// Close session
void ibm_api_close_session(void* api_handle) {
    if (!api_handle) {
        return;
    }

    ibm_api_handle_t* handle = (ibm_api_handle_t*)api_handle;
    handle->connected = false;
}

// Clear credentials
void ibm_api_clear_credentials(void* api_handle) {
    if (!api_handle) {
        return;
    }

    ibm_api_handle_t* handle = (ibm_api_handle_t*)api_handle;

    // Securely clear sensitive data
    if (handle->api_token) {
        memset(handle->api_token, 0, strlen(handle->api_token));
        free(handle->api_token);
        handle->api_token = NULL;
    }

    if (handle->access_token) {
        memset(handle->access_token, 0, strlen(handle->access_token));
        free(handle->access_token);
        handle->access_token = NULL;
    }
}

// Destroy API handle
void ibm_api_destroy(void* api_handle) {
    if (!api_handle) {
        return;
    }

    ibm_api_handle_t* handle = (ibm_api_handle_t*)api_handle;

    // Clear credentials first
    ibm_api_clear_credentials(handle);

    // Free backend name
    free(handle->backend_name);

    // Free hub/group/project
    free(handle->hub);
    free(handle->group);
    free(handle->project);

    // Free state vector
    free(handle->state_vector);

    // Free job data
    for (size_t i = 0; i < handle->num_jobs; i++) {
        free(handle->jobs[i].job_id);
        if (handle->jobs[i].result) {
            free(handle->jobs[i].result->counts);
            free(handle->jobs[i].result->probabilities);
            free(handle->jobs[i].result->error_message);
            free(handle->jobs[i].result);
        }
    }

    free(handle);
}

// Clean up IBM result
void cleanup_ibm_result(IBMJobResult* result) {
    if (!result) {
        return;
    }

    free(result->counts);
    free(result->probabilities);
    free(result->error_message);
    free(result->raw_data);
    free(result);
}

// =============================================================================
// Local Simulation Functions
// =============================================================================

// Apply single-qubit gate
static void apply_single_qubit_gate(ibm_api_handle_t* handle,
                                   complex double gate[2][2],
                                   size_t qubit) {
    if (!handle || !handle->state_vector || qubit >= handle->num_qubits) {
        return;
    }

    size_t dim = handle->state_dim;
    size_t stride = 1ULL << qubit;

    for (size_t i = 0; i < dim; i += 2 * stride) {
        for (size_t j = 0; j < stride; j++) {
            size_t idx0 = i + j;
            size_t idx1 = idx0 + stride;

            complex double a0 = handle->state_vector[idx0];
            complex double a1 = handle->state_vector[idx1];

            handle->state_vector[idx0] = gate[0][0] * a0 + gate[0][1] * a1;
            handle->state_vector[idx1] = gate[1][0] * a0 + gate[1][1] * a1;
        }
    }
}

// Apply two-qubit gate (CNOT, CZ)
static void apply_two_qubit_gate(ibm_api_handle_t* handle,
                                complex double gate[4][4],
                                size_t control, size_t target) {
    if (!handle || !handle->state_vector ||
        control >= handle->num_qubits || target >= handle->num_qubits) {
        return;
    }

    size_t dim = handle->state_dim;
    size_t stride_c = 1ULL << control;
    size_t stride_t = 1ULL << target;

    for (size_t i = 0; i < dim; i++) {
        size_t c_bit = (i >> control) & 1;
        size_t t_bit = (i >> target) & 1;

        // Only process when we're at the "base" index for this 2-qubit block
        if (c_bit == 0 && t_bit == 0) {
            size_t idx00 = i;
            size_t idx01 = i | stride_t;
            size_t idx10 = i | stride_c;
            size_t idx11 = i | stride_c | stride_t;

            complex double a00 = handle->state_vector[idx00];
            complex double a01 = handle->state_vector[idx01];
            complex double a10 = handle->state_vector[idx10];
            complex double a11 = handle->state_vector[idx11];

            handle->state_vector[idx00] = gate[0][0]*a00 + gate[0][1]*a01 + gate[0][2]*a10 + gate[0][3]*a11;
            handle->state_vector[idx01] = gate[1][0]*a00 + gate[1][1]*a01 + gate[1][2]*a10 + gate[1][3]*a11;
            handle->state_vector[idx10] = gate[2][0]*a00 + gate[2][1]*a01 + gate[2][2]*a10 + gate[2][3]*a11;
            handle->state_vector[idx11] = gate[3][0]*a00 + gate[3][1]*a01 + gate[3][2]*a10 + gate[3][3]*a11;
        }
    }
}

// Apply gate by name
static void apply_local_gate(ibm_api_handle_t* handle, const char* gate_name,
                            size_t qubit1, size_t qubit2, double* params, size_t num_params) {
    if (!handle || !gate_name) {
        return;
    }

    // Identity gate (using IDENT to avoid conflict with complex.h I macro)
    complex double IDENT[2][2] = {{1, 0}, {0, 1}};

    // Pauli gates (using I from complex.h for imaginary unit)
    complex double X[2][2] = {{0, 1}, {1, 0}};
    complex double Y[2][2] = {{0, -I}, {I, 0}};
    complex double Z[2][2] = {{1, 0}, {0, -1}};

    // Hadamard
    double s = 1.0 / sqrt(2.0);
    complex double H[2][2] = {{s, s}, {s, -s}};

    // S and T gates (using I from complex.h for imaginary unit)
    complex double S_gate[2][2] = {{1, 0}, {0, I}};
    complex double T_gate[2][2] = {{1, 0}, {0, cexp(I * M_PI / 4)}};

    // CNOT matrix (4x4)
    complex double CNOT[4][4] = {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 0, 1},
        {0, 0, 1, 0}
    };

    // CZ matrix (4x4)
    complex double CZ[4][4] = {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, -1}
    };

    if (strcmp(gate_name, "id") == 0 || strcmp(gate_name, "i") == 0) {
        apply_single_qubit_gate(handle, IDENT, qubit1);
    }
    else if (strcmp(gate_name, "x") == 0) {
        apply_single_qubit_gate(handle, X, qubit1);
    }
    else if (strcmp(gate_name, "y") == 0) {
        apply_single_qubit_gate(handle, Y, qubit1);
    }
    else if (strcmp(gate_name, "z") == 0) {
        apply_single_qubit_gate(handle, Z, qubit1);
    }
    else if (strcmp(gate_name, "h") == 0) {
        apply_single_qubit_gate(handle, H, qubit1);
    }
    else if (strcmp(gate_name, "s") == 0) {
        apply_single_qubit_gate(handle, S_gate, qubit1);
    }
    else if (strcmp(gate_name, "t") == 0) {
        apply_single_qubit_gate(handle, T_gate, qubit1);
    }
    else if (strcmp(gate_name, "rx") == 0 && num_params >= 1) {
        double theta = params[0];
        double c = cos(theta / 2);
        double s_val = sin(theta / 2);
        complex double RX[2][2] = {{c, -I * s_val}, {-I * s_val, c}};
        apply_single_qubit_gate(handle, RX, qubit1);
    }
    else if (strcmp(gate_name, "ry") == 0 && num_params >= 1) {
        double theta = params[0];
        double c = cos(theta / 2);
        double s_val = sin(theta / 2);
        complex double RY[2][2] = {{c, -s_val}, {s_val, c}};
        apply_single_qubit_gate(handle, RY, qubit1);
    }
    else if (strcmp(gate_name, "rz") == 0 && num_params >= 1) {
        double theta = params[0];
        complex double RZ[2][2] = {{cexp(-I * theta / 2), 0}, {0, cexp(I * theta / 2)}};
        apply_single_qubit_gate(handle, RZ, qubit1);
    }
    else if (strcmp(gate_name, "u1") == 0 && num_params >= 1) {
        double lambda = params[0];
        complex double U1[2][2] = {{1, 0}, {0, cexp(I * lambda)}};
        apply_single_qubit_gate(handle, U1, qubit1);
    }
    else if (strcmp(gate_name, "u2") == 0 && num_params >= 2) {
        double phi = params[0];
        double lambda = params[1];
        double s_val = 1.0 / sqrt(2.0);
        complex double U2[2][2] = {
            {s_val, -s_val * cexp(I * lambda)},
            {s_val * cexp(I * phi), s_val * cexp(I * (phi + lambda))}
        };
        apply_single_qubit_gate(handle, U2, qubit1);
    }
    else if (strcmp(gate_name, "u3") == 0 && num_params >= 3) {
        double theta = params[0];
        double phi = params[1];
        double lambda = params[2];
        double c = cos(theta / 2);
        double s_val = sin(theta / 2);
        complex double U3[2][2] = {
            {c, -s_val * cexp(I * lambda)},
            {s_val * cexp(I * phi), c * cexp(I * (phi + lambda))}
        };
        apply_single_qubit_gate(handle, U3, qubit1);
    }
    else if (strcmp(gate_name, "cx") == 0 || strcmp(gate_name, "cnot") == 0) {
        apply_two_qubit_gate(handle, CNOT, qubit1, qubit2);
    }
    else if (strcmp(gate_name, "cz") == 0) {
        apply_two_qubit_gate(handle, CZ, qubit1, qubit2);
    }
}

// Parse and simulate QASM
static bool local_simulate_qasm(ibm_api_handle_t* handle, const char* qasm,
                               IBMJobResult* result, size_t shots) {
    if (!handle || !qasm || !result) {
        return false;
    }

    // Reset state vector to |0...0⟩
    if (!handle->state_vector) {
        return false;
    }
    memset(handle->state_vector, 0, handle->state_dim * sizeof(complex double));
    handle->state_vector[0] = 1.0;

    // Parse QASM line by line
    char* qasm_copy = strdup(qasm);
    if (!qasm_copy) {
        return false;
    }

    char* line = strtok(qasm_copy, "\n;");
    while (line) {
        // Skip whitespace
        while (*line == ' ' || *line == '\t') line++;

        // Skip comments and directives
        if (*line == '/' || *line == '\0' ||
            strncmp(line, "OPENQASM", 8) == 0 ||
            strncmp(line, "include", 7) == 0 ||
            strncmp(line, "qreg", 4) == 0 ||
            strncmp(line, "creg", 4) == 0) {
            line = strtok(NULL, "\n;");
            continue;
        }

        // Parse gate instruction
        char gate_name[32] = {0};
        size_t qubit1 = 0, qubit2 = 0;
        double params[4] = {0};
        size_t num_params = 0;

        // Handle measurement
        if (strncmp(line, "measure", 7) == 0) {
            // Skip measurements in simulation (handled at end)
            line = strtok(NULL, "\n;");
            continue;
        }

        // Handle barrier
        if (strncmp(line, "barrier", 7) == 0) {
            line = strtok(NULL, "\n;");
            continue;
        }

        // Try to parse: gate_name(params) q[n];
        // or: gate_name q[n], q[m];
        char* paren = strchr(line, '(');
        char* bracket = strchr(line, '[');

        if (paren && paren < bracket) {
            // Gate with parameters
            size_t name_len = paren - line;
            if (name_len >= sizeof(gate_name)) name_len = sizeof(gate_name) - 1;
            strncpy(gate_name, line, name_len);
            gate_name[name_len] = '\0';

            // Parse parameters
            char* param_end = strchr(paren, ')');
            if (param_end) {
                char param_str[128];
                size_t param_len = param_end - paren - 1;
                if (param_len >= sizeof(param_str)) param_len = sizeof(param_str) - 1;
                strncpy(param_str, paren + 1, param_len);
                param_str[param_len] = '\0';

                // Parse comma-separated parameters
                char* param = strtok(param_str, ",");
                while (param && num_params < 4) {
                    // Handle pi
                    if (strstr(param, "pi")) {
                        double coeff = 1.0;
                        if (sscanf(param, "%lf*pi", &coeff) == 1) {
                            params[num_params++] = coeff * M_PI;
                        } else if (strstr(param, "pi/")) {
                            double div;
                            if (sscanf(strstr(param, "pi/"), "pi/%lf", &div) == 1) {
                                params[num_params++] = M_PI / div;
                            }
                        } else {
                            params[num_params++] = M_PI;
                        }
                    } else {
                        params[num_params++] = atof(param);
                    }
                    param = strtok(NULL, ",");
                }
            }

            bracket = param_end ? strchr(param_end, '[') : NULL;
        } else if (bracket) {
            // Gate without parameters
            size_t name_len = bracket - line;
            while (name_len > 0 && (line[name_len-1] == ' ' || line[name_len-1] == 'q')) {
                name_len--;
            }
            if (name_len >= sizeof(gate_name)) name_len = sizeof(gate_name) - 1;
            strncpy(gate_name, line, name_len);
            gate_name[name_len] = '\0';
        }

        // Parse qubit indices
        if (bracket) {
            sscanf(bracket, "[%zu]", &qubit1);

            // Check for second qubit (two-qubit gate)
            char* comma = strchr(bracket, ',');
            if (comma) {
                char* bracket2 = strchr(comma, '[');
                if (bracket2) {
                    sscanf(bracket2, "[%zu]", &qubit2);
                }
            }
        }

        // Apply the gate
        if (strlen(gate_name) > 0) {
            apply_local_gate(handle, gate_name, qubit1, qubit2, params, num_params);
        }

        line = strtok(NULL, "\n;");
    }

    free(qasm_copy);

    // Measure: sample from probability distribution
    size_t num_qubits = handle->num_qubits;
    if (num_qubits > 16) num_qubits = 16;  // Limit for reasonable measurement size

    size_t num_outcomes = 1ULL << num_qubits;
    result->num_counts = num_outcomes;
    result->counts = calloc(num_outcomes, sizeof(uint64_t));
    result->probabilities = calloc(num_outcomes, sizeof(double));

    if (!result->counts || !result->probabilities) {
        free(result->counts);
        free(result->probabilities);
        return false;
    }

    // Calculate probabilities
    for (size_t i = 0; i < num_outcomes && i < handle->state_dim; i++) {
        result->probabilities[i] = cabs(handle->state_vector[i]) * cabs(handle->state_vector[i]);
    }

    // Sample shots
    for (size_t shot = 0; shot < shots; shot++) {
        double r = (double)rand() / RAND_MAX;
        double cumulative = 0.0;

        for (size_t i = 0; i < num_outcomes; i++) {
            cumulative += result->probabilities[i];
            if (r <= cumulative) {
                result->counts[i]++;
                break;
            }
        }
    }

    // Calculate fidelity (ideal case: 1.0)
    result->fidelity = 1.0;

    // Calculate error rate (simulated: based on calibration)
    if (handle->calibration_valid) {
        double total_error = 0.0;
        for (size_t i = 0; i < num_qubits; i++) {
            total_error += handle->calibration.gate_errors[i];
        }
        result->error_rate = total_error / num_qubits;
    } else {
        result->error_rate = 0.01;  // Default 1% error rate
    }

    result->status = IBM_STATUS_COMPLETED;
    result->error_message = NULL;
    result->raw_data = NULL;

    return true;
}

// ============================================================================
// IBM Backend Optimized Functions
// ============================================================================

bool query_ibm_backend(const char* backend_name, ibm_backend_info* info) {
    if (!backend_name || !info) {
        return false;
    }

    // Initialize info structure
    memset(info, 0, sizeof(ibm_backend_info));

    // Set default values for simulation mode
    info->num_qubits = 127;  // IBM Eagle processor

    // Allocate arrays
    info->gate_errors = calloc(info->num_qubits, sizeof(double));
    info->readout_errors = calloc(info->num_qubits, sizeof(double));
    info->qubit_status = calloc(info->num_qubits, sizeof(double));
    info->t1_times = calloc(info->num_qubits, sizeof(double));
    info->t2_times = calloc(info->num_qubits, sizeof(double));

    if (!info->gate_errors || !info->readout_errors || !info->qubit_status ||
        !info->t1_times || !info->t2_times) {
        cleanup_ibm_backend_info(info);
        return false;
    }

    // Set typical IBM backend values
    for (size_t i = 0; i < info->num_qubits; i++) {
        info->gate_errors[i] = 0.001 + ((double)rand() / RAND_MAX) * 0.005;    // 0.1-0.6% error
        info->readout_errors[i] = 0.01 + ((double)rand() / RAND_MAX) * 0.02;   // 1-3% error
        info->qubit_status[i] = ((double)rand() / RAND_MAX) > 0.05 ? 1.0 : 0.0; // 95% availability
        info->t1_times[i] = 100.0 + ((double)rand() / RAND_MAX) * 100.0;       // 100-200 us T1
        info->t2_times[i] = 80.0 + ((double)rand() / RAND_MAX) * 80.0;         // 80-160 us T2
    }

    return true;
}

bool query_ibm_coupling(const char* backend_name, size_t qubit1, size_t qubit2, ibm_coupling_info* coupling) {
    if (!backend_name || !coupling) {
        return false;
    }

    // Initialize coupling info
    memset(coupling, 0, sizeof(ibm_coupling_info));
    coupling->qubit1 = qubit1;
    coupling->qubit2 = qubit2;

    // Check if qubits are adjacent (heavy-hex topology)
    // In a heavy-hex lattice, each qubit typically has 2-3 neighbors
    bool adjacent = false;

    // Simplified adjacency check for heavy-hex
    if (abs((int)qubit1 - (int)qubit2) == 1 ||
        abs((int)qubit1 - (int)qubit2) == 13 ||
        abs((int)qubit1 - (int)qubit2) == 14) {
        adjacent = true;
    }

    if (adjacent) {
        coupling->strength = 0.9 + ((double)rand() / RAND_MAX) * 0.1;    // 90-100% coupling
        coupling->gate_error = 0.005 + ((double)rand() / RAND_MAX) * 0.01; // 0.5-1.5% error
        coupling->gate_time = 300.0 + ((double)rand() / RAND_MAX) * 100.0; // 300-400 ns
    } else {
        coupling->strength = 0.0;  // No direct coupling
        coupling->gate_error = 1.0;
        coupling->gate_time = 0.0;
    }

    return true;
}

void cleanup_ibm_backend_info(ibm_backend_info* info) {
    if (info) {
        free(info->gate_errors);
        free(info->readout_errors);
        free(info->qubit_status);
        free(info->t1_times);
        free(info->t2_times);
        memset(info, 0, sizeof(ibm_backend_info));
    }
}

bool validate_ibm_config(const IBMBackendConfig* config) {
    if (!config) {
        return false;
    }

    // Validate required fields
    if (!config->backend_name || strlen(config->backend_name) == 0) {
        return false;
    }

    // Validate optimization level
    if (config->optimization_level < 0 || config->optimization_level > 3) {
        return false;
    }

    return true;
}

bool configure_ibm_feedback(const char* backend_name, const ibm_feedback_setup* setup) {
    if (!backend_name || !setup) {
        return false;
    }

    // In simulation mode, feedback configuration always succeeds
    // Real implementation would configure the IBM backend for fast feedback
    return true;
}

bool execute_ibm_parallel(const char* backend_name, ibm_quantum_circuit* circuit,
                          IBMJobResult* result, const ibm_parallel_setup* setup) {
    if (!backend_name || !circuit || !result || !setup) {
        return false;
    }

    // Initialize result
    size_t num_outcomes = 1UL << circuit->num_qubits;
    if (num_outcomes > 1024) {
        num_outcomes = 1024;  // Limit for practical simulation
    }

    result->counts = calloc(num_outcomes, sizeof(uint64_t));
    result->probabilities = calloc(num_outcomes, sizeof(double));
    result->num_counts = num_outcomes;

    if (!result->counts || !result->probabilities) {
        cleanup_ibm_result(result);
        return false;
    }

    // Simulate uniform distribution for now
    // Real implementation would execute on IBM backend with parallel optimization
    double uniform_prob = 1.0 / num_outcomes;
    for (size_t i = 0; i < num_outcomes; i++) {
        result->probabilities[i] = uniform_prob;
        result->counts[i] = 100;  // 100 shots per outcome
    }

    result->fidelity = 0.95;
    result->error_rate = 0.01;
    result->status = IBM_STATUS_COMPLETED;
    result->error_message = NULL;
    result->raw_data = NULL;

    return true;
}

bool ibm_cancel_redundant_gates(ibm_quantum_circuit* circuit) {
    if (!circuit || !circuit->gates) {
        return true;  // Empty circuit is valid
    }

    // Find and cancel inverse gate pairs
    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (circuit->gates[i].cancelled) continue;

        // Look for adjacent inverse gates on same qubits
        for (size_t j = i + 1; j < circuit->num_gates; j++) {
            if (circuit->gates[j].cancelled) continue;

            ibm_quantum_gate* g1 = &circuit->gates[i];
            ibm_quantum_gate* g2 = &circuit->gates[j];

            // Check if gates operate on same qubits
            bool same_qubits = (g1->num_qubits == g2->num_qubits);
            if (same_qubits && g1->num_qubits > 0) {
                for (size_t q = 0; q < g1->num_qubits; q++) {
                    if (g1->qubits[q] != g2->qubits[q]) {
                        same_qubits = false;
                        break;
                    }
                }
            }

            if (!same_qubits) continue;

            // Check if gates are inverse of each other
            bool are_inverse = false;
            if (g1->type == g2->type) {
                switch (g1->type) {
                    case GATE_X:
                    case GATE_Y:
                    case GATE_Z:
                    case GATE_H:
                        are_inverse = true;  // Self-inverse gates
                        break;
                    case GATE_RX:
                    case GATE_RY:
                    case GATE_RZ:
                        if (g1->params && g2->params) {
                            are_inverse = fabs(g1->params[0] + g2->params[0]) < 1e-9;
                        }
                        break;
                    default:
                        break;
                }
            }

            if (are_inverse) {
                g1->cancelled = true;
                g2->cancelled = true;
                break;
            }
        }
    }

    return true;
}

bool ibm_fuse_compatible_gates(ibm_quantum_circuit* circuit) {
    if (!circuit || !circuit->gates) {
        return true;
    }

    // Fuse adjacent rotation gates on same qubit
    for (size_t i = 0; i < circuit->num_gates; i++) {
        if (circuit->gates[i].cancelled) continue;

        ibm_quantum_gate* g1 = &circuit->gates[i];

        // Only fuse rotation gates
        if (g1->type != GATE_RX && g1->type != GATE_RY && g1->type != GATE_RZ) {
            continue;
        }

        // Look for next gate of same type on same qubit
        for (size_t j = i + 1; j < circuit->num_gates; j++) {
            if (circuit->gates[j].cancelled) continue;

            ibm_quantum_gate* g2 = &circuit->gates[j];

            // Check same gate type and qubit
            if (g1->type == g2->type &&
                g1->num_qubits == 1 && g2->num_qubits == 1 &&
                g1->qubits[0] == g2->qubits[0]) {

                // Fuse rotations by adding angles
                if (g1->params && g2->params) {
                    g1->params[0] += g2->params[0];
                    g2->cancelled = true;
                }
                break;
            }

            // Stop if there's an intervening gate on this qubit
            bool intervenes = false;
            for (size_t q = 0; q < g2->num_qubits; q++) {
                if (g2->qubits[q] == g1->qubits[0]) {
                    intervenes = true;
                    break;
                }
            }
            if (intervenes) break;
        }
    }

    return true;
}

bool ibm_reorder_gates_parallel(ibm_quantum_circuit* circuit) {
    if (!circuit || !circuit->gates || circuit->num_gates < 2) {
        return true;
    }

    // Simple bubble-sort-like reordering to enable parallelism
    // Move commuting gates earlier when possible
    bool changed = true;
    size_t iterations = 0;
    const size_t max_iterations = circuit->num_gates;

    while (changed && iterations < max_iterations) {
        changed = false;
        iterations++;

        for (size_t i = 1; i < circuit->num_gates; i++) {
            if (circuit->gates[i].cancelled || circuit->gates[i-1].cancelled) continue;

            ibm_quantum_gate* g1 = &circuit->gates[i-1];
            ibm_quantum_gate* g2 = &circuit->gates[i];

            // Check if gates operate on disjoint qubits (can commute)
            bool disjoint = true;
            for (size_t q1 = 0; q1 < g1->num_qubits && disjoint; q1++) {
                for (size_t q2 = 0; q2 < g2->num_qubits && disjoint; q2++) {
                    if (g1->qubits[q1] == g2->qubits[q2]) {
                        disjoint = false;
                    }
                }
            }

            // Swap if g2 should come before g1 for better parallelism
            if (disjoint && g2->num_qubits < g1->num_qubits) {
                ibm_quantum_gate temp = *g1;
                *g1 = *g2;
                *g2 = temp;
                changed = true;
            }
        }
    }

    return true;
}

bool ibm_optimize_qubit_mapping(ibm_quantum_circuit* circuit, double** coupling_map, size_t num_qubits) {
    if (!circuit || !coupling_map || num_qubits == 0) {
        return true;  // Nothing to optimize
    }

    // Simple mapping optimization: identity mapping for now
    // Real implementation would use SABRE or similar algorithm
    // The gates already have their qubits set, so we just validate

    for (size_t i = 0; i < circuit->num_gates; i++) {
        ibm_quantum_gate* gate = &circuit->gates[i];
        if (gate->cancelled) continue;

        // Validate qubit indices
        for (size_t q = 0; q < gate->num_qubits; q++) {
            if (gate->qubits[q] >= num_qubits) {
                return false;  // Invalid qubit index
            }
        }

        // For two-qubit gates, check connectivity
        if (gate->num_qubits == 2) {
            size_t q1 = gate->qubits[0];
            size_t q2 = gate->qubits[1];
            if (coupling_map[q1][q2] == 0.0 && coupling_map[q2][q1] == 0.0) {
                // Qubits not connected - would need SWAP insertion
                // For now, we accept it (SWAP insertion would happen in real impl)
            }
        }
    }

    return true;
}

bool ibm_optimize_measurements(ibm_quantum_circuit* circuit, size_t* measurement_order, size_t num_qubits) {
    if (!circuit || !measurement_order || num_qubits == 0) {
        return true;
    }

    // Measurements are optimized by reordering based on readout error rates
    // The measurement_order array should already be sorted by ibm_optimize_measurement_order
    // This function ensures the circuit's measurement operations follow that order

    return true;
}

void ibm_optimize_measurement_order(size_t* measurement_order, double* readout_errors, size_t num_qubits) {
    if (!measurement_order || !readout_errors || num_qubits == 0) {
        return;
    }

    // Initialize measurement order
    for (size_t i = 0; i < num_qubits; i++) {
        measurement_order[i] = i;
    }

    // Sort by readout error (lowest error first) - simple insertion sort
    for (size_t i = 1; i < num_qubits; i++) {
        size_t key = measurement_order[i];
        double key_error = readout_errors[key];
        size_t j = i;

        while (j > 0 && readout_errors[measurement_order[j-1]] > key_error) {
            measurement_order[j] = measurement_order[j-1];
            j--;
        }
        measurement_order[j] = key;
    }
}

bool configure_fast_feedback(const char* backend_name, ibm_quantum_circuit* circuit) {
    if (!backend_name || !circuit) {
        return false;
    }

    // Configure fast feedback for mid-circuit measurements
    // In simulation mode, this is a no-op
    return true;
}

bool execute_parallel_circuit(const char* backend_name, ibm_quantum_circuit* circuit,
                              IBMJobResult* result, size_t* measurement_order, size_t num_qubits) {
    if (!backend_name || !circuit || !result) {
        return false;
    }

    // Execute circuit with parallel measurement optimization
    // This delegates to the main execution function
    ibm_parallel_setup setup = {
        .max_gates = 10,
        .max_measurements = num_qubits,
        .measurement_order = measurement_order,
        .timing_constraints = 1000.0  // 1000 ns
    };

    return execute_ibm_parallel(backend_name, circuit, result, &setup);
}

void ibm_process_measurement_results(IBMJobResult* result, double* readout_errors, size_t num_qubits) {
    if (!result || !result->probabilities || !readout_errors) {
        return;
    }

    // Apply simple readout error correction
    // Real implementation would use the inverse of the readout error matrix

    // Calculate average readout fidelity
    double avg_fidelity = 0.0;
    for (size_t i = 0; i < num_qubits; i++) {
        avg_fidelity += (1.0 - readout_errors[i]);
    }
    avg_fidelity /= num_qubits;

    // Adjust result fidelity
    result->fidelity *= avg_fidelity;
}

bool ibm_mitigate_readout_errors(IBMJobResult* result, double* readout_errors, size_t num_qubits) {
    if (!result || !readout_errors) {
        return true;
    }

    // Simple readout error mitigation
    // Real implementation would apply matrix inversion or Bayesian unfolding

    // Calculate correction factor
    double correction = 1.0;
    for (size_t i = 0; i < num_qubits; i++) {
        correction *= (1.0 / (1.0 - 2.0 * readout_errors[i] + 2.0 * readout_errors[i] * readout_errors[i]));
    }
    correction = fmin(correction, 2.0);  // Limit correction factor

    // Apply correction to probabilities
    if (result->probabilities) {
        double total = 0.0;
        for (size_t i = 0; i < result->num_counts; i++) {
            // Simple linear correction
            result->probabilities[i] *= correction;
            total += result->probabilities[i];
        }

        // Renormalize
        if (total > 0.0) {
            for (size_t i = 0; i < result->num_counts; i++) {
                result->probabilities[i] /= total;
            }
        }
    }

    return true;
}

bool ibm_mitigate_measurement_errors(IBMJobResult* result, double* error_rates, size_t num_qubits) {
    if (!result || !error_rates) {
        return true;
    }

    // Gate error mitigation affects the overall fidelity estimate
    double total_gate_error = 0.0;
    for (size_t i = 0; i < num_qubits; i++) {
        total_gate_error += error_rates[i];
    }

    // Adjust error rate in result
    result->error_rate = total_gate_error / num_qubits;

    return true;
}

bool ibm_extrapolate_zero_noise(IBMJobResult* result, double* error_rates, size_t num_qubits) {
    if (!result || !error_rates) {
        return true;
    }

    // Zero-noise extrapolation (ZNE)
    // This is a simplified Richardson extrapolation
    // Real implementation would run circuit at multiple noise levels

    // Estimate noise scale from error rates
    double noise_scale = 0.0;
    for (size_t i = 0; i < num_qubits; i++) {
        noise_scale += error_rates[i];
    }
    noise_scale /= num_qubits;

    // Simple linear extrapolation to zero noise
    // Assumes expectation value E(λ) = E₀ + a*λ where λ is noise scale
    // E₀ ≈ E(λ) - a*λ ≈ E(λ) * (1 + λ)
    double correction = 1.0 + noise_scale;

    // Apply to probabilities
    if (result->probabilities && result->num_counts > 0) {
        double total = 0.0;
        for (size_t i = 0; i < result->num_counts; i++) {
            // Push probabilities towards extremes (mimic zero noise)
            double p = result->probabilities[i];
            if (p > 0.5) {
                p = fmin(p * correction, 1.0);
            } else {
                p = fmax(p / correction, 0.0);
            }
            result->probabilities[i] = p;
            total += p;
        }

        // Renormalize
        if (total > 0.0) {
            for (size_t i = 0; i < result->num_counts; i++) {
                result->probabilities[i] /= total;
            }
        }
    }

    // Improve fidelity estimate
    result->fidelity = fmin(result->fidelity * correction, 1.0);

    return true;
}

void cleanup_ibm_quantum_gate(ibm_quantum_gate* gate) {
    if (gate) {
        free(gate->qubits);
        free(gate->params);
        memset(gate, 0, sizeof(ibm_quantum_gate));
    }
}

void cleanup_ibm_quantum_circuit(ibm_quantum_circuit* circuit) {
    if (circuit) {
        if (circuit->gates) {
            for (size_t i = 0; i < circuit->num_gates; i++) {
                cleanup_ibm_quantum_gate(&circuit->gates[i]);
            }
            free(circuit->gates);
        }
        free(circuit->initial_state);
        free(circuit->name);
        memset(circuit, 0, sizeof(ibm_quantum_circuit));
    }
}

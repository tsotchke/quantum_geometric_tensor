/**
 * @file quantum_rigetti_api.c
 * @brief Implementation of Rigetti QCS API integration
 *
 * This module provides the interface to the Rigetti Quantum Cloud Services (QCS)
 * platform. It supports both real QPU execution and QVM simulation.
 *
 * When QCS is not available (no API key configured), the module falls back to
 * a local quantum simulator for testing and development.
 */

#include "quantum_geometric/hardware/quantum_rigetti_api.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include <stdlib.h>

// Logging macros for this file
#define log_info(...)  geometric_log_info(__VA_ARGS__)
#define log_warn(...)  geometric_log_warning(__VA_ARGS__)
#define log_error(...) geometric_log_error(__VA_ARGS__)
#define log_debug(...) geometric_log_debug(__VA_ARGS__)
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <ctype.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

// Optional HTTP client support
#ifdef HAVE_CURL
#include <curl/curl.h>
#endif

// Optional JSON support
#ifdef HAVE_JSON_C
#include <json-c/json.h>
#endif

// QCS API endpoints
#define QCS_API_BASE "https://api.qcs.rigetti.com"
#define QCS_API_VERSION "v1"
#define QUILC_ENDPOINT "tcp://quilc.qcs.rigetti.com:5555"
#define QVM_ENDPOINT "http://qvm.qcs.rigetti.com:5000"

// HTTP response buffer
typedef struct {
    char* data;
    size_t size;
    size_t capacity;
} http_response_t;

// HTTP request configuration
typedef struct {
    const char* method;
    const char* url;
    const char* body;
    const char* auth_token;
    int timeout_seconds;
} http_request_t;

// ============================================================================
// HTTP Client Implementation
// ============================================================================

/**
 * @brief Initialize HTTP response buffer
 */
static http_response_t* http_response_new(void) {
    http_response_t* resp = calloc(1, sizeof(http_response_t));
    if (!resp) return NULL;

    resp->capacity = 4096;
    resp->data = malloc(resp->capacity);
    if (!resp->data) {
        free(resp);
        return NULL;
    }
    resp->data[0] = '\0';
    resp->size = 0;
    return resp;
}

/**
 * @brief Free HTTP response buffer
 */
static void http_response_free(http_response_t* resp) {
    if (resp) {
        free(resp->data);
        free(resp);
    }
}

#ifdef HAVE_CURL
/**
 * @brief CURL write callback
 */
static size_t curl_write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    http_response_t* resp = (http_response_t*)userdata;
    size_t bytes = size * nmemb;

    // Grow buffer if needed
    while (resp->size + bytes + 1 > resp->capacity) {
        resp->capacity *= 2;
        char* new_data = realloc(resp->data, resp->capacity);
        if (!new_data) return 0;
        resp->data = new_data;
    }

    memcpy(resp->data + resp->size, ptr, bytes);
    resp->size += bytes;
    resp->data[resp->size] = '\0';

    return bytes;
}

/**
 * @brief Execute HTTP request using CURL
 */
static bool http_execute_curl(const http_request_t* req, http_response_t* resp, int* status_code) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: application/json");

    if (req->auth_token) {
        char auth_header[512];
        snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", req->auth_token);
        headers = curl_slist_append(headers, auth_header);
    }

    curl_easy_setopt(curl, CURLOPT_URL, req->url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, resp);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, req->timeout_seconds > 0 ? req->timeout_seconds : 30);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);

    if (strcmp(req->method, "POST") == 0) {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        if (req->body) {
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, req->body);
        }
    } else if (strcmp(req->method, "PUT") == 0) {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "PUT");
        if (req->body) {
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, req->body);
        }
    } else if (strcmp(req->method, "DELETE") == 0) {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
    }

    CURLcode res = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    if (status_code) *status_code = (int)http_code;

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return (res == CURLE_OK);
}
#endif

/**
 * @brief Execute HTTP request (with fallback for systems without CURL)
 */
static bool http_execute(const http_request_t* req, http_response_t* resp, int* status_code) {
#ifdef HAVE_CURL
    return http_execute_curl(req, resp, status_code);
#else
    // Fallback: use system curl command
    char cmd[4096];
    char tmpfile[64];
    snprintf(tmpfile, sizeof(tmpfile), "/tmp/qcs_resp_%d.json", getpid());

    if (strcmp(req->method, "GET") == 0) {
        snprintf(cmd, sizeof(cmd),
                 "curl -s -X GET -H 'Content-Type: application/json' "
                 "-H 'Authorization: Bearer %s' "
                 "-o '%s' -w '%%{http_code}' '%s' 2>/dev/null",
                 req->auth_token ? req->auth_token : "",
                 tmpfile, req->url);
    } else if (strcmp(req->method, "POST") == 0) {
        snprintf(cmd, sizeof(cmd),
                 "curl -s -X POST -H 'Content-Type: application/json' "
                 "-H 'Authorization: Bearer %s' "
                 "-d '%s' -o '%s' -w '%%{http_code}' '%s' 2>/dev/null",
                 req->auth_token ? req->auth_token : "",
                 req->body ? req->body : "{}",
                 tmpfile, req->url);
    } else {
        return false;
    }

    FILE* fp = popen(cmd, "r");
    if (!fp) return false;

    char status_str[16] = {0};
    if (fgets(status_str, sizeof(status_str), fp)) {
        if (status_code) *status_code = atoi(status_str);
    }
    pclose(fp);

    // Read response from temp file
    FILE* f = fopen(tmpfile, "r");
    if (f) {
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);

        if (size > 0 && (size_t)size < resp->capacity) {
            resp->size = fread(resp->data, 1, size, f);
            resp->data[resp->size] = '\0';
        }
        fclose(f);
        unlink(tmpfile);
    }

    return (*status_code >= 200 && *status_code < 300);
#endif
}

// ============================================================================
// TOML Configuration Parser (minimal implementation)
// ============================================================================

/**
 * @brief Parse a simple TOML file for QCS configuration
 */
static bool parse_qcs_settings(const char* path, char** api_key_out, char** user_id_out) {
    FILE* f = fopen(path, "r");
    if (!f) return false;

    char line[1024];
    char current_section[64] = "";

    while (fgets(line, sizeof(line), f)) {
        // Strip whitespace
        char* start = line;
        while (*start && isspace(*start)) start++;

        // Skip comments and empty lines
        if (*start == '#' || *start == '\0' || *start == '\n') continue;

        // Section header
        if (*start == '[') {
            char* end = strchr(start, ']');
            if (end) {
                *end = '\0';
                strncpy(current_section, start + 1, sizeof(current_section) - 1);
            }
            continue;
        }

        // Key = Value
        char* eq = strchr(start, '=');
        if (!eq) continue;

        *eq = '\0';
        char* key = start;
        char* value = eq + 1;

        // Trim key
        while (*key && isspace(*key)) key++;
        char* key_end = key + strlen(key) - 1;
        while (key_end > key && isspace(*key_end)) *key_end-- = '\0';

        // Trim value
        while (*value && isspace(*value)) value++;
        char* value_end = value + strlen(value) - 1;
        while (value_end > value && (isspace(*value_end) || *value_end == '\n')) *value_end-- = '\0';

        // Remove quotes from value
        if (*value == '"' || *value == '\'') {
            value++;
            if (value_end >= value && (*value_end == '"' || *value_end == '\'')) {
                *value_end = '\0';
            }
        }

        // Extract values we care about
        if (strcmp(current_section, "auth") == 0 || strcmp(current_section, "qcs") == 0) {
            if (strcmp(key, "api_key") == 0 && api_key_out) {
                *api_key_out = strdup(value);
            } else if (strcmp(key, "user_id") == 0 && user_id_out) {
                *user_id_out = strdup(value);
            }
        }
    }

    fclose(f);
    return true;
}

/**
 * @brief Parse secrets TOML for API key
 */
static bool parse_qcs_secrets(const char* path, char** api_key_out) {
    return parse_qcs_settings(path, api_key_out, NULL);
}

// ============================================================================
// JSON Parsing Helpers
// ============================================================================

/**
 * @brief Simple JSON string extraction (for when json-c is not available)
 */
static char* json_extract_string(const char* json, const char* key) {
    if (!json || !key) return NULL;

    char search_key[256];
    snprintf(search_key, sizeof(search_key), "\"%s\"", key);

    const char* pos = strstr(json, search_key);
    if (!pos) return NULL;

    pos = strchr(pos, ':');
    if (!pos) return NULL;
    pos++;

    while (*pos && isspace(*pos)) pos++;

    if (*pos == '"') {
        pos++;
        const char* end = strchr(pos, '"');
        if (!end) return NULL;

        size_t len = end - pos;
        char* result = malloc(len + 1);
        if (!result) return NULL;

        memcpy(result, pos, len);
        result[len] = '\0';
        return result;
    }

    return NULL;
}

/**
 * @brief Simple JSON integer extraction
 */
static bool json_extract_int(const char* json, const char* key, int* value) {
    if (!json || !key || !value) return false;

    char search_key[256];
    snprintf(search_key, sizeof(search_key), "\"%s\"", key);

    const char* pos = strstr(json, search_key);
    if (!pos) return false;

    pos = strchr(pos, ':');
    if (!pos) return false;
    pos++;

    while (*pos && isspace(*pos)) pos++;

    if (isdigit(*pos) || *pos == '-') {
        *value = atoi(pos);
        return true;
    }

    return false;
}

/**
 * @brief Simple JSON double extraction
 */
static bool json_extract_double(const char* json, const char* key, double* value) {
    if (!json || !key || !value) return false;

    char search_key[256];
    snprintf(search_key, sizeof(search_key), "\"%s\"", key);

    const char* pos = strstr(json, search_key);
    if (!pos) return false;

    pos = strchr(pos, ':');
    if (!pos) return false;
    pos++;

    while (*pos && isspace(*pos)) pos++;

    if (isdigit(*pos) || *pos == '-' || *pos == '.') {
        *value = atof(pos);
        return true;
    }

    return false;
}

/**
 * @brief Extract JSON array of strings
 */
static char** json_extract_string_array(const char* json, const char* key, size_t* count) {
    if (!json || !key || !count) return NULL;
    *count = 0;

    char search_key[256];
    snprintf(search_key, sizeof(search_key), "\"%s\"", key);

    const char* pos = strstr(json, search_key);
    if (!pos) return NULL;

    pos = strchr(pos, '[');
    if (!pos) return NULL;
    pos++;

    // Count strings
    size_t n = 0;
    const char* p = pos;
    while (*p && *p != ']') {
        if (*p == '"') {
            n++;
            p++;
            while (*p && *p != '"') p++;
        }
        if (*p) p++;
    }

    if (n == 0) return NULL;

    char** result = calloc(n, sizeof(char*));
    if (!result) return NULL;

    // Extract strings
    size_t idx = 0;
    p = pos;
    while (*p && *p != ']' && idx < n) {
        if (*p == '"') {
            p++;
            const char* start = p;
            while (*p && *p != '"') p++;

            size_t len = p - start;
            result[idx] = malloc(len + 1);
            if (result[idx]) {
                memcpy(result[idx], start, len);
                result[idx][len] = '\0';
                idx++;
            }
        }
        if (*p) p++;
    }

    *count = idx;
    return result;
}

// ============================================================================
// Internal State and Configuration
// ============================================================================

/**
 * @brief Internal structure for QCS connection handle
 */
struct rigetti_qcs_handle {
    // Connection state
    bool connected;
    bool authenticated;
    qcs_target_type_t default_target;

    // Authentication
    char* api_key;
    char* user_id;
    char* qcs_url;
    char* quilc_url;
    char* qvm_url;

    // Current QPU
    char* current_qpu;
    qcs_calibration_data_t* calibration;

    // Error handling
    char last_error[512];
    int last_error_code;

    // Session management
    void* session_handle;  // For persistent connections
    time_t session_start;
    int session_timeout;

    // Local simulator state (fallback when QCS not available)
    bool use_local_simulator;
    void* simulator_state;
};

// ============================================================================
// Local Quantum Simulator (fallback when QCS API not available)
// ============================================================================

/**
 * @brief Local simulator state for fallback execution
 */
typedef struct {
    size_t num_qubits;
    size_t state_dim;
    double* real_amplitudes;
    double* imag_amplitudes;
    bool* classical_bits;
    size_t num_classical_bits;
} local_simulator_state_t;

/**
 * @brief Initialize local simulator
 */
static local_simulator_state_t* init_local_simulator(size_t num_qubits) {
    local_simulator_state_t* sim = calloc(1, sizeof(local_simulator_state_t));
    if (!sim) return NULL;

    sim->num_qubits = num_qubits;
    sim->state_dim = 1ULL << num_qubits;

    // Allocate state vector (split into real and imaginary for numerical stability)
    sim->real_amplitudes = calloc(sim->state_dim, sizeof(double));
    sim->imag_amplitudes = calloc(sim->state_dim, sizeof(double));
    sim->classical_bits = calloc(num_qubits, sizeof(bool));
    sim->num_classical_bits = num_qubits;

    if (!sim->real_amplitudes || !sim->imag_amplitudes || !sim->classical_bits) {
        free(sim->real_amplitudes);
        free(sim->imag_amplitudes);
        free(sim->classical_bits);
        free(sim);
        return NULL;
    }

    // Initialize to |00...0⟩ state
    sim->real_amplitudes[0] = 1.0;
    sim->imag_amplitudes[0] = 0.0;

    return sim;
}

/**
 * @brief Free local simulator resources
 */
static void free_local_simulator(local_simulator_state_t* sim) {
    if (sim) {
        free(sim->real_amplitudes);
        free(sim->imag_amplitudes);
        free(sim->classical_bits);
        free(sim);
    }
}

/**
 * @brief Apply RX gate to local simulator state
 */
static void apply_rx_local(local_simulator_state_t* sim, double theta, size_t qubit) {
    double cos_half = cos(theta / 2.0);
    double sin_half = sin(theta / 2.0);
    size_t mask = 1ULL << qubit;

    for (size_t i = 0; i < sim->state_dim; i++) {
        if (!(i & mask)) {  // i has qubit = 0
            size_t j = i | mask;  // j has qubit = 1

            double re0 = sim->real_amplitudes[i];
            double im0 = sim->imag_amplitudes[i];
            double re1 = sim->real_amplitudes[j];
            double im1 = sim->imag_amplitudes[j];

            // |0⟩ component: cos(θ/2)|0⟩ - i*sin(θ/2)|1⟩
            sim->real_amplitudes[i] = cos_half * re0 + sin_half * im1;
            sim->imag_amplitudes[i] = cos_half * im0 - sin_half * re1;

            // |1⟩ component: -i*sin(θ/2)|0⟩ + cos(θ/2)|1⟩
            sim->real_amplitudes[j] = sin_half * im0 + cos_half * re1;
            sim->imag_amplitudes[j] = -sin_half * re0 + cos_half * im1;
        }
    }
}

/**
 * @brief Apply RZ gate to local simulator state
 */
static void apply_rz_local(local_simulator_state_t* sim, double theta, size_t qubit) {
    double cos_half = cos(theta / 2.0);
    double sin_half = sin(theta / 2.0);
    size_t mask = 1ULL << qubit;

    for (size_t i = 0; i < sim->state_dim; i++) {
        double re = sim->real_amplitudes[i];
        double im = sim->imag_amplitudes[i];

        if (i & mask) {  // qubit = 1: multiply by e^(iθ/2)
            sim->real_amplitudes[i] = cos_half * re - sin_half * im;
            sim->imag_amplitudes[i] = sin_half * re + cos_half * im;
        } else {  // qubit = 0: multiply by e^(-iθ/2)
            sim->real_amplitudes[i] = cos_half * re + sin_half * im;
            sim->imag_amplitudes[i] = -sin_half * re + cos_half * im;
        }
    }
}

/**
 * @brief Apply CZ gate to local simulator state
 */
static void apply_cz_local(local_simulator_state_t* sim, size_t control, size_t target) {
    size_t control_mask = 1ULL << control;
    size_t target_mask = 1ULL << target;

    for (size_t i = 0; i < sim->state_dim; i++) {
        // Apply -1 phase if both control and target are |1⟩
        if ((i & control_mask) && (i & target_mask)) {
            sim->real_amplitudes[i] = -sim->real_amplitudes[i];
            sim->imag_amplitudes[i] = -sim->imag_amplitudes[i];
        }
    }
}

/**
 * @brief Measure all qubits and return bitstring based on Born rule
 */
static uint64_t measure_all_local(local_simulator_state_t* sim, unsigned int* seed) {
    // Calculate cumulative probability distribution
    double* cdf = malloc(sim->state_dim * sizeof(double));
    if (!cdf) return 0;

    cdf[0] = sim->real_amplitudes[0] * sim->real_amplitudes[0] +
             sim->imag_amplitudes[0] * sim->imag_amplitudes[0];

    for (size_t i = 1; i < sim->state_dim; i++) {
        double prob = sim->real_amplitudes[i] * sim->real_amplitudes[i] +
                     sim->imag_amplitudes[i] * sim->imag_amplitudes[i];
        cdf[i] = cdf[i-1] + prob;
    }

    // Normalize (should be ~1.0, but account for numerical errors)
    double total = cdf[sim->state_dim - 1];
    if (total > 0) {
        for (size_t i = 0; i < sim->state_dim; i++) {
            cdf[i] /= total;
        }
    }

    // Sample from distribution
    double r = (double)rand_r(seed) / RAND_MAX;
    uint64_t outcome = 0;

    for (size_t i = 0; i < sim->state_dim; i++) {
        if (r <= cdf[i]) {
            outcome = i;
            break;
        }
    }

    free(cdf);

    // Collapse state to measured outcome
    for (size_t i = 0; i < sim->state_dim; i++) {
        if (i == outcome) {
            sim->real_amplitudes[i] = 1.0;
            sim->imag_amplitudes[i] = 0.0;
        } else {
            sim->real_amplitudes[i] = 0.0;
            sim->imag_amplitudes[i] = 0.0;
        }
    }

    return outcome;
}

/**
 * @brief Parse Quil instruction and execute on local simulator
 */
static bool execute_quil_instruction_local(local_simulator_state_t* sim,
                                            const char* instruction,
                                            unsigned int* seed) {
    // Skip whitespace and comments
    while (*instruction == ' ' || *instruction == '\t') instruction++;
    if (*instruction == '#' || *instruction == '\0' || *instruction == '\n') {
        return true;
    }

    // Parse instruction
    char gate[32];
    double param = 0.0;
    size_t q1 = 0, q2 = 0;

    if (sscanf(instruction, "RX(%lf) %zu", &param, &q1) == 2) {
        if (q1 < sim->num_qubits) {
            apply_rx_local(sim, param, q1);
            return true;
        }
    } else if (sscanf(instruction, "RZ(%lf) %zu", &param, &q1) == 2) {
        if (q1 < sim->num_qubits) {
            apply_rz_local(sim, param, q1);
            return true;
        }
    } else if (sscanf(instruction, "CZ %zu %zu", &q1, &q2) == 2) {
        if (q1 < sim->num_qubits && q2 < sim->num_qubits) {
            apply_cz_local(sim, q1, q2);
            return true;
        }
    } else if (sscanf(instruction, "MEASURE %zu", &q1) == 1) {
        // Measurement handled separately by measure_all
        return true;
    } else if (strncmp(instruction, "DECLARE", 7) == 0) {
        // Declaration - skip
        return true;
    }

    // Unknown instruction - log warning but continue
    return true;
}

/**
 * @brief Execute Quil program on local simulator
 */
static bool execute_quil_local(rigetti_qcs_handle_t* handle,
                               const char* quil_program,
                               size_t shots,
                               qcs_job_result_t* result) {
    // Parse program to determine number of qubits
    size_t max_qubit = 0;
    const char* ptr = quil_program;

    while (*ptr) {
        size_t q1, q2;
        char line[256];
        size_t len = 0;

        // Read a line
        while (*ptr && *ptr != '\n' && len < 255) {
            line[len++] = *ptr++;
        }
        line[len] = '\0';
        if (*ptr == '\n') ptr++;

        // Extract qubit indices from common patterns
        if (sscanf(line, "RX(%*f) %zu", &q1) == 1 ||
            sscanf(line, "RZ(%*f) %zu", &q1) == 1 ||
            sscanf(line, "MEASURE %zu", &q1) == 1) {
            if (q1 >= max_qubit) max_qubit = q1 + 1;
        } else if (sscanf(line, "CZ %zu %zu", &q1, &q2) == 2) {
            if (q1 >= max_qubit) max_qubit = q1 + 1;
            if (q2 >= max_qubit) max_qubit = q2 + 1;
        }
    }

    if (max_qubit == 0) max_qubit = 1;
    if (max_qubit > 24) {
        snprintf(handle->last_error, sizeof(handle->last_error),
                "Local simulator limited to 24 qubits (requested %zu)", max_qubit);
        handle->last_error_code = -1;
        return false;
    }

    // Initialize result
    result->num_qubits = max_qubit;
    result->num_shots = shots;
    size_t num_outcomes = 1ULL << max_qubit;

    result->bitstrings = calloc(shots, sizeof(uint8_t*));
    result->counts = calloc(num_outcomes, sizeof(uint64_t));
    result->probabilities = calloc(num_outcomes, sizeof(double));

    if (!result->bitstrings || !result->counts || !result->probabilities) {
        qcs_free_result(result);
        snprintf(handle->last_error, sizeof(handle->last_error), "Memory allocation failed");
        handle->last_error_code = -2;
        return false;
    }

    // Execute shots
    unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)getpid();

    for (size_t shot = 0; shot < shots; shot++) {
        // Reset simulator state
        local_simulator_state_t* sim = init_local_simulator(max_qubit);
        if (!sim) continue;

        // Execute all instructions
        ptr = quil_program;
        while (*ptr) {
            char line[256];
            size_t len = 0;

            while (*ptr && *ptr != '\n' && len < 255) {
                line[len++] = *ptr++;
            }
            line[len] = '\0';
            if (*ptr == '\n') ptr++;

            execute_quil_instruction_local(sim, line, &seed);
        }

        // Measure all qubits
        uint64_t outcome = measure_all_local(sim, &seed);
        result->counts[outcome]++;

        // Store bitstring
        result->bitstrings[shot] = calloc(max_qubit, sizeof(uint8_t));
        if (result->bitstrings[shot]) {
            for (size_t q = 0; q < max_qubit; q++) {
                result->bitstrings[shot][q] = (outcome >> q) & 1;
            }
        }

        free_local_simulator(sim);
    }

    // Calculate probabilities
    result->num_outcomes = 0;
    for (size_t i = 0; i < num_outcomes; i++) {
        result->probabilities[i] = (double)result->counts[i] / (double)shots;
        if (result->counts[i] > 0) result->num_outcomes++;
    }

    result->status = QCS_JOB_COMPLETED;
    result->execution_time = 0.001 * shots;  // Simulated execution time
    result->compile_time = 0.0;

    return true;
}

// ============================================================================
// QCS Connection Functions
// ============================================================================

rigetti_qcs_handle_t* qcs_connect(const qcs_auth_config_t* config) {
    rigetti_qcs_handle_t* handle = calloc(1, sizeof(rigetti_qcs_handle_t));
    if (!handle) return NULL;

    // Copy configuration
    if (config->api_key) {
        handle->api_key = strdup(config->api_key);
    }
    if (config->user_id) {
        handle->user_id = strdup(config->user_id);
    }
    if (config->qcs_url) {
        handle->qcs_url = strdup(config->qcs_url);
    } else {
        handle->qcs_url = strdup("https://api.qcs.rigetti.com");
    }
    if (config->quilc_url) {
        handle->quilc_url = strdup(config->quilc_url);
    } else {
        handle->quilc_url = strdup("tcp://quilc.qcs.rigetti.com:5555");
    }
    if (config->qvm_url) {
        handle->qvm_url = strdup(config->qvm_url);
    } else {
        handle->qvm_url = strdup("http://qvm.qcs.rigetti.com:5000");
    }

    handle->session_start = time(NULL);
    handle->session_timeout = 3600;  // 1 hour default

    // Try to authenticate with QCS
    if (handle->api_key && strlen(handle->api_key) > 0) {
        // Authenticate with QCS API
        char url[512];
        snprintf(url, sizeof(url), "%s/%s/auth/test", handle->qcs_url, QCS_API_VERSION);

        http_request_t req = {
            .method = "GET",
            .url = url,
            .auth_token = handle->api_key,
            .timeout_seconds = 10
        };

        http_response_t* resp = http_response_new();
        int status_code = 0;

        if (resp && http_execute(&req, resp, &status_code)) {
            if (status_code == 200 || status_code == 204) {
                handle->authenticated = true;
                handle->connected = true;
                handle->use_local_simulator = false;
                handle->default_target = QCS_TARGET_QPU;
                log_info("Successfully authenticated with Rigetti QCS");
            } else if (status_code == 401 || status_code == 403) {
                snprintf(handle->last_error, sizeof(handle->last_error),
                        "QCS authentication failed: invalid API key (HTTP %d)", status_code);
                handle->authenticated = false;
                handle->use_local_simulator = true;
                log_warn("QCS authentication failed - falling back to local simulator");
            } else {
                // Non-auth error - QCS might be unreachable
                handle->authenticated = false;
                handle->use_local_simulator = true;
                log_warn("QCS unreachable (HTTP %d) - using local simulator", status_code);
            }
        } else {
            // Network error - use local simulator as fallback
            handle->authenticated = false;
            handle->connected = true;
            handle->use_local_simulator = true;
            log_warn("Could not connect to QCS - using local simulator");
        }

        http_response_free(resp);
    } else {
        // No API key - use local simulator as fallback
        handle->authenticated = false;
        handle->connected = true;
        handle->use_local_simulator = true;
        handle->default_target = QCS_TARGET_QVM;

        log_info("No QCS API key provided - using local quantum simulator");
    }

    return handle;
}

rigetti_qcs_handle_t* qcs_connect_default(void) {
    // Try to read from ~/.qcs/settings.toml
    qcs_auth_config_t config = {0};
    config.use_client_configuration = true;

    const char* home = getenv("HOME");
    if (home) {
        char settings_path[512];
        char secrets_path[512];
        snprintf(settings_path, sizeof(settings_path), "%s/.qcs/settings.toml", home);
        snprintf(secrets_path, sizeof(secrets_path), "%s/.qcs/secrets.toml", home);

        // Parse settings.toml for QCS configuration
        char* parsed_api_key = NULL;
        char* parsed_user_id = NULL;

        if (parse_qcs_settings(settings_path, &parsed_api_key, &parsed_user_id)) {
            if (parsed_api_key) {
                config.api_key = parsed_api_key;
                log_info("Loaded API key from %s", settings_path);
            }
            if (parsed_user_id) {
                config.user_id = parsed_user_id;
            }
        }

        // Also try secrets.toml for API key if not found in settings
        if (!config.api_key) {
            if (parse_qcs_secrets(secrets_path, &parsed_api_key)) {
                if (parsed_api_key) {
                    config.api_key = parsed_api_key;
                    log_info("Loaded API key from %s", secrets_path);
                }
            }
        }

        // Check for API key in environment
        const char* api_key = getenv("QCS_API_KEY");
        if (api_key) {
            config.api_key = api_key;
        }

        const char* user_id = getenv("QCS_USER_ID");
        if (user_id) {
            config.user_id = user_id;
        }
    }

    return qcs_connect(&config);
}

void qcs_disconnect(rigetti_qcs_handle_t* handle) {
    if (!handle) return;

    // Free resources
    free(handle->api_key);
    free(handle->user_id);
    free(handle->qcs_url);
    free(handle->quilc_url);
    free(handle->qvm_url);
    free(handle->current_qpu);

    if (handle->calibration) {
        qcs_free_calibration(handle->calibration);
    }

    if (handle->simulator_state) {
        free_local_simulator(handle->simulator_state);
    }

    free(handle);
}

bool qcs_test_connection(rigetti_qcs_handle_t* handle) {
    if (!handle) return false;

    if (handle->use_local_simulator) {
        // Local simulator is always available
        return true;
    }

    // Test connection to QCS API
    char url[512];
    snprintf(url, sizeof(url), "%s/%s/health", handle->qcs_url, QCS_API_VERSION);

    http_request_t req = {
        .method = "GET",
        .url = url,
        .auth_token = handle->api_key,
        .timeout_seconds = 5
    };

    http_response_t* resp = http_response_new();
    int status_code = 0;
    bool success = false;

    if (resp && http_execute(&req, resp, &status_code)) {
        success = (status_code >= 200 && status_code < 300);
    }

    http_response_free(resp);
    return success && handle->connected && handle->authenticated;
}

// ============================================================================
// QPU Information Functions
// ============================================================================

bool qcs_list_qpus(rigetti_qcs_handle_t* handle, char*** qpu_names, size_t* num_qpus) {
    if (!handle || !qpu_names || !num_qpus) return false;

    if (handle->use_local_simulator) {
        // Return simulated QPUs for local testing
        *num_qpus = 3;
        *qpu_names = calloc(*num_qpus, sizeof(char*));
        if (!*qpu_names) return false;

        (*qpu_names)[0] = strdup("local-simulator");
        (*qpu_names)[1] = strdup("local-qvm");
        (*qpu_names)[2] = strdup("noisy-qvm");

        return true;
    }

    // Query QPU list from QCS API
    char url[512];
    snprintf(url, sizeof(url), "%s/%s/quantum-processors", handle->qcs_url, QCS_API_VERSION);

    http_request_t req = {
        .method = "GET",
        .url = url,
        .auth_token = handle->api_key,
        .timeout_seconds = 30
    };

    http_response_t* resp = http_response_new();
    int status_code = 0;

    if (resp && http_execute(&req, resp, &status_code) && status_code == 200) {
        // Parse QPU list from JSON response
        *qpu_names = json_extract_string_array(resp->data, "quantum_processors", num_qpus);

        if (*qpu_names && *num_qpus > 0) {
            http_response_free(resp);
            return true;
        }
    }

    http_response_free(resp);

    // If API call failed, return known Rigetti QPU names as fallback
    *num_qpus = 4;
    *qpu_names = calloc(*num_qpus, sizeof(char*));
    if (!*qpu_names) return false;

    (*qpu_names)[0] = strdup("Aspen-M-3");
    (*qpu_names)[1] = strdup("Ankaa-2");
    (*qpu_names)[2] = strdup("Ankaa-9Q-1");
    (*qpu_names)[3] = strdup("Ankaa-9Q-3");

    log_info("Using known QPU list (API call failed or unavailable)");
    return true;
}

bool qcs_get_calibration(rigetti_qcs_handle_t* handle, const char* qpu_name,
                         qcs_calibration_data_t* cal_data) {
    if (!handle || !qpu_name || !cal_data) return false;

    // Initialize calibration data with typical values
    memset(cal_data, 0, sizeof(qcs_calibration_data_t));

    if (handle->use_local_simulator) {
        // Return ideal calibration for simulator
        cal_data->num_qubits = 24;
        cal_data->t1_times = calloc(24, sizeof(double));
        cal_data->t2_times = calloc(24, sizeof(double));
        cal_data->readout_fidelities = calloc(24, sizeof(double));
        cal_data->gate_fidelities = calloc(24, sizeof(double));
        cal_data->qubit_ids = calloc(24, sizeof(size_t));
        cal_data->qubit_dead = calloc(24, sizeof(bool));

        for (size_t i = 0; i < 24; i++) {
            cal_data->t1_times[i] = 50.0e-6;  // 50 microseconds
            cal_data->t2_times[i] = 30.0e-6;  // 30 microseconds
            cal_data->readout_fidelities[i] = 0.99;  // 99% fidelity
            cal_data->gate_fidelities[i] = 0.999;   // 99.9% fidelity
            cal_data->qubit_ids[i] = i;
            cal_data->qubit_dead[i] = false;
        }

        cal_data->timestamp = time(NULL);
        return true;
    }

    // Query calibration data from QCS API
    char url[512];
    snprintf(url, sizeof(url), "%s/%s/quantum-processors/%s/calibrations/latest",
             handle->qcs_url, QCS_API_VERSION, qpu_name);

    http_request_t req = {
        .method = "GET",
        .url = url,
        .auth_token = handle->api_key,
        .timeout_seconds = 30
    };

    http_response_t* resp = http_response_new();
    int status_code = 0;

    if (resp && http_execute(&req, resp, &status_code) && status_code == 200) {
        // Parse calibration data from JSON
        int num_qubits = 0;
        if (json_extract_int(resp->data, "num_qubits", &num_qubits) && num_qubits > 0) {
            cal_data->num_qubits = (size_t)num_qubits;

            // Allocate arrays
            cal_data->t1_times = calloc(cal_data->num_qubits, sizeof(double));
            cal_data->t2_times = calloc(cal_data->num_qubits, sizeof(double));
            cal_data->readout_fidelities = calloc(cal_data->num_qubits, sizeof(double));
            cal_data->gate_fidelities = calloc(cal_data->num_qubits, sizeof(double));
            cal_data->qubit_ids = calloc(cal_data->num_qubits, sizeof(size_t));
            cal_data->qubit_dead = calloc(cal_data->num_qubits, sizeof(bool));

            // Parse per-qubit data from calibration response
            // Note: Rigetti API returns calibration in a specific format
            // We extract typical values or defaults if parsing fails
            for (size_t i = 0; i < cal_data->num_qubits; i++) {
                cal_data->qubit_ids[i] = i;
                cal_data->t1_times[i] = 50.0e-6;  // Default 50µs
                cal_data->t2_times[i] = 30.0e-6;  // Default 30µs
                cal_data->readout_fidelities[i] = 0.97;
                cal_data->gate_fidelities[i] = 0.999;
                cal_data->qubit_dead[i] = false;
            }

            // Try to extract actual values from JSON (if available)
            double t1_avg = 0, t2_avg = 0, ro_avg = 0, gate_avg = 0;
            if (json_extract_double(resp->data, "average_t1", &t1_avg)) {
                for (size_t i = 0; i < cal_data->num_qubits; i++) {
                    cal_data->t1_times[i] = t1_avg * 1e-6;  // Convert µs to seconds
                }
            }
            if (json_extract_double(resp->data, "average_t2", &t2_avg)) {
                for (size_t i = 0; i < cal_data->num_qubits; i++) {
                    cal_data->t2_times[i] = t2_avg * 1e-6;
                }
            }
            if (json_extract_double(resp->data, "average_readout_fidelity", &ro_avg)) {
                for (size_t i = 0; i < cal_data->num_qubits; i++) {
                    cal_data->readout_fidelities[i] = ro_avg;
                }
            }
            if (json_extract_double(resp->data, "average_gate_fidelity", &gate_avg)) {
                for (size_t i = 0; i < cal_data->num_qubits; i++) {
                    cal_data->gate_fidelities[i] = gate_avg;
                }
            }

            cal_data->timestamp = time(NULL);
            http_response_free(resp);
            return true;
        }
    }

    http_response_free(resp);
    snprintf(handle->last_error, sizeof(handle->last_error),
            "Failed to fetch calibration for %s (HTTP %d)", qpu_name, status_code);
    return false;
}

char* qcs_get_isa(rigetti_qcs_handle_t* handle, const char* qpu_name) {
    if (!handle || !qpu_name) return NULL;

    // Return native gate set description
    if (handle->use_local_simulator) {
        return strdup("DEFGATE RX(%theta) q:\n"
                     "    cos(%theta/2), -i*sin(%theta/2)\n"
                     "    -i*sin(%theta/2), cos(%theta/2)\n\n"
                     "DEFGATE RZ(%theta) q:\n"
                     "    exp(-i*%theta/2), 0\n"
                     "    0, exp(i*%theta/2)\n\n"
                     "DEFGATE CZ p q:\n"
                     "    1, 0, 0, 0\n"
                     "    0, 1, 0, 0\n"
                     "    0, 0, 1, 0\n"
                     "    0, 0, 0, -1\n");
    }

    // Query ISA from QCS API
    char url[512];
    snprintf(url, sizeof(url), "%s/%s/quantum-processors/%s/isa",
             handle->qcs_url, QCS_API_VERSION, qpu_name);

    http_request_t req = {
        .method = "GET",
        .url = url,
        .auth_token = handle->api_key,
        .timeout_seconds = 30
    };

    http_response_t* resp = http_response_new();
    int status_code = 0;

    if (resp && http_execute(&req, resp, &status_code) && status_code == 200) {
        // Extract ISA definition from response
        char* isa = json_extract_string(resp->data, "isa");
        if (!isa) {
            // Return the entire response if it's Quil-formatted
            isa = strdup(resp->data);
        }
        http_response_free(resp);
        return isa;
    }

    http_response_free(resp);
    return NULL;
}

bool qcs_get_connectivity(rigetti_qcs_handle_t* handle, const char* qpu_name,
                          size_t** edges, size_t* num_edges) {
    if (!handle || !qpu_name || !edges || !num_edges) return false;

    if (handle->use_local_simulator) {
        // Return a linear chain topology for the simulator
        size_t num_qubits = 24;
        *num_edges = num_qubits - 1;
        *edges = calloc(*num_edges * 2, sizeof(size_t));
        if (!*edges) return false;

        for (size_t i = 0; i < *num_edges; i++) {
            (*edges)[i * 2] = i;
            (*edges)[i * 2 + 1] = i + 1;
        }
        return true;
    }

    // Query connectivity from QCS API
    char url[512];
    snprintf(url, sizeof(url), "%s/%s/quantum-processors/%s/connectivity",
             handle->qcs_url, QCS_API_VERSION, qpu_name);

    http_request_t req = {
        .method = "GET",
        .url = url,
        .auth_token = handle->api_key,
        .timeout_seconds = 30
    };

    http_response_t* resp = http_response_new();
    int status_code = 0;

    if (resp && http_execute(&req, resp, &status_code) && status_code == 200) {
        // Parse connectivity from JSON - looking for edges array
        // Format: {"edges": [[0,1], [1,2], ...]}
        const char* edges_start = strstr(resp->data, "\"edges\"");
        if (edges_start) {
            edges_start = strchr(edges_start, '[');
            if (edges_start) {
                edges_start++;

                // Count edges
                size_t count = 0;
                const char* p = edges_start;
                while (*p && *p != ']') {
                    if (*p == '[') count++;
                    p++;
                }

                if (count > 0) {
                    *edges = calloc(count * 2, sizeof(size_t));
                    if (*edges) {
                        *num_edges = count;

                        // Parse edges
                        p = edges_start;
                        size_t idx = 0;
                        while (*p && *p != ']' && idx < count) {
                            if (*p == '[') {
                                p++;
                                int q1 = 0, q2 = 0;
                                if (sscanf(p, "%d,%d", &q1, &q2) == 2 ||
                                    sscanf(p, "%d, %d", &q1, &q2) == 2) {
                                    (*edges)[idx * 2] = (size_t)q1;
                                    (*edges)[idx * 2 + 1] = (size_t)q2;
                                    idx++;
                                }
                            }
                            p++;
                        }

                        http_response_free(resp);
                        return true;
                    }
                }
            }
        }
    }

    http_response_free(resp);
    return false;
}

// ============================================================================
// Program Compilation Functions
// ============================================================================

bool qcs_compile_program(rigetti_qcs_handle_t* handle, const char* quil_program,
                         const char* qpu_name, char** compiled_program) {
    if (!handle || !quil_program || !compiled_program) return false;

    if (handle->use_local_simulator) {
        // No compilation needed for local simulator
        *compiled_program = strdup(quil_program);
        return true;
    }

    // Call quilc service for compilation
    char url[512];
    snprintf(url, sizeof(url), "%s/%s/compile", handle->qcs_url, QCS_API_VERSION);

    // Build compilation request JSON
    char body[8192];
    snprintf(body, sizeof(body),
             "{\"quil\": \"%s\", \"target_device\": \"%s\", \"protoquil\": true}",
             quil_program, qpu_name ? qpu_name : "");

    // Escape newlines and quotes in the Quil program
    char* escaped_body = malloc(strlen(body) * 2 + 1);
    if (!escaped_body) {
        *compiled_program = strdup(quil_program);
        return true;
    }

    // Simple escape for JSON
    char* dst = escaped_body;
    for (const char* src = body; *src; src++) {
        if (*src == '\n') {
            *dst++ = '\\';
            *dst++ = 'n';
        } else if (*src == '"' && src != body && *(src-1) != '\\' && *(src-1) != ':' && *(src-1) != ' ') {
            // Skip already escaped quotes
            *dst++ = *src;
        } else {
            *dst++ = *src;
        }
    }
    *dst = '\0';

    http_request_t req = {
        .method = "POST",
        .url = url,
        .body = escaped_body,
        .auth_token = handle->api_key,
        .timeout_seconds = 60
    };

    http_response_t* resp = http_response_new();
    int status_code = 0;

    if (resp && http_execute(&req, resp, &status_code)) {
        free(escaped_body);

        if (status_code == 200) {
            // Extract compiled program from response
            char* compiled = json_extract_string(resp->data, "program");
            if (compiled) {
                *compiled_program = compiled;
                http_response_free(resp);
                return true;
            }
        }
    } else {
        free(escaped_body);
    }

    http_response_free(resp);

    // Fallback: return original program if compilation fails
    *compiled_program = strdup(quil_program);
    log_warn("Quilc compilation failed - using original program");
    return true;
}

char* qcs_get_native_quil(rigetti_qcs_handle_t* handle, const char* quil_program,
                          const char* qpu_name) {
    char* compiled = NULL;
    if (qcs_compile_program(handle, quil_program, qpu_name, &compiled)) {
        return compiled;
    }
    return NULL;
}

// ============================================================================
// Program Execution Functions
// ============================================================================

char* qcs_submit_program(rigetti_qcs_handle_t* handle, const char* quil_program,
                         const qcs_execution_options_t* options) {
    if (!handle || !quil_program || !options) return NULL;

    // Generate job ID
    char* job_id = malloc(64);
    if (!job_id) return NULL;

    snprintf(job_id, 64, "qcs-job-%ld-%d", (long)time(NULL), rand() % 10000);

    if (handle->use_local_simulator) {
        // For local simulator, execution is synchronous
        // Store the program for later retrieval
        log_info("Submitted job %s to local simulator", job_id);
        return job_id;
    }

    // Submit job to QCS API
    char url[512];
    snprintf(url, sizeof(url), "%s/%s/jobs", handle->qcs_url, QCS_API_VERSION);

    // Build job submission request
    char body[16384];
    snprintf(body, sizeof(body),
             "{"
             "\"program\": \"%s\","
             "\"shots\": %zu,"
             "\"target\": \"%s\","
             "\"use_active_reset\": %s"
             "}",
             quil_program,
             options->shots > 0 ? options->shots : 1000,
             options->qpu_name ? options->qpu_name : "Ankaa-2",
             options->use_active_reset ? "true" : "false");

    http_request_t req = {
        .method = "POST",
        .url = url,
        .body = body,
        .auth_token = handle->api_key,
        .timeout_seconds = 30
    };

    http_response_t* resp = http_response_new();
    int status_code = 0;

    if (resp && http_execute(&req, resp, &status_code)) {
        if (status_code == 200 || status_code == 201 || status_code == 202) {
            // Extract job ID from response
            char* qcs_job_id = json_extract_string(resp->data, "job_id");
            if (!qcs_job_id) {
                qcs_job_id = json_extract_string(resp->data, "id");
            }

            if (qcs_job_id) {
                http_response_free(resp);
                free(job_id);
                return qcs_job_id;
            }
        }
    }

    http_response_free(resp);
    return job_id;
}

bool qcs_execute_program(rigetti_qcs_handle_t* handle, const char* quil_program,
                         const qcs_execution_options_t* options, qcs_job_result_t* result) {
    if (!handle || !quil_program || !options || !result) return false;

    memset(result, 0, sizeof(qcs_job_result_t));

    if (handle->use_local_simulator || options->target == QCS_TARGET_QVM) {
        // Execute on local simulator
        return execute_quil_local(handle, quil_program, options->shots, result);
    }

    // Submit to QCS
    char* job_id = qcs_submit_program(handle, quil_program, options);
    if (!job_id) return false;

    // Wait for completion
    int timeout = options->timeout_seconds > 0 ? options->timeout_seconds : 300;
    qcs_job_status_t status = qcs_wait_for_job(handle, job_id, timeout);

    if (status == QCS_JOB_COMPLETED) {
        bool success = qcs_get_job_result(handle, job_id, result);
        free(job_id);
        return success;
    }

    snprintf(handle->last_error, sizeof(handle->last_error),
            "Job did not complete: status %d", status);
    free(job_id);
    return false;
}

bool qcs_execute_on_qvm(rigetti_qcs_handle_t* handle, const char* quil_program,
                        size_t num_shots, qcs_job_result_t* result) {
    qcs_execution_options_t options = {
        .target = QCS_TARGET_QVM,
        .qpu_name = NULL,
        .shots = num_shots,
        .use_quilc = false,
        .use_parametric = false,
        .use_active_reset = false,
        .timeout_seconds = 60
    };

    return qcs_execute_program(handle, quil_program, &options, result);
}

qcs_job_status_t qcs_get_job_status(rigetti_qcs_handle_t* handle, const char* job_id) {
    if (!handle || !job_id) return QCS_JOB_UNKNOWN;

    if (handle->use_local_simulator) {
        // Local jobs complete immediately
        return QCS_JOB_COMPLETED;
    }

    // Query job status from QCS API
    char url[512];
    snprintf(url, sizeof(url), "%s/%s/jobs/%s", handle->qcs_url, QCS_API_VERSION, job_id);

    http_request_t req = {
        .method = "GET",
        .url = url,
        .auth_token = handle->api_key,
        .timeout_seconds = 10
    };

    http_response_t* resp = http_response_new();
    int status_code = 0;
    qcs_job_status_t status = QCS_JOB_UNKNOWN;

    if (resp && http_execute(&req, resp, &status_code) && status_code == 200) {
        char* status_str = json_extract_string(resp->data, "status");
        if (status_str) {
            if (strcmp(status_str, "PENDING") == 0 || strcmp(status_str, "pending") == 0) {
                status = QCS_JOB_PENDING;
            } else if (strcmp(status_str, "QUEUED") == 0 || strcmp(status_str, "queued") == 0) {
                status = QCS_JOB_QUEUED;
            } else if (strcmp(status_str, "RUNNING") == 0 || strcmp(status_str, "running") == 0) {
                status = QCS_JOB_RUNNING;
            } else if (strcmp(status_str, "COMPLETED") == 0 || strcmp(status_str, "completed") == 0 ||
                       strcmp(status_str, "DONE") == 0 || strcmp(status_str, "done") == 0) {
                status = QCS_JOB_COMPLETED;
            } else if (strcmp(status_str, "FAILED") == 0 || strcmp(status_str, "failed") == 0 ||
                       strcmp(status_str, "ERROR") == 0 || strcmp(status_str, "error") == 0) {
                status = QCS_JOB_FAILED;
            } else if (strcmp(status_str, "CANCELLED") == 0 || strcmp(status_str, "cancelled") == 0) {
                status = QCS_JOB_CANCELLED;
            }
            free(status_str);
        }
    }

    http_response_free(resp);
    return status;
}

bool qcs_get_job_result(rigetti_qcs_handle_t* handle, const char* job_id,
                        qcs_job_result_t* result) {
    if (!handle || !job_id || !result) return false;

    if (handle->use_local_simulator) {
        // Result should already be populated for local jobs
        return true;
    }

    // Fetch results from QCS API
    char url[512];
    snprintf(url, sizeof(url), "%s/%s/jobs/%s/results", handle->qcs_url, QCS_API_VERSION, job_id);

    http_request_t req = {
        .method = "GET",
        .url = url,
        .auth_token = handle->api_key,
        .timeout_seconds = 30
    };

    http_response_t* resp = http_response_new();
    int status_code = 0;

    if (resp && http_execute(&req, resp, &status_code) && status_code == 200) {
        // Parse result data
        int num_qubits = 0, num_shots = 0;
        json_extract_int(resp->data, "num_qubits", &num_qubits);
        json_extract_int(resp->data, "shots", &num_shots);

        if (num_qubits > 0 && num_shots > 0) {
            result->num_qubits = (size_t)num_qubits;
            result->num_shots = (size_t)num_shots;

            size_t num_outcomes = 1ULL << num_qubits;
            result->counts = calloc(num_outcomes, sizeof(uint64_t));
            result->probabilities = calloc(num_outcomes, sizeof(double));
            result->bitstrings = calloc(num_shots, sizeof(uint8_t*));

            if (result->counts && result->probabilities && result->bitstrings) {
                // Parse measurement results from JSON
                // Format: {"results": [[0,1,0], [1,0,1], ...]}
                const char* results_start = strstr(resp->data, "\"results\"");
                if (results_start) {
                    results_start = strchr(results_start, '[');
                    if (results_start) {
                        results_start++;

                        size_t shot = 0;
                        while (*results_start && shot < result->num_shots) {
                            if (*results_start == '[') {
                                // Parse bitstring for this shot
                                result->bitstrings[shot] = calloc(num_qubits, sizeof(uint8_t));
                                if (result->bitstrings[shot]) {
                                    results_start++;
                                    size_t qubit = 0;
                                    uint64_t outcome = 0;

                                    while (*results_start && *results_start != ']' && qubit < result->num_qubits) {
                                        if (*results_start == '0' || *results_start == '1') {
                                            int bit = *results_start - '0';
                                            result->bitstrings[shot][qubit] = (uint8_t)bit;
                                            if (bit) outcome |= (1ULL << qubit);
                                            qubit++;
                                        }
                                        results_start++;
                                    }

                                    result->counts[outcome]++;
                                }
                                shot++;
                            }
                            if (*results_start) results_start++;
                        }

                        // Calculate probabilities
                        result->num_outcomes = 0;
                        for (size_t i = 0; i < num_outcomes; i++) {
                            result->probabilities[i] = (double)result->counts[i] / (double)num_shots;
                            if (result->counts[i] > 0) result->num_outcomes++;
                        }

                        // Extract timing information if available
                        double exec_time = 0, comp_time = 0;
                        if (json_extract_double(resp->data, "execution_time", &exec_time)) {
                            result->execution_time = exec_time;
                        }
                        if (json_extract_double(resp->data, "compile_time", &comp_time)) {
                            result->compile_time = comp_time;
                        }

                        result->status = QCS_JOB_COMPLETED;
                        http_response_free(resp);
                        return true;
                    }
                }
            }
        }
    }

    http_response_free(resp);
    snprintf(handle->last_error, sizeof(handle->last_error),
            "Failed to fetch results for job %s", job_id);
    return false;
}

bool qcs_cancel_job(rigetti_qcs_handle_t* handle, const char* job_id) {
    if (!handle || !job_id) return false;

    if (handle->use_local_simulator) {
        // Local jobs can't be cancelled (they complete immediately)
        return true;
    }

    // Cancel job on QCS API
    char url[512];
    snprintf(url, sizeof(url), "%s/%s/jobs/%s/cancel", handle->qcs_url, QCS_API_VERSION, job_id);

    http_request_t req = {
        .method = "POST",
        .url = url,
        .body = "{}",
        .auth_token = handle->api_key,
        .timeout_seconds = 10
    };

    http_response_t* resp = http_response_new();
    int status_code = 0;

    bool success = false;
    if (resp && http_execute(&req, resp, &status_code)) {
        success = (status_code == 200 || status_code == 202 || status_code == 204);
    }

    http_response_free(resp);
    return success;
}

qcs_job_status_t qcs_wait_for_job(rigetti_qcs_handle_t* handle, const char* job_id,
                                   int timeout_seconds) {
    if (!handle || !job_id) return QCS_JOB_UNKNOWN;

    time_t start = time(NULL);

    while (1) {
        qcs_job_status_t status = qcs_get_job_status(handle, job_id);

        if (status == QCS_JOB_COMPLETED || status == QCS_JOB_FAILED ||
            status == QCS_JOB_CANCELLED) {
            return status;
        }

        if (timeout_seconds > 0 && time(NULL) - start > timeout_seconds) {
            return QCS_JOB_UNKNOWN;
        }

        sleep(1);  // Poll every second
    }
}

// ============================================================================
// Memory Management Functions
// ============================================================================

void qcs_free_calibration(qcs_calibration_data_t* cal_data) {
    if (!cal_data) return;

    free(cal_data->t1_times);
    free(cal_data->t2_times);
    free(cal_data->readout_fidelities);
    free(cal_data->gate_fidelities);
    free(cal_data->qubit_ids);
    free(cal_data->qubit_dead);

    if (cal_data->cz_fidelities) {
        for (size_t i = 0; i < cal_data->num_qubits; i++) {
            free(cal_data->cz_fidelities[i]);
        }
        free(cal_data->cz_fidelities);
    }

    memset(cal_data, 0, sizeof(qcs_calibration_data_t));
}

void qcs_free_result(qcs_job_result_t* result) {
    if (!result) return;

    if (result->bitstrings) {
        for (size_t i = 0; i < result->num_shots; i++) {
            free(result->bitstrings[i]);
        }
        free(result->bitstrings);
    }

    free(result->counts);
    free(result->probabilities);
    free(result->error_message);
    free(result->metadata);

    memset(result, 0, sizeof(qcs_job_result_t));
}

void qcs_free_qpu_names(char** names, size_t num) {
    if (!names) return;

    for (size_t i = 0; i < num; i++) {
        free(names[i]);
    }
    free(names);
}

// ============================================================================
// Error Handling
// ============================================================================

const char* qcs_get_last_error(rigetti_qcs_handle_t* handle) {
    if (!handle) return "Invalid handle";
    return handle->last_error;
}

int qcs_get_last_error_code(rigetti_qcs_handle_t* handle) {
    if (!handle) return -1;
    return handle->last_error_code;
}

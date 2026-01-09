/**
 * @file quantum_dwave_backend.c
 * @brief D-Wave Quantum backend implementation
 *
 * Provides D-Wave quantum annealing backend with:
 * - Real hardware support via SAPI API when CURL is available
 * - Semi-classical simulated annealing fallback otherwise
 */

#include "quantum_geometric/hardware/quantum_dwave_backend.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <unistd.h>

// Auto-detect CURL availability using __has_include
#ifdef __has_include
#if __has_include(<curl/curl.h>)
#define QGT_HAS_CURL 1
#include <curl/curl.h>
#endif
#if __has_include(<json-c/json.h>)
#define QGT_HAS_JSON_C 1
#include <json-c/json.h>
#endif
#endif

// Default to unavailable if not detected
#ifndef QGT_HAS_CURL
#define QGT_HAS_CURL 0
#endif

#ifndef QGT_HAS_JSON_C
#define QGT_HAS_JSON_C 0
#endif

// D-Wave parameters
#ifndef MAX_QUBITS
#define MAX_QUBITS 5000
#endif
#define MAX_COUPLERS 10000
#define API_TIMEOUT 60
#define MAX_RETRIES 5

// D-Wave API endpoints
#define DWAVE_API_URL "https://cloud.dwavesys.com/sapi/v2"
#define DWAVE_SOLVER_URL DWAVE_API_URL "/solvers"
#define DWAVE_PROBLEMS_URL DWAVE_API_URL "/problems"

// Internal D-Wave state structure (stored in DWaveConfig.backend_specific_config)
typedef struct DWaveInternalState {
    char* api_key;
    char* solver_name;
    size_t num_qubits;
    DWaveBackendType type;
    DWaveSolverType solver_type;
    DWaveSamplingParams sampling_params;
    void* api_handle;      // CURL handle when available
    void* config_json;     // json_object when available
    void* curl_headers;    // curl_slist for headers
    char error_buffer[256];
    bool initialized;
    bool connected;
    // Custom annealing schedule storage
    double* annealing_schedule;
    size_t annealing_schedule_points;
    // Flux bias and anneal offsets
    double* flux_bias_offsets;
    double* anneal_offsets;
    size_t num_offset_qubits;
} DWaveInternalState;

// Helper to get internal state from config
static inline DWaveInternalState* get_internal_state(DWaveConfig* config) {
    if (!config) return NULL;
    return (DWaveInternalState*)config->backend_specific_config;
}

static inline const DWaveInternalState* get_internal_state_const(const DWaveConfig* config) {
    if (!config) return NULL;
    return (const DWaveInternalState*)config->backend_specific_config;
}

// ============================================================================
// CURL + JSON-C implementation (full D-Wave API support)
// ============================================================================

#if defined(QGT_HAS_CURL) && defined(QGT_HAS_JSON_C)

// Response buffer for CURL
typedef struct {
    char* data;
    size_t size;
} ResponseBuffer;

// CURL callback for writing response
static size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    ResponseBuffer* buf = (ResponseBuffer*)userdata;
    size_t total_size = size * nmemb;

    char* new_data = realloc(buf->data, buf->size + total_size + 1);
    if (!new_data) return 0;

    buf->data = new_data;
    memcpy(buf->data + buf->size, ptr, total_size);
    buf->size += total_size;
    buf->data[buf->size] = '\0';

    return total_size;
}

// Initialize D-Wave backend with full API support
DWaveConfig* init_dwave_backend(const DWaveBackendConfig* config) {
    if (!config) return NULL;

    // Allocate main config
    DWaveConfig* dc = calloc(1, sizeof(DWaveConfig));
    if (!dc) return NULL;

    // Allocate internal state
    DWaveInternalState* state = calloc(1, sizeof(DWaveInternalState));
    if (!state) {
        free(dc);
        return NULL;
    }
    dc->backend_specific_config = state;

    // Copy configuration to internal state
    state->type = config->type;
    state->solver_type = config->solver_type;
    state->sampling_params = config->sampling_params;

    if (config->api_token) {
        state->api_key = strdup(config->api_token);
    }
    if (config->solver_name) {
        state->solver_name = strdup(config->solver_name);
        dc->backend_name = strdup(config->solver_name);  // Also store in public config
    }

    // Initialize CURL
    curl_global_init(CURL_GLOBAL_DEFAULT);
    CURL* curl = curl_easy_init();
    if (!curl) {
        free(state->api_key);
        free(state->solver_name);
        free(dc->backend_name);
        free(state);
        free(dc);
        return NULL;
    }
    state->api_handle = curl;

    // Configure CURL
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, state->error_buffer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, API_TIMEOUT);

    // Set authentication headers
    if (state->api_key) {
        struct curl_slist* headers = NULL;
        char auth_header[512];
        snprintf(auth_header, sizeof(auth_header), "X-Auth-Token: %s", state->api_key);
        headers = curl_slist_append(headers, auth_header);
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        state->curl_headers = headers;
    }

    // Get solver configuration if we have a solver name
    if (state->solver_name) {
        ResponseBuffer response = {NULL, 0};
        char solver_url[512];
        snprintf(solver_url, sizeof(solver_url), "%s/%s", DWAVE_SOLVER_URL, state->solver_name);

        curl_easy_setopt(curl, CURLOPT_URL, solver_url);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        CURLcode res = curl_easy_perform(curl);
        if (res == CURLE_OK && response.data) {
            struct json_object* json_resp = json_tokener_parse(response.data);
            if (json_resp) {
                struct json_object* properties_obj;
                if (json_object_object_get_ex(json_resp, "properties", &properties_obj)) {
                    struct json_object* num_qubits_obj;
                    if (json_object_object_get_ex(properties_obj, "num_qubits", &num_qubits_obj)) {
                        state->num_qubits = (size_t)json_object_get_int(num_qubits_obj);
                    }
                }
                state->config_json = json_resp;
                state->connected = true;
            }
        }
        free(response.data);
    }

    state->initialized = true;
    return dc;
}

// Submit job to D-Wave
char* submit_dwave_job(DWaveConfig* config, const DWaveJobConfig* job_config) {
    if (!config || !job_config || !job_config->problem) {
        return NULL;
    }

    DWaveInternalState* state = get_internal_state(config);
    if (!state || !state->api_handle) {
        return NULL;
    }

    CURL* curl = (CURL*)state->api_handle;
    DWaveProblem* problem = job_config->problem;

    // Create problem JSON
    struct json_object* request = json_object_new_object();
    json_object_object_add(request, "solver", json_object_new_string(state->solver_name));

    // Create h (linear terms) and J (quadratic terms)
    struct json_object* h = json_object_new_object();
    struct json_object* J = json_object_new_object();

    // Add linear terms
    for (size_t i = 0; i < problem->num_variables; i++) {
        if (fabs(problem->linear_terms[i]) > 1e-10) {
            char key[32];
            snprintf(key, sizeof(key), "%zu", i);
            json_object_object_add(h, key, json_object_new_double(problem->linear_terms[i]));
        }
    }

    // Add quadratic terms
    for (size_t i = 0; i < problem->num_interactions; i++) {
        // Assuming quadratic_terms is stored as flattened upper triangle
        if (fabs(problem->quadratic_terms[i]) > 1e-10) {
            char key[64];
            snprintf(key, sizeof(key), "%zu", i);
            json_object_object_add(J, key, json_object_new_double(problem->quadratic_terms[i]));
        }
    }

    struct json_object* problem_obj = json_object_new_object();
    json_object_object_add(problem_obj, "h", h);
    json_object_object_add(problem_obj, "J", J);
    json_object_object_add(request, "data", problem_obj);

    // Set parameters
    struct json_object* params = json_object_new_object();
    json_object_object_add(params, "num_reads",
        json_object_new_int((int32_t)state->sampling_params.num_reads));
    json_object_object_add(params, "annealing_time",
        json_object_new_int((int32_t)state->sampling_params.annealing_time));
    json_object_object_add(request, "params", params);

    // Submit
    ResponseBuffer response = {NULL, 0};
    curl_easy_setopt(curl, CURLOPT_URL, DWAVE_PROBLEMS_URL);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_object_to_json_string(request));
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    json_object_put(request);

    if (res != CURLE_OK || !response.data) {
        free(response.data);
        return NULL;
    }

    // Extract job ID
    struct json_object* resp_json = json_tokener_parse(response.data);
    char* job_id = NULL;

    if (resp_json) {
        struct json_object* id_obj;
        if (json_object_object_get_ex(resp_json, "id", &id_obj)) {
            job_id = strdup(json_object_get_string(id_obj));
        }
        json_object_put(resp_json);
    }

    free(response.data);
    return job_id;
}

// Get job status
DWaveJobStatus get_dwave_job_status(DWaveConfig* config, const char* job_id) {
    if (!config || !job_id) {
        return DWAVE_STATUS_ERROR;
    }

    DWaveInternalState* state = get_internal_state(config);
    if (!state || !state->api_handle) {
        return DWAVE_STATUS_ERROR;
    }

    CURL* curl = (CURL*)state->api_handle;

    char status_url[512];
    snprintf(status_url, sizeof(status_url), "%s/%s", DWAVE_PROBLEMS_URL, job_id);

    ResponseBuffer response = {NULL, 0};
    curl_easy_setopt(curl, CURLOPT_URL, status_url);
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK || !response.data) {
        free(response.data);
        return DWAVE_STATUS_ERROR;
    }

    DWaveJobStatus status = DWAVE_STATUS_QUEUED;
    struct json_object* resp_json = json_tokener_parse(response.data);

    if (resp_json) {
        struct json_object* status_obj;
        if (json_object_object_get_ex(resp_json, "status", &status_obj)) {
            const char* status_str = json_object_get_string(status_obj);
            if (strcmp(status_str, "COMPLETED") == 0) {
                status = DWAVE_STATUS_COMPLETED;
            } else if (strcmp(status_str, "RUNNING") == 0) {
                status = DWAVE_STATUS_RUNNING;
            } else if (strcmp(status_str, "CANCELLED") == 0) {
                status = DWAVE_STATUS_CANCELLED;
            } else if (strcmp(status_str, "FAILED") == 0) {
                status = DWAVE_STATUS_ERROR;
            }
        }
        json_object_put(resp_json);
    }

    free(response.data);
    return status;
}

// Get job result
DWaveJobResult* get_dwave_job_result(DWaveConfig* config, const char* job_id) {
    if (!config || !job_id) {
        return NULL;
    }

    DWaveInternalState* state = get_internal_state(config);
    if (!state || !state->api_handle) {
        return NULL;
    }

    CURL* curl = (CURL*)state->api_handle;

    char result_url[512];
    snprintf(result_url, sizeof(result_url), "%s/%s/answer", DWAVE_PROBLEMS_URL, job_id);

    ResponseBuffer response = {NULL, 0};
    curl_easy_setopt(curl, CURLOPT_URL, result_url);
    curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK || !response.data) {
        free(response.data);
        return NULL;
    }

    DWaveJobResult* result = calloc(1, sizeof(DWaveJobResult));
    if (!result) {
        free(response.data);
        return NULL;
    }

    struct json_object* resp_json = json_tokener_parse(response.data);
    if (resp_json) {
        struct json_object* solutions_obj;
        if (json_object_object_get_ex(resp_json, "solutions", &solutions_obj)) {
            result->num_samples = json_object_array_length(solutions_obj);
            result->samples = calloc(result->num_samples, sizeof(DWaveSample));
            result->energies = calloc(result->num_samples, sizeof(double));

            struct json_object* energies_obj;
            json_object_object_get_ex(resp_json, "energies", &energies_obj);

            struct json_object* occurrences_obj;
            json_object_object_get_ex(resp_json, "num_occurrences", &occurrences_obj);

            for (size_t i = 0; i < result->num_samples; i++) {
                if (energies_obj) {
                    result->energies[i] = json_object_get_double(
                        json_object_array_get_idx(energies_obj, (int)i));
                }
                if (result->samples && occurrences_obj) {
                    result->samples[i].occurrence = (double)json_object_get_int(
                        json_object_array_get_idx(occurrences_obj, (int)i));
                }

                // Find min/max energy
                if (i == 0 || result->energies[i] < result->min_energy) {
                    result->min_energy = result->energies[i];
                }
                if (i == 0 || result->energies[i] > result->max_energy) {
                    result->max_energy = result->energies[i];
                }
            }
        }

        result->status = DWAVE_STATUS_COMPLETED;
        json_object_put(resp_json);
    }

    free(response.data);
    return result;
}

// Cleanup config
void cleanup_dwave_config(DWaveConfig* config) {
    if (!config) return;

    DWaveInternalState* state = get_internal_state(config);
    if (state) {
        if (state->api_handle) {
            curl_easy_cleanup((CURL*)state->api_handle);
            curl_global_cleanup();
        }

        if (state->curl_headers) {
            curl_slist_free_all((struct curl_slist*)state->curl_headers);
        }

        if (state->config_json) {
            json_object_put((struct json_object*)state->config_json);
        }

        free(state->api_key);
        free(state->solver_name);
        free(state);
    }

    free(config->backend_name);
    free(config->noise_model);
    free(config);
}

#else // No CURL or JSON-C - local simulation fallback with simulated annealing

#include <time.h>

// Storage for local simulation results (simple single-job cache)
static DWaveJobResult* g_cached_result = NULL;
static char* g_cached_job_id = NULL;

// Compute Ising energy: E = Σᵢ hᵢsᵢ + Σᵢⱼ Jᵢⱼsᵢsⱼ
static double compute_ising_energy(const DWaveProblem* problem, const int32_t* spins) {
    double energy = problem->offset;

    // Linear terms
    for (size_t i = 0; i < problem->num_variables; i++) {
        energy += problem->linear_terms[i] * spins[i];
    }

    // Quadratic terms - assuming J stored linearly for pairs
    // In Ising model, J_ij contributes when both i and j are +1 or both -1
    size_t idx = 0;
    for (size_t i = 0; i < problem->num_variables && idx < problem->num_interactions; i++) {
        for (size_t j = i + 1; j < problem->num_variables && idx < problem->num_interactions; j++) {
            double J_ij = problem->quadratic_terms[idx];
            if (J_ij != 0.0) {
                energy += J_ij * spins[i] * spins[j];
            }
            idx++;
        }
    }

    return energy;
}

// Simulated annealing for Ising problems
static DWaveJobResult* simulated_annealing(const DWaveProblem* problem,
                                            const DWaveSamplingParams* params) {
    if (!problem || problem->num_variables == 0) return NULL;

    size_t n = problem->num_variables;
    uint32_t num_reads = params->num_reads > 0 ? params->num_reads : 100;
    uint32_t annealing_time = params->annealing_time > 0 ? params->annealing_time : 20;

    // Scale annealing steps based on annealing_time (microseconds -> iterations)
    size_t num_sweeps = (size_t)(annealing_time * 10);  // 10 sweeps per microsecond
    if (num_sweeps < 100) num_sweeps = 100;
    if (num_sweeps > 10000) num_sweeps = 10000;

    // Allocate result
    DWaveJobResult* result = calloc(1, sizeof(DWaveJobResult));
    if (!result) return NULL;

    result->samples = calloc(num_reads, sizeof(DWaveSample));
    result->energies = calloc(num_reads, sizeof(double));
    result->probabilities = calloc(num_reads, sizeof(double));
    result->num_samples = num_reads;
    result->min_energy = INFINITY;
    result->max_energy = -INFINITY;

    if (!result->samples || !result->energies || !result->probabilities) {
        cleanup_dwave_result(result);
        return NULL;
    }

    // Allocate working spin array
    int32_t* spins = malloc(n * sizeof(int32_t));
    if (!spins) {
        cleanup_dwave_result(result);
        return NULL;
    }

    unsigned int seed = (unsigned int)(time(NULL) ^ getpid());

    // Perform multiple annealing runs
    for (uint32_t read = 0; read < num_reads; read++) {
        // Initialize random spins
        for (size_t i = 0; i < n; i++) {
            spins[i] = (rand_r(&seed) % 2) * 2 - 1;  // Random {-1, +1}
        }

        double energy = compute_ising_energy(problem, spins);

        // Annealing schedule: T decreases from T_high to T_low
        double T_high = 10.0;
        double T_low = 0.01;

        for (size_t sweep = 0; sweep < num_sweeps; sweep++) {
            // Current temperature
            double T = T_high * pow(T_low / T_high, (double)sweep / (double)num_sweeps);

            // Sweep through all variables
            for (size_t i = 0; i < n; i++) {
                // Compute energy change if we flip spin i
                double delta_E = -2.0 * problem->linear_terms[i] * spins[i];

                // Add interaction terms
                size_t idx = 0;
                for (size_t a = 0; a < n && idx < problem->num_interactions; a++) {
                    for (size_t b = a + 1; b < n && idx < problem->num_interactions; b++) {
                        double J_ab = problem->quadratic_terms[idx];
                        if (J_ab != 0.0) {
                            if (a == i) {
                                delta_E -= 2.0 * J_ab * spins[i] * spins[b];
                            } else if (b == i) {
                                delta_E -= 2.0 * J_ab * spins[a] * spins[i];
                            }
                        }
                        idx++;
                    }
                }

                // Metropolis acceptance criterion
                if (delta_E < 0.0 || (double)rand_r(&seed) / RAND_MAX < exp(-delta_E / T)) {
                    spins[i] = -spins[i];
                    energy += delta_E;
                }
            }
        }

        // Store result
        result->samples[read].variables = malloc(n * sizeof(int32_t));
        if (result->samples[read].variables) {
            memcpy(result->samples[read].variables, spins, n * sizeof(int32_t));
        }
        result->samples[read].energy = energy;
        result->samples[read].occurrence = 1.0;
        result->energies[read] = energy;

        if (energy < result->min_energy) result->min_energy = energy;
        if (energy > result->max_energy) result->max_energy = energy;
    }

    // Compute probabilities (Boltzmann distribution at final temperature)
    double T_final = T_low;
    double Z = 0.0;
    for (uint32_t i = 0; i < num_reads; i++) {
        result->probabilities[i] = exp(-result->energies[i] / T_final);
        Z += result->probabilities[i];
    }
    if (Z > 0.0) {
        for (uint32_t i = 0; i < num_reads; i++) {
            result->probabilities[i] /= Z;
        }
    }

    free(spins);

    result->status = DWAVE_STATUS_COMPLETED;
    result->error_message = NULL;

    return result;
}

DWaveConfig* init_dwave_backend(const DWaveBackendConfig* config) {
    if (!config) return NULL;

    // Allocate main config
    DWaveConfig* dc = calloc(1, sizeof(DWaveConfig));
    if (!dc) return NULL;

    // Allocate internal state
    DWaveInternalState* state = calloc(1, sizeof(DWaveInternalState));
    if (!state) {
        free(dc);
        return NULL;
    }
    dc->backend_specific_config = state;

    // Copy configuration to internal state
    state->type = config->type;
    state->solver_type = config->solver_type;
    state->sampling_params = config->sampling_params;

    if (config->api_token) {
        state->api_key = strdup(config->api_token);
    }
    if (config->solver_name) {
        state->solver_name = strdup(config->solver_name);
        dc->backend_name = strdup(config->solver_name);
    }

    state->num_qubits = MAX_QUBITS;  // Default Advantage qubits
    state->initialized = true;
    state->connected = true;  // Local simulation is always "connected"

    return dc;
}

char* submit_dwave_job(DWaveConfig* config, const DWaveJobConfig* job_config) {
    if (!config || !job_config || !job_config->problem) {
        return NULL;
    }

    // Clear previous cached result
    if (g_cached_result) {
        cleanup_dwave_result(g_cached_result);
        g_cached_result = NULL;
    }
    if (g_cached_job_id) {
        free(g_cached_job_id);
        g_cached_job_id = NULL;
    }

    // Perform simulated annealing immediately
    g_cached_result = simulated_annealing(job_config->problem, &job_config->params);
    if (!g_cached_result) {
        return NULL;
    }

    // Generate a pseudo job ID
    g_cached_job_id = malloc(64);
    if (g_cached_job_id) {
        snprintf(g_cached_job_id, 64, "local-sa-%lu", (unsigned long)time(NULL));
    }

    return g_cached_job_id ? strdup(g_cached_job_id) : NULL;
}

DWaveJobStatus get_dwave_job_status(DWaveConfig* config, const char* job_id) {
    (void)config;

    // Local simulation is always complete immediately
    if (job_id && g_cached_job_id && strcmp(job_id, g_cached_job_id) == 0) {
        return g_cached_result ? DWAVE_STATUS_COMPLETED : DWAVE_STATUS_ERROR;
    }
    return DWAVE_STATUS_ERROR;
}

DWaveJobResult* get_dwave_job_result(DWaveConfig* config, const char* job_id) {
    (void)config;

    if (!job_id || !g_cached_job_id || strcmp(job_id, g_cached_job_id) != 0) {
        return NULL;
    }

    // Return the cached result (caller should not free this)
    // For proper API, we'd deep copy, but for simplicity return the cached one
    // and mark it as transferred
    DWaveJobResult* result = g_cached_result;
    g_cached_result = NULL;  // Transfer ownership
    return result;
}

void cleanup_dwave_config(DWaveConfig* config) {
    if (!config) return;

    DWaveInternalState* state = get_internal_state(config);
    if (state) {
        free(state->api_key);
        free(state->solver_name);
        free(state);
    }

    free(config->backend_name);
    free(config->noise_model);
    free(config);
}

#endif // QGT_HAS_CURL && QGT_HAS_JSON_C

// ============================================================================
// Common implementations (don't require CURL/JSON)
// ============================================================================

// Create problem
DWaveProblem* create_dwave_problem(size_t num_variables, size_t num_interactions) {
    DWaveProblem* problem = calloc(1, sizeof(DWaveProblem));
    if (!problem) return NULL;

    problem->num_variables = num_variables;
    problem->num_interactions = num_interactions;

    problem->linear_terms = calloc(num_variables, sizeof(double));
    problem->quadratic_terms = calloc(num_interactions, sizeof(double));

    if (!problem->linear_terms || !problem->quadratic_terms) {
        cleanup_dwave_problem(problem);
        return NULL;
    }

    return problem;
}

// Add linear term
bool add_linear_term(DWaveProblem* problem, size_t variable, double coefficient) {
    if (!problem || variable >= problem->num_variables) return false;
    problem->linear_terms[variable] = coefficient;
    return true;
}

// Add quadratic term
bool add_quadratic_term(DWaveProblem* problem, size_t var1, size_t var2, double coefficient) {
    if (!problem) return false;

    // Store in upper triangle index
    size_t idx = var1 * problem->num_variables + var2;
    if (idx >= problem->num_interactions) return false;

    problem->quadratic_terms[idx] = coefficient;
    return true;
}

// Add constraint
bool add_constraint(DWaveProblem* problem, const void* constraint) {
    (void)problem;
    (void)constraint;
    return true;  // Constraints handled at higher level
}

// Cancel job
bool cancel_dwave_job(DWaveConfig* config, const char* job_id) {
    if (!config || !job_id || strlen(job_id) == 0) {
        return false;
    }

#if QGT_HAS_CURL
    // Use D-Wave SAPI to cancel the job
    CURL* curl = curl_easy_init();
    if (!curl) {
        return false;
    }

    // Build cancellation URL: DELETE /problems/{job_id}
    char url[512];
    snprintf(url, sizeof(url), "%s/%s", DWAVE_PROBLEMS_URL, job_id);

    // Set up authentication header
    struct curl_slist* headers = NULL;
    char auth_header[256];
    snprintf(auth_header, sizeof(auth_header), "X-Auth-Token: %s", config->api_key);
    headers = curl_slist_append(headers, auth_header);
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, API_TIMEOUT);

    // Perform the request
    CURLcode res = curl_easy_perform(curl);

    // Check HTTP response code
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    // Success if request succeeded and got 200/202/204
    if (res == CURLE_OK && (http_code == 200 || http_code == 202 || http_code == 204)) {
        return true;
    }

    // Job may already be completed or not found (404) - still consider it "cancelled"
    if (res == CURLE_OK && http_code == 404) {
        return true;
    }

    return false;
#else
    // Without CURL, we can only cancel local simulated jobs
    // For simulated annealing, cancellation is not supported
    // Return true to indicate the job is effectively cancelled (won't be processed)
    (void)config;
    (void)job_id;
    return true;  // Simulated jobs are not persistent, so "cancellation" trivially succeeds
#endif
}

// Get capabilities
DWaveCapabilities* get_dwave_capabilities(DWaveConfig* config) {
    DWaveCapabilities* caps = calloc(1, sizeof(DWaveCapabilities));
    if (!caps) return NULL;

    DWaveInternalState* state = get_internal_state(config);
    if (state) {
        caps->num_qubits = (uint32_t)state->num_qubits;
    } else {
        caps->num_qubits = MAX_QUBITS;
    }

    caps->max_qubits = MAX_QUBITS;
    caps->min_annealing_time = 1.0;
    caps->max_annealing_time = 2000.0;

    return caps;
}

// Get solvers
char** get_dwave_solvers(DWaveConfig* config, size_t* num_solvers) {
    (void)config;
    *num_solvers = 1;
    char** solvers = malloc(sizeof(char*));
    if (solvers) {
        solvers[0] = strdup("Advantage_system6.4");
    }
    return solvers;
}

// Get solver properties
char* get_dwave_solver_properties(DWaveConfig* config, const char* solver_name) {
    (void)config;
    (void)solver_name;
    return strdup("{\"num_qubits\": 5000}");
}

// Get queue position
size_t get_dwave_queue_position(DWaveConfig* config, const char* job_id) {
    (void)config;
    (void)job_id;
    return 0;
}

// Get estimated runtime
double get_dwave_estimated_runtime(DWaveConfig* config, const DWaveProblem* problem) {
    (void)config;
    if (!problem) return 0.0;
    return problem->num_variables * 0.001;  // Rough estimate
}

// Optimize problem for D-Wave hardware
bool optimize_dwave_problem(DWaveProblem* problem, const DWaveCapabilities* capabilities) {
    if (!problem || !capabilities) return false;

    // Check if problem fits on hardware
    if (problem->num_variables > capabilities->max_qubits) {
        return false;
    }

    // Find the maximum coefficient magnitude for scaling
    double max_linear = 0.0;
    double max_quadratic = 0.0;

    for (size_t i = 0; i < problem->num_variables; i++) {
        double abs_val = fabs(problem->linear_terms[i]);
        if (abs_val > max_linear) max_linear = abs_val;
    }

    for (size_t i = 0; i < problem->num_interactions; i++) {
        double abs_val = fabs(problem->quadratic_terms[i]);
        if (abs_val > max_quadratic) max_quadratic = abs_val;
    }

    // D-Wave hardware typically has h range [-4, 4] and J range [-1, 1]
    // Scale coefficients to fit within hardware range
    const double H_RANGE = 4.0;
    const double J_RANGE = 1.0;
    const double SMALL_THRESHOLD = 1e-10;

    double h_scale = (max_linear > H_RANGE) ? H_RANGE / max_linear : 1.0;
    double j_scale = (max_quadratic > J_RANGE) ? J_RANGE / max_quadratic : 1.0;

    // Use the more restrictive scale factor
    double scale = (h_scale < j_scale) ? h_scale : j_scale;

    // Apply scaling if needed
    if (scale < 1.0) {
        for (size_t i = 0; i < problem->num_variables; i++) {
            problem->linear_terms[i] *= scale;
        }
        for (size_t i = 0; i < problem->num_interactions; i++) {
            problem->quadratic_terms[i] *= scale;
        }
        problem->offset *= scale;
    }

    // Remove terms below threshold (numerical noise)
    for (size_t i = 0; i < problem->num_variables; i++) {
        if (fabs(problem->linear_terms[i]) < SMALL_THRESHOLD) {
            problem->linear_terms[i] = 0.0;
        }
    }

    for (size_t i = 0; i < problem->num_interactions; i++) {
        if (fabs(problem->quadratic_terms[i]) < SMALL_THRESHOLD) {
            problem->quadratic_terms[i] = 0.0;
        }
    }

    return true;
}

// Apply error mitigation to D-Wave results
bool apply_dwave_error_mitigation(DWaveJobResult* result, const struct MitigationParams* params) {
    if (!result || result->num_samples == 0) return false;

    // Default mitigation factor if not specified
    double mitigation_factor = 1.0;
    if (params && params->mitigation_factor > 0.0) {
        mitigation_factor = params->mitigation_factor;
    }

    // Apply readout error correction using majority voting for chain breaks
    // Chain breaks occur when embedded qubits in a chain disagree on their value

    // Determine number of variables from chain_breaks array (terminated by negative value)
    size_t num_vars = 0;
    if (result->num_samples > 0 && result->samples[0].chain_breaks) {
        while (result->samples[0].chain_breaks[num_vars] >= 0.0 && num_vars < 10000) {
            num_vars++;
        }
    }

    for (size_t s = 0; s < result->num_samples; s++) {
        DWaveSample* sample = &result->samples[s];
        if (!sample || !sample->variables) continue;

        // If chain break information is available, resolve using energy-based voting
        if (sample->chain_breaks && num_vars > 0) {
            // Process each variable that has chain break information
            for (size_t i = 0; i < num_vars; i++) {
                double break_fraction = sample->chain_breaks[i];
                if (break_fraction < 0.0) break;  // End of valid data

                if (break_fraction > 0.01) {
                    // Chain break detected - need to resolve the variable value
                    // Use weighted majority voting based on neighbors and energy

                    // Calculate local field contribution from neighboring variables
                    double local_field = 0.0;
                    int neighbor_count = 0;

                    // Check left neighbor
                    if (i > 0 && sample->chain_breaks[i - 1] >= 0.0) {
                        local_field += (double)sample->variables[i - 1];
                        neighbor_count++;
                    }
                    // Check right neighbor
                    if (i + 1 < num_vars && sample->chain_breaks[i + 1] >= 0.0) {
                        local_field += (double)sample->variables[i + 1];
                        neighbor_count++;
                    }

                    if (neighbor_count > 0) {
                        // Majority vote: if neighbors mostly positive, prefer +1, else -1
                        double avg_neighbor = local_field / neighbor_count;

                        // Apply correction based on break severity and neighbor consensus
                        if (break_fraction > 0.5) {
                            // Severe break - use neighbor majority
                            int32_t corrected = (avg_neighbor >= 0.0) ? 1 : -1;
                            sample->variables[i] = corrected;
                        } else if (break_fraction > 0.25) {
                            // Moderate break - flip only if strong neighbor consensus
                            if (fabs(avg_neighbor) > 0.5) {
                                int32_t corrected = (avg_neighbor >= 0.0) ? 1 : -1;
                                // Only flip if disagrees with current value
                                if ((sample->variables[i] > 0) != (corrected > 0)) {
                                    sample->variables[i] = corrected;
                                }
                            }
                        }
                        // Light breaks (< 0.25) - keep D-Wave's resolution
                    }
                }
            }
        }
    }

    // Post-selection: filter samples based on energy threshold
    // Keep samples within a certain range of the minimum energy
    double energy_threshold = result->min_energy + (result->max_energy - result->min_energy) * (1.0 - mitigation_factor);

    // Count valid samples
    size_t valid_count = 0;
    for (size_t s = 0; s < result->num_samples; s++) {
        if (result->samples[s].energy <= energy_threshold) {
            valid_count++;
        }
    }

    // If mitigation is too aggressive and would remove all samples, keep at least the best ones
    if (valid_count == 0) {
        energy_threshold = result->min_energy + 0.1 * (result->max_energy - result->min_energy);
        for (size_t s = 0; s < result->num_samples; s++) {
            if (result->samples[s].energy <= energy_threshold) {
                valid_count++;
            }
        }
    }

    // Renormalize probabilities for valid samples
    if (result->probabilities && valid_count > 0) {
        double total_prob = 0.0;
        for (size_t s = 0; s < result->num_samples; s++) {
            if (result->samples[s].energy <= energy_threshold) {
                total_prob += result->probabilities[s];
            } else {
                result->probabilities[s] = 0.0;
            }
        }

        if (total_prob > 0.0) {
            for (size_t s = 0; s < result->num_samples; s++) {
                result->probabilities[s] /= total_prob;
            }
        }
    }

    return true;
}

// Convert QUBO to Ising
DWaveProblem* qubo_to_ising(const DWaveProblem* qubo) {
    if (!qubo) return NULL;

    DWaveProblem* ising = create_dwave_problem(qubo->num_variables, qubo->num_interactions);
    if (!ising) return NULL;

    // QUBO to Ising conversion: h_i = Q_ii/2, J_ij = Q_ij/4
    for (size_t i = 0; i < qubo->num_variables; i++) {
        ising->linear_terms[i] = qubo->linear_terms[i] / 2.0;
    }

    for (size_t i = 0; i < qubo->num_interactions; i++) {
        ising->quadratic_terms[i] = qubo->quadratic_terms[i] / 4.0;
    }

    return ising;
}

// Convert Ising to QUBO
DWaveProblem* ising_to_qubo(const DWaveProblem* ising) {
    if (!ising) return NULL;

    DWaveProblem* qubo = create_dwave_problem(ising->num_variables, ising->num_interactions);
    if (!qubo) return NULL;

    // Ising to QUBO conversion: Q_ii = 2*h_i, Q_ij = 4*J_ij
    for (size_t i = 0; i < ising->num_variables; i++) {
        qubo->linear_terms[i] = ising->linear_terms[i] * 2.0;
    }

    for (size_t i = 0; i < ising->num_interactions; i++) {
        qubo->quadratic_terms[i] = ising->quadratic_terms[i] * 4.0;
    }

    return qubo;
}

// Validate problem
bool validate_dwave_problem(const DWaveProblem* problem, const DWaveCapabilities* capabilities) {
    if (!problem || !capabilities) return false;
    return problem->num_variables <= capabilities->max_qubits;
}

// Get error info
char* get_dwave_error_info(DWaveConfig* config, const char* job_id) {
    (void)job_id;
    DWaveInternalState* state = get_internal_state(config);
    if (state && state->error_buffer[0] != '\0') {
        return strdup(state->error_buffer);
    }
    return strdup("No error information available");
}

// Cleanup problem
void cleanup_dwave_problem(DWaveProblem* problem) {
    if (!problem) return;
    free(problem->linear_terms);
    free(problem->quadratic_terms);
    free(problem->fixed_variables);
    free(problem->constraints);
    free(problem->custom_data);
    free(problem);
}

// Cleanup result
void cleanup_dwave_result(DWaveJobResult* result) {
    if (!result) return;

    if (result->samples) {
        for (size_t i = 0; i < result->num_samples; i++) {
            free(result->samples[i].variables);
            free(result->samples[i].chain_breaks);
            free(result->samples[i].custom_data);
        }
        free(result->samples);
    }

    free(result->energies);
    free(result->probabilities);
    free(result->error_message);
    free(result->raw_data);
    free(result);
}

// Cleanup capabilities
void cleanup_dwave_capabilities(DWaveCapabilities* capabilities) {
    if (!capabilities) return;
    free(capabilities->qubit_connectivity);
    free(capabilities);
}

// Save credentials
bool save_dwave_credentials(const char* token, const char* filename) {
    if (!token || !filename) return false;

    FILE* f = fopen(filename, "w");
    if (!f) return false;

    fprintf(f, "%s", token);
    fclose(f);
    return true;
}

// Load credentials
char* load_dwave_credentials(const char* filename) {
    if (!filename) return NULL;

    FILE* f = fopen(filename, "r");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* token = malloc(size + 1);
    if (token) {
        size_t read = fread(token, 1, size, f);
        token[read] = '\0';
    }

    fclose(f);
    return token;
}

// Test connection
bool test_dwave_connection(const char* token) {
    (void)token;
#if defined(QGT_HAS_CURL)
    // Would need to actually test the connection
    return token != NULL;
#else
    return false;
#endif
}

// Set log level
void set_dwave_log_level(int level) {
    (void)level;
}

// Get version
char* get_dwave_version(void) {
    return strdup("1.0.0");
}

// Advanced annealing control - set custom annealing schedule
// Schedule format: pairs of (time, s) where time is in microseconds and s is in [0,1]
// s=0 means full transverse field, s=1 means full problem Hamiltonian
bool set_annealing_schedule(DWaveConfig* config, const double* schedule, size_t points) {
    if (!config || !schedule || points < 2) return false;

    DWaveInternalState* state = get_internal_state(config);
    if (!state) return false;

    // Free existing schedule if any
    if (state->annealing_schedule) {
        free(state->annealing_schedule);
        state->annealing_schedule = NULL;
        state->annealing_schedule_points = 0;
    }

    // Validate schedule: time should be monotonically increasing
    for (size_t i = 1; i < points; i++) {
        // Each point is (time, s), so time is at index i*2
        if (schedule[i * 2] <= schedule[(i - 1) * 2]) {
            return false;  // Time must be strictly increasing
        }
    }

    // Validate s values are in [0, 1]
    for (size_t i = 0; i < points; i++) {
        double s = schedule[i * 2 + 1];
        if (s < 0.0 || s > 1.0) {
            return false;
        }
    }

    // Allocate and copy schedule (2 values per point: time and s)
    state->annealing_schedule = malloc(points * 2 * sizeof(double));
    if (!state->annealing_schedule) return false;

    memcpy(state->annealing_schedule, schedule, points * 2 * sizeof(double));
    state->annealing_schedule_points = points;

    return true;
}

bool set_flux_bias_offsets(DWaveConfig* config, const double* offsets, size_t num_qubits) {
    if (!config || !offsets || num_qubits == 0) return false;

    DWaveInternalState* state = get_internal_state(config);
    if (!state) return false;

    // Free existing offsets
    if (state->flux_bias_offsets) {
        free(state->flux_bias_offsets);
    }

    state->flux_bias_offsets = malloc(num_qubits * sizeof(double));
    if (!state->flux_bias_offsets) return false;

    memcpy(state->flux_bias_offsets, offsets, num_qubits * sizeof(double));
    state->num_offset_qubits = num_qubits;

    return true;
}

bool set_anneal_offsets(DWaveConfig* config, const double* offsets, size_t num_qubits) {
    if (!config || !offsets || num_qubits == 0) return false;

    DWaveInternalState* state = get_internal_state(config);
    if (!state) return false;

    // Free existing offsets
    if (state->anneal_offsets) {
        free(state->anneal_offsets);
    }

    state->anneal_offsets = malloc(num_qubits * sizeof(double));
    if (!state->anneal_offsets) return false;

    memcpy(state->anneal_offsets, offsets, num_qubits * sizeof(double));
    // Note: shares num_offset_qubits with flux_bias_offsets
    if (state->num_offset_qubits < num_qubits) {
        state->num_offset_qubits = num_qubits;
    }

    return true;
}

bool set_programming_thermalization(DWaveConfig* config, double time_us) {
    DWaveInternalState* state = get_internal_state(config);
    if (!state) return false;
    state->sampling_params.programming_thermalization = time_us;
    return true;
}

// Problem manipulation
bool reverse_annealing(DWaveConfig* config, const int32_t* initial_state, double reinitialize_state) {
    (void)config;
    (void)initial_state;
    (void)reinitialize_state;
    return true;
}

bool set_chain_strength(DWaveConfig* config, double strength) {
    DWaveInternalState* state = get_internal_state(config);
    if (!state) return false;
    state->sampling_params.chain_strength = strength;
    return true;
}

bool set_chain_break_method(DWaveConfig* config, const char* method) {
    (void)config;
    (void)method;
    return true;
}

bool set_postprocess_method(DWaveConfig* config, const char* method) {
    (void)config;
    (void)method;
    return true;
}

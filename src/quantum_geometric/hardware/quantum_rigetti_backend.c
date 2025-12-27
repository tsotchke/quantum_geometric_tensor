#include "quantum_geometric/hardware/quantum_rigetti_backend.h"
#include "quantum_geometric/hardware/quantum_rigetti_api.h"
#include "quantum_geometric/hardware/quantum_hardware_abstraction.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include "quantum_geometric/core/quantum_circuit_types.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include <curl/curl.h>
#include <json-c/json.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

// Logging macros
#define log_info(...)  geometric_log_info(__VA_ARGS__)
#define log_warn(...)  geometric_log_warning(__VA_ARGS__)
#define log_error(...) geometric_log_error(__VA_ARGS__)

// Gate field access macros for compatibility
// quantum_gate_t uses arrays: control_qubits[], target_qubits[]
// This code expects simple: control, target fields
// We use macros to bridge the difference
#define GATE_GET_TARGET(gate) ((gate)->target_qubits ? (gate)->target_qubits[0] : 0)
#define GATE_GET_CONTROL(gate) ((gate)->control_qubits && (gate)->num_controls > 0 ? (gate)->control_qubits[0] : SIZE_MAX)
#define GATE_IS_SINGLE_QUBIT(gate) ((gate)->num_controls == 0 || !(gate)->control_qubits)

// Rigetti parameters - undefine to avoid conflicts
#undef MAX_QUBITS
#define RIGETTI_MAX_QUBITS 80
#define MAX_CIRCUITS 500
#define API_TIMEOUT 30
#define MAX_RETRIES 3
#define MIN_SHOTS 1000
#define LOCAL_MAX_SHOTS 100000
#define DEFAULT_SHOTS 10000

// Error mitigation parameters
#define ZNE_SCALE_FACTORS {1.0, 1.5, 2.0, 3.0}
#define NUM_ZNE_SCALES 4
#define READOUT_CAL_SHOTS 5000
#define SYMMETRY_SHOTS 2000
#define ERROR_BOUND_THRESHOLD 0.05

// Rigetti API endpoints
#define RIGETTI_API_URL "https://api.qcs.rigetti.com/v1"
#define RIGETTI_AUTH_URL RIGETTI_API_URL "/auth"
#define RIGETTI_JOBS_URL RIGETTI_API_URL "/jobs"
#define RIGETTI_DEVICES_URL RIGETTI_API_URL "/devices"

// Error mitigation configuration (internal)
typedef struct {
    double zne_scales[NUM_ZNE_SCALES];
    double readout_threshold;
    double symmetry_threshold;
    double error_bound;
    // Additional fields for error estimation
    size_t num_qubits;
    size_t num_shots;
    double readout_error_rate;
    double gate_error_rate;
    size_t circuit_depth;
    double t1_time;
    double t2_time;
} RigettiErrorMitigationConfig;

// Alias for backward compatibility
typedef RigettiErrorMitigationConfig ErrorMitigationConfig;

// Internal Rigetti backend state
typedef struct RigettiInternalBackend {
    char* api_key;
    char* device_name;
    size_t num_qubits;
    bool is_simulator;
    CURL* curl;
    struct json_object* json_config;
    char error_buffer[CURL_ERROR_SIZE];
    RigettiErrorMitigationConfig error_config;
    // QCS API handle for proper job management
    rigetti_qcs_handle_t* qcs_handle;
    // Job tracking
    char** active_job_ids;
    size_t num_active_jobs;
    size_t max_active_jobs;
    // Cached job results
    struct {
        char job_id[64];
        qcs_job_result_t result;
        bool valid;
    } cached_results[16];
    size_t num_cached_results;
    // Last error info
    char last_error[512];
    int last_error_code;
} RigettiInternalBackend;

// Static internal backend for current session
static RigettiInternalBackend* g_rigetti_backend = NULL;

// Forward declarations
static void cleanup_internal_backend(RigettiInternalBackend* backend);
static int gate_to_native_quil(const quantum_gate_t* gate, const size_t* qubit_mapping,
                               char* quil, size_t size);

// CURL callback for writing response
static size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    struct json_object** response = (struct json_object**)userdata;
    size_t total_size = size * nmemb;

    // Parse JSON response
    struct json_object* obj = json_tokener_parse_ex(
        json_tokener_new(), ptr, (int)total_size);

    if (obj) {
        *response = obj;
        return total_size;
    }

    return 0;
}

// Legacy Rigetti backend initialization (canonical init_rigetti_backend is in quantum_rigetti_backend_optimized.c)
struct RigettiConfig* init_rigetti_backend_legacy(const struct RigettiBackendConfig* backend_config) {
    if (!backend_config) return NULL;

    // Allocate RigettiConfig
    struct RigettiConfig* config = calloc(1, sizeof(struct RigettiConfig));
    if (!config) return NULL;

    // Copy configuration
    if (backend_config->api_key) {
        config->api_key = strdup(backend_config->api_key);
    }
    if (backend_config->backend_name) {
        config->backend_name = strdup(backend_config->backend_name);
        config->url = strdup("https://api.qcs.rigetti.com/v1");
    }
    config->max_shots = backend_config->max_shots > 0 ? backend_config->max_shots : DEFAULT_SHOTS;
    config->max_qubits = RIGETTI_MAX_QUBITS;
    config->optimize_mapping = backend_config->optimize_mapping;

    // Create internal backend
    RigettiInternalBackend* internal = calloc(1, sizeof(RigettiInternalBackend));
    if (!internal) {
        cleanup_rigetti_config(config);
        return NULL;
    }

    // Initialize CURL
    curl_global_init(CURL_GLOBAL_DEFAULT);
    internal->curl = curl_easy_init();
    if (!internal->curl) {
        free(internal);
        cleanup_rigetti_config(config);
        return NULL;
    }

    // Set API key and device name
    if (backend_config->api_key) {
        internal->api_key = strdup(backend_config->api_key);
    }
    if (backend_config->backend_name) {
        internal->device_name = strdup(backend_config->backend_name);
        internal->is_simulator = (strstr(backend_config->backend_name, "qvm") != NULL);
    }

    // Configure error mitigation
    double zne_scales[] = ZNE_SCALE_FACTORS;
    memcpy(internal->error_config.zne_scales, zne_scales, sizeof(zne_scales));
    internal->error_config.readout_threshold = 0.98;
    internal->error_config.symmetry_threshold = 0.95;
    internal->error_config.error_bound = ERROR_BOUND_THRESHOLD;
    // Default values for error estimation
    internal->error_config.num_qubits = RIGETTI_MAX_QUBITS;
    internal->error_config.num_shots = DEFAULT_SHOTS;
    internal->error_config.readout_error_rate = 0.02;  // 2% default
    internal->error_config.gate_error_rate = 0.001;    // 0.1% default
    internal->error_config.circuit_depth = 10;         // Default circuit depth
    internal->error_config.t1_time = 50e-6;            // 50µs default
    internal->error_config.t2_time = 30e-6;            // 30µs default

    // Configure CURL
    curl_easy_setopt(internal->curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(internal->curl, CURLOPT_ERRORBUFFER, internal->error_buffer);
    curl_easy_setopt(internal->curl, CURLOPT_TIMEOUT, API_TIMEOUT);

    // Set authentication headers
    if (backend_config->api_key) {
        struct curl_slist* headers = NULL;
        char auth_header[256];
        snprintf(auth_header, sizeof(auth_header),
                 "Authorization: Bearer %s", backend_config->api_key);
        headers = curl_slist_append(headers, auth_header);
        headers = curl_slist_append(headers, "Content-Type: application/json");
        curl_easy_setopt(internal->curl, CURLOPT_HTTPHEADER, headers);
    }

    // Store internal backend pointer
    config->backend_specific_config = internal;
    g_rigetti_backend = internal;

    // Default qubit count for simulation
    internal->num_qubits = RIGETTI_MAX_QUBITS;

    return config;
}

// ============================================================================
// Qubit Mapping Optimization with Hardware Awareness
// ============================================================================

// Edge in the hardware connectivity graph
typedef struct {
    size_t qubit1;
    size_t qubit2;
    double error_rate;    // Two-qubit gate error rate
    double crosstalk;     // Crosstalk coefficient
} HardwareEdge;

// Hardware topology representation
typedef struct {
    size_t num_qubits;
    HardwareEdge* edges;
    size_t num_edges;
    double* t1_times;       // T1 coherence times per qubit
    double* t2_times;       // T2 coherence times per qubit
    double* readout_errors; // Readout error rates per qubit
    double* gate_errors;    // Single-qubit gate error rates
} HardwareTopology;

// Parse hardware topology from JSON
static HardwareTopology* parse_topology(const struct json_object* topology_json,
                                         size_t num_qubits) {
    HardwareTopology* topo = calloc(1, sizeof(HardwareTopology));
    if (!topo) return NULL;

    topo->num_qubits = num_qubits;

    // Allocate arrays
    topo->t1_times = calloc(num_qubits, sizeof(double));
    topo->t2_times = calloc(num_qubits, sizeof(double));
    topo->readout_errors = calloc(num_qubits, sizeof(double));
    topo->gate_errors = calloc(num_qubits, sizeof(double));

    if (!topo->t1_times || !topo->t2_times ||
        !topo->readout_errors || !topo->gate_errors) {
        free(topo->t1_times);
        free(topo->t2_times);
        free(topo->readout_errors);
        free(topo->gate_errors);
        free(topo);
        return NULL;
    }

    // Parse qubit properties
    struct json_object* qubits_obj;
    if (json_object_object_get_ex(topology_json, "qubits", &qubits_obj)) {
        size_t n = json_object_array_length(qubits_obj);
        for (size_t i = 0; i < n && i < num_qubits; i++) {
            struct json_object* q = json_object_array_get_idx(qubits_obj, i);
            struct json_object* prop;

            if (json_object_object_get_ex(q, "t1", &prop)) {
                topo->t1_times[i] = json_object_get_double(prop);
            } else {
                topo->t1_times[i] = 50e-6;  // Default 50µs
            }

            if (json_object_object_get_ex(q, "t2", &prop)) {
                topo->t2_times[i] = json_object_get_double(prop);
            } else {
                topo->t2_times[i] = 30e-6;  // Default 30µs
            }

            if (json_object_object_get_ex(q, "readout_error", &prop)) {
                topo->readout_errors[i] = json_object_get_double(prop);
            } else {
                topo->readout_errors[i] = 0.02;  // Default 2%
            }

            if (json_object_object_get_ex(q, "gate_error", &prop)) {
                topo->gate_errors[i] = json_object_get_double(prop);
            } else {
                topo->gate_errors[i] = 0.001;  // Default 0.1%
            }
        }
    }

    // Parse edges (connectivity)
    struct json_object* edges_obj;
    if (json_object_object_get_ex(topology_json, "edges", &edges_obj)) {
        size_t n = json_object_array_length(edges_obj);
        topo->edges = calloc(n, sizeof(HardwareEdge));
        if (!topo->edges) {
            free(topo->t1_times);
            free(topo->t2_times);
            free(topo->readout_errors);
            free(topo->gate_errors);
            free(topo);
            return NULL;
        }

        topo->num_edges = n;
        for (size_t i = 0; i < n; i++) {
            struct json_object* e = json_object_array_get_idx(edges_obj, i);
            struct json_object* prop;

            if (json_object_object_get_ex(e, "qubit1", &prop)) {
                topo->edges[i].qubit1 = json_object_get_int(prop);
            }
            if (json_object_object_get_ex(e, "qubit2", &prop)) {
                topo->edges[i].qubit2 = json_object_get_int(prop);
            }
            if (json_object_object_get_ex(e, "error_rate", &prop)) {
                topo->edges[i].error_rate = json_object_get_double(prop);
            } else {
                topo->edges[i].error_rate = 0.01;  // Default 1%
            }
            if (json_object_object_get_ex(e, "crosstalk", &prop)) {
                topo->edges[i].crosstalk = json_object_get_double(prop);
            } else {
                topo->edges[i].crosstalk = 0.001;  // Default 0.1%
            }
        }
    }

    return topo;
}

// Free hardware topology
static void free_topology(HardwareTopology* topo) {
    if (topo) {
        free(topo->t1_times);
        free(topo->t2_times);
        free(topo->readout_errors);
        free(topo->gate_errors);
        free(topo->edges);
        free(topo);
    }
}

// Check if two qubits are connected in hardware
static bool are_connected(const HardwareTopology* topo, size_t q1, size_t q2) {
    for (size_t i = 0; i < topo->num_edges; i++) {
        if ((topo->edges[i].qubit1 == q1 && topo->edges[i].qubit2 == q2) ||
            (topo->edges[i].qubit1 == q2 && topo->edges[i].qubit2 == q1)) {
            return true;
        }
    }
    return false;
}

// Get edge error rate
static double get_edge_error(const HardwareTopology* topo, size_t q1, size_t q2) {
    for (size_t i = 0; i < topo->num_edges; i++) {
        if ((topo->edges[i].qubit1 == q1 && topo->edges[i].qubit2 == q2) ||
            (topo->edges[i].qubit1 == q2 && topo->edges[i].qubit2 == q1)) {
            return topo->edges[i].error_rate;
        }
    }
    return 1.0;  // Not connected - maximum error
}

// Compute mapping cost based on circuit and hardware
static double compute_mapping_cost(const struct quantum_circuit* circuit,
                                   const HardwareTopology* topo,
                                   const size_t* mapping,
                                   size_t num_qubits) {
    double cost = 0.0;

    // Cost from two-qubit gates requiring SWAPs
    for (size_t i = 0; i < circuit->num_gates; i++) {
        const quantum_gate_t* gate = circuit->gates[i];
        if (!gate) continue;

        // Single-qubit gate cost based on error rate
        if (GATE_IS_SINGLE_QUBIT(gate)) {
            size_t phys = mapping[GATE_GET_TARGET(gate)];
            cost += topo->gate_errors[phys];
        } else {
            // Two-qubit gate
            size_t phys_ctrl = mapping[GATE_GET_CONTROL(gate)];
            size_t phys_tgt = mapping[GATE_GET_TARGET(gate)];

            if (are_connected(topo, phys_ctrl, phys_tgt)) {
                // Connected - use edge error rate
                cost += get_edge_error(topo, phys_ctrl, phys_tgt);
            } else {
                // Not connected - need SWAP routing (high cost)
                cost += 3.0 * 0.01;  // 3 CZ gates worth of error
            }
        }
    }

    // Cost from readout errors
    for (size_t i = 0; i < num_qubits; i++) {
        size_t phys = mapping[i];
        cost += topo->readout_errors[phys];
    }

    // Bonus for using qubits with longer coherence times
    double avg_depth = (double)circuit->num_gates / num_qubits;
    double gate_time = 50e-9;  // 50ns per gate typical

    for (size_t i = 0; i < num_qubits; i++) {
        size_t phys = mapping[i];
        double circuit_time = avg_depth * gate_time;

        // T1 decay probability
        double t1_decay = 1.0 - exp(-circuit_time / topo->t1_times[phys]);
        // T2 dephasing probability
        double t2_dephase = 1.0 - exp(-circuit_time / topo->t2_times[phys]);

        cost += 0.5 * (t1_decay + t2_dephase);
    }

    return cost;
}

// Greedy initial mapping based on qubit quality
static void greedy_initial_mapping(const struct quantum_circuit* circuit,
                                   const HardwareTopology* topo,
                                   size_t* mapping,
                                   size_t num_qubits) {
    // Score each physical qubit by quality
    double* qubit_scores = malloc(topo->num_qubits * sizeof(double));
    bool* used = calloc(topo->num_qubits, sizeof(bool));

    if (!qubit_scores || !used) {
        free(qubit_scores);
        free(used);
        // Fall back to identity mapping
        for (size_t i = 0; i < num_qubits; i++) {
            mapping[i] = i;
        }
        return;
    }

    // Compute quality score for each physical qubit
    // Higher score = better quality
    for (size_t i = 0; i < topo->num_qubits; i++) {
        qubit_scores[i] = 0.0;

        // Prefer low gate error
        qubit_scores[i] += 1.0 / (topo->gate_errors[i] + 1e-10);

        // Prefer low readout error
        qubit_scores[i] += 0.5 / (topo->readout_errors[i] + 1e-10);

        // Prefer high coherence times
        qubit_scores[i] += topo->t1_times[i] * 1e4;  // Normalize ~µs to ~1
        qubit_scores[i] += topo->t2_times[i] * 1e4;

        // Prefer well-connected qubits
        size_t connectivity = 0;
        double edge_quality = 0.0;
        for (size_t j = 0; j < topo->num_edges; j++) {
            if (topo->edges[j].qubit1 == i || topo->edges[j].qubit2 == i) {
                connectivity++;
                edge_quality += 1.0 / (topo->edges[j].error_rate + 1e-10);
            }
        }
        qubit_scores[i] += connectivity * 0.5 + edge_quality * 0.1;
    }

    // Assign logical qubits to physical qubits greedily
    for (size_t logical = 0; logical < num_qubits; logical++) {
        // Find best available physical qubit
        size_t best_phys = 0;
        double best_score = -1e9;

        for (size_t phys = 0; phys < topo->num_qubits; phys++) {
            if (!used[phys] && qubit_scores[phys] > best_score) {
                best_score = qubit_scores[phys];
                best_phys = phys;
            }
        }

        mapping[logical] = best_phys;
        used[best_phys] = true;
    }

    free(qubit_scores);
    free(used);
}

// Simulated annealing to optimize mapping
static void optimize_mapping_sa(const struct quantum_circuit* circuit,
                                const HardwareTopology* topo,
                                size_t* mapping,
                                size_t num_qubits) {
    double temperature = 1.0;
    double cooling_rate = 0.995;
    double min_temp = 0.001;
    size_t max_iterations = 1000;

    double current_cost = compute_mapping_cost(circuit, topo, mapping, num_qubits);

    size_t* best_mapping = malloc(num_qubits * sizeof(size_t));
    if (!best_mapping) return;
    memcpy(best_mapping, mapping, num_qubits * sizeof(size_t));
    double best_cost = current_cost;

    srand((unsigned int)time(NULL));

    for (size_t iter = 0; iter < max_iterations && temperature > min_temp; iter++) {
        // Generate neighbor by swapping two random qubits
        size_t i = rand() % num_qubits;
        size_t j = rand() % num_qubits;
        if (i == j) continue;

        // Swap
        size_t temp = mapping[i];
        mapping[i] = mapping[j];
        mapping[j] = temp;

        double new_cost = compute_mapping_cost(circuit, topo, mapping, num_qubits);
        double delta = new_cost - current_cost;

        // Accept or reject
        if (delta < 0 || (double)rand() / RAND_MAX < exp(-delta / temperature)) {
            current_cost = new_cost;
            if (current_cost < best_cost) {
                best_cost = current_cost;
                memcpy(best_mapping, mapping, num_qubits * sizeof(size_t));
            }
        } else {
            // Reject - swap back
            mapping[j] = mapping[i];
            mapping[i] = temp;
        }

        temperature *= cooling_rate;
    }

    // Use best mapping found
    memcpy(mapping, best_mapping, num_qubits * sizeof(size_t));
    free(best_mapping);
}

// Optimize qubit mapping for hardware connectivity
static size_t* optimize_qubit_mapping(const struct quantum_circuit* circuit,
                                    const struct json_object* topology,
                                    size_t num_qubits) {
    size_t* mapping = malloc(num_qubits * sizeof(size_t));
    if (!mapping) return NULL;

    // Parse hardware topology
    HardwareTopology* topo = parse_topology(topology, num_qubits);
    if (!topo) {
        // Fall back to identity mapping
        for (size_t i = 0; i < num_qubits; i++) {
            mapping[i] = i;
        }
        return mapping;
    }

    // Start with greedy initial mapping based on qubit quality
    greedy_initial_mapping(circuit, topo, mapping, num_qubits);

    // Optimize with simulated annealing
    optimize_mapping_sa(circuit, topo, mapping, num_qubits);

    free_topology(topo);
    return mapping;
}

// Add readout error calibration circuits
static int add_calibration_circuits(char* quil,
                                  size_t size,
                                  size_t num_qubits) {
    int offset = 0;
    
    // Add calibration sequences for |0⟩ state
    for (size_t q = 0; q < num_qubits; q++) {
        offset += snprintf(quil + offset, size - offset,
                         "RESET\n"
                         "MEASURE %zu calibration[%zu]\n",
                         q, q);
    }
    
    // Add calibration sequences for |1⟩ state
    for (size_t q = 0; q < num_qubits; q++) {
        offset += snprintf(quil + offset, size - offset,
                         "RESET\n"
                         "X %zu\n"
                         "MEASURE %zu calibration[%zu]\n",
                         q, q, q + num_qubits);
    }
    
    return offset;
}

// Add symmetry verification circuits
static int add_symmetry_circuits(char* quil,
                               size_t size,
                               size_t num_qubits) {
    int offset = 0;
    
    // Add symmetry verification sequences
    for (size_t q = 0; q < num_qubits; q++) {
        offset += snprintf(quil + offset, size - offset,
                         "RESET\n"
                         "H %zu\n"
                         "MEASURE %zu symmetry[%zu]\n",
                         q, q, q);
    }
    
    return offset;
}

// Internal: Convert quantum_circuit to hardware-efficient Quil
static char* circuit_to_quil_internal(const struct quantum_circuit* circuit) {
    if (!circuit) return NULL;

    // Get internal backend state
    RigettiInternalBackend* rb = g_rigetti_backend;
    if (!rb) return NULL;

    // Get device topology
    struct json_object* topology_obj = NULL;
    if (rb->json_config) {
        json_object_object_get_ex(rb->json_config, "topology", &topology_obj);
    }

    // Create qubit mapping to respect connectivity
    size_t* qubit_mapping = optimize_qubit_mapping(
        circuit, topology_obj, rb->num_qubits);
    if (!qubit_mapping) return NULL;

    // Allocate Quil string
    size_t max_size = 1024 * 1024;  // 1MB should be enough
    char* quil = malloc(max_size);
    if (!quil) {
        free(qubit_mapping);
        return NULL;
    }

    // Add header with error mitigation declarations
    int offset = snprintf(quil, max_size,
                         "DECLARE ro BIT[%zu]\n"
                         "DECLARE calibration BIT[%zu]\n"
                         "DECLARE symmetry BIT[%zu]\n\n",
                         circuit->num_qubits,
                         circuit->num_qubits * 2,  // For both |0⟩ and |1⟩
                         circuit->num_qubits);

    // Add readout error calibration circuits
    offset += add_calibration_circuits(quil + offset,
                                     max_size - offset,
                                     circuit->num_qubits);

    // Add symmetry verification circuits
    offset += add_symmetry_circuits(quil + offset,
                                  max_size - offset,
                                  circuit->num_qubits);

    // Add main circuit with hardware-efficient gates
    for (size_t i = 0; i < circuit->num_gates; i++) {
        const quantum_gate_t* gate = circuit->gates[i];
        if (!gate) continue;
        offset += gate_to_native_quil(gate,
                                    qubit_mapping,
                                    quil + offset,
                                    max_size - offset);
    }

    // Add measurements with error mitigation
    for (size_t i = 0; i < circuit->num_qubits; i++) {
        size_t physical_qubit = qubit_mapping[i];
        offset += snprintf(quil + offset, max_size - offset,
                         "MEASURE %zu ro[%zu]\n",
                         physical_qubit, i);
    }

    free(qubit_mapping);
    return quil;
}

// Helper: Convert HardwareGate to Quil instruction
static int hardware_gate_to_quil(const HardwareGate* gate, char* quil, size_t size) {
    if (!gate || !quil) return 0;

    switch (gate->type) {
        case GATE_H:
            return snprintf(quil, size,
                          "RZ(pi/2) %u\nRX(pi/2) %u\nRZ(pi/2) %u\n",
                          gate->target, gate->target, gate->target);
        case GATE_X:
            return snprintf(quil, size, "RX(pi) %u\n", gate->target);
        case GATE_Y:
            return snprintf(quil, size, "RY(pi) %u\n", gate->target);
        case GATE_Z:
            return snprintf(quil, size, "RZ(pi) %u\n", gate->target);
        case GATE_CNOT:
            return snprintf(quil, size,
                          "RX(pi/2) %u\nCZ %u %u\nRX(-pi/2) %u\n",
                          gate->target, gate->control, gate->target, gate->target);
        case GATE_CZ:
            return snprintf(quil, size, "CZ %u %u\n", gate->control, gate->target);
        case GATE_RX:
            return snprintf(quil, size, "RX(%g) %u\n", gate->parameter, gate->target);
        case GATE_RY:
            return snprintf(quil, size, "RY(%g) %u\n", gate->parameter, gate->target);
        case GATE_RZ:
            return snprintf(quil, size, "RZ(%g) %u\n", gate->parameter, gate->target);
        default:
            return 0;
    }
}

// Public API: Convert QuantumCircuit (HAL type) to Quil
// Matches header: char* circuit_to_quil(const struct QuantumCircuit* circuit);
char* circuit_to_quil(const struct QuantumCircuit* circuit) {
    if (!circuit) return NULL;

    // Allocate Quil string
    size_t max_size = 1024 * 1024;  // 1MB
    char* quil = malloc(max_size);
    if (!quil) return NULL;

    // Add header with declarations
    int offset = snprintf(quil, max_size,
                         "DECLARE ro BIT[%zu]\n\n",
                         circuit->num_qubits);

    // Add calibration circuits if backend available
    RigettiInternalBackend* rb = g_rigetti_backend;
    if (rb) {
        offset += add_calibration_circuits(quil + offset,
                                         max_size - offset,
                                         circuit->num_qubits);
        offset += add_symmetry_circuits(quil + offset,
                                      max_size - offset,
                                      circuit->num_qubits);
    }

    // Convert each HardwareGate to Quil
    for (size_t i = 0; i < circuit->num_gates; i++) {
        offset += hardware_gate_to_quil(&circuit->gates[i],
                                       quil + offset,
                                       max_size - offset);
    }

    // Add measurements
    for (size_t i = 0; i < circuit->num_qubits; i++) {
        offset += snprintf(quil + offset, max_size - offset,
                         "MEASURE %zu ro[%zu]\n", i, i);
    }

    return quil;
}

// Convert gate to native Rigetti gates
static int gate_to_native_quil(const quantum_gate_t* gate,
                             const size_t* qubit_mapping,
                             char* quil,
                             size_t size) {
    size_t physical_target = qubit_mapping[GATE_GET_TARGET(gate)];
    size_t physical_control = !GATE_IS_SINGLE_QUBIT(gate) ?
                             qubit_mapping[GATE_GET_CONTROL(gate)] : SIZE_MAX;
    
    switch (gate->type) {
        case GATE_H:
            // Decompose H into native RX and RZ gates
            return snprintf(quil, size,
                          "RZ(pi/2) %zu\n"
                          "RX(pi/2) %zu\n"
                          "RZ(pi/2) %zu\n",
                          physical_target,
                          physical_target,
                          physical_target);
        case GATE_X:
            return snprintf(quil, size,
                          "RX(pi) %zu\n",
                          physical_target);
        case GATE_Y:
            return snprintf(quil, size,
                          "RY(pi) %zu\n",
                          physical_target);
        case GATE_Z:
            return snprintf(quil, size,
                          "RZ(pi) %zu\n",
                          physical_target);
        case GATE_CNOT:
            // Decompose CNOT into native CZ and RX gates
            return snprintf(quil, size,
                          "RX(pi/2) %zu\n"
                          "CZ %zu %zu\n"
                          "RX(-pi/2) %zu\n",
                          physical_target,
                          physical_control,
                          physical_target,
                          physical_target);
        case GATE_CZ:
            return snprintf(quil, size,
                          "CZ %zu %zu\n",
                          physical_control,
                          physical_target);
        default:
            return 0;
    }
}

// ============================================================================
// Error Mitigation Implementation
// ============================================================================

// Build readout error matrix from calibration data
// The matrix M is 2^n x 2^n where M[i][j] = P(measure i | prepared j)
static double* build_readout_matrix(const struct json_object* calibration,
                                     size_t num_qubits) {
    size_t dim = 1UL << num_qubits;  // 2^n
    double* matrix = calloc(dim * dim, sizeof(double));
    if (!matrix) return NULL;

    // Initialize to identity (perfect readout)
    for (size_t i = 0; i < dim; i++) {
        matrix[i * dim + i] = 1.0;
    }

    // Parse calibration data to build the confusion matrix
    // calibration contains measurement results for each qubit prepared in |0⟩ and |1⟩
    struct json_object* cal_data;
    if (!json_object_object_get_ex(calibration, "data", &cal_data)) {
        // No calibration data - use identity
        return matrix;
    }

    // For each qubit, get P(0|0) and P(1|1) from calibration
    double* p00 = malloc(num_qubits * sizeof(double));  // P(measure 0 | prep 0)
    double* p11 = malloc(num_qubits * sizeof(double));  // P(measure 1 | prep 1)

    if (!p00 || !p11) {
        free(p00);
        free(p11);
        return matrix;
    }

    // Initialize with typical values
    for (size_t q = 0; q < num_qubits; q++) {
        p00[q] = 0.97;  // 3% readout error
        p11[q] = 0.95;  // 5% readout error (typically asymmetric)
    }

    // Parse actual calibration values
    size_t cal_len = json_object_array_length(cal_data);
    for (size_t i = 0; i < cal_len && i < num_qubits * 2; i++) {
        struct json_object* cal_entry = json_object_array_get_idx(cal_data, i);
        struct json_object* val_obj;

        size_t qubit = i % num_qubits;
        bool is_one_prep = (i >= num_qubits);

        if (json_object_object_get_ex(cal_entry, "success_rate", &val_obj)) {
            double rate = json_object_get_double(val_obj);
            if (is_one_prep) {
                p11[qubit] = rate;
            } else {
                p00[qubit] = rate;
            }
        }
    }

    // Build full confusion matrix using tensor product structure
    // For each bitstring i and j, compute probability
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            double prob = 1.0;
            for (size_t q = 0; q < num_qubits; q++) {
                int bit_i = (i >> q) & 1;  // Measured bit
                int bit_j = (j >> q) & 1;  // Prepared bit

                if (bit_j == 0) {
                    prob *= (bit_i == 0) ? p00[q] : (1.0 - p00[q]);
                } else {
                    prob *= (bit_i == 1) ? p11[q] : (1.0 - p11[q]);
                }
            }
            matrix[i * dim + j] = prob;
        }
    }

    free(p00);
    free(p11);
    return matrix;
}

// Invert readout matrix using regularized pseudo-inverse
static double* invert_readout_matrix(const double* matrix, size_t dim) {
    double* inverse = malloc(dim * dim * sizeof(double));
    if (!inverse) return NULL;

    // Copy matrix for in-place operations
    double* work = malloc(dim * dim * sizeof(double));
    double* identity = malloc(dim * dim * sizeof(double));

    if (!work || !identity) {
        free(work);
        free(identity);
        free(inverse);
        return NULL;
    }

    memcpy(work, matrix, dim * dim * sizeof(double));

    // Initialize identity matrix (will become inverse)
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            identity[i * dim + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Gauss-Jordan elimination with regularization
    double eps = 1e-10;  // Regularization for near-singular matrices

    for (size_t col = 0; col < dim; col++) {
        // Find pivot
        size_t pivot_row = col;
        double max_val = fabs(work[col * dim + col]);

        for (size_t row = col + 1; row < dim; row++) {
            if (fabs(work[row * dim + col]) > max_val) {
                max_val = fabs(work[row * dim + col]);
                pivot_row = row;
            }
        }

        // Swap rows
        if (pivot_row != col) {
            for (size_t j = 0; j < dim; j++) {
                double tmp = work[col * dim + j];
                work[col * dim + j] = work[pivot_row * dim + j];
                work[pivot_row * dim + j] = tmp;

                tmp = identity[col * dim + j];
                identity[col * dim + j] = identity[pivot_row * dim + j];
                identity[pivot_row * dim + j] = tmp;
            }
        }

        // Regularize if pivot is too small
        double pivot = work[col * dim + col];
        if (fabs(pivot) < eps) {
            pivot = eps * (pivot >= 0 ? 1 : -1);
            work[col * dim + col] = pivot;
        }

        // Scale pivot row
        double scale = 1.0 / pivot;
        for (size_t j = 0; j < dim; j++) {
            work[col * dim + j] *= scale;
            identity[col * dim + j] *= scale;
        }

        // Eliminate column
        for (size_t row = 0; row < dim; row++) {
            if (row != col) {
                double factor = work[row * dim + col];
                for (size_t j = 0; j < dim; j++) {
                    work[row * dim + j] -= factor * work[col * dim + j];
                    identity[row * dim + j] -= factor * identity[col * dim + j];
                }
            }
        }
    }

    memcpy(inverse, identity, dim * dim * sizeof(double));

    free(work);
    free(identity);
    return inverse;
}

// Apply readout error correction
static double* apply_readout_correction(const struct json_object* results,
                                      const struct json_object* calibration,
                                      const ErrorMitigationConfig* config) {
    if (!results || !config) return NULL;

    // Get number of qubits from config
    size_t num_qubits = config->num_qubits;
    size_t dim = 1UL << num_qubits;

    // Build readout error matrix from calibration
    double* readout_matrix = build_readout_matrix(calibration, num_qubits);
    if (!readout_matrix) return NULL;

    // Invert the readout matrix
    double* inverse_matrix = invert_readout_matrix(readout_matrix, dim);
    free(readout_matrix);
    if (!inverse_matrix) return NULL;

    // Extract measured probabilities from results
    double* measured = calloc(dim, sizeof(double));
    double* corrected = calloc(dim, sizeof(double));

    if (!measured || !corrected) {
        free(inverse_matrix);
        free(measured);
        free(corrected);
        return NULL;
    }

    // Parse results to get measurement counts
    size_t total_shots = 0;
    struct json_object* counts_obj;
    if (json_object_object_get_ex(results, "counts", &counts_obj)) {
        struct json_object_iterator it = json_object_iter_begin(counts_obj);
        struct json_object_iterator end = json_object_iter_end(counts_obj);

        while (!json_object_iter_equal(&it, &end)) {
            const char* bitstring = json_object_iter_peek_name(&it);
            struct json_object* count_obj = json_object_iter_peek_value(&it);
            size_t count = json_object_get_int(count_obj);

            // Convert bitstring to index
            size_t idx = 0;
            size_t len = strlen(bitstring);
            for (size_t i = 0; i < len && i < num_qubits; i++) {
                if (bitstring[len - 1 - i] == '1') {
                    idx |= (1UL << i);
                }
            }

            measured[idx] = (double)count;
            total_shots += count;

            json_object_iter_next(&it);
        }
    }

    // Normalize to probabilities
    if (total_shots > 0) {
        for (size_t i = 0; i < dim; i++) {
            measured[i] /= total_shots;
        }
    }

    // Apply inverse matrix: corrected = M^{-1} * measured
    for (size_t i = 0; i < dim; i++) {
        corrected[i] = 0.0;
        for (size_t j = 0; j < dim; j++) {
            corrected[i] += inverse_matrix[i * dim + j] * measured[j];
        }
    }

    // Clip negative probabilities and renormalize
    double total = 0.0;
    for (size_t i = 0; i < dim; i++) {
        if (corrected[i] < 0) corrected[i] = 0;
        total += corrected[i];
    }
    if (total > 0) {
        for (size_t i = 0; i < dim; i++) {
            corrected[i] /= total;
        }
    }

    free(inverse_matrix);
    free(measured);

    return corrected;
}

// Apply zero-noise extrapolation
static double* apply_zne_correction(const double* results,
                                  const ErrorMitigationConfig* config) {
    if (!results || !config) return NULL;

    size_t dim = 1UL << config->num_qubits;

    // ZNE works by running the circuit at multiple noise scales
    // and extrapolating to zero noise
    // The results array should contain results at different noise scales:
    // results[0..dim-1]: noise scale 1 (original)
    // results[dim..2*dim-1]: noise scale c1 (e.g., 1.5)
    // results[2*dim..3*dim-1]: noise scale c2 (e.g., 2.0)

    // Noise scale factors (pulse stretching)
    double noise_scales[] = {1.0, 1.5, 2.0};
    size_t num_scales = 3;

    double* corrected = calloc(dim, sizeof(double));
    if (!corrected) return NULL;

    // For each bitstring probability, fit polynomial and extrapolate to λ=0
    for (size_t i = 0; i < dim; i++) {
        // Collect data points: (λ_k, E(λ_k))
        double x[3], y[3];
        for (size_t k = 0; k < num_scales; k++) {
            x[k] = noise_scales[k];
            y[k] = results[k * dim + i];
        }

        // Richardson extrapolation using polynomial fit
        // For 3 points, we fit a quadratic: y = a + b*x + c*x^2
        // and extrapolate to x=0, which gives y(0) = a

        // Using Lagrange interpolation for extrapolation to x=0:
        // L(0) = sum_k y_k * prod_{j!=k} (-x_j) / (x_k - x_j)

        double y0 = 0.0;
        for (size_t k = 0; k < num_scales; k++) {
            double term = y[k];
            for (size_t j = 0; j < num_scales; j++) {
                if (j != k) {
                    term *= (0.0 - x[j]) / (x[k] - x[j]);
                }
            }
            y0 += term;
        }

        corrected[i] = y0;
    }

    // Clip and renormalize
    double total = 0.0;
    for (size_t i = 0; i < dim; i++) {
        if (corrected[i] < 0) corrected[i] = 0;
        if (corrected[i] > 1) corrected[i] = 1;
        total += corrected[i];
    }
    if (total > 0) {
        for (size_t i = 0; i < dim; i++) {
            corrected[i] /= total;
        }
    }

    return corrected;
}

// Apply symmetry verification
static double* apply_symmetry_correction(const double* results,
                                       const struct json_object* symmetry,
                                       const ErrorMitigationConfig* config) {
    if (!results || !config) return NULL;

    size_t dim = 1UL << config->num_qubits;
    size_t num_qubits = config->num_qubits;

    double* corrected = calloc(dim, sizeof(double));
    if (!corrected) return NULL;

    // Parse symmetry constraints from config
    // Default symmetry: total parity conservation
    // Only accept bitstrings with even parity if initial state had even parity

    // Get symmetry type from JSON
    int symmetry_type = 0;  // 0 = parity, 1 = hamming weight, 2 = custom
    int target_parity = 0;
    int target_hamming = -1;

    if (symmetry) {
        struct json_object* type_obj;
        if (json_object_object_get_ex(symmetry, "type", &type_obj)) {
            const char* type_str = json_object_get_string(type_obj);
            if (strcmp(type_str, "parity") == 0) symmetry_type = 0;
            else if (strcmp(type_str, "hamming") == 0) symmetry_type = 1;
            else if (strcmp(type_str, "custom") == 0) symmetry_type = 2;
        }

        struct json_object* parity_obj;
        if (json_object_object_get_ex(symmetry, "target_parity", &parity_obj)) {
            target_parity = json_object_get_int(parity_obj);
        }

        struct json_object* hamming_obj;
        if (json_object_object_get_ex(symmetry, "target_hamming", &hamming_obj)) {
            target_hamming = json_object_get_int(hamming_obj);
        }
    }

    // Post-select results based on symmetry
    double accepted_total = 0.0;

    for (size_t i = 0; i < dim; i++) {
        bool accept = false;

        switch (symmetry_type) {
            case 0: {  // Parity symmetry
                // Count bits in i
                int parity = 0;
                size_t bits = i;
                while (bits) {
                    parity ^= (bits & 1);
                    bits >>= 1;
                }
                accept = (parity == target_parity);
                break;
            }

            case 1: {  // Hamming weight symmetry
                if (target_hamming >= 0) {
                    int weight = 0;
                    size_t bits = i;
                    while (bits) {
                        weight += (bits & 1);
                        bits >>= 1;
                    }
                    accept = (weight == target_hamming);
                } else {
                    accept = true;  // No constraint
                }
                break;
            }

            case 2:  // Custom symmetry - accept all for now
            default:
                accept = true;
                break;
        }

        if (accept) {
            corrected[i] = results[i];
            accepted_total += results[i];
        } else {
            corrected[i] = 0.0;
        }
    }

    // Renormalize
    if (accepted_total > 0) {
        for (size_t i = 0; i < dim; i++) {
            corrected[i] /= accepted_total;
        }
    }

    return corrected;
}

// Estimate error bounds
static double estimate_error_bound(double probability,
                                 const ErrorMitigationConfig* config) {
    if (!config) return 0.1;  // Default 10% error

    // Combined error estimation considering multiple sources

    // 1. Statistical error from finite shots (binomial)
    // σ = sqrt(p(1-p)/N)
    double shots = (double)config->num_shots;
    if (shots < 1) shots = 1000;  // Default

    double stat_error = sqrt(probability * (1.0 - probability) / shots);

    // 2. Readout error contribution
    // Estimated from typical readout fidelities
    double readout_error = config->readout_error_rate;
    if (readout_error <= 0) readout_error = 0.02;  // Default 2%

    // Readout error propagates as approximately linear
    double readout_contribution = probability * readout_error +
                                  (1.0 - probability) * readout_error;

    // 3. Gate error contribution
    // Accumulates with circuit depth
    double gate_error = config->gate_error_rate;
    if (gate_error <= 0) gate_error = 0.001;  // Default 0.1%

    size_t depth = config->circuit_depth;
    if (depth < 1) depth = 10;  // Default

    // Gate errors accumulate roughly linearly for small errors
    double gate_contribution = gate_error * depth * probability;

    // 4. Decoherence contribution
    double circuit_time = (double)depth * 50e-9;  // 50ns per gate
    double t1 = config->t1_time;
    double t2 = config->t2_time;
    if (t1 <= 0) t1 = 50e-6;  // Default 50µs
    if (t2 <= 0) t2 = 30e-6;  // Default 30µs

    double decay_error = probability * (1.0 - exp(-circuit_time / t1));
    double dephase_error = probability * (1.0 - exp(-circuit_time / t2));
    double decoherence_contribution = decay_error + dephase_error;

    // 5. Combine errors in quadrature (assuming independence)
    double total_error = sqrt(stat_error * stat_error +
                              readout_contribution * readout_contribution +
                              gate_contribution * gate_contribution +
                              decoherence_contribution * decoherence_contribution);

    // Apply confidence factor for 95% confidence interval
    double confidence_factor = 1.96;  // 95% CI for normal distribution
    total_error *= confidence_factor;

    // Clip to reasonable range
    if (total_error > 1.0) total_error = 1.0;
    if (total_error < 1e-10) total_error = 1e-10;

    return total_error;
}

// Submit quantum circuit to Rigetti backend with error mitigation
// Matches header: struct QuantumCircuit* (HAL type)
int submit_rigetti_circuit(struct RigettiConfig* config,
                         struct QuantumCircuit* circuit,
                         struct MitigationParams* mitigation,
                         struct ExecutionResult* result) {
    if (!config || !circuit || !result) return -1;

    // Get internal backend from config
    RigettiInternalBackend* rb = (RigettiInternalBackend*)config->backend_specific_config;
    if (!rb) rb = g_rigetti_backend;
    if (!rb) return -1;

    // Update error config with circuit-specific values
    rb->error_config.num_qubits = circuit->num_qubits;
    rb->error_config.circuit_depth = circuit->num_gates;

    // Convert QuantumCircuit (HAL type) to hardware-efficient Quil
    char* quil = circuit_to_quil(circuit);
    if (!quil) return -1;
    
    // Create job request with error mitigation
    struct json_object* request = json_object_new_object();
    json_object_object_add(request, "device",
                          json_object_new_string(rb->device_name));
    json_object_object_add(request, "program",
                          json_object_new_string(quil));
    json_object_object_add(request, "shots",
                          json_object_new_int(DEFAULT_SHOTS));
    
    // Add error mitigation parameters
    struct json_object* mit_config = json_object_new_object();
    json_object_object_add(mit_config, "readout_correction",
                          json_object_new_boolean(true));
    json_object_object_add(mit_config, "symmetry_verification",
                          json_object_new_boolean(true));
    json_object_object_add(request, "error_mitigation", mit_config);
    
    // Submit job
    struct json_object* response = NULL;
    curl_easy_setopt(rb->curl, CURLOPT_URL, RIGETTI_JOBS_URL);
    curl_easy_setopt(rb->curl, CURLOPT_POSTFIELDS,
                    json_object_to_json_string(request));
    curl_easy_setopt(rb->curl, CURLOPT_WRITEDATA, &response);
    
    CURLcode res = curl_easy_perform(rb->curl);
    free(quil);
    json_object_put(request);
    
    if (res != CURLE_OK) return -1;
    
    // Get job ID
    struct json_object* job_id_obj;
    if (!json_object_object_get_ex(response, "id", &job_id_obj)) {
        json_object_put(response);
        return -1;
    }
    
    const char* job_id = json_object_get_string(job_id_obj);
    
    // Wait for job completion
    bool completed = false;
    int retries = 0;
    
    while (!completed && retries < MAX_RETRIES) {
        // Check job status
        char status_url[256];
        snprintf(status_url, sizeof(status_url),
                 "%s/%s", RIGETTI_JOBS_URL, job_id);
        
        curl_easy_setopt(rb->curl, CURLOPT_URL, status_url);
        struct json_object* status_response = NULL;
        res = curl_easy_perform(rb->curl);
        
        if (res == CURLE_OK) {
            struct json_object* status_obj;
            if (json_object_object_get_ex(status_response,
                                        "status",
                                        &status_obj)) {
                const char* status = json_object_get_string(status_obj);
                if (strcmp(status, "COMPLETED") == 0) {
                    completed = true;
                    
                    // Get results with error mitigation
                    struct json_object* results_obj;
                    struct json_object* calibration_obj;
                    struct json_object* symmetry_obj;
                    
                    if (json_object_object_get_ex(status_response,
                                                "results",
                                                &results_obj) &&
                        json_object_object_get_ex(status_response,
                                                "calibration",
                                                &calibration_obj) &&
                        json_object_object_get_ex(status_response,
                                                "symmetry",
                                                &symmetry_obj)) {
                        
                        // Apply error mitigation pipeline
                        double* corrected = apply_readout_correction(
                            results_obj,
                            calibration_obj,
                            &rb->error_config);
                            
                        double* zne = apply_zne_correction(
                            corrected,
                            &rb->error_config);
                            
                        double* final = apply_symmetry_correction(
                            zne,
                            symmetry_obj,
                            &rb->error_config);
                            
                        // Store final results with error bounds
                        // ExecutionResult uses num_results, not num_states
                        size_t dim = 1UL << circuit->num_qubits;
                        if (!result->probabilities) {
                            result->probabilities = calloc(dim, sizeof(double));
                        }
                        result->num_results = dim;

                        // Store error bounds in backend_data
                        double* error_bounds = calloc(dim, sizeof(double));
                        double max_error = 0.0;

                        for (size_t i = 0; i < result->num_results; i++) {
                            result->probabilities[i] = final[i];
                            if (error_bounds) {
                                error_bounds[i] = estimate_error_bound(
                                    final[i],
                                    &rb->error_config);
                                if (error_bounds[i] > max_error) {
                                    max_error = error_bounds[i];
                                }
                            }
                        }

                        // Store error bounds in backend_data and max error in error_rate
                        result->error_rate = max_error;
                        result->backend_data = error_bounds;
                        
                        free(corrected);
                        free(zne);
                        free(final);
                    }
                }
            }
            json_object_put(status_response);
        }
        
        if (!completed) {
            sleep(1);  // Wait before retrying
            retries++;
        }
    }
    
    json_object_put(response);
    return completed ? 0 : -1;
}

// Clean up Rigetti config (matches header)
void cleanup_rigetti_config(struct RigettiConfig* config) {
    if (!config) return;

    // Clean up internal backend if present
    if (config->backend_specific_config) {
        RigettiInternalBackend* internal = (RigettiInternalBackend*)config->backend_specific_config;

        if (internal->curl) {
            curl_easy_cleanup(internal->curl);
            curl_global_cleanup();
        }

        if (internal->json_config) {
            json_object_put(internal->json_config);
        }

        free(internal->api_key);
        free(internal->device_name);
        free(internal);
        config->backend_specific_config = NULL;
    }

    // Clean up RigettiConfig fields
    free(config->api_key);
    free(config->url);
    free(config->backend_name);
    free(config->noise_model);
    free(config);

    // Clear global reference
    g_rigetti_backend = NULL;
}

// ============================================================================
// Job Management Functions
// ============================================================================

/**
 * @brief Initialize QCS handle if not already done
 */
static bool ensure_qcs_handle(RigettiInternalBackend* internal) {
    if (!internal) return false;
    if (internal->qcs_handle) return true;

    // Try to connect using API key or default settings
    if (internal->api_key) {
        qcs_auth_config_t auth = {
            .api_key = internal->api_key,
            .user_id = NULL,
            .qcs_url = NULL,
            .quilc_url = NULL,
            .qvm_url = NULL,
            .use_client_configuration = false
        };
        internal->qcs_handle = qcs_connect(&auth);
    } else {
        internal->qcs_handle = qcs_connect_default();
    }

    if (!internal->qcs_handle) {
        snprintf(internal->last_error, sizeof(internal->last_error),
                 "Failed to establish QCS connection");
        internal->last_error_code = -1;
        return false;
    }

    return true;
}

/**
 * @brief Add job ID to tracking list
 */
static bool track_job(RigettiInternalBackend* internal, const char* job_id) {
    if (!internal || !job_id) return false;

    // Initialize job tracking if needed
    if (!internal->active_job_ids) {
        internal->max_active_jobs = 64;
        internal->active_job_ids = calloc(internal->max_active_jobs, sizeof(char*));
        if (!internal->active_job_ids) return false;
    }

    // Grow if needed
    if (internal->num_active_jobs >= internal->max_active_jobs) {
        size_t new_max = internal->max_active_jobs * 2;
        char** new_ids = realloc(internal->active_job_ids, new_max * sizeof(char*));
        if (!new_ids) return false;
        internal->active_job_ids = new_ids;
        internal->max_active_jobs = new_max;
    }

    internal->active_job_ids[internal->num_active_jobs++] = strdup(job_id);
    return true;
}

/**
 * @brief Remove job ID from tracking list
 */
static void untrack_job(RigettiInternalBackend* internal, const char* job_id) {
    if (!internal || !job_id || !internal->active_job_ids) return;

    for (size_t i = 0; i < internal->num_active_jobs; i++) {
        if (internal->active_job_ids[i] && strcmp(internal->active_job_ids[i], job_id) == 0) {
            free(internal->active_job_ids[i]);
            // Shift remaining jobs
            for (size_t j = i; j < internal->num_active_jobs - 1; j++) {
                internal->active_job_ids[j] = internal->active_job_ids[j + 1];
            }
            internal->num_active_jobs--;
            return;
        }
    }
}

/**
 * @brief Cache a job result for later retrieval
 */
static void cache_result(RigettiInternalBackend* internal, const char* job_id,
                         const qcs_job_result_t* qcs_result) {
    if (!internal || !job_id || !qcs_result) return;

    // Find empty slot or oldest entry
    size_t slot = internal->num_cached_results;
    if (slot >= 16) {
        // Overwrite oldest (slot 0) and shift
        qcs_free_result((qcs_job_result_t*)&internal->cached_results[0].result);
        for (size_t i = 0; i < 15; i++) {
            internal->cached_results[i] = internal->cached_results[i + 1];
        }
        slot = 15;
    } else {
        internal->num_cached_results++;
    }

    // Copy result
    strncpy(internal->cached_results[slot].job_id, job_id, 63);
    internal->cached_results[slot].job_id[63] = '\0';
    memcpy(&internal->cached_results[slot].result, qcs_result, sizeof(qcs_job_result_t));
    internal->cached_results[slot].valid = true;
}

/**
 * @brief Find cached result by job ID
 */
static qcs_job_result_t* find_cached_result(RigettiInternalBackend* internal, const char* job_id) {
    if (!internal || !job_id) return NULL;

    for (size_t i = 0; i < internal->num_cached_results; i++) {
        if (internal->cached_results[i].valid &&
            strcmp(internal->cached_results[i].job_id, job_id) == 0) {
            return &internal->cached_results[i].result;
        }
    }
    return NULL;
}

/**
 * @brief Map QCS job status to RigettiJobStatus
 */
static RigettiJobStatus map_qcs_status(qcs_job_status_t qcs_status) {
    switch (qcs_status) {
        case QCS_JOB_PENDING:  return RIGETTI_STATUS_QUEUED;
        case QCS_JOB_QUEUED:   return RIGETTI_STATUS_QUEUED;
        case QCS_JOB_RUNNING:  return RIGETTI_STATUS_RUNNING;
        case QCS_JOB_COMPLETED: return RIGETTI_STATUS_COMPLETED;
        case QCS_JOB_FAILED:   return RIGETTI_STATUS_ERROR;
        case QCS_JOB_CANCELLED: return RIGETTI_STATUS_CANCELLED;
        default:               return RIGETTI_STATUS_ERROR;
    }
}

/**
 * Submit a job to Rigetti QCS
 *
 * This function converts the circuit to Quil format and submits it
 * to the Rigetti QCS platform for execution on QPU or QVM.
 */
char* submit_rigetti_job(struct RigettiConfig* config, const struct RigettiJobConfig* job_config) {
    if (!job_config) {
        return NULL;
    }
    (void)config;  // Use global backend state

    RigettiInternalBackend* internal = g_rigetti_backend;
    if (!internal) {
        log_error("Rigetti backend not initialized");
        return NULL;
    }

    // Ensure QCS connection
    if (!ensure_qcs_handle(internal)) {
        log_error("Failed to connect to QCS: %s", internal->last_error);
        return NULL;
    }

    // Convert circuit to Quil
    char* quil_program = NULL;
    if (job_config->circuit) {
        quil_program = circuit_to_quil(job_config->circuit);
        if (!quil_program) {
            snprintf(internal->last_error, sizeof(internal->last_error),
                     "Failed to convert circuit to Quil format");
            return NULL;
        }
    } else {
        snprintf(internal->last_error, sizeof(internal->last_error),
                 "No circuit provided in job configuration");
        return NULL;
    }

    // Configure execution options
    qcs_execution_options_t exec_options = {
        .target = internal->is_simulator ? QCS_TARGET_QVM : QCS_TARGET_QPU,
        .qpu_name = internal->device_name,
        .shots = job_config->shots > 0 ? job_config->shots : DEFAULT_SHOTS,
        .use_quilc = job_config->optimize,
        .use_parametric = job_config->use_parametric_compilation,
        .use_active_reset = false,
        .timeout_seconds = API_TIMEOUT
    };

    // Submit job to QCS
    char* job_id = qcs_submit_program(internal->qcs_handle, quil_program, &exec_options);
    free(quil_program);

    if (!job_id) {
        const char* qcs_error = qcs_get_last_error(internal->qcs_handle);
        snprintf(internal->last_error, sizeof(internal->last_error),
                 "Failed to submit job to QCS: %s",
                 qcs_error ? qcs_error : "Unknown error");
        return NULL;
    }

    // Track the job
    track_job(internal, job_id);

    log_info("Submitted job %s to Rigetti %s (%zu shots)",
             job_id,
             internal->is_simulator ? "QVM" : internal->device_name,
             exec_options.shots);

    return job_id;
}

/**
 * Get status of a Rigetti job
 */
RigettiJobStatus get_rigetti_job_status(struct RigettiConfig* config, const char* job_id) {
    (void)config;

    if (!job_id) {
        return RIGETTI_STATUS_ERROR;
    }

    RigettiInternalBackend* internal = g_rigetti_backend;
    if (!internal) {
        return RIGETTI_STATUS_ERROR;
    }

    // Check cached results first
    qcs_job_result_t* cached = find_cached_result(internal, job_id);
    if (cached) {
        return map_qcs_status(cached->status);
    }

    // Query QCS for status
    if (!internal->qcs_handle) {
        return RIGETTI_STATUS_ERROR;
    }

    qcs_job_status_t qcs_status = qcs_get_job_status(internal->qcs_handle, job_id);
    return map_qcs_status(qcs_status);
}

/**
 * Get result of a Rigetti job
 *
 * This function retrieves the execution results from QCS, including
 * measurement counts, probabilities, and execution metadata.
 */
RigettiJobResult* get_rigetti_job_result(struct RigettiConfig* config, const char* job_id) {
    (void)config;

    if (!job_id) {
        return NULL;
    }

    RigettiInternalBackend* internal = g_rigetti_backend;
    if (!internal || !internal->qcs_handle) {
        return NULL;
    }

    // Check cache first
    qcs_job_result_t* cached = find_cached_result(internal, job_id);
    qcs_job_result_t qcs_result;
    qcs_job_result_t* result_ptr = cached;

    if (!cached) {
        // Fetch from QCS
        memset(&qcs_result, 0, sizeof(qcs_result));
        if (!qcs_get_job_result(internal->qcs_handle, job_id, &qcs_result)) {
            snprintf(internal->last_error, sizeof(internal->last_error),
                     "Failed to retrieve job result from QCS");
            return NULL;
        }
        result_ptr = &qcs_result;

        // Cache the result
        cache_result(internal, job_id, &qcs_result);
    }

    // Create RigettiJobResult from qcs_job_result_t
    RigettiJobResult* result = calloc(1, sizeof(RigettiJobResult));
    if (!result) {
        return NULL;
    }

    result->status = map_qcs_status(result_ptr->status);

    // Copy counts if available
    if (result_ptr->counts && result_ptr->num_outcomes > 0) {
        result->counts = calloc(result_ptr->num_outcomes, sizeof(uint64_t));
        if (result->counts) {
            memcpy(result->counts, result_ptr->counts,
                   result_ptr->num_outcomes * sizeof(uint64_t));
        }
    }

    // Copy probabilities if available
    if (result_ptr->probabilities && result_ptr->num_outcomes > 0) {
        result->probabilities = calloc(result_ptr->num_outcomes, sizeof(double));
        if (result->probabilities) {
            memcpy(result->probabilities, result_ptr->probabilities,
                   result_ptr->num_outcomes * sizeof(double));
        }
    }

    // Calculate fidelity estimate from probabilities
    if (result->probabilities && result_ptr->num_outcomes > 0) {
        // Use max probability as rough fidelity estimate
        double max_prob = 0.0;
        for (size_t i = 0; i < result_ptr->num_outcomes; i++) {
            if (result->probabilities[i] > max_prob) {
                max_prob = result->probabilities[i];
            }
        }
        result->fidelity = max_prob;
        result->error_rate = 1.0 - max_prob;
    }

    // Copy error message if present
    if (result_ptr->error_message) {
        result->error_message = strdup(result_ptr->error_message);
    }

    // Store raw data reference
    result->raw_data = result_ptr->metadata;

    // Remove from active tracking
    untrack_job(internal, job_id);

    return result;
}

/**
 * Cancel a Rigetti job
 */
bool cancel_rigetti_job(struct RigettiConfig* config, const char* job_id) {
    (void)config;

    if (!job_id) {
        return false;
    }

    RigettiInternalBackend* internal = g_rigetti_backend;
    if (!internal || !internal->qcs_handle) {
        return false;
    }

    bool success = qcs_cancel_job(internal->qcs_handle, job_id);
    if (success) {
        untrack_job(internal, job_id);
        log_info("Cancelled job %s", job_id);
    } else {
        const char* error = qcs_get_last_error(internal->qcs_handle);
        snprintf(internal->last_error, sizeof(internal->last_error),
                 "Failed to cancel job: %s", error ? error : "Unknown error");
    }

    return success;
}

/**
 * Get error information for a Rigetti job
 */
char* get_rigetti_error_info(struct RigettiConfig* config, const char* job_id) {
    (void)config;
    (void)job_id;

    RigettiInternalBackend* internal = g_rigetti_backend;
    if (!internal) {
        return strdup("Rigetti backend not initialized");
    }

    // First check QCS handle for latest error
    if (internal->qcs_handle) {
        const char* qcs_error = qcs_get_last_error(internal->qcs_handle);
        if (qcs_error && qcs_error[0]) {
            return strdup(qcs_error);
        }
    }

    // Fall back to cached error
    if (internal->last_error[0]) {
        return strdup(internal->last_error);
    }

    return NULL;
}

/**
 * Clean up a Rigetti job result
 */
void cleanup_rigetti_result(RigettiJobResult* result) {
    if (!result) {
        return;
    }

    free(result->counts);
    free(result->probabilities);
    free(result->error_message);
    free(result->parametric_values);
    // raw_data is owned by cache, don't free
    free(result);
}

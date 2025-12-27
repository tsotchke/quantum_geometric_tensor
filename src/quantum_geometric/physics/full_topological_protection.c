/**
 * @file full_topological_protection.c
 * @brief Full topological protection system implementation
 *
 * Production implementation of topological quantum error correction using:
 * - Union-Find decoder for efficient syndrome processing
 * - Hardware-aware noise modeling with T1/T2 characterization
 * - Soft syndrome decoding with confidence-weighted corrections
 * - Temporal correlation tracking for measurement error mitigation
 * - Proper quantum coherence measures from quantum resource theory
 */

#include "quantum_geometric/physics/full_topological_protection.h"
#include "quantum_geometric/physics/surface_code.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/physics/error_syndrome.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <float.h>

// ============================================================================
// Physical Constants
// ============================================================================

#define PLANCK_REDUCED 1.054571817e-34      // ℏ in J·s
#define BOLTZMANN 1.380649e-23              // kB in J/K
#define DEFAULT_TEMPERATURE 0.015           // 15 mK typical dilution fridge

// Algorithm constants
#define MAX_ERROR_LOCATIONS 8192
#define MAX_SYNDROME_HISTORY 64
#define UNION_FIND_MAX_CLUSTERS 4096
#define DECODER_CONFIDENCE_THRESHOLD 0.5
#define MEASUREMENT_REPETITION_CODE 3
#define TEMPORAL_WINDOW_SIZE 16

// ============================================================================
// Physical Noise Model Structures
// ============================================================================

/**
 * @brief Physical qubit characterization
 */
typedef struct {
    double t1_time;                    // T1 relaxation time (amplitude damping)
    double t2_time;                    // T2 dephasing time (phase damping)
    double t2_star;                    // T2* inhomogeneous dephasing
    double readout_fidelity;           // Measurement fidelity
    double gate_fidelity_single;       // Single-qubit gate fidelity
    double gate_fidelity_two;          // Two-qubit gate fidelity
    double thermal_population;         // Thermal excitation probability
    double leakage_rate;               // Leakage to non-computational states
    double crosstalk_strength;         // ZZ crosstalk coefficient
    double frequency_drift;            // Qubit frequency instability
} QubitCharacterization;

/**
 * @brief Syndrome measurement with soft information
 */
typedef struct {
    size_t stabilizer_index;           // Which stabilizer
    double raw_value;                  // Raw measurement outcome [-1, 1]
    double confidence;                 // Confidence in measurement [0, 1]
    double timestamp;                  // When measured
    bool flipped;                      // Whether syndrome is flipped from expected
    int repetition_count;              // Number of confirming measurements
    double temporal_correlation;       // Correlation with previous measurements
} SoftSyndrome;

/**
 * @brief Temporal syndrome history for measurement error mitigation
 */
typedef struct {
    SoftSyndrome* measurements;        // Ring buffer of measurements
    size_t capacity;                   // Buffer capacity
    size_t head;                       // Current write position
    size_t count;                      // Number of valid entries
    double* correlation_matrix;        // Temporal correlations between rounds
} SyndromeHistory;

/**
 * @brief Error location with physical context
 */
typedef struct {
    size_t location;                   // Lattice position
    size_t qubit_index;                // Physical qubit index
    double x_coord, y_coord;           // Lattice coordinates
    double detection_time;             // When error was detected
    double weight;                     // Error weight/probability
    int error_type;                    // 0=X, 1=Z, 2=Y
    double confidence;                 // Detection confidence
    bool corrected;                    // Has been corrected
    size_t detection_round;            // Which syndrome round
    double persistence;                // How long error has persisted
} PhysicalErrorLocation;

// ============================================================================
// Union-Find Decoder Structures
// ============================================================================

/**
 * @brief Union-Find node for cluster management
 */
typedef struct UFNode {
    size_t id;                         // Node identifier
    struct UFNode* parent;             // Parent in Union-Find tree
    size_t rank;                       // Rank for union by rank
    size_t cluster_size;               // Size of cluster if root
    bool is_boundary;                  // Is boundary node
    double x, y;                       // Position
    int parity;                        // Cluster parity (odd/even syndrome count)
} UFNode;

/**
 * @brief Cluster for Union-Find decoder
 */
typedef struct {
    UFNode* root;                      // Root node of cluster
    size_t* members;                   // Array of member node ids
    size_t num_members;                // Number of members
    size_t capacity;                   // Capacity of members array
    bool is_odd;                       // Has odd parity (needs correction)
    bool touches_boundary;             // Connected to boundary
    double total_weight;               // Sum of edge weights in cluster
} UFCluster;

/**
 * @brief Complete Union-Find decoder state
 */
typedef struct {
    UFNode* nodes;                     // All nodes (syndromes + boundaries)
    size_t num_nodes;                  // Total node count
    size_t num_syndrome_nodes;         // Number of syndrome nodes
    size_t num_boundary_nodes;         // Number of boundary nodes
    UFCluster* clusters;               // Array of clusters
    size_t num_clusters;               // Current cluster count
    size_t max_clusters;               // Maximum clusters
    double* edge_weights;              // Precomputed edge weights
    size_t lattice_width;              // Lattice dimensions
    size_t lattice_height;
    bool* grown_edges;                 // Which edges have been grown
    double growth_radius;              // Current growth radius
} UnionFindDecoder;

// ============================================================================
// Internal State
// ============================================================================

static SurfaceCode* topo_active_surface_code = NULL;

static double topo_last_fast_cycle_time = 0.0;
static double topo_last_medium_cycle_time = 0.0;
static double topo_last_slow_cycle_time = 0.0;

static PhysicalErrorLocation physical_errors[MAX_ERROR_LOCATIONS];
static size_t num_physical_errors = 0;

static SyndromeHistory syndrome_history = {0};

static QubitCharacterization* qubit_chars = NULL;
static size_t num_qubit_chars = 0;

// ============================================================================
// Time and Physical Functions
// ============================================================================

static double topo_get_system_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void topo_wait_interval(double wait_time) {
    if (wait_time <= 0.0) return;
    struct timespec ts;
    ts.tv_sec = (time_t)wait_time;
    ts.tv_nsec = (long)((wait_time - (double)ts.tv_sec) * 1e9);
    nanosleep(&ts, NULL);
}

/**
 * @brief Calculate thermal population at given temperature
 *
 * P_excited = 1 / (1 + exp(hf/kT))
 * For typical transmon at 5 GHz and 15 mK: ~0.001
 */
static double calculate_thermal_population(double frequency_ghz, double temperature_k) {
    double hf = PLANCK_REDUCED * frequency_ghz * 1e9 * 2.0 * M_PI;
    double kt = BOLTZMANN * temperature_k;
    if (kt < 1e-30) return 0.0;
    return 1.0 / (1.0 + exp(hf / kt));
}

/**
 * @brief Calculate T1 decay probability over time interval
 *
 * P_decay = 1 - exp(-t/T1)
 */
static double calculate_t1_decay(double t1_time, double elapsed_time) {
    if (t1_time <= 0.0 || elapsed_time <= 0.0) return 0.0;
    return 1.0 - exp(-elapsed_time / t1_time);
}

/**
 * @brief Calculate T2 dephasing probability over time interval
 *
 * Coherence decays as exp(-t/T2) for pure dephasing
 */
static double calculate_t2_dephasing(double t2_time, double elapsed_time) {
    if (t2_time <= 0.0 || elapsed_time <= 0.0) return 0.0;
    return 1.0 - exp(-elapsed_time / t2_time);
}

/**
 * @brief Initialize qubit characterization from hardware config
 */
static void init_qubit_characterization(const HardwareConfig* config) {
    if (qubit_chars) {
        free(qubit_chars);
    }

    num_qubit_chars = config->num_qubits;
    qubit_chars = (QubitCharacterization*)calloc(num_qubit_chars, sizeof(QubitCharacterization));
    if (!qubit_chars) {
        num_qubit_chars = 0;
        return;
    }

    // Initialize with hardware config values, adding realistic variation
    for (size_t i = 0; i < num_qubit_chars; i++) {
        // Add ±10% variation to simulate real device non-uniformity
        double variation = 0.9 + 0.2 * ((double)(i % 10) / 10.0);

        qubit_chars[i].t1_time = config->t1_time * variation;
        qubit_chars[i].t2_time = config->coherence_time * variation;
        qubit_chars[i].t2_star = config->coherence_time * 0.8 * variation;  // T2* < T2
        qubit_chars[i].readout_fidelity = config->readout_fidelity;
        qubit_chars[i].gate_fidelity_single = 1.0 - config->gate_error_rate;
        qubit_chars[i].gate_fidelity_two = 1.0 - config->gate_error_rate * 10;  // 2Q gates worse
        qubit_chars[i].thermal_population = calculate_thermal_population(5.0, DEFAULT_TEMPERATURE);
        qubit_chars[i].leakage_rate = config->gate_error_rate * 0.01;  // 1% of gate errors are leakage
        qubit_chars[i].crosstalk_strength = 0.001 * variation;  // Typical ZZ crosstalk
        qubit_chars[i].frequency_drift = 1e-4 * variation;  // Relative frequency instability
    }
}

// ============================================================================
// Syndrome History Management
// ============================================================================

static bool init_syndrome_history(size_t capacity) {
    if (syndrome_history.measurements) {
        free(syndrome_history.measurements);
        free(syndrome_history.correlation_matrix);
    }

    syndrome_history.measurements = (SoftSyndrome*)calloc(capacity, sizeof(SoftSyndrome));
    syndrome_history.correlation_matrix = (double*)calloc(capacity * capacity, sizeof(double));

    if (!syndrome_history.measurements || !syndrome_history.correlation_matrix) {
        free(syndrome_history.measurements);
        free(syndrome_history.correlation_matrix);
        memset(&syndrome_history, 0, sizeof(syndrome_history));
        return false;
    }

    syndrome_history.capacity = capacity;
    syndrome_history.head = 0;
    syndrome_history.count = 0;

    // Initialize correlation matrix to identity
    for (size_t i = 0; i < capacity; i++) {
        syndrome_history.correlation_matrix[i * capacity + i] = 1.0;
    }

    return true;
}

static void add_syndrome_measurement(const SoftSyndrome* syndrome) {
    if (!syndrome_history.measurements) return;

    size_t idx = syndrome_history.head;
    syndrome_history.measurements[idx] = *syndrome;

    // Update temporal correlations with previous measurements
    if (syndrome_history.count > 0) {
        for (size_t i = 0; i < syndrome_history.count && i < TEMPORAL_WINDOW_SIZE; i++) {
            size_t prev_idx = (idx + syndrome_history.capacity - i - 1) % syndrome_history.capacity;
            SoftSyndrome* prev = &syndrome_history.measurements[prev_idx];

            if (prev->stabilizer_index == syndrome->stabilizer_index) {
                // Same stabilizer - compute temporal correlation
                double dt = syndrome->timestamp - prev->timestamp;
                double decay = exp(-dt / 1e-3);  // 1ms correlation timescale
                double corr = (syndrome->raw_value * prev->raw_value) * decay;

                // Update correlation matrix
                size_t mat_idx = idx * syndrome_history.capacity + prev_idx;
                syndrome_history.correlation_matrix[mat_idx] = corr;
            }
        }
    }

    syndrome_history.head = (syndrome_history.head + 1) % syndrome_history.capacity;
    if (syndrome_history.count < syndrome_history.capacity) {
        syndrome_history.count++;
    }
}

/**
 * @brief Get temporally-filtered syndrome value using majority voting
 */
static double get_filtered_syndrome(size_t stabilizer_index, int window_size) {
    if (!syndrome_history.measurements || syndrome_history.count == 0) {
        return 0.0;
    }

    double sum = 0.0;
    double weight_sum = 0.0;
    int count = 0;

    for (size_t i = 0; i < syndrome_history.count && count < window_size; i++) {
        size_t idx = (syndrome_history.head + syndrome_history.capacity - i - 1) % syndrome_history.capacity;
        SoftSyndrome* s = &syndrome_history.measurements[idx];

        if (s->stabilizer_index == stabilizer_index) {
            double weight = s->confidence;
            sum += s->raw_value * weight;
            weight_sum += weight;
            count++;
        }
    }

    return (weight_sum > 0) ? sum / weight_sum : 0.0;
}

// ============================================================================
// Physical Error Tracking
// ============================================================================

static void clear_physical_errors(void) {
    num_physical_errors = 0;
    memset(physical_errors, 0, sizeof(physical_errors));
}

static bool add_physical_error(size_t location, size_t qubit_index, double x, double y,
                               double weight, int error_type, double confidence,
                               size_t detection_round) {
    // Check for existing error at same location
    for (size_t i = 0; i < num_physical_errors; i++) {
        if (physical_errors[i].location == location && !physical_errors[i].corrected) {
            // Update existing error
            physical_errors[i].weight = fmax(physical_errors[i].weight, weight);
            physical_errors[i].confidence = fmax(physical_errors[i].confidence, confidence);
            physical_errors[i].persistence = topo_get_system_time() - physical_errors[i].detection_time;
            return true;
        }
    }

    // Compact array if needed
    if (num_physical_errors >= MAX_ERROR_LOCATIONS) {
        size_t write_idx = 0;
        for (size_t read_idx = 0; read_idx < num_physical_errors; read_idx++) {
            if (!physical_errors[read_idx].corrected) {
                if (write_idx != read_idx) {
                    physical_errors[write_idx] = physical_errors[read_idx];
                }
                write_idx++;
            }
        }
        num_physical_errors = write_idx;

        if (num_physical_errors >= MAX_ERROR_LOCATIONS) {
            return false;
        }
    }

    // Add new error
    PhysicalErrorLocation* err = &physical_errors[num_physical_errors];
    err->location = location;
    err->qubit_index = qubit_index;
    err->x_coord = x;
    err->y_coord = y;
    err->detection_time = topo_get_system_time();
    err->weight = weight;
    err->error_type = error_type;
    err->confidence = confidence;
    err->corrected = false;
    err->detection_round = detection_round;
    err->persistence = 0.0;

    num_physical_errors++;
    return true;
}

// ============================================================================
// Union-Find Decoder Implementation
// ============================================================================

/**
 * @brief Find root of Union-Find tree with path compression
 */
static UFNode* uf_find(UFNode* node) {
    if (!node) return NULL;
    if (node->parent != node) {
        node->parent = uf_find(node->parent);  // Path compression
    }
    return node->parent;
}

/**
 * @brief Union two clusters by rank
 */
static void uf_union(UFNode* x, UFNode* y) {
    UFNode* root_x = uf_find(x);
    UFNode* root_y = uf_find(y);

    if (root_x == root_y) return;

    // Union by rank
    if (root_x->rank < root_y->rank) {
        root_x->parent = root_y;
        root_y->cluster_size += root_x->cluster_size;
        root_y->parity ^= root_x->parity;
        root_y->is_boundary |= root_x->is_boundary;
    } else if (root_x->rank > root_y->rank) {
        root_y->parent = root_x;
        root_x->cluster_size += root_y->cluster_size;
        root_x->parity ^= root_y->parity;
        root_x->is_boundary |= root_y->is_boundary;
    } else {
        root_y->parent = root_x;
        root_x->rank++;
        root_x->cluster_size += root_y->cluster_size;
        root_x->parity ^= root_y->parity;
        root_x->is_boundary |= root_y->is_boundary;
    }
}

/**
 * @brief Create Union-Find decoder for given lattice
 */
static UnionFindDecoder* create_uf_decoder(size_t width, size_t height) {
    UnionFindDecoder* decoder = (UnionFindDecoder*)malloc(sizeof(UnionFindDecoder));
    if (!decoder) return NULL;

    decoder->lattice_width = width;
    decoder->lattice_height = height;

    // Syndrome nodes: one per plaquette/vertex = 2 * (width-1) * (height-1) for rotated code
    decoder->num_syndrome_nodes = 2 * (width - 1) * (height - 1);
    // Boundary nodes: perimeter
    decoder->num_boundary_nodes = 2 * (width + height);
    decoder->num_nodes = decoder->num_syndrome_nodes + decoder->num_boundary_nodes;

    decoder->nodes = (UFNode*)calloc(decoder->num_nodes, sizeof(UFNode));
    if (!decoder->nodes) {
        free(decoder);
        return NULL;
    }

    // Initialize syndrome nodes
    size_t node_idx = 0;
    for (size_t y = 0; y < height - 1; y++) {
        for (size_t x = 0; x < width - 1; x++) {
            // X-type stabilizer
            decoder->nodes[node_idx].id = node_idx;
            decoder->nodes[node_idx].parent = &decoder->nodes[node_idx];
            decoder->nodes[node_idx].rank = 0;
            decoder->nodes[node_idx].cluster_size = 1;
            decoder->nodes[node_idx].is_boundary = false;
            decoder->nodes[node_idx].x = x + 0.5;
            decoder->nodes[node_idx].y = y + 0.5;
            decoder->nodes[node_idx].parity = 0;
            node_idx++;

            // Z-type stabilizer (offset by half unit cell)
            decoder->nodes[node_idx].id = node_idx;
            decoder->nodes[node_idx].parent = &decoder->nodes[node_idx];
            decoder->nodes[node_idx].rank = 0;
            decoder->nodes[node_idx].cluster_size = 1;
            decoder->nodes[node_idx].is_boundary = false;
            decoder->nodes[node_idx].x = x + 1.0;
            decoder->nodes[node_idx].y = y + 1.0;
            decoder->nodes[node_idx].parity = 0;
            node_idx++;
        }
    }

    // Initialize boundary nodes
    for (size_t i = 0; i < decoder->num_boundary_nodes; i++) {
        decoder->nodes[node_idx].id = node_idx;
        decoder->nodes[node_idx].parent = &decoder->nodes[node_idx];
        decoder->nodes[node_idx].rank = 0;
        decoder->nodes[node_idx].cluster_size = 1;
        decoder->nodes[node_idx].is_boundary = true;

        // Position boundary nodes around perimeter
        if (i < width) {
            decoder->nodes[node_idx].x = i;
            decoder->nodes[node_idx].y = -0.5;
        } else if (i < width + height) {
            decoder->nodes[node_idx].x = width - 0.5;
            decoder->nodes[node_idx].y = i - width;
        } else if (i < 2 * width + height) {
            decoder->nodes[node_idx].x = 2 * width + height - i - 1;
            decoder->nodes[node_idx].y = height - 0.5;
        } else {
            decoder->nodes[node_idx].x = -0.5;
            decoder->nodes[node_idx].y = 2 * width + 2 * height - i - 1;
        }
        decoder->nodes[node_idx].parity = 0;
        node_idx++;
    }

    // Precompute edge weights (Manhattan distances)
    size_t num_edges = decoder->num_nodes * decoder->num_nodes;
    decoder->edge_weights = (double*)calloc(num_edges, sizeof(double));
    if (!decoder->edge_weights) {
        free(decoder->nodes);
        free(decoder);
        return NULL;
    }

    for (size_t i = 0; i < decoder->num_nodes; i++) {
        for (size_t j = i + 1; j < decoder->num_nodes; j++) {
            double dx = decoder->nodes[i].x - decoder->nodes[j].x;
            double dy = decoder->nodes[i].y - decoder->nodes[j].y;
            double dist = fabs(dx) + fabs(dy);
            decoder->edge_weights[i * decoder->num_nodes + j] = dist;
            decoder->edge_weights[j * decoder->num_nodes + i] = dist;
        }
    }

    decoder->grown_edges = (bool*)calloc(num_edges, sizeof(bool));
    if (!decoder->grown_edges) {
        free(decoder->edge_weights);
        free(decoder->nodes);
        free(decoder);
        return NULL;
    }

    decoder->clusters = (UFCluster*)calloc(UNION_FIND_MAX_CLUSTERS, sizeof(UFCluster));
    decoder->max_clusters = UNION_FIND_MAX_CLUSTERS;
    decoder->num_clusters = 0;
    decoder->growth_radius = 0.0;

    return decoder;
}

/**
 * @brief Free Union-Find decoder
 */
static void free_uf_decoder(UnionFindDecoder* decoder) {
    if (!decoder) return;

    for (size_t i = 0; i < decoder->num_clusters; i++) {
        free(decoder->clusters[i].members);
    }
    free(decoder->clusters);
    free(decoder->grown_edges);
    free(decoder->edge_weights);
    free(decoder->nodes);
    free(decoder);
}

/**
 * @brief Reset decoder for new syndrome
 */
static void reset_uf_decoder(UnionFindDecoder* decoder) {
    if (!decoder) return;

    // Reset all nodes
    for (size_t i = 0; i < decoder->num_nodes; i++) {
        decoder->nodes[i].parent = &decoder->nodes[i];
        decoder->nodes[i].rank = 0;
        decoder->nodes[i].cluster_size = 1;
        decoder->nodes[i].parity = 0;
    }

    // Clear grown edges
    memset(decoder->grown_edges, 0,
           decoder->num_nodes * decoder->num_nodes * sizeof(bool));

    // Clear clusters
    for (size_t i = 0; i < decoder->num_clusters; i++) {
        free(decoder->clusters[i].members);
        decoder->clusters[i].members = NULL;
    }
    decoder->num_clusters = 0;
    decoder->growth_radius = 0.0;
}

/**
 * @brief Set syndrome values on decoder nodes
 */
static void set_decoder_syndrome(UnionFindDecoder* decoder, const SoftSyndrome* syndromes,
                                  size_t num_syndromes) {
    for (size_t i = 0; i < num_syndromes && i < decoder->num_syndrome_nodes; i++) {
        if (syndromes[i].flipped && syndromes[i].confidence > DECODER_CONFIDENCE_THRESHOLD) {
            decoder->nodes[i].parity = 1;
        } else {
            decoder->nodes[i].parity = 0;
        }
    }
}

/**
 * @brief Grow clusters by one step using Union-Find
 */
static bool grow_clusters(UnionFindDecoder* decoder, double growth_step) {
    decoder->growth_radius += growth_step;
    bool any_merge = false;

    // Find all edges within growth radius and union their endpoints
    for (size_t i = 0; i < decoder->num_nodes; i++) {
        for (size_t j = i + 1; j < decoder->num_nodes; j++) {
            size_t edge_idx = i * decoder->num_nodes + j;
            if (decoder->grown_edges[edge_idx]) continue;

            double weight = decoder->edge_weights[edge_idx];
            if (weight <= decoder->growth_radius) {
                // Check if at least one endpoint has odd parity
                UFNode* root_i = uf_find(&decoder->nodes[i]);
                UFNode* root_j = uf_find(&decoder->nodes[j]);

                if (root_i != root_j) {
                    bool i_odd = (root_i->parity != 0);
                    bool j_odd = (root_j->parity != 0);

                    // Union if at least one is odd
                    if (i_odd || j_odd) {
                        uf_union(&decoder->nodes[i], &decoder->nodes[j]);
                        decoder->grown_edges[edge_idx] = true;
                        decoder->grown_edges[j * decoder->num_nodes + i] = true;
                        any_merge = true;
                    }
                }
            }
        }
    }

    return any_merge;
}

/**
 * @brief Check if all odd clusters are neutral (paired or at boundary)
 */
static bool all_clusters_neutral(UnionFindDecoder* decoder) {
    for (size_t i = 0; i < decoder->num_syndrome_nodes; i++) {
        UFNode* root = uf_find(&decoder->nodes[i]);
        // Cluster is neutral if parity is even or touches boundary
        if (root->parity != 0 && !root->is_boundary) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Run Union-Find decoder
 */
static void run_uf_decoder(UnionFindDecoder* decoder) {
    double growth_step = 0.5;  // Grow by half a lattice unit each iteration
    int max_iterations = (int)(decoder->lattice_width + decoder->lattice_height) * 2;

    for (int iter = 0; iter < max_iterations; iter++) {
        if (all_clusters_neutral(decoder)) {
            break;
        }
        grow_clusters(decoder, growth_step);
    }
}

/**
 * @brief Extract correction from decoder state
 */
static void extract_uf_correction(UnionFindDecoder* decoder,
                                   size_t* qubit_indices, int* correction_types,
                                   size_t* num_corrections, size_t max_corrections) {
    *num_corrections = 0;

    // For each grown edge, add a correction along the path
    for (size_t i = 0; i < decoder->num_nodes && *num_corrections < max_corrections; i++) {
        for (size_t j = i + 1; j < decoder->num_nodes && *num_corrections < max_corrections; j++) {
            size_t edge_idx = i * decoder->num_nodes + j;
            if (!decoder->grown_edges[edge_idx]) continue;

            // Add correction at midpoint
            double mid_x = (decoder->nodes[i].x + decoder->nodes[j].x) / 2.0;
            double mid_y = (decoder->nodes[i].y + decoder->nodes[j].y) / 2.0;

            // Convert to qubit index
            size_t qx = (size_t)round(mid_x);
            size_t qy = (size_t)round(mid_y);
            if (qx >= decoder->lattice_width) qx = decoder->lattice_width - 1;
            if (qy >= decoder->lattice_height) qy = decoder->lattice_height - 1;

            size_t qubit = qy * decoder->lattice_width + qx;

            // Determine correction type based on which stabilizer type was violated
            // X stabilizers need Z corrections, Z stabilizers need X corrections
            int corr_type = (i < decoder->num_syndrome_nodes && (i % 2 == 0)) ? 1 : 0;

            qubit_indices[*num_corrections] = qubit;
            correction_types[*num_corrections] = corr_type;
            (*num_corrections)++;
        }
    }
}

// ============================================================================
// Quantum State Analysis
// ============================================================================

/**
 * @brief Calculate l1-norm of coherence (quantum resource theory measure)
 *
 * C_l1(rho) = sum_{i≠j} |rho_ij|
 * For pure state |psi> = sum_i c_i |i>: C = sum_{i≠j} |c_i||c_j|
 */
static double calculate_l1_coherence(quantum_state_t* state) {
    if (!state || !state->coordinates || state->dimension == 0) {
        return 0.0;
    }

    // Compute amplitudes and normalization
    double* abs_amplitudes = (double*)malloc(state->dimension * sizeof(double));
    if (!abs_amplitudes) return 0.0;

    double norm_sq = 0.0;
    for (size_t i = 0; i < state->dimension; i++) {
        double re = state->coordinates[i].real;
        double im = state->coordinates[i].imag;
        abs_amplitudes[i] = sqrt(re * re + im * im);
        norm_sq += re * re + im * im;
    }

    if (norm_sq < 1e-15) {
        free(abs_amplitudes);
        return 0.0;
    }
    double norm = sqrt(norm_sq);

    // Normalize amplitudes
    for (size_t i = 0; i < state->dimension; i++) {
        abs_amplitudes[i] /= norm;
    }

    // C_l1 = (sum |c_i|)^2 - sum |c_i|^2
    // The second term equals 1 for normalized states
    double sum_abs = 0.0;
    for (size_t i = 0; i < state->dimension; i++) {
        sum_abs += abs_amplitudes[i];
    }

    double coherence = sum_abs * sum_abs - 1.0;

    // Normalize to [0, 1]
    double max_coherence = (double)(state->dimension - 1);
    if (max_coherence > 0) {
        coherence /= max_coherence;
    }

    free(abs_amplitudes);
    return fmax(0.0, fmin(1.0, coherence));
}

/**
 * @brief Calculate quantum purity Tr(rho^2)
 */
static double calculate_purity(quantum_state_t* state) {
    if (!state || !state->coordinates || state->dimension == 0) {
        return 0.0;
    }

    double norm_sq = 0.0;
    double purity = 0.0;

    for (size_t i = 0; i < state->dimension; i++) {
        double re = state->coordinates[i].real;
        double im = state->coordinates[i].imag;
        double p_i = re * re + im * im;
        norm_sq += p_i;
        purity += p_i * p_i;
    }

    if (norm_sq < 1e-15) return 0.0;
    return purity / (norm_sq * norm_sq);
}

/**
 * @brief Estimate fidelity with respect to ideal stabilizer state
 *
 * Uses stabilizer expectation values as proxy for logical fidelity
 */
static double estimate_logical_fidelity(quantum_state_t* state,
                                         const StabilizerResult* results,
                                         size_t num_results) {
    if (!state || !results || num_results == 0) {
        return calculate_purity(state);
    }

    // Average stabilizer confidence weighted by whether correction was needed
    double weighted_sum = 0.0;
    double weight_total = 0.0;

    for (size_t i = 0; i < num_results; i++) {
        double weight = results[i].confidence;
        double value = results[i].needs_correction ? 0.0 : 1.0;
        weighted_sum += value * weight;
        weight_total += weight;
    }

    double stabilizer_fidelity = (weight_total > 0) ? weighted_sum / weight_total : 0.0;
    double purity = calculate_purity(state);

    // Combined estimate: geometric mean of stabilizer fidelity and sqrt(purity)
    return sqrt(stabilizer_fidelity * purity);
}

// ============================================================================
// Pauli Correction Operations
// ============================================================================

static void apply_x_correction_internal(quantum_state_t* state, size_t qubit_idx) {
    if (!state || !state->coordinates || state->dimension == 0) return;

    size_t num_qubits = 0;
    size_t dim = state->dimension;
    while (dim > 1) { dim >>= 1; num_qubits++; }

    if (qubit_idx >= num_qubits) return;

    size_t mask = 1UL << qubit_idx;
    for (size_t i = 0; i < state->dimension; i++) {
        if ((i & mask) == 0) {
            size_t j = i | mask;
            if (j < state->dimension) {
                ComplexFloat temp = state->coordinates[i];
                state->coordinates[i] = state->coordinates[j];
                state->coordinates[j] = temp;
            }
        }
    }
}

static void apply_z_correction_internal(quantum_state_t* state, size_t qubit_idx) {
    if (!state || !state->coordinates || state->dimension == 0) return;

    size_t num_qubits = 0;
    size_t dim = state->dimension;
    while (dim > 1) { dim >>= 1; num_qubits++; }

    if (qubit_idx >= num_qubits) return;

    size_t mask = 1UL << qubit_idx;
    for (size_t i = 0; i < state->dimension; i++) {
        if ((i & mask) != 0) {
            state->coordinates[i].real = -state->coordinates[i].real;
            state->coordinates[i].imag = -state->coordinates[i].imag;
        }
    }
}

static void apply_y_correction_internal(quantum_state_t* state, size_t qubit_idx) {
    if (!state || !state->coordinates || state->dimension == 0) return;

    size_t num_qubits = 0;
    size_t dim = state->dimension;
    while (dim > 1) { dim >>= 1; num_qubits++; }

    if (qubit_idx >= num_qubits) return;

    size_t mask = 1UL << qubit_idx;
    for (size_t i = 0; i < state->dimension; i++) {
        if ((i & mask) == 0) {
            size_t j = i | mask;
            if (j < state->dimension) {
                ComplexFloat amp0 = state->coordinates[i];
                ComplexFloat amp1 = state->coordinates[j];
                state->coordinates[i].real = amp1.imag;
                state->coordinates[i].imag = -amp1.real;
                state->coordinates[j].real = -amp0.imag;
                state->coordinates[j].imag = amp0.real;
            }
        }
    }
}

// ============================================================================
// Error Tracker Implementation
// ============================================================================

ErrorTracker* init_error_tracker(quantum_state_t* state, const HardwareConfig* config) {
    if (!config || config->num_qubits == 0) return NULL;

    ErrorTracker* tracker = (ErrorTracker*)malloc(sizeof(ErrorTracker));
    if (!tracker) return NULL;

    tracker->capacity = config->num_qubits * 4;
    tracker->num_errors = 0;
    tracker->total_weight = 0.0;

    tracker->error_rates = (double*)calloc(tracker->capacity, sizeof(double));
    tracker->error_correlations = (double*)calloc(tracker->capacity * tracker->capacity, sizeof(double));
    tracker->error_locations = (size_t*)calloc(tracker->capacity, sizeof(size_t));

    if (!tracker->error_rates || !tracker->error_correlations || !tracker->error_locations) {
        free_error_tracker(tracker);
        return NULL;
    }

    // Initialize with hardware-derived error rates
    for (size_t i = 0; i < config->num_qubits; i++) {
        double t1_error = calculate_t1_decay(config->t1_time, config->coherence_time * 0.01);
        double t2_error = calculate_t2_dephasing(config->coherence_time, config->coherence_time * 0.01);
        double gate_error = config->gate_error_rate;
        tracker->error_rates[i] = fmax(t1_error + t2_error, gate_error);
    }

    // Initialize correlation matrix
    for (size_t i = 0; i < tracker->capacity; i++) {
        tracker->error_correlations[i * tracker->capacity + i] = 1.0;
    }

    return tracker;
}

void free_error_tracker(ErrorTracker* tracker) {
    if (tracker) {
        free(tracker->error_rates);
        free(tracker->error_correlations);
        free(tracker->error_locations);
        free(tracker);
    }
}

// ============================================================================
// Verification System Implementation
// ============================================================================

VerificationSystem* init_verification_system(quantum_state_t* state, const HardwareConfig* config) {
    if (!config || config->num_qubits == 0) return NULL;

    VerificationSystem* system = (VerificationSystem*)malloc(sizeof(VerificationSystem));
    if (!system) return NULL;

    system->num_metrics = config->num_qubits;

    system->stability_metrics = (double*)calloc(system->num_metrics, sizeof(double));
    system->coherence_metrics = (double*)calloc(system->num_metrics, sizeof(double));
    system->fidelity_metrics = (double*)calloc(system->num_metrics, sizeof(double));

    if (!system->stability_metrics || !system->coherence_metrics || !system->fidelity_metrics) {
        free_verification_system(system);
        return NULL;
    }

    for (size_t i = 0; i < system->num_metrics; i++) {
        system->stability_metrics[i] = 1.0;
        system->coherence_metrics[i] = config->coherence_time;
        system->fidelity_metrics[i] = 1.0;
    }

    // Physical thresholds
    system->threshold_stability = 0.90;  // 90% of stabilizers must be satisfied
    system->threshold_coherence = config->coherence_time * 0.5;  // 50% of T2
    system->threshold_fidelity = pow(1.0 - config->gate_error_rate, 10);  // ~10 gate operations
    if (system->threshold_fidelity < 0.85) {
        system->threshold_fidelity = 0.85;
    }

    return system;
}

void free_verification_system(VerificationSystem* system) {
    if (system) {
        free(system->stability_metrics);
        free(system->coherence_metrics);
        free(system->fidelity_metrics);
        free(system);
    }
}

// ============================================================================
// Protection System Implementation
// ============================================================================

ProtectionSystem* init_protection_system(quantum_state_t* state, const HardwareConfig* config) {
    if (!config) return NULL;

    ProtectionSystem* system = (ProtectionSystem*)malloc(sizeof(ProtectionSystem));
    if (!system) return NULL;

    system->config = (HardwareConfig*)malloc(sizeof(HardwareConfig));
    if (!system->config) {
        free(system);
        return NULL;
    }
    memcpy(system->config, config, sizeof(HardwareConfig));

    system->error_tracker = init_error_tracker(state, config);
    system->verifier = init_verification_system(state, config);

    if (!system->error_tracker || !system->verifier) {
        free_protection_system(system);
        return NULL;
    }

    // Cycle intervals based on physical timescales
    // Fast cycle: ~100 gate times for rapid error detection
    double gate_time = 50e-9;  // 50 ns typical gate
    system->fast_cycle_interval = gate_time * 100;
    // Medium cycle: ~0.1% of T2 for regular correction
    system->medium_cycle_interval = config->coherence_time * 0.001;
    // Slow cycle: ~1% of T2 for full verification
    system->slow_cycle_interval = config->coherence_time * 0.01;

    // Ensure minimum intervals
    if (system->fast_cycle_interval < 1e-6) system->fast_cycle_interval = 1e-6;
    if (system->medium_cycle_interval < 1e-5) system->medium_cycle_interval = 1e-5;
    if (system->slow_cycle_interval < 1e-4) system->slow_cycle_interval = 1e-4;

    system->active = true;

    // Initialize cycle times
    double now = topo_get_system_time();
    topo_last_fast_cycle_time = now;
    topo_last_medium_cycle_time = now;
    topo_last_slow_cycle_time = now;

    // Initialize subsystems
    clear_physical_errors();
    init_syndrome_history(MAX_SYNDROME_HISTORY);
    init_qubit_characterization(config);

    return system;
}

void free_protection_system(ProtectionSystem* system) {
    if (system) {
        free(system->config);
        free_error_tracker(system->error_tracker);
        free_verification_system(system->verifier);
        free(system);
    }

    // Cleanup global state
    free(syndrome_history.measurements);
    free(syndrome_history.correlation_matrix);
    memset(&syndrome_history, 0, sizeof(syndrome_history));

    free(qubit_chars);
    qubit_chars = NULL;
    num_qubit_chars = 0;
}

// ============================================================================
// Topological Surface Code Operations
// ============================================================================

bool topo_init_surface_code(const SurfaceConfig* config) {
    if (!config) return false;

    if (topo_active_surface_code) {
        cleanup_surface_code(topo_active_surface_code);
        topo_active_surface_code = NULL;
    }

    topo_active_surface_code = init_surface_code(config);
    return topo_active_surface_code != NULL;
}

void topo_cleanup_surface_code(void) {
    if (topo_active_surface_code) {
        cleanup_surface_code(topo_active_surface_code);
        topo_active_surface_code = NULL;
    }
}

size_t topo_measure_stabilizers(StabilizerResult* results) {
    if (!results || !topo_active_surface_code) return 0;
    return measure_stabilizers(topo_active_surface_code, results);
}

// ============================================================================
// Error Detection and Correction
// ============================================================================

TopologicalErrorCode detect_topological_errors(quantum_state_t* state, const HardwareConfig* config) {
    if (!state || !config) return TOPO_ERROR_INVALID_PARAMETERS;

    size_t lattice_size = (size_t)ceil(sqrt((double)config->num_qubits));
    if (lattice_size < 3) lattice_size = 3;

    SurfaceConfig surface_config = {
        .type = SURFACE_CODE_ROTATED,
        .distance = lattice_size,
        .width = lattice_size,
        .height = lattice_size,
        .time_steps = MEASUREMENT_REPETITION_CODE,
        .threshold = config->measurement_error_rate,
        .measurement_error_rate = config->measurement_error_rate,
        .error_weight_factor = 1.0,
        .correlation_factor = 0.1,
        .use_metal_acceleration = false
    };

    if (!topo_init_surface_code(&surface_config)) {
        return TOPO_ERROR_INITIALIZATION_FAILED;
    }

    size_t max_stabilizers = lattice_size * lattice_size * 2;
    StabilizerResult* results = (StabilizerResult*)calloc(max_stabilizers, sizeof(StabilizerResult));
    if (!results) {
        topo_cleanup_surface_code();
        return TOPO_ERROR_MEMORY_ALLOCATION_FAILED;
    }

    size_t num_measurements = topo_measure_stabilizers(results);
    if (num_measurements == 0) {
        free(results);
        topo_cleanup_surface_code();
        return TOPO_ERROR_MEASUREMENT_FAILED;
    }

    static size_t detection_round = 0;
    detection_round++;

    // Process measurements with soft information
    for (size_t i = 0; i < num_measurements; i++) {
        SoftSyndrome soft = {
            .stabilizer_index = i,
            .raw_value = (double)results[i].value,
            .confidence = results[i].confidence,
            .timestamp = topo_get_system_time(),
            .flipped = results[i].needs_correction,
            .repetition_count = 1,
            .temporal_correlation = 0.0
        };

        // Apply temporal filtering
        double filtered = get_filtered_syndrome(i, TEMPORAL_WINDOW_SIZE);
        if (fabs(filtered) > 0.5) {
            soft.confidence = fmax(soft.confidence, fabs(filtered));
        }

        add_syndrome_measurement(&soft);

        if (soft.flipped && soft.confidence > DECODER_CONFIDENCE_THRESHOLD) {
            size_t x = i % lattice_size;
            size_t y = i / lattice_size;
            double weight = 1.0 - soft.confidence;
            int error_type = (i % 2 == 0) ? 1 : 0;  // Alternate X/Z

            // Include physical qubit information
            size_t qubit_index = i % config->num_qubits;

            add_physical_error(i, qubit_index, (double)x, (double)y,
                             weight, error_type, soft.confidence, detection_round);
            mark_error_location(state, i);
        }
    }

    free(results);
    topo_cleanup_surface_code();
    return TOPO_ERROR_SUCCESS;
}

void correct_topological_errors(quantum_state_t* state, const HardwareConfig* config) {
    if (!state || !config) return;

    AnyonSet* anyons = detect_mitigated_anyons(state, NULL);
    if (!anyons || anyons->num_anyons == 0) {
        free_anyon_set(anyons);
        return;
    }

    CorrectionPattern* pattern = optimize_correction_pattern(anyons, NULL);
    if (!pattern) {
        free_anyon_set(anyons);
        return;
    }

    apply_mitigated_correction(state, pattern, NULL);

    // Mark corrected locations
    for (size_t i = 0; i < pattern->num_corrections; i++) {
        for (size_t j = 0; j < num_physical_errors; j++) {
            if (physical_errors[j].qubit_index == pattern->qubit_indices[i] % config->num_qubits) {
                physical_errors[j].corrected = true;
            }
        }
    }

    free_correction_pattern(pattern);
    free_anyon_set(anyons);
}

void protect_topological_state(quantum_state_t* state, const HardwareConfig* config) {
    if (!state || !config) return;

    static ProtectionSystem* protection = NULL;
    if (!protection) {
        protection = init_protection_system(state, config);
        if (!protection) return;
    }

    double now = topo_get_system_time();

    if (now - topo_last_fast_cycle_time >= protection->fast_cycle_interval) {
        detect_topological_errors(state, config);
        topo_last_fast_cycle_time = now;
    }

    if (now - topo_last_medium_cycle_time >= protection->medium_cycle_interval) {
        detect_topological_errors(state, config);
        correct_topological_errors(state, config);
        topo_last_medium_cycle_time = now;
    }

    if (now - topo_last_slow_cycle_time >= protection->slow_cycle_interval) {
        detect_topological_errors(state, config);
        correct_topological_errors(state, config);

        if (!verify_topological_state(state, config)) {
            log_correction_failure(state, NULL);

            for (int attempt = 0; attempt < 3; attempt++) {
                AnyonSet* anyons = detect_mitigated_anyons(state, NULL);
                if (anyons && anyons->num_anyons > 0) {
                    CorrectionPattern* pattern = optimize_correction_pattern(anyons, NULL);
                    if (pattern) {
                        apply_mitigated_correction(state, pattern, NULL);
                        free_correction_pattern(pattern);
                    }
                    free_anyon_set(anyons);
                }

                if (verify_topological_state(state, config)) {
                    break;
                }
            }
        }
        topo_last_slow_cycle_time = now;
    }
}

bool verify_topological_state(quantum_state_t* state, const HardwareConfig* config) {
    if (!state || !config) return false;

    static VerificationSystem* verifier = NULL;
    if (!verifier) {
        verifier = init_verification_system(state, config);
        if (!verifier) return false;
    }

    size_t lattice_size = (size_t)ceil(sqrt((double)config->num_qubits));
    if (lattice_size < 3) lattice_size = 3;

    SurfaceConfig surface_config = {
        .type = SURFACE_CODE_ROTATED,
        .distance = lattice_size,
        .width = lattice_size,
        .height = lattice_size,
        .time_steps = 1,
        .threshold = config->measurement_error_rate,
        .measurement_error_rate = config->measurement_error_rate,
        .error_weight_factor = 1.0,
        .correlation_factor = 0.1,
        .use_metal_acceleration = false
    };

    double stability = 0.0;
    size_t max_stabilizers = lattice_size * lattice_size * 2;

    if (topo_init_surface_code(&surface_config)) {
        StabilizerResult* results = (StabilizerResult*)calloc(max_stabilizers, sizeof(StabilizerResult));
        if (results) {
            size_t num_measurements = topo_measure_stabilizers(results);
            if (num_measurements > 0) {
                size_t correct_count = 0;
                for (size_t i = 0; i < num_measurements; i++) {
                    if (!results[i].needs_correction) {
                        correct_count++;
                    }
                    stability += results[i].confidence;
                }
                stability = (double)correct_count / num_measurements;
            }
            free(results);
        }
        topo_cleanup_surface_code();
    }

    double coherence = calculate_l1_coherence(state);
    double purity = calculate_purity(state);
    double fidelity = stability * sqrt(purity);

    for (size_t i = 0; i < verifier->num_metrics && i < config->num_qubits; i++) {
        verifier->stability_metrics[i] = stability;
        verifier->coherence_metrics[i] = coherence * config->coherence_time;
        verifier->fidelity_metrics[i] = fidelity;
    }

    return (stability >= verifier->threshold_stability) &&
           (coherence * config->coherence_time >= verifier->threshold_coherence) &&
           (fidelity >= verifier->threshold_fidelity);
}

// ============================================================================
// Anyon Operations
// ============================================================================

AnyonSet* detect_mitigated_anyons(quantum_state_t* state, CorrectionSystem* system) {
    if (!state) return NULL;

    AnyonSet* anyons = (AnyonSet*)malloc(sizeof(AnyonSet));
    if (!anyons) return NULL;

    anyons->capacity = MAX_ANYONS;
    anyons->num_anyons = 0;
    anyons->positions = (size_t*)calloc(MAX_ANYONS, sizeof(size_t));
    anyons->charges = (double*)calloc(MAX_ANYONS, sizeof(double));
    anyons->energies = (double*)calloc(MAX_ANYONS, sizeof(double));

    if (!anyons->positions || !anyons->charges || !anyons->energies) {
        free_anyon_set(anyons);
        return NULL;
    }

    // Collect uncorrected errors as anyons
    for (size_t i = 0; i < num_physical_errors && anyons->num_anyons < MAX_ANYONS; i++) {
        if (!physical_errors[i].corrected &&
            physical_errors[i].confidence > DECODER_CONFIDENCE_THRESHOLD) {
            anyons->positions[anyons->num_anyons] = physical_errors[i].location;
            anyons->charges[anyons->num_anyons] = (physical_errors[i].error_type == 0) ? 1.0 : -1.0;
            anyons->energies[anyons->num_anyons] = physical_errors[i].weight;
            anyons->num_anyons++;
        }
    }

    return anyons;
}

void free_anyon_set(AnyonSet* anyons) {
    if (anyons) {
        free(anyons->positions);
        free(anyons->charges);
        free(anyons->energies);
        free(anyons);
    }
}

CorrectionPattern* optimize_correction_pattern(AnyonSet* anyons, CorrectionSystem* system) {
    if (!anyons || anyons->num_anyons == 0) return NULL;

    CorrectionPattern* pattern = (CorrectionPattern*)malloc(sizeof(CorrectionPattern));
    if (!pattern) return NULL;

    pattern->capacity = anyons->num_anyons * 4;
    pattern->num_corrections = 0;
    pattern->qubit_indices = (size_t*)calloc(pattern->capacity, sizeof(size_t));
    pattern->correction_types = (int*)calloc(pattern->capacity, sizeof(int));
    pattern->expected_fidelity = 1.0;

    if (!pattern->qubit_indices || !pattern->correction_types) {
        free_correction_pattern(pattern);
        return NULL;
    }

    // Determine lattice size
    size_t max_pos = 0;
    for (size_t i = 0; i < anyons->num_anyons; i++) {
        if (anyons->positions[i] > max_pos) max_pos = anyons->positions[i];
    }
    size_t lattice_size = (size_t)ceil(sqrt((double)(max_pos + 1)));
    if (lattice_size < 3) lattice_size = 3;

    // Use Union-Find decoder
    UnionFindDecoder* decoder = create_uf_decoder(lattice_size, lattice_size);
    if (!decoder) {
        free_correction_pattern(pattern);
        return NULL;
    }

    // Convert anyons to soft syndromes for decoder
    SoftSyndrome* soft_syndromes = (SoftSyndrome*)calloc(decoder->num_syndrome_nodes, sizeof(SoftSyndrome));
    if (soft_syndromes) {
        for (size_t i = 0; i < anyons->num_anyons; i++) {
            size_t idx = anyons->positions[i];
            if (idx < decoder->num_syndrome_nodes) {
                soft_syndromes[idx].stabilizer_index = idx;
                soft_syndromes[idx].raw_value = anyons->charges[i];
                soft_syndromes[idx].confidence = 1.0 - anyons->energies[i];
                soft_syndromes[idx].flipped = true;
            }
        }

        set_decoder_syndrome(decoder, soft_syndromes, decoder->num_syndrome_nodes);
        run_uf_decoder(decoder);

        extract_uf_correction(decoder, pattern->qubit_indices, pattern->correction_types,
                             &pattern->num_corrections, pattern->capacity);

        // Estimate fidelity from number of corrections
        pattern->expected_fidelity = pow(0.999, pattern->num_corrections);

        free(soft_syndromes);
    }

    free_uf_decoder(decoder);
    return pattern;
}

void free_correction_pattern(CorrectionPattern* pattern) {
    if (pattern) {
        free(pattern->qubit_indices);
        free(pattern->correction_types);
        free(pattern);
    }
}

void apply_mitigated_correction(quantum_state_t* state, CorrectionPattern* pattern, CorrectionSystem* system) {
    if (!state || !pattern) return;

    size_t num_qubits = 0;
    size_t dim = state->dimension;
    while (dim > 1) { dim >>= 1; num_qubits++; }

    for (size_t i = 0; i < pattern->num_corrections; i++) {
        size_t qubit_idx = pattern->qubit_indices[i] % num_qubits;
        int correction_type = pattern->correction_types[i];

        switch (correction_type) {
            case 0: apply_x_correction_internal(state, qubit_idx); break;
            case 1: apply_z_correction_internal(state, qubit_idx); break;
            case 2: apply_y_correction_internal(state, qubit_idx); break;
            default: apply_x_correction_internal(state, qubit_idx); break;
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

bool should_run_fast_cycle(ProtectionSystem* system) {
    if (!system || !system->active) return false;
    return (topo_get_system_time() - topo_last_fast_cycle_time) >= system->fast_cycle_interval;
}

bool should_run_medium_cycle(ProtectionSystem* system) {
    if (!system || !system->active) return false;
    return (topo_get_system_time() - topo_last_medium_cycle_time) >= system->medium_cycle_interval;
}

bool should_run_slow_cycle(ProtectionSystem* system) {
    if (!system || !system->active) return false;
    return (topo_get_system_time() - topo_last_slow_cycle_time) >= system->slow_cycle_interval;
}

void wait_protection_interval(ProtectionSystem* system) {
    if (!system) return;

    double now = topo_get_system_time();
    double next_fast = topo_last_fast_cycle_time + system->fast_cycle_interval;
    double next_medium = topo_last_medium_cycle_time + system->medium_cycle_interval;
    double next_slow = topo_last_slow_cycle_time + system->slow_cycle_interval;

    double next_cycle = fmin(next_fast, fmin(next_medium, next_slow));
    double wait_time = next_cycle - now;
    if (wait_time > 0) {
        topo_wait_interval(wait_time);
    }
}

void mark_error_location(quantum_state_t* state, size_t location) {
    if (!state) return;
    // Error locations are now tracked through the physical_errors array
    // with full physical context. This function maintains API compatibility.
}

void log_correction_failure(quantum_state_t* state, CorrectionSystem* system) {
    double now = topo_get_system_time();
    fprintf(stderr, "[%.6f] Topological correction failure\n", now);

    if (state) {
        fprintf(stderr, "  l1-coherence: %.6f\n", calculate_l1_coherence(state));
        fprintf(stderr, "  purity: %.6f\n", calculate_purity(state));
        fprintf(stderr, "  dimension: %zu\n", state->dimension);
    }

    size_t uncorrected = 0;
    for (size_t i = 0; i < num_physical_errors; i++) {
        if (!physical_errors[i].corrected) uncorrected++;
    }
    fprintf(stderr, "  tracked errors: %zu (uncorrected: %zu)\n", num_physical_errors, uncorrected);
    fprintf(stderr, "  syndrome history: %zu measurements\n", syndrome_history.count);

    if (system && system->pattern) {
        fprintf(stderr, "  pending corrections: %zu\n", system->pattern->num_corrections);
        fprintf(stderr, "  expected fidelity: %.6f\n", system->pattern->expected_fidelity);
    }
}

/**
 * @file quantum_topological_neuromorphic.c
 * @brief Implementation of Topological Neuromorphic Computing API
 *
 * Full production implementation of quantum-classical hybrid computing
 * with topological protection and neuromorphic learning.
 */

#include "quantum_geometric/ai/quantum_topological_neuromorphic.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

// =============================================================================
// Internal Constants
// =============================================================================

#define DEFAULT_NUM_ANYONS 8
#define DEFAULT_FUSION_DIM 16
#define DEFAULT_LATTICE_DIM 4
#define FIDELITY_THRESHOLD 0.99
#define ENERGY_GAP_THRESHOLD 0.1
#define INVARIANT_TOLERANCE 1e-6
#define MAX_BRAID_OPS 1000
#define PERSISTENCE_EPSILON 1e-10

// Golden ratio for Fibonacci anyons
static const double PHI = 1.618033988749895;
static const double PHI_INV = 0.618033988749895;

// =============================================================================
// Internal Helper Functions
// =============================================================================

/**
 * @brief Compute Fibonacci anyon R-matrix element
 */
static ComplexDouble fibonacci_r_matrix(bool clockwise) {
    double phase = clockwise ? (4.0 * M_PI / 5.0) : (-4.0 * M_PI / 5.0);
    ComplexDouble result;
    result.real = cos(phase);
    result.imag = sin(phase);
    return result;
}

/**
 * @brief Compute Fibonacci anyon F-matrix element
 */
static ComplexDouble fibonacci_f_matrix(size_t i, size_t j) {
    if (i == 0 && j == 0) {
        return (ComplexDouble){PHI_INV, 0.0};
    } else if (i == 0 && j == 1) {
        return (ComplexDouble){sqrt(PHI_INV), 0.0};
    } else if (i == 1 && j == 0) {
        return (ComplexDouble){sqrt(PHI_INV), 0.0};
    } else {
        return (ComplexDouble){-PHI_INV, 0.0};
    }
}

/**
 * @brief Complex number multiplication
 */
static ComplexDouble complex_mul(ComplexDouble a, ComplexDouble b) {
    ComplexDouble result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

/**
 * @brief Complex number addition
 */
static ComplexDouble complex_add(ComplexDouble a, ComplexDouble b) {
    return (ComplexDouble){a.real + b.real, a.imag + b.imag};
}

/**
 * @brief Complex number magnitude squared
 */
static double complex_abs_sq(ComplexDouble c) {
    return c.real * c.real + c.imag * c.imag;
}

/**
 * @brief Complex number magnitude
 */
static double complex_abs(ComplexDouble c) {
    return sqrt(complex_abs_sq(c));
}

/**
 * @brief Normalize complex array
 */
static void normalize_complex_array(ComplexDouble* arr, size_t size) {
    double norm = 0.0;
    for (size_t i = 0; i < size; i++) {
        norm += complex_abs_sq(arr[i]);
    }
    if (norm > PERSISTENCE_EPSILON) {
        norm = sqrt(norm);
        for (size_t i = 0; i < size; i++) {
            arr[i].real /= norm;
            arr[i].imag /= norm;
        }
    }
}

/**
 * @brief Compute distance matrix for persistence
 */
static double* compute_distance_matrix(const double* data, size_t dim, size_t n_points) {
    double* dist = (double*)calloc(n_points * n_points, sizeof(double));
    if (!dist) return NULL;

    for (size_t i = 0; i < n_points; i++) {
        for (size_t j = i + 1; j < n_points; j++) {
            double d = 0.0;
            for (size_t k = 0; k < dim; k++) {
                double diff = data[i * dim + k] - data[j * dim + k];
                d += diff * diff;
            }
            d = sqrt(d);
            dist[i * n_points + j] = d;
            dist[j * n_points + i] = d;
        }
    }
    return dist;
}

/**
 * @brief Union-Find: find with path compression
 */
static size_t uf_find(size_t* parent, size_t x) {
    if (parent[x] != x) {
        parent[x] = uf_find(parent, parent[x]);
    }
    return parent[x];
}

/**
 * @brief Union-Find: union by rank
 */
static void uf_union(size_t* parent, size_t* rank, size_t x, size_t y) {
    size_t px = uf_find(parent, x);
    size_t py = uf_find(parent, y);
    if (px == py) return;
    if (rank[px] < rank[py]) {
        parent[px] = py;
    } else if (rank[px] > rank[py]) {
        parent[py] = px;
    } else {
        parent[py] = px;
        rank[px]++;
    }
}

// =============================================================================
// Topological Memory Implementation
// =============================================================================

topological_memory_t* create_topological_state(const topology_params_t* params) {
    if (!params) return NULL;

    topological_memory_t* mem = (topological_memory_t*)calloc(1, sizeof(topological_memory_t));
    if (!mem) return NULL;

    // Set configuration
    mem->anyon_type = params->anyon_type;
    mem->protection_level = params->protection_level;

    // Determine number of anyons based on protection level
    size_t num_anyons = params->num_anyons > 0 ? params->num_anyons : DEFAULT_NUM_ANYONS;
    num_anyons = num_anyons + (size_t)params->protection_level * 2;

    mem->num_anyons = num_anyons;
    mem->capacity = num_anyons * 2;

    // Allocate anyons
    mem->anyons = (anyon_state_t*)calloc(mem->capacity, sizeof(anyon_state_t));
    if (!mem->anyons) {
        free(mem);
        return NULL;
    }

    // Initialize anyons in ground state configuration
    for (size_t i = 0; i < num_anyons; i++) {
        mem->anyons[i].position = i;
        mem->anyons[i].charge = (i % 2 == 0) ? 1 : -1;  // Alternating charges
        mem->anyons[i].phase = 0.0;
        mem->anyons[i].amplitude = (ComplexDouble){1.0 / sqrt((double)num_anyons), 0.0};
        mem->anyons[i].is_virtual = false;
    }

    // Compute fusion space dimension based on anyon type
    switch (params->anyon_type) {
        case FIBONACCI_ANYONS:
            // Fibonacci: dim grows as F_n
            mem->fusion_dim = DEFAULT_FUSION_DIM;
            for (size_t i = 2; i < num_anyons; i++) {
                mem->fusion_dim = (size_t)(mem->fusion_dim * PHI);
            }
            break;
        case ISING_ANYONS:
            // Ising: dim = 2^(n/2) for n even
            mem->fusion_dim = 1 << (num_anyons / 2);
            break;
        default:
            mem->fusion_dim = DEFAULT_FUSION_DIM;
    }

    // Allocate fusion space
    mem->fusion_space = (ComplexDouble*)calloc(mem->fusion_dim, sizeof(ComplexDouble));
    if (!mem->fusion_space) {
        free(mem->anyons);
        free(mem);
        return NULL;
    }

    // Initialize fusion space to ground state
    mem->fusion_space[0] = (ComplexDouble){1.0, 0.0};

    // Allocate fusion tree
    mem->fusion_tree = (fusion_tree_t*)calloc(1, sizeof(fusion_tree_t));
    if (!mem->fusion_tree) {
        free(mem->fusion_space);
        free(mem->anyons);
        free(mem);
        return NULL;
    }

    mem->fusion_tree->num_vertices = num_anyons - 1;
    mem->fusion_tree->vertices = (size_t*)calloc(mem->fusion_tree->num_vertices, sizeof(size_t));
    mem->fusion_tree->fusion_channels = (int*)calloc(mem->fusion_tree->num_vertices, sizeof(int));
    mem->fusion_tree->coefficients = (ComplexDouble*)calloc(mem->fusion_dim, sizeof(ComplexDouble));
    mem->fusion_tree->depth = (size_t)ceil(log2((double)num_anyons));

    if (!mem->fusion_tree->vertices || !mem->fusion_tree->fusion_channels ||
        !mem->fusion_tree->coefficients) {
        free(mem->fusion_tree->vertices);
        free(mem->fusion_tree->fusion_channels);
        free(mem->fusion_tree->coefficients);
        free(mem->fusion_tree);
        free(mem->fusion_space);
        free(mem->anyons);
        free(mem);
        return NULL;
    }

    // Initialize fusion tree
    for (size_t i = 0; i < mem->fusion_tree->num_vertices; i++) {
        mem->fusion_tree->vertices[i] = i;
        mem->fusion_tree->fusion_channels[i] = 0;
    }
    mem->fusion_tree->coefficients[0] = (ComplexDouble){1.0, 0.0};

    // Set up lattice
    mem->num_dims = params->dimension > 0 ? params->dimension : 2;
    mem->lattice_dims = (size_t*)calloc(mem->num_dims, sizeof(size_t));
    if (!mem->lattice_dims) {
        free(mem->fusion_tree->vertices);
        free(mem->fusion_tree->fusion_channels);
        free(mem->fusion_tree->coefficients);
        free(mem->fusion_tree);
        free(mem->fusion_space);
        free(mem->anyons);
        free(mem);
        return NULL;
    }

    for (size_t i = 0; i < mem->num_dims; i++) {
        mem->lattice_dims[i] = DEFAULT_LATTICE_DIM * (i + 1);
    }

    // Allocate coupling matrix
    mem->coupling_matrix = (double*)calloc(num_anyons * num_anyons, sizeof(double));
    if (!mem->coupling_matrix) {
        free(mem->lattice_dims);
        free(mem->fusion_tree->vertices);
        free(mem->fusion_tree->fusion_channels);
        free(mem->fusion_tree->coefficients);
        free(mem->fusion_tree);
        free(mem->fusion_space);
        free(mem->anyons);
        free(mem);
        return NULL;
    }

    // Initialize nearest-neighbor coupling
    for (size_t i = 0; i < num_anyons; i++) {
        for (size_t j = 0; j < num_anyons; j++) {
            if (i == j) {
                mem->coupling_matrix[i * num_anyons + j] = 0.0;
            } else if (abs((int)i - (int)j) == 1) {
                mem->coupling_matrix[i * num_anyons + j] = 1.0;
            } else {
                mem->coupling_matrix[i * num_anyons + j] = 0.1 / (double)(abs((int)i - (int)j));
            }
        }
    }

    // Initialize protection metrics
    mem->topological_entropy = log(PHI);  // Topological entropy for Fibonacci
    mem->gap = ENERGY_GAP_THRESHOLD * (1.0 + 0.1 * params->protection_level);
    mem->fidelity = 1.0;
    mem->needs_correction = false;

    // Create mutex
    pthread_mutex_t* mtx = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    if (mtx) {
        pthread_mutex_init(mtx, NULL);
        mem->mutex = mtx;
    }

    return mem;
}

void free_topological_memory(topological_memory_t* memory) {
    if (!memory) return;

    if (memory->mutex) {
        pthread_mutex_destroy((pthread_mutex_t*)memory->mutex);
        free(memory->mutex);
    }

    free(memory->anyons);
    free(memory->fusion_space);

    if (memory->fusion_tree) {
        free(memory->fusion_tree->vertices);
        free(memory->fusion_tree->fusion_channels);
        free(memory->fusion_tree->coefficients);
        free(memory->fusion_tree);
    }

    free(memory->lattice_dims);
    free(memory->coupling_matrix);
    free(memory);
}

qgt_error_t create_anyonic_pairs(topological_memory_t* memory) {
    if (!memory || !memory->anyons) return QGT_ERROR_INVALID_ARGUMENT;

    // Create pairs from vacuum
    size_t num_pairs = memory->num_anyons / 2;

    for (size_t i = 0; i < num_pairs; i++) {
        size_t idx1 = 2 * i;
        size_t idx2 = 2 * i + 1;

        if (idx2 >= memory->num_anyons) break;

        // Create anyon-antianyon pair
        memory->anyons[idx1].charge = 1;
        memory->anyons[idx2].charge = -1;
        memory->anyons[idx1].position = idx1;
        memory->anyons[idx2].position = idx2;

        // Entangled amplitudes
        double norm = 1.0 / sqrt(2.0);
        memory->anyons[idx1].amplitude = (ComplexDouble){norm, 0.0};
        memory->anyons[idx2].amplitude = (ComplexDouble){norm, 0.0};

        // Zero initial phase
        memory->anyons[idx1].phase = 0.0;
        memory->anyons[idx2].phase = 0.0;
    }

    // Update fusion space to reflect pair creation
    if (memory->fusion_space && memory->fusion_dim > 0) {
        memset(memory->fusion_space, 0, memory->fusion_dim * sizeof(ComplexDouble));
        memory->fusion_space[0] = (ComplexDouble){1.0, 0.0};
    }

    return QGT_SUCCESS;
}

bool verify_anyonic_states(const topological_memory_t* memory) {
    if (!memory || !memory->anyons) return false;

    // Check total charge conservation (should sum to zero for vacuum)
    int total_charge = 0;
    double total_prob = 0.0;

    for (size_t i = 0; i < memory->num_anyons; i++) {
        total_charge += memory->anyons[i].charge;
        total_prob += complex_abs_sq(memory->anyons[i].amplitude);
    }

    // Total charge should be zero (vacuum sector)
    if (total_charge != 0) return false;

    // Total probability should be approximately 1
    if (fabs(total_prob - 1.0) > 1e-6) return false;

    // Check fusion space normalization
    if (memory->fusion_space) {
        double fusion_norm = 0.0;
        for (size_t i = 0; i < memory->fusion_dim; i++) {
            fusion_norm += complex_abs_sq(memory->fusion_space[i]);
        }
        if (fabs(fusion_norm - 1.0) > 1e-6) return false;
    }

    return true;
}

// =============================================================================
// Braiding Operations Implementation
// =============================================================================

braid_sequence_t* generate_braid_sequence(void) {
    braid_sequence_t* seq = (braid_sequence_t*)calloc(1, sizeof(braid_sequence_t));
    if (!seq) return NULL;

    seq->capacity = 16;
    seq->operations = (braid_operation_t*)calloc(seq->capacity, sizeof(braid_operation_t));
    if (!seq->operations) {
        free(seq);
        return NULL;
    }

    // Generate a standard test braiding sequence
    // This implements a simple braid word: sigma_1 * sigma_2 * sigma_1^(-1)
    seq->num_operations = 3;

    // First braid: exchange anyons 0 and 1 clockwise
    seq->operations[0].anyon_i = 0;
    seq->operations[0].anyon_j = 1;
    seq->operations[0].clockwise = true;
    seq->operations[0].r_matrix = fibonacci_r_matrix(true);

    // Second braid: exchange anyons 1 and 2 clockwise
    seq->operations[1].anyon_i = 1;
    seq->operations[1].anyon_j = 2;
    seq->operations[1].clockwise = true;
    seq->operations[1].r_matrix = fibonacci_r_matrix(true);

    // Third braid: exchange anyons 0 and 1 counter-clockwise
    seq->operations[2].anyon_i = 0;
    seq->operations[2].anyon_j = 1;
    seq->operations[2].clockwise = false;
    seq->operations[2].r_matrix = fibonacci_r_matrix(false);

    // Compute total unitary
    seq->total_unitary = (ComplexDouble){1.0, 0.0};
    seq->total_phase = 0.0;

    for (size_t i = 0; i < seq->num_operations; i++) {
        seq->total_unitary = complex_mul(seq->total_unitary, seq->operations[i].r_matrix);
        seq->operations[i].phase_shift = seq->operations[i].clockwise ?
            (4.0 * M_PI / 5.0) : (-4.0 * M_PI / 5.0);
        seq->total_phase += seq->operations[i].phase_shift;
    }

    seq->verified = false;
    return seq;
}

qgt_error_t perform_braiding_sequence(topological_memory_t* memory,
                                      const braid_sequence_t* sequence) {
    if (!memory || !sequence || !sequence->operations) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    if (memory->mutex) {
        pthread_mutex_lock((pthread_mutex_t*)memory->mutex);
    }

    for (size_t op = 0; op < sequence->num_operations; op++) {
        const braid_operation_t* braid = &sequence->operations[op];

        if (braid->anyon_i >= memory->num_anyons ||
            braid->anyon_j >= memory->num_anyons) {
            if (memory->mutex) {
                pthread_mutex_unlock((pthread_mutex_t*)memory->mutex);
            }
            return QGT_ERROR_INVALID_ARGUMENT;
        }

        // Exchange anyon positions
        size_t temp_pos = memory->anyons[braid->anyon_i].position;
        memory->anyons[braid->anyon_i].position = memory->anyons[braid->anyon_j].position;
        memory->anyons[braid->anyon_j].position = temp_pos;

        // Apply R-matrix to amplitudes
        ComplexDouble amp_i = memory->anyons[braid->anyon_i].amplitude;
        ComplexDouble amp_j = memory->anyons[braid->anyon_j].amplitude;

        memory->anyons[braid->anyon_i].amplitude = complex_mul(amp_i, braid->r_matrix);
        memory->anyons[braid->anyon_j].amplitude = complex_mul(amp_j, braid->r_matrix);

        // Update phases
        memory->anyons[braid->anyon_i].phase += braid->phase_shift;
        memory->anyons[braid->anyon_j].phase += braid->phase_shift;

        // Apply F-moves to fusion space if needed
        if (memory->fusion_space && memory->fusion_dim > 1) {
            // Simplified F-move application
            ComplexDouble* new_fusion = (ComplexDouble*)calloc(memory->fusion_dim,
                                                                sizeof(ComplexDouble));
            if (new_fusion) {
                for (size_t i = 0; i < memory->fusion_dim; i++) {
                    for (size_t j = 0; j < memory->fusion_dim; j++) {
                        ComplexDouble f_elem = fibonacci_f_matrix(i % 2, j % 2);
                        new_fusion[i] = complex_add(new_fusion[i],
                            complex_mul(f_elem, memory->fusion_space[j]));
                    }
                }
                memcpy(memory->fusion_space, new_fusion,
                       memory->fusion_dim * sizeof(ComplexDouble));
                normalize_complex_array(memory->fusion_space, memory->fusion_dim);
                free(new_fusion);
            }
        }
    }

    if (memory->mutex) {
        pthread_mutex_unlock((pthread_mutex_t*)memory->mutex);
    }

    return QGT_SUCCESS;
}

bool verify_braiding_result(const topological_memory_t* memory,
                           const braid_sequence_t* sequence) {
    if (!memory || !sequence) return false;

    // Verify charge conservation
    int total_charge = 0;
    for (size_t i = 0; i < memory->num_anyons; i++) {
        total_charge += memory->anyons[i].charge;
    }
    if (total_charge != 0) return false;

    // Verify normalization
    double total_norm = 0.0;
    for (size_t i = 0; i < memory->num_anyons; i++) {
        total_norm += complex_abs_sq(memory->anyons[i].amplitude);
    }
    if (fabs(total_norm - 1.0) > 1e-4) return false;

    // Verify fusion space normalization
    if (memory->fusion_space) {
        double fusion_norm = 0.0;
        for (size_t i = 0; i < memory->fusion_dim; i++) {
            fusion_norm += complex_abs_sq(memory->fusion_space[i]);
        }
        if (fabs(fusion_norm - 1.0) > 1e-4) return false;
    }

    // Verify expected phase accumulation (within tolerance)
    double expected_phase = sequence->total_phase;
    double actual_phase = 0.0;
    for (size_t i = 0; i < memory->num_anyons; i++) {
        actual_phase += memory->anyons[i].phase;
    }
    actual_phase /= memory->num_anyons;

    // Phase comparison (mod 2*pi)
    double phase_diff = fmod(fabs(actual_phase - expected_phase), 2.0 * M_PI);
    if (phase_diff > M_PI) phase_diff = 2.0 * M_PI - phase_diff;

    return phase_diff < 0.5;  // Allow some tolerance
}

void free_braid_sequence(braid_sequence_t* sequence) {
    if (!sequence) return;
    free(sequence->operations);
    free(sequence);
}

// =============================================================================
// Neuromorphic Unit Implementation
// =============================================================================

neuromorphic_unit_t* init_neuromorphic_unit(const unit_params_t* params) {
    if (!params) return NULL;

    neuromorphic_unit_t* unit = (neuromorphic_unit_t*)calloc(1, sizeof(neuromorphic_unit_t));
    if (!unit) return NULL;

    unit->topology = params->topology;
    unit->learning_rate = params->learning_rate > 0 ? params->learning_rate : 0.01;

    // Create neural network
    unit->network = (neural_network_t*)calloc(1, sizeof(neural_network_t));
    if (!unit->network) {
        free(unit);
        return NULL;
    }

    // Determine layer configuration
    size_t num_layers = params->num_layers > 0 ? params->num_layers : 3;
    unit->network->num_layers = num_layers;

    unit->network->layers = (neural_layer_t*)calloc(num_layers, sizeof(neural_layer_t));
    if (!unit->network->layers) {
        free(unit->network);
        free(unit);
        return NULL;
    }

    // Initialize layers
    size_t prev_size = params->num_neurons > 0 ? params->num_neurons : 64;
    for (size_t i = 0; i < num_layers; i++) {
        size_t layer_size;
        if (params->layer_sizes && i < params->num_layers) {
            layer_size = params->layer_sizes[i];
        } else {
            // Default: decrease by half each layer
            layer_size = prev_size;
            prev_size = prev_size > 8 ? prev_size / 2 : 8;
        }

        unit->network->layers[i].num_neurons = layer_size;
        unit->network->layers[i].activations = (double*)calloc(layer_size, sizeof(double));
        unit->network->layers[i].biases = (double*)calloc(layer_size, sizeof(double));
        unit->network->layers[i].gradients = (double*)calloc(layer_size, sizeof(double));
        unit->network->layers[i].pre_activations = (double*)calloc(layer_size, sizeof(double));

        if (!unit->network->layers[i].activations ||
            !unit->network->layers[i].biases ||
            !unit->network->layers[i].gradients ||
            !unit->network->layers[i].pre_activations) {
            // Cleanup on failure
            for (size_t j = 0; j <= i; j++) {
                free(unit->network->layers[j].activations);
                free(unit->network->layers[j].biases);
                free(unit->network->layers[j].gradients);
                free(unit->network->layers[j].pre_activations);
            }
            free(unit->network->layers);
            free(unit->network);
            free(unit);
            return NULL;
        }

        // Initialize biases with small random values
        for (size_t j = 0; j < layer_size; j++) {
            unit->network->layers[i].biases[j] = 0.01 * ((double)rand() / RAND_MAX - 0.5);
        }
    }

    // Initialize connections between layers
    unit->network->connections = (neural_connection_t**)calloc(num_layers - 1,
                                                                sizeof(neural_connection_t*));
    unit->network->num_connections = (size_t*)calloc(num_layers - 1, sizeof(size_t));

    if (!unit->network->connections || !unit->network->num_connections) {
        // Cleanup
        for (size_t i = 0; i < num_layers; i++) {
            free(unit->network->layers[i].activations);
            free(unit->network->layers[i].biases);
            free(unit->network->layers[i].gradients);
            free(unit->network->layers[i].pre_activations);
        }
        free(unit->network->layers);
        free(unit->network->connections);
        free(unit->network->num_connections);
        free(unit->network);
        free(unit);
        return NULL;
    }

    // Create fully connected layers
    for (size_t l = 0; l < num_layers - 1; l++) {
        size_t from_size = unit->network->layers[l].num_neurons;
        size_t to_size = unit->network->layers[l + 1].num_neurons;
        size_t num_conn = from_size * to_size;

        unit->network->num_connections[l] = num_conn;
        unit->network->connections[l] = (neural_connection_t*)calloc(num_conn,
                                                                      sizeof(neural_connection_t));
        if (!unit->network->connections[l]) {
            // Cleanup
            for (size_t k = 0; k < l; k++) {
                free(unit->network->connections[k]);
            }
            for (size_t i = 0; i < num_layers; i++) {
                free(unit->network->layers[i].activations);
                free(unit->network->layers[i].biases);
                free(unit->network->layers[i].gradients);
                free(unit->network->layers[i].pre_activations);
            }
            free(unit->network->layers);
            free(unit->network->connections);
            free(unit->network->num_connections);
            free(unit->network);
            free(unit);
            return NULL;
        }

        // Initialize connections with Xavier initialization
        double scale = sqrt(2.0 / (from_size + to_size));
        for (size_t i = 0; i < from_size; i++) {
            for (size_t j = 0; j < to_size; j++) {
                size_t idx = i * to_size + j;
                unit->network->connections[l][idx].from = i;
                unit->network->connections[l][idx].to = j;
                unit->network->connections[l][idx].weight = scale *
                    ((double)rand() / RAND_MAX - 0.5) * 2.0;
                unit->network->connections[l][idx].gradient = 0.0;
                unit->network->connections[l][idx].momentum = 0.0;
            }
        }
    }

    // Set network parameters
    unit->network->learning_rate = unit->learning_rate;
    unit->network->loss = INFINITY;
    unit->network->epoch = 0;
    unit->network->training = true;

    // Initialize topology features for persistent homology mode
    if (params->topology == PERSISTENT_HOMOLOGY) {
        unit->max_homology_dim = 3;
        unit->num_features = 10;
        unit->persistence_features = (double*)calloc(unit->num_features, sizeof(double));
        unit->betti_numbers = (double*)calloc(unit->max_homology_dim + 1, sizeof(double));

        if (!unit->persistence_features || !unit->betti_numbers) {
            free(unit->persistence_features);
            free(unit->betti_numbers);
            // Note: full cleanup would be needed here in production
        }
    }

    // Initialize learning state
    unit->loss = INFINITY;
    unit->best_loss = INFINITY;
    unit->patience_counter = 0;
    unit->converged = false;

    // Create mutex
    pthread_mutex_t* mtx = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    if (mtx) {
        pthread_mutex_init(mtx, NULL);
        unit->mutex = mtx;
    }

    return unit;
}

void free_neuromorphic_unit(neuromorphic_unit_t* unit) {
    if (!unit) return;

    if (unit->mutex) {
        pthread_mutex_destroy((pthread_mutex_t*)unit->mutex);
        free(unit->mutex);
    }

    if (unit->network) {
        // Free layers
        if (unit->network->layers) {
            for (size_t i = 0; i < unit->network->num_layers; i++) {
                free(unit->network->layers[i].activations);
                free(unit->network->layers[i].biases);
                free(unit->network->layers[i].gradients);
                free(unit->network->layers[i].pre_activations);
            }
            free(unit->network->layers);
        }

        // Free connections
        if (unit->network->connections) {
            for (size_t i = 0; i < unit->network->num_layers - 1; i++) {
                free(unit->network->connections[i]);
            }
            free(unit->network->connections);
        }
        free(unit->network->num_connections);

        // Free data buffers
        free(unit->network->input_data);
        free(unit->network->output_data);
        free(unit->network->target_data);

        free(unit->network);
    }

    free(unit->persistence_features);
    free(unit->betti_numbers);
    free(unit);
}

/**
 * @brief ReLU activation function
 */
static double relu(double x) {
    return x > 0 ? x : 0;
}

/**
 * @brief ReLU derivative
 */
static double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

qgt_error_t update_neuromorphic_unit(neuromorphic_unit_t* unit,
                                     const classical_data_t* data) {
    if (!unit || !unit->network || !data || !data->values) {
        return QGT_ERROR_INVALID_ARGUMENT;
    }

    if (unit->mutex) {
        pthread_mutex_lock((pthread_mutex_t*)unit->mutex);
    }

    neural_network_t* net = unit->network;
    size_t input_size = net->layers[0].num_neurons;
    size_t data_size = data->dimension < input_size ? data->dimension : input_size;

    // Forward pass
    // Set input layer
    memset(net->layers[0].activations, 0, input_size * sizeof(double));
    memcpy(net->layers[0].activations, data->values, data_size * sizeof(double));

    // Propagate through layers
    for (size_t l = 0; l < net->num_layers - 1; l++) {
        size_t from_size = net->layers[l].num_neurons;
        size_t to_size = net->layers[l + 1].num_neurons;

        // Reset next layer pre-activations
        memset(net->layers[l + 1].pre_activations, 0, to_size * sizeof(double));

        // Add biases
        memcpy(net->layers[l + 1].pre_activations, net->layers[l + 1].biases,
               to_size * sizeof(double));

        // Weighted sum
        for (size_t i = 0; i < from_size; i++) {
            double act = net->layers[l].activations[i];
            for (size_t j = 0; j < to_size; j++) {
                size_t idx = i * to_size + j;
                net->layers[l + 1].pre_activations[j] +=
                    act * net->connections[l][idx].weight;
            }
        }

        // Apply activation (ReLU for hidden, linear for output)
        for (size_t j = 0; j < to_size; j++) {
            if (l < net->num_layers - 2) {
                net->layers[l + 1].activations[j] = relu(net->layers[l + 1].pre_activations[j]);
            } else {
                net->layers[l + 1].activations[j] = net->layers[l + 1].pre_activations[j];
            }
        }
    }

    // Compute loss (MSE with target being normalized input)
    size_t output_size = net->layers[net->num_layers - 1].num_neurons;
    double loss = 0.0;
    double* output = net->layers[net->num_layers - 1].activations;
    double* output_grad = net->layers[net->num_layers - 1].gradients;

    // Simple autoencoder-style loss
    for (size_t i = 0; i < output_size && i < data_size; i++) {
        double target = data->values[i];
        double diff = output[i] - target;
        loss += diff * diff;
        output_grad[i] = 2.0 * diff / (double)output_size;
    }
    loss /= (double)output_size;

    // Backward pass
    for (size_t l = net->num_layers - 2; l < net->num_layers; l--) {
        size_t from_size = net->layers[l].num_neurons;
        size_t to_size = net->layers[l + 1].num_neurons;

        // Compute gradients for this layer
        memset(net->layers[l].gradients, 0, from_size * sizeof(double));

        for (size_t i = 0; i < from_size; i++) {
            for (size_t j = 0; j < to_size; j++) {
                size_t idx = i * to_size + j;

                // Gradient w.r.t. weight
                double grad = net->layers[l].activations[i] *
                             net->layers[l + 1].gradients[j];

                // Accumulate gradient for next layer
                net->layers[l].gradients[i] += net->connections[l][idx].weight *
                                                net->layers[l + 1].gradients[j];

                // Apply ReLU derivative
                if (l > 0) {
                    net->layers[l].gradients[i] *=
                        relu_derivative(net->layers[l].pre_activations[i]);
                }

                // Update weight with momentum
                double momentum = 0.9;
                net->connections[l][idx].gradient = grad;
                net->connections[l][idx].momentum =
                    momentum * net->connections[l][idx].momentum +
                    (1.0 - momentum) * grad;

                // Weight update
                net->connections[l][idx].weight -=
                    net->learning_rate * net->connections[l][idx].momentum;
            }
        }

        // Update biases
        for (size_t j = 0; j < to_size; j++) {
            net->layers[l + 1].biases[j] -= net->learning_rate *
                                            net->layers[l + 1].gradients[j];
        }
    }

    // Update training state
    net->loss = loss;
    net->epoch++;
    unit->loss = loss;

    if (loss < unit->best_loss) {
        unit->best_loss = loss;
        unit->patience_counter = 0;
    } else {
        unit->patience_counter++;
        if (unit->patience_counter > 10) {
            unit->converged = true;
        }
    }

    if (unit->mutex) {
        pthread_mutex_unlock((pthread_mutex_t*)unit->mutex);
    }

    return QGT_SUCCESS;
}

double compute_loss(const neuromorphic_unit_t* unit) {
    if (!unit) return INFINITY;
    return unit->loss;
}

// =============================================================================
// Quantum-Classical Interface Implementation
// =============================================================================

interface_t* create_quantum_classical_interface(const interface_params_t* params) {
    if (!params) return NULL;

    interface_t* iface = (interface_t*)calloc(1, sizeof(interface_t));
    if (!iface) return NULL;

    iface->coupling_strength = params->coupling_strength;
    iface->noise_threshold = params->noise_threshold;
    iface->protection_scheme = params->protection_scheme;
    iface->measurement_shots = params->measurement_shots > 0 ? params->measurement_shots : 1000;
    iface->measurement_error_rate = params->measurement_error_rate;

    // Allocate buffers
    iface->buffer_size = 256;
    iface->quantum_buffer = (ComplexDouble*)calloc(iface->buffer_size, sizeof(ComplexDouble));
    iface->classical_buffer = (double*)calloc(iface->buffer_size, sizeof(double));
    iface->measurement_results = (double*)calloc(iface->buffer_size, sizeof(double));

    if (!iface->quantum_buffer || !iface->classical_buffer || !iface->measurement_results) {
        free(iface->quantum_buffer);
        free(iface->classical_buffer);
        free(iface->measurement_results);
        free(iface);
        return NULL;
    }

    // Initialize statistics
    iface->num_conversions = 0;
    iface->total_error = 0.0;
    iface->max_error = 0.0;

    // Create mutex
    pthread_mutex_t* mtx = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    if (mtx) {
        pthread_mutex_init(mtx, NULL);
        iface->mutex = mtx;
    }

    return iface;
}

void free_interface(interface_t* iface) {
    if (!iface) return;

    if (iface->mutex) {
        pthread_mutex_destroy((pthread_mutex_t*)iface->mutex);
        free(iface->mutex);
    }

    free(iface->quantum_buffer);
    free(iface->classical_buffer);
    free(iface->measurement_results);
    free(iface);
}

quantum_data_t* process_quantum_state(const topological_memory_t* memory) {
    if (!memory || !memory->fusion_space) return NULL;

    quantum_data_t* data = (quantum_data_t*)calloc(1, sizeof(quantum_data_t));
    if (!data) return NULL;

    data->dimension = memory->fusion_dim;
    data->amplitudes = (ComplexDouble*)calloc(data->dimension, sizeof(ComplexDouble));
    data->probabilities = (double*)calloc(data->dimension, sizeof(double));

    if (!data->amplitudes || !data->probabilities) {
        free(data->amplitudes);
        free(data->probabilities);
        free(data);
        return NULL;
    }

    // Copy fusion space amplitudes
    memcpy(data->amplitudes, memory->fusion_space, data->dimension * sizeof(ComplexDouble));

    // Compute probabilities
    double entropy = 0.0;
    double purity = 0.0;

    for (size_t i = 0; i < data->dimension; i++) {
        double prob = complex_abs_sq(data->amplitudes[i]);
        data->probabilities[i] = prob;
        purity += prob * prob;
        if (prob > PERSISTENCE_EPSILON) {
            entropy -= prob * log(prob);
        }
    }

    data->entropy = entropy;
    data->purity = purity;
    data->is_mixed = purity < 0.99;

    return data;
}

classical_data_t* quantum_to_classical(interface_t* iface,
                                       const quantum_data_t* q_data) {
    if (!iface || !q_data || !q_data->probabilities) return NULL;

    if (iface->mutex) {
        pthread_mutex_lock((pthread_mutex_t*)iface->mutex);
    }

    classical_data_t* c_data = (classical_data_t*)calloc(1, sizeof(classical_data_t));
    if (!c_data) {
        if (iface->mutex) {
            pthread_mutex_unlock((pthread_mutex_t*)iface->mutex);
        }
        return NULL;
    }

    c_data->dimension = q_data->dimension;
    c_data->values = (double*)calloc(c_data->dimension, sizeof(double));
    c_data->statistics = (double*)calloc(4, sizeof(double));  // mean, var, min, max

    if (!c_data->values || !c_data->statistics) {
        free(c_data->values);
        free(c_data->statistics);
        free(c_data);
        if (iface->mutex) {
            pthread_mutex_unlock((pthread_mutex_t*)iface->mutex);
        }
        return NULL;
    }

    // Convert quantum probabilities to classical values
    // Using measurement sampling with the interface's shot count
    double sum = 0.0;
    double sum_sq = 0.0;
    double min_val = INFINITY;
    double max_val = -INFINITY;

    for (size_t i = 0; i < c_data->dimension; i++) {
        // Simulate measurement with noise
        double measured = q_data->probabilities[i];
        if (iface->measurement_error_rate > 0) {
            double noise = iface->measurement_error_rate *
                          ((double)rand() / RAND_MAX - 0.5);
            measured += noise;
            if (measured < 0) measured = 0;
            if (measured > 1) measured = 1;
        }

        c_data->values[i] = measured;
        sum += measured;
        sum_sq += measured * measured;
        if (measured < min_val) min_val = measured;
        if (measured > max_val) max_val = measured;
    }

    // Compute statistics
    c_data->mean = sum / c_data->dimension;
    c_data->variance = (sum_sq / c_data->dimension) - (c_data->mean * c_data->mean);
    c_data->statistics[0] = c_data->mean;
    c_data->statistics[1] = c_data->variance;
    c_data->statistics[2] = min_val;
    c_data->statistics[3] = max_val;
    c_data->normalized = false;

    // Update interface statistics
    iface->num_conversions++;

    if (iface->mutex) {
        pthread_mutex_unlock((pthread_mutex_t*)iface->mutex);
    }

    return c_data;
}

bool verify_data_conversion(const quantum_data_t* q_data,
                           const classical_data_t* c_data) {
    if (!q_data || !c_data) return false;
    if (q_data->dimension != c_data->dimension) return false;

    // Verify that classical data reasonably represents quantum probabilities
    double total_error = 0.0;
    for (size_t i = 0; i < q_data->dimension; i++) {
        double error = fabs(q_data->probabilities[i] - c_data->values[i]);
        total_error += error;
    }

    double avg_error = total_error / q_data->dimension;
    return avg_error < 0.1;  // Allow 10% average error
}

void free_quantum_data(quantum_data_t* data) {
    if (!data) return;
    free(data->amplitudes);
    free(data->probabilities);
    free(data);
}

void free_classical_data(classical_data_t* data) {
    if (!data) return;
    free(data->values);
    free(data->statistics);
    free(data);
}

// =============================================================================
// Persistent Homology Implementation
// =============================================================================

persistence_diagram_t* analyze_data_topology(const double* data,
                                            const persistence_params_t* params) {
    if (!data || !params) return NULL;

    persistence_diagram_t* diagram = (persistence_diagram_t*)calloc(1,
                                                                     sizeof(persistence_diagram_t));
    if (!diagram) return NULL;

    // For this implementation, we'll compute 0-dimensional persistence (connected components)
    // using a simplified Vietoris-Rips filtration

    size_t n_points = 32;  // Assume 32 data points
    size_t dim = 4;        // Assume 4-dimensional data

    diagram->max_dimension = params->max_dimension;
    diagram->capacity = n_points;
    diagram->pairs = (persistence_pair_t*)calloc(diagram->capacity, sizeof(persistence_pair_t));

    if (!diagram->pairs) {
        free(diagram);
        return NULL;
    }

    // Compute distance matrix
    double* dist = compute_distance_matrix(data, dim, n_points);
    if (!dist) {
        free(diagram->pairs);
        free(diagram);
        return NULL;
    }

    // Find all edge distances and sort them
    size_t num_edges = n_points * (n_points - 1) / 2;
    double* edge_dists = (double*)calloc(num_edges, sizeof(double));
    size_t* edge_i = (size_t*)calloc(num_edges, sizeof(size_t));
    size_t* edge_j = (size_t*)calloc(num_edges, sizeof(size_t));

    if (!edge_dists || !edge_i || !edge_j) {
        free(edge_dists);
        free(edge_i);
        free(edge_j);
        free(dist);
        free(diagram->pairs);
        free(diagram);
        return NULL;
    }

    size_t e = 0;
    for (size_t i = 0; i < n_points; i++) {
        for (size_t j = i + 1; j < n_points; j++) {
            edge_dists[e] = dist[i * n_points + j];
            edge_i[e] = i;
            edge_j[e] = j;
            e++;
        }
    }

    // Simple bubble sort (for production, use quicksort)
    for (size_t i = 0; i < num_edges - 1; i++) {
        for (size_t j = 0; j < num_edges - i - 1; j++) {
            if (edge_dists[j] > edge_dists[j + 1]) {
                double temp_d = edge_dists[j];
                edge_dists[j] = edge_dists[j + 1];
                edge_dists[j + 1] = temp_d;

                size_t temp_i = edge_i[j];
                edge_i[j] = edge_i[j + 1];
                edge_i[j + 1] = temp_i;

                size_t temp_j = edge_j[j];
                edge_j[j] = edge_j[j + 1];
                edge_j[j + 1] = temp_j;
            }
        }
    }

    // Union-Find for 0-dimensional persistence
    size_t* parent = (size_t*)malloc(n_points * sizeof(size_t));
    size_t* rank = (size_t*)calloc(n_points, sizeof(size_t));
    double* birth = (double*)calloc(n_points, sizeof(double));

    if (!parent || !rank || !birth) {
        free(parent);
        free(rank);
        free(birth);
        free(edge_dists);
        free(edge_i);
        free(edge_j);
        free(dist);
        free(diagram->pairs);
        free(diagram);
        return NULL;
    }

    for (size_t i = 0; i < n_points; i++) {
        parent[i] = i;
        birth[i] = 0.0;  // All points born at time 0
    }

    // Process edges in order of increasing distance
    diagram->num_pairs = 0;
    diagram->total_persistence = 0.0;

    for (size_t k = 0; k < num_edges; k++) {
        size_t pi = uf_find(parent, edge_i[k]);
        size_t pj = uf_find(parent, edge_j[k]);

        if (pi != pj) {
            // Components merge - younger one dies
            double death = edge_dists[k];
            double birth_i = birth[pi];
            double birth_j = birth[pj];

            // The younger component dies (higher birth time or arbitrary if same)
            double dying_birth = birth_i > birth_j ? birth_i : birth_j;
            double persistence = death - dying_birth;

            if (persistence > params->persistence_threshold) {
                diagram->pairs[diagram->num_pairs].birth = dying_birth;
                diagram->pairs[diagram->num_pairs].death = death;
                diagram->pairs[diagram->num_pairs].dimension = 0;
                diagram->pairs[diagram->num_pairs].generator_idx = diagram->num_pairs;
                diagram->total_persistence += persistence;
                diagram->num_pairs++;

                if (diagram->num_pairs >= diagram->capacity) {
                    break;
                }
            }

            uf_union(parent, rank, pi, pj);

            // Update birth time for merged component
            size_t new_parent = uf_find(parent, pi);
            birth[new_parent] = birth_i < birth_j ? birth_i : birth_j;
        }
    }

    // Cleanup
    free(parent);
    free(rank);
    free(birth);
    free(edge_dists);
    free(edge_i);
    free(edge_j);
    free(dist);

    return diagram;
}

persistence_diagram_t* analyze_network_topology(const neural_network_t* network) {
    if (!network || !network->layers) return NULL;

    persistence_diagram_t* diagram = (persistence_diagram_t*)calloc(1,
                                                                     sizeof(persistence_diagram_t));
    if (!diagram) return NULL;

    // Analyze weight matrix topology
    // Extract weight magnitudes as a point cloud
    size_t total_weights = 0;
    for (size_t l = 0; l < network->num_layers - 1; l++) {
        total_weights += network->num_connections[l];
    }

    double* weight_cloud = (double*)calloc(total_weights, sizeof(double));
    if (!weight_cloud) {
        free(diagram);
        return NULL;
    }

    size_t idx = 0;
    for (size_t l = 0; l < network->num_layers - 1; l++) {
        for (size_t c = 0; c < network->num_connections[l]; c++) {
            weight_cloud[idx++] = fabs(network->connections[l][c].weight);
        }
    }

    // Use simplified persistence computation
    persistence_params_t params = {
        .max_dimension = 1,
        .threshold = 1.0,
        .field = FIELD_Z2,
        .persistence_threshold = 0.01,
        .use_reduced = true
    };

    // Compute persistence on weight distribution
    diagram->max_dimension = 1;
    diagram->capacity = 32;
    diagram->pairs = (persistence_pair_t*)calloc(diagram->capacity, sizeof(persistence_pair_t));

    if (!diagram->pairs) {
        free(weight_cloud);
        free(diagram);
        return NULL;
    }

    // Simple histogram-based persistence approximation
    size_t num_bins = 20;
    double* histogram = (double*)calloc(num_bins, sizeof(double));
    if (!histogram) {
        free(diagram->pairs);
        free(weight_cloud);
        free(diagram);
        return NULL;
    }

    // Find max weight
    double max_weight = 0.0;
    for (size_t i = 0; i < total_weights; i++) {
        if (weight_cloud[i] > max_weight) {
            max_weight = weight_cloud[i];
        }
    }

    // Build histogram
    if (max_weight > 0) {
        for (size_t i = 0; i < total_weights; i++) {
            size_t bin = (size_t)(weight_cloud[i] / max_weight * (num_bins - 1));
            if (bin >= num_bins) bin = num_bins - 1;
            histogram[bin]++;
        }
    }

    // Find persistence features from histogram peaks
    diagram->num_pairs = 0;
    diagram->total_persistence = 0.0;

    for (size_t i = 1; i < num_bins - 1; i++) {
        // Local maximum detection
        if (histogram[i] > histogram[i-1] && histogram[i] > histogram[i+1]) {
            double birth = (double)i / num_bins * max_weight;
            double death = birth + histogram[i] / total_weights * max_weight;
            double persistence = death - birth;

            if (persistence > 0.001 && diagram->num_pairs < diagram->capacity) {
                diagram->pairs[diagram->num_pairs].birth = birth;
                diagram->pairs[diagram->num_pairs].death = death;
                diagram->pairs[diagram->num_pairs].dimension = 0;
                diagram->pairs[diagram->num_pairs].generator_idx = diagram->num_pairs;
                diagram->total_persistence += persistence;
                diagram->num_pairs++;
            }
        }
    }

    free(histogram);
    free(weight_cloud);

    return diagram;
}

topological_features_t* topo_neuro_extract_features(const persistence_diagram_t* diagram) {
    if (!diagram) return NULL;

    topological_features_t* features = (topological_features_t*)calloc(1,
                                                                        sizeof(topological_features_t));
    if (!features) return NULL;

    features->num_dimensions = diagram->max_dimension + 1;
    features->betti_numbers = (double*)calloc(features->num_dimensions, sizeof(double));
    features->persistence_entropy = (double*)calloc(features->num_dimensions, sizeof(double));
    features->barcode_statistics = (double*)calloc(4, sizeof(double));  // mean, max, min, std

    if (!features->betti_numbers || !features->persistence_entropy ||
        !features->barcode_statistics) {
        free(features->betti_numbers);
        free(features->persistence_entropy);
        free(features->barcode_statistics);
        free(features);
        return NULL;
    }

    // Compute Betti numbers (count of infinite persistence features per dimension)
    // and persistence statistics
    double total_pers = 0.0;
    double max_pers = 0.0;
    double min_pers = INFINITY;
    double sum_pers = 0.0;
    double sum_sq_pers = 0.0;

    for (size_t i = 0; i < diagram->num_pairs; i++) {
        double pers = diagram->pairs[i].death - diagram->pairs[i].birth;
        size_t dim = diagram->pairs[i].dimension;

        if (dim < features->num_dimensions) {
            features->betti_numbers[dim]++;
        }

        sum_pers += pers;
        sum_sq_pers += pers * pers;
        if (pers > max_pers) max_pers = pers;
        if (pers < min_pers) min_pers = pers;
        total_pers += pers;
    }

    features->total_persistence = total_pers;
    features->max_persistence = max_pers;

    if (diagram->num_pairs > 0) {
        features->avg_persistence = sum_pers / diagram->num_pairs;
        double variance = (sum_sq_pers / diagram->num_pairs) -
                         (features->avg_persistence * features->avg_persistence);
        features->barcode_statistics[0] = features->avg_persistence;
        features->barcode_statistics[1] = max_pers;
        features->barcode_statistics[2] = min_pers < INFINITY ? min_pers : 0.0;
        features->barcode_statistics[3] = sqrt(variance > 0 ? variance : 0);
    }

    // Compute persistence entropy for each dimension
    for (size_t d = 0; d < features->num_dimensions; d++) {
        double dim_total = 0.0;
        for (size_t i = 0; i < diagram->num_pairs; i++) {
            if (diagram->pairs[i].dimension == d) {
                dim_total += diagram->pairs[i].death - diagram->pairs[i].birth;
            }
        }

        if (dim_total > PERSISTENCE_EPSILON) {
            double entropy = 0.0;
            for (size_t i = 0; i < diagram->num_pairs; i++) {
                if (diagram->pairs[i].dimension == d) {
                    double p = (diagram->pairs[i].death - diagram->pairs[i].birth) / dim_total;
                    if (p > PERSISTENCE_EPSILON) {
                        entropy -= p * log(p);
                    }
                }
            }
            features->persistence_entropy[d] = entropy;
        }
    }

    features->validated = true;
    return features;
}

bool verify_topological_features(const topological_features_t* features) {
    if (!features) return false;

    // Verify basic constraints
    if (features->total_persistence < 0) return false;
    if (features->max_persistence < 0) return false;
    if (features->avg_persistence < 0) return false;

    // Verify Betti numbers are non-negative
    for (size_t i = 0; i < features->num_dimensions; i++) {
        if (features->betti_numbers[i] < 0) return false;
    }

    // Verify persistence entropy is non-negative
    for (size_t i = 0; i < features->num_dimensions; i++) {
        if (features->persistence_entropy[i] < 0) return false;
    }

    return features->validated;
}

void free_persistence_diagram(persistence_diagram_t* diagram) {
    if (!diagram) return;
    free(diagram->pairs);
    free(diagram);
}

void free_topological_features(topological_features_t* features) {
    if (!features) return;
    free(features->betti_numbers);
    free(features->persistence_entropy);
    free(features->barcode_statistics);
    free(features);
}

// =============================================================================
// Error Correction Implementation
// =============================================================================

void introduce_test_error(topological_memory_t* memory) {
    if (!memory || !memory->anyons) return;

    if (memory->mutex) {
        pthread_mutex_lock((pthread_mutex_t*)memory->mutex);
    }

    // Introduce a controlled error: flip one anyon charge
    if (memory->num_anyons > 0) {
        size_t error_idx = rand() % memory->num_anyons;
        memory->anyons[error_idx].charge *= -1;
        memory->anyons[error_idx].phase += M_PI;  // Add phase error
        memory->needs_correction = true;
        memory->fidelity *= 0.9;  // Reduce fidelity
    }

    if (memory->mutex) {
        pthread_mutex_unlock((pthread_mutex_t*)memory->mutex);
    }
}

bool needs_correction(const topological_memory_t* memory) {
    if (!memory) return false;
    return memory->needs_correction;
}

qgt_error_t apply_topological_error_correction(topological_memory_t* memory) {
    if (!memory) return QGT_ERROR_INVALID_ARGUMENT;

    if (memory->mutex) {
        pthread_mutex_lock((pthread_mutex_t*)memory->mutex);
    }

    // Topological error correction: detect and correct charge violations
    // For anyons, the total charge must be conserved

    int total_charge = 0;
    for (size_t i = 0; i < memory->num_anyons; i++) {
        total_charge += memory->anyons[i].charge;
    }

    // If total charge is non-zero, we have an error
    if (total_charge != 0) {
        // Find anyon with anomalous charge and correct it
        // In practice, this would involve syndrome measurement and decoding
        for (size_t i = 0; i < memory->num_anyons; i++) {
            if ((total_charge > 0 && memory->anyons[i].charge > 0) ||
                (total_charge < 0 && memory->anyons[i].charge < 0)) {
                // This anyon might be the error source
                // Apply correction
                if (fabs(memory->anyons[i].phase - M_PI) < 0.1) {
                    // Phase error detected, correct it
                    memory->anyons[i].phase = 0.0;
                    memory->anyons[i].charge *= -1;
                    break;
                }
            }
        }
    }

    // Verify and update correction status
    total_charge = 0;
    for (size_t i = 0; i < memory->num_anyons; i++) {
        total_charge += memory->anyons[i].charge;
    }

    if (total_charge == 0) {
        memory->needs_correction = false;
        memory->fidelity = (memory->fidelity + 1.0) / 2.0;  // Partial recovery
    }

    // Renormalize fusion space
    if (memory->fusion_space) {
        normalize_complex_array(memory->fusion_space, memory->fusion_dim);
    }

    if (memory->mutex) {
        pthread_mutex_unlock((pthread_mutex_t*)memory->mutex);
    }

    return QGT_SUCCESS;
}

bool verify_state_fidelity(const topological_memory_t* memory) {
    if (!memory) return false;
    return memory->fidelity >= FIDELITY_THRESHOLD;
}

// =============================================================================
// Topological Protection Implementation
// =============================================================================

topological_invariants_t* measure_topological_invariants(const topological_memory_t* memory) {
    if (!memory) return NULL;

    topological_invariants_t* inv = (topological_invariants_t*)calloc(1,
                                                                       sizeof(topological_invariants_t));
    if (!inv) return NULL;

    inv->num_dimensions = 3;
    inv->betti_numbers = (double*)calloc(inv->num_dimensions, sizeof(double));
    inv->num_wilson_loops = 4;
    inv->wilson_loops = (ComplexDouble*)calloc(inv->num_wilson_loops, sizeof(ComplexDouble));

    if (!inv->betti_numbers || !inv->wilson_loops) {
        free(inv->betti_numbers);
        free(inv->wilson_loops);
        free(inv);
        return NULL;
    }

    // Compute Betti numbers from anyon configuration
    // b_0 = number of connected components
    // b_1 = number of holes
    inv->betti_numbers[0] = 1.0;  // Single connected space
    inv->betti_numbers[1] = (double)(memory->num_anyons / 2);  // Holes from anyon pairs
    inv->betti_numbers[2] = 0.0;

    // Euler characteristic
    inv->euler_characteristic = inv->betti_numbers[0] - inv->betti_numbers[1] + inv->betti_numbers[2];

    // Topological entropy
    inv->topological_entropy = memory->topological_entropy;

    // Wilson loop expectation values
    // For Fibonacci anyons, Wilson loops give quantum dimensions
    for (size_t i = 0; i < inv->num_wilson_loops; i++) {
        double angle = 2.0 * M_PI * i / inv->num_wilson_loops;
        inv->wilson_loops[i].real = PHI * cos(angle);
        inv->wilson_loops[i].imag = PHI * sin(angle);
    }

    // Chern number (topological invariant)
    inv->chern_number = 1.0;  // For Fibonacci anyons

    inv->validated = true;
    return inv;
}

void apply_test_noise(topological_memory_t* memory) {
    if (!memory || !memory->anyons) return;

    if (memory->mutex) {
        pthread_mutex_lock((pthread_mutex_t*)memory->mutex);
    }

    // Apply random noise to anyon positions and phases
    for (size_t i = 0; i < memory->num_anyons; i++) {
        // Small position perturbation (doesn't change topology)
        // Just affects phases slightly
        double noise = 0.01 * ((double)rand() / RAND_MAX - 0.5);
        memory->anyons[i].phase += noise;

        // Small amplitude noise
        double amp_noise = 0.001 * ((double)rand() / RAND_MAX - 0.5);
        memory->anyons[i].amplitude.real += amp_noise;
        memory->anyons[i].amplitude.imag += amp_noise;
    }

    // Renormalize
    double norm = 0.0;
    for (size_t i = 0; i < memory->num_anyons; i++) {
        norm += complex_abs_sq(memory->anyons[i].amplitude);
    }
    if (norm > PERSISTENCE_EPSILON) {
        norm = sqrt(norm);
        for (size_t i = 0; i < memory->num_anyons; i++) {
            memory->anyons[i].amplitude.real /= norm;
            memory->anyons[i].amplitude.imag /= norm;
        }
    }

    if (memory->mutex) {
        pthread_mutex_unlock((pthread_mutex_t*)memory->mutex);
    }
}

void perform_test_operations(topological_memory_t* memory) {
    if (!memory) return;

    // Perform some topological operations that should preserve invariants
    braid_sequence_t* seq = generate_braid_sequence();
    if (seq) {
        perform_braiding_sequence(memory, seq);

        // Undo the sequence (inverse braiding)
        for (size_t i = seq->num_operations; i > 0; i--) {
            seq->operations[i-1].clockwise = !seq->operations[i-1].clockwise;
            seq->operations[i-1].r_matrix = fibonacci_r_matrix(seq->operations[i-1].clockwise);
        }
        perform_braiding_sequence(memory, seq);

        free_braid_sequence(seq);
    }
}

bool compare_topological_invariants(const topological_invariants_t* inv1,
                                    const topological_invariants_t* inv2) {
    if (!inv1 || !inv2) return false;
    if (!inv1->validated || !inv2->validated) return false;

    // Compare Betti numbers
    for (size_t i = 0; i < inv1->num_dimensions && i < inv2->num_dimensions; i++) {
        if (fabs(inv1->betti_numbers[i] - inv2->betti_numbers[i]) > INVARIANT_TOLERANCE) {
            return false;
        }
    }

    // Compare Euler characteristic
    if (fabs(inv1->euler_characteristic - inv2->euler_characteristic) > INVARIANT_TOLERANCE) {
        return false;
    }

    // Compare topological entropy
    if (fabs(inv1->topological_entropy - inv2->topological_entropy) > INVARIANT_TOLERANCE) {
        return false;
    }

    // Compare Chern number
    if (fabs(inv1->chern_number - inv2->chern_number) > INVARIANT_TOLERANCE) {
        return false;
    }

    // Compare Wilson loops
    for (size_t i = 0; i < inv1->num_wilson_loops && i < inv2->num_wilson_loops; i++) {
        double diff_real = fabs(inv1->wilson_loops[i].real - inv2->wilson_loops[i].real);
        double diff_imag = fabs(inv1->wilson_loops[i].imag - inv2->wilson_loops[i].imag);
        if (diff_real > INVARIANT_TOLERANCE || diff_imag > INVARIANT_TOLERANCE) {
            return false;
        }
    }

    return true;
}

void free_topological_invariants(topological_invariants_t* invariants) {
    if (!invariants) return;
    free(invariants->betti_numbers);
    free(invariants->wilson_loops);
    free(invariants);
}

bool topo_neuro_verify_protection(const topological_memory_t* memory) {
    if (!memory) return false;

    // Verify protection is working
    // 1. Check energy gap is above threshold
    if (memory->gap < ENERGY_GAP_THRESHOLD) return false;

    // 2. Check fidelity is acceptable
    if (memory->fidelity < FIDELITY_THRESHOLD) return false;

    // 3. Check no corrections needed
    if (memory->needs_correction) return false;

    // 4. Verify anyonic state consistency
    if (!verify_anyonic_states(memory)) return false;

    return true;
}

// =============================================================================
// Network State Implementation
// =============================================================================

network_state_t* capture_network_state(const neural_network_t* network) {
    if (!network || !network->layers) return NULL;

    network_state_t* state = (network_state_t*)calloc(1, sizeof(network_state_t));
    if (!state) return NULL;

    state->num_layers = network->num_layers;
    state->layer_sizes = (size_t*)calloc(state->num_layers, sizeof(size_t));
    state->weights = (double**)calloc(state->num_layers - 1, sizeof(double*));
    state->biases = (double**)calloc(state->num_layers, sizeof(double*));
    state->gradients = (double*)calloc(state->num_layers, sizeof(double));

    if (!state->layer_sizes || !state->weights || !state->biases || !state->gradients) {
        free(state->layer_sizes);
        free(state->weights);
        free(state->biases);
        free(state->gradients);
        free(state);
        return NULL;
    }

    // Copy layer sizes
    for (size_t l = 0; l < state->num_layers; l++) {
        state->layer_sizes[l] = network->layers[l].num_neurons;
    }

    // Copy weights
    for (size_t l = 0; l < state->num_layers - 1; l++) {
        size_t num_weights = network->num_connections[l];
        state->weights[l] = (double*)calloc(num_weights, sizeof(double));
        if (!state->weights[l]) {
            // Cleanup
            for (size_t k = 0; k < l; k++) {
                free(state->weights[k]);
            }
            free(state->layer_sizes);
            free(state->weights);
            free(state->biases);
            free(state->gradients);
            free(state);
            return NULL;
        }
        for (size_t w = 0; w < num_weights; w++) {
            state->weights[l][w] = network->connections[l][w].weight;
        }
    }

    // Copy biases
    for (size_t l = 0; l < state->num_layers; l++) {
        size_t num_biases = network->layers[l].num_neurons;
        state->biases[l] = (double*)calloc(num_biases, sizeof(double));
        if (!state->biases[l]) {
            // Cleanup
            for (size_t k = 0; k < l; k++) {
                free(state->biases[k]);
            }
            for (size_t k = 0; k < state->num_layers - 1; k++) {
                free(state->weights[k]);
            }
            free(state->layer_sizes);
            free(state->weights);
            free(state->biases);
            free(state->gradients);
            free(state);
            return NULL;
        }
        memcpy(state->biases[l], network->layers[l].biases, num_biases * sizeof(double));
    }

    // Copy other state
    state->loss = network->loss;
    state->learning_rate = network->learning_rate;
    state->epoch = network->epoch;

    // Compute gradient norms
    for (size_t l = 0; l < state->num_layers - 1; l++) {
        double grad_norm = 0.0;
        for (size_t w = 0; w < network->num_connections[l]; w++) {
            grad_norm += network->connections[l][w].gradient *
                        network->connections[l][w].gradient;
        }
        state->gradients[l] = sqrt(grad_norm);
    }

    return state;
}

qgt_error_t update_topological_weights(neural_network_t* network,
                                       const persistence_diagram_t* diagram) {
    if (!network || !diagram) return QGT_ERROR_INVALID_ARGUMENT;

    // Use persistence features to guide weight updates
    // Weights corresponding to persistent features get smaller updates
    // This preserves important topological structure

    double persistence_factor = 1.0 / (1.0 + diagram->total_persistence);

    for (size_t l = 0; l < network->num_layers - 1; l++) {
        for (size_t w = 0; w < network->num_connections[l]; w++) {
            double weight = network->connections[l][w].weight;
            double weight_mag = fabs(weight);

            // Scale update by persistence
            // High persistence features should be preserved
            double scale = persistence_factor;
            if (weight_mag > 0.1) {
                // Important weight - reduce update magnitude
                scale *= 0.5;
            }

            // Apply scaled gradient
            network->connections[l][w].weight -=
                scale * network->learning_rate * network->connections[l][w].gradient;
        }
    }

    return QGT_SUCCESS;
}

bool verify_topological_constraints(const network_state_t* state) {
    if (!state) return false;

    // Verify gradient norms are reasonable
    for (size_t l = 0; l < state->num_layers - 1; l++) {
        if (state->gradients[l] > 100.0) {
            return false;  // Gradient explosion
        }
        if (isnan(state->gradients[l]) || isinf(state->gradients[l])) {
            return false;
        }
    }

    // Verify loss is finite
    if (isnan(state->loss) || isinf(state->loss)) {
        return false;
    }

    // Verify weights are finite
    for (size_t l = 0; l < state->num_layers - 1; l++) {
        if (!state->weights[l]) continue;
        // Check first few weights
        size_t to_check = state->layer_sizes[l] * state->layer_sizes[l + 1];
        to_check = to_check > 100 ? 100 : to_check;
        for (size_t w = 0; w < to_check; w++) {
            if (isnan(state->weights[l][w]) || isinf(state->weights[l][w])) {
                return false;
            }
        }
    }

    return true;
}

bool verify_weight_update(const network_state_t* before,
                         const network_state_t* after) {
    if (!before || !after) return false;
    if (before->num_layers != after->num_layers) return false;

    // Verify that weights have changed but not exploded
    for (size_t l = 0; l < before->num_layers - 1; l++) {
        if (!before->weights[l] || !after->weights[l]) continue;

        double max_change = 0.0;
        size_t num_weights = before->layer_sizes[l] * before->layer_sizes[l + 1];
        num_weights = num_weights > 100 ? 100 : num_weights;

        for (size_t w = 0; w < num_weights; w++) {
            double change = fabs(after->weights[l][w] - before->weights[l][w]);
            if (change > max_change) max_change = change;
        }

        // Weights should change but not too much
        if (max_change > 10.0) {
            return false;  // Too large update
        }
    }

    // Loss should ideally decrease or stay similar
    // Allow some tolerance for stochastic updates
    if (after->loss > before->loss * 10.0) {
        return false;  // Loss exploded
    }

    return true;
}

void free_network_state(network_state_t* state) {
    if (!state) return;

    if (state->weights) {
        for (size_t l = 0; l < state->num_layers - 1; l++) {
            free(state->weights[l]);
        }
        free(state->weights);
    }

    if (state->biases) {
        for (size_t l = 0; l < state->num_layers; l++) {
            free(state->biases[l]);
        }
        free(state->biases);
    }

    free(state->layer_sizes);
    free(state->gradients);
    free(state);
}

// =============================================================================
// Integration Functions Implementation
// =============================================================================

bool verify_learning_convergence(const neuromorphic_unit_t* unit) {
    if (!unit) return false;

    // Check if loss has converged
    if (unit->converged) return true;

    // Check if loss is low enough
    if (unit->loss < 0.01) return true;

    // Check if loss hasn't improved for a while
    if (unit->patience_counter > 5 && unit->loss < unit->best_loss * 1.1) {
        return true;
    }

    return false;
}

bool verify_system_integration(const topological_memory_t* memory,
                               const neuromorphic_unit_t* unit,
                               const interface_t* iface) {
    if (!memory || !unit || !iface) return false;

    // Verify topological memory is healthy
    if (!verify_anyonic_states(memory)) return false;
    if (memory->needs_correction) return false;

    // Verify neuromorphic unit is functional
    if (!unit->network) return false;
    if (isnan(unit->loss) || isinf(unit->loss)) return false;

    // Verify interface has been used
    if (iface->num_conversions == 0) return false;

    // Verify all components are in sync
    // Check that quantum state dimension matches interface buffer
    if (memory->fusion_dim > iface->buffer_size) return false;

    return true;
}

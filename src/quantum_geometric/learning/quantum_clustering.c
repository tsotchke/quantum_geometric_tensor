/**
 * @file quantum_clustering.c
 * @brief Implementation of quantum clustering algorithms
 */

#include "quantum_geometric/learning/quantum_clustering.h"
#include "quantum_geometric/core/quantum_state.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdio.h>

// =============================================================================
// Internal Helper Functions
// =============================================================================

/**
 * @brief Calculate quantum fidelity between two states
 */
static double calculate_fidelity(quantum_state_t* a, quantum_state_t* b) {
    if (!a || !b || a->dimension != b->dimension) return 0.0;

    double fidelity = 0.0;
    for (size_t i = 0; i < a->dimension; i++) {
        // F = |<a|b>|^2
        double re_a = a->coordinates[i].real;
        double im_a = a->coordinates[i].imag;
        double re_b = b->coordinates[i].real;
        double im_b = b->coordinates[i].imag;

        // Inner product contribution
        fidelity += re_a * re_b + im_a * im_b;
    }

    return fidelity * fidelity;  // Square for fidelity
}

/**
 * @brief Calculate quantum distance (1 - fidelity)
 */
static double calculate_quantum_distance(quantum_state_t* a, quantum_state_t* b,
                                         DistanceMetricType metric) {
    switch (metric) {
        case DISTANCE_QUANTUM_FIDELITY:
            return 1.0 - calculate_fidelity(a, b);
        case DISTANCE_TRACE:
            // Trace distance = sqrt(1 - F)
            return sqrt(1.0 - calculate_fidelity(a, b));
        case DISTANCE_BURES:
            // Bures distance = sqrt(2(1 - sqrt(F)))
            return sqrt(2.0 * (1.0 - sqrt(calculate_fidelity(a, b))));
        case DISTANCE_EUCLIDEAN:
        default: {
            // Euclidean distance in Hilbert space
            double dist = 0.0;
            for (size_t i = 0; i < a->dimension; i++) {
                double dr = a->coordinates[i].real - b->coordinates[i].real;
                double di = a->coordinates[i].imag - b->coordinates[i].imag;
                dist += dr * dr + di * di;
            }
            return sqrt(dist);
        }
    }
}

/**
 * @brief Update centroid from assigned states (quantum version)
 */
static void update_centroid(quantum_state_t* centroid,
                           quantum_state_t** states,
                           int* assignments,
                           size_t num_states,
                           int cluster_id) {
    if (!centroid || !states || !assignments) return;

    // Zero out centroid
    for (size_t i = 0; i < centroid->dimension; i++) {
        centroid->coordinates[i].real = 0.0f;
        centroid->coordinates[i].imag = 0.0f;
    }

    // Sum all states assigned to this cluster
    size_t count = 0;
    for (size_t s = 0; s < num_states; s++) {
        if (assignments[s] == cluster_id && states[s]) {
            for (size_t i = 0; i < centroid->dimension; i++) {
                centroid->coordinates[i].real += states[s]->coordinates[i].real;
                centroid->coordinates[i].imag += states[s]->coordinates[i].imag;
            }
            count++;
        }
    }

    // Normalize the centroid
    if (count > 0) {
        double norm = 0.0;
        for (size_t i = 0; i < centroid->dimension; i++) {
            norm += centroid->coordinates[i].real * centroid->coordinates[i].real +
                    centroid->coordinates[i].imag * centroid->coordinates[i].imag;
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            float inv_norm = 1.0f / (float)norm;
            for (size_t i = 0; i < centroid->dimension; i++) {
                centroid->coordinates[i].real *= inv_norm;
                centroid->coordinates[i].imag *= inv_norm;
            }
        }
    }
}

// =============================================================================
// Core API Implementation
// =============================================================================

quantum_clustering_t* quantum_clustering_create(const quantum_clustering_config_t* config) {
    if (!config || config->num_clusters == 0 || config->input_dim == 0) {
        return NULL;
    }

    quantum_clustering_t* model = calloc(1, sizeof(quantum_clustering_t));
    if (!model) return NULL;

    model->num_clusters = config->num_clusters;
    model->input_dim = config->input_dim;
    model->quantum_depth = config->quantum_depth;
    model->config = *config;
    model->is_trained = false;
    model->num_samples = 0;
    model->assignments = NULL;

    // Allocate centroids
    model->centroids = calloc(config->num_clusters, sizeof(quantum_state_t*));
    if (!model->centroids) {
        free(model);
        return NULL;
    }

    // Initialize centroids (will be properly initialized during training)
    size_t dim = 1ULL << config->input_dim;  // 2^num_qubits
    for (size_t i = 0; i < config->num_clusters; i++) {
        quantum_state_create(&model->centroids[i], QUANTUM_STATE_PURE, dim);
        if (model->centroids[i]) {
            // Initialize to random state
            double norm = 0.0;
            for (size_t j = 0; j < dim; j++) {
                model->centroids[i]->coordinates[j].real = (float)(rand() / (double)RAND_MAX - 0.5);
                model->centroids[i]->coordinates[j].imag = (float)(rand() / (double)RAND_MAX - 0.5);
                norm += model->centroids[i]->coordinates[j].real * model->centroids[i]->coordinates[j].real +
                        model->centroids[i]->coordinates[j].imag * model->centroids[i]->coordinates[j].imag;
            }
            // Normalize
            norm = sqrt(norm);
            for (size_t j = 0; j < dim; j++) {
                model->centroids[i]->coordinates[j].real /= (float)norm;
                model->centroids[i]->coordinates[j].imag /= (float)norm;
            }
        }
    }

    return model;
}

void quantum_clustering_destroy(quantum_clustering_t* model) {
    if (!model) return;

    // Free centroids
    if (model->centroids) {
        for (size_t i = 0; i < model->num_clusters; i++) {
            if (model->centroids[i]) {
                quantum_state_destroy(model->centroids[i]);
            }
        }
        free(model->centroids);
    }

    // Free assignments
    if (model->assignments) {
        free(model->assignments);
    }

    // Free internal data
    if (model->internal_data) {
        free(model->internal_data);
    }

    free(model);
}

// =============================================================================
// Data Preparation Implementation
// =============================================================================

dataset_t* quantum_generate_synthetic_data(size_t num_samples,
                                           size_t feature_dim,
                                           DataType type) {
    if (num_samples == 0 || feature_dim == 0) return NULL;

    dataset_t* data = calloc(1, sizeof(dataset_t));
    if (!data) return NULL;

    data->num_samples = num_samples;
    data->feature_dim = feature_dim;

    // Allocate feature matrix
    data->features = calloc(num_samples, sizeof(double*));
    if (!data->features) {
        free(data);
        return NULL;
    }

    // Seed random number generator
    static bool seeded = false;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = true;
    }

    // Generate data based on type
    for (size_t i = 0; i < num_samples; i++) {
        data->features[i] = calloc(feature_dim, sizeof(double));
        if (!data->features[i]) {
            // Cleanup on failure
            for (size_t j = 0; j < i; j++) {
                free(data->features[j]);
            }
            free(data->features);
            free(data);
            return NULL;
        }

        // Generate random features in [-1, 1]
        for (size_t j = 0; j < feature_dim; j++) {
            data->features[i][j] = 2.0 * rand() / (double)RAND_MAX - 1.0;
        }
    }

    // Optionally allocate labels
    if (type == DATA_TYPE_CLASSIFICATION) {
        data->labels = calloc(num_samples, sizeof(int));
        if (data->labels) {
            for (size_t i = 0; i < num_samples; i++) {
                data->labels[i] = rand() % 3;  // Random 0-2 labels
            }
        }
    }

    return data;
}

void quantum_destroy_dataset(dataset_t* data) {
    if (!data) return;

    if (data->features) {
        for (size_t i = 0; i < data->num_samples; i++) {
            free(data->features[i]);
        }
        free(data->features);
    }

    if (data->labels) {
        free(data->labels);
    }

    free(data);
}

quantum_dataset_t* quantum_prepare_states(dataset_t* classical_data,
                                          quantum_system_t* system) {
    if (!classical_data || classical_data->num_samples == 0) return NULL;

    quantum_dataset_t* qdata = calloc(1, sizeof(quantum_dataset_t));
    if (!qdata) return NULL;

    qdata->num_samples = classical_data->num_samples;
    qdata->state_dim = classical_data->feature_dim;

    // Allocate state array
    qdata->states = calloc(qdata->num_samples, sizeof(quantum_state_t*));
    if (!qdata->states) {
        free(qdata);
        return NULL;
    }

    // Determine quantum dimension (2^num_qubits)
    size_t num_qubits = classical_data->feature_dim;
    size_t dim = 1ULL << num_qubits;

    // Create quantum states from classical data
    for (size_t i = 0; i < qdata->num_samples; i++) {
        // Create state
        if (quantum_state_create(&qdata->states[i], QUANTUM_STATE_PURE, dim) != QGT_SUCCESS) {
            // Cleanup on failure
            for (size_t j = 0; j < i; j++) {
                quantum_state_destroy(qdata->states[j]);
            }
            free(qdata->states);
            free(qdata);
            return NULL;
        }

        // Amplitude encoding: normalize classical data into quantum amplitudes
        double norm = 0.0;
        for (size_t j = 0; j < classical_data->feature_dim && j < dim; j++) {
            norm += classical_data->features[i][j] * classical_data->features[i][j];
        }
        norm = sqrt(norm);
        if (norm < 1e-10) norm = 1.0;

        // Set amplitudes (encoding classical features as quantum amplitudes)
        for (size_t j = 0; j < dim; j++) {
            if (j < classical_data->feature_dim) {
                qdata->states[i]->coordinates[j].real = (float)(classical_data->features[i][j] / norm);
                qdata->states[i]->coordinates[j].imag = 0.0f;
            } else {
                qdata->states[i]->coordinates[j].real = 0.0f;
                qdata->states[i]->coordinates[j].imag = 0.0f;
            }
        }

        // Ensure normalization
        norm = 0.0;
        for (size_t j = 0; j < dim; j++) {
            norm += qdata->states[i]->coordinates[j].real * qdata->states[i]->coordinates[j].real +
                    qdata->states[i]->coordinates[j].imag * qdata->states[i]->coordinates[j].imag;
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (size_t j = 0; j < dim; j++) {
                qdata->states[i]->coordinates[j].real /= (float)norm;
                qdata->states[i]->coordinates[j].imag /= (float)norm;
            }
        } else {
            // Default to |0> state
            qdata->states[i]->coordinates[0].real = 1.0f;
        }

        qdata->states[i]->is_normalized = true;
    }

    return qdata;
}

void quantum_destroy_quantum_dataset(quantum_dataset_t* data) {
    if (!data) return;

    if (data->states) {
        for (size_t i = 0; i < data->num_samples; i++) {
            if (data->states[i]) {
                quantum_state_destroy(data->states[i]);
            }
        }
        free(data->states);
    }

    free(data);
}

// =============================================================================
// Clustering Operations Implementation
// =============================================================================

clustering_result_t quantum_cluster_distributed(quantum_clustering_t* model,
                                                quantum_dataset_t* data,
                                                struct distributed_manager_t* manager,
                                                void* options) {
    clustering_result_t result = {0};
    result.status = CLUSTERING_ERROR;

    if (!model || !data || data->num_samples == 0) {
        result.status = CLUSTERING_INVALID_INPUT;
        return result;
    }

    // Allocate assignments array
    if (model->assignments) free(model->assignments);
    model->assignments = calloc(data->num_samples, sizeof(int));
    if (!model->assignments) {
        result.status = CLUSTERING_MEMORY_ERROR;
        return result;
    }
    model->num_samples = data->num_samples;

    // Initialize centroids using k-means++ style initialization
    // (Select first centroid randomly, then select subsequent centroids
    // with probability proportional to squared distance from nearest centroid)

    // Copy first random state as first centroid
    size_t first_idx = rand() % data->num_samples;
    if (model->centroids[0] && data->states[first_idx]) {
        for (size_t i = 0; i < model->centroids[0]->dimension; i++) {
            model->centroids[0]->coordinates[i] = data->states[first_idx]->coordinates[i];
        }
    }

    // Initialize remaining centroids with k-means++ probability
    for (size_t k = 1; k < model->num_clusters; k++) {
        double total_dist = 0.0;
        double* dists = calloc(data->num_samples, sizeof(double));
        if (!dists) continue;

        // Calculate distance to nearest centroid for each point
        for (size_t i = 0; i < data->num_samples; i++) {
            double min_dist = INFINITY;
            for (size_t c = 0; c < k; c++) {
                double d = calculate_quantum_distance(data->states[i], model->centroids[c],
                                                     model->config.algorithm.distance);
                if (d < min_dist) min_dist = d;
            }
            dists[i] = min_dist * min_dist;  // Square for probability weighting
            total_dist += dists[i];
        }

        // Select next centroid with probability proportional to squared distance
        double r = (rand() / (double)RAND_MAX) * total_dist;
        double cumsum = 0.0;
        size_t selected = 0;
        for (size_t i = 0; i < data->num_samples; i++) {
            cumsum += dists[i];
            if (cumsum >= r) {
                selected = i;
                break;
            }
        }

        // Copy selected state as centroid
        if (model->centroids[k] && data->states[selected]) {
            for (size_t i = 0; i < model->centroids[k]->dimension; i++) {
                model->centroids[k]->coordinates[i] = data->states[selected]->coordinates[i];
            }
        }

        free(dists);
    }

    // Main k-means loop
    size_t max_iter = model->config.optimization.convergence.max_iterations;
    double tolerance = model->config.optimization.convergence.tolerance;

    for (size_t iter = 0; iter < max_iter; iter++) {
        // Assignment step: assign each point to nearest centroid
        int changes = 0;
        for (size_t i = 0; i < data->num_samples; i++) {
            int best_cluster = 0;
            double min_dist = INFINITY;

            for (size_t c = 0; c < model->num_clusters; c++) {
                double d = calculate_quantum_distance(data->states[i], model->centroids[c],
                                                     model->config.algorithm.distance);
                if (d < min_dist) {
                    min_dist = d;
                    best_cluster = (int)c;
                }
            }

            if (model->assignments[i] != best_cluster) {
                model->assignments[i] = best_cluster;
                changes++;
            }
        }

        // Update step: recalculate centroids
        for (size_t c = 0; c < model->num_clusters; c++) {
            update_centroid(model->centroids[c], data->states,
                           model->assignments, data->num_samples, (int)c);
        }

        result.iterations = iter + 1;

        // Check convergence
        if (changes == 0 || (changes / (double)data->num_samples) < tolerance) {
            break;
        }
    }

    // Calculate final loss (inertia)
    result.final_loss = 0.0;
    for (size_t i = 0; i < data->num_samples; i++) {
        int c = model->assignments[i];
        double d = calculate_quantum_distance(data->states[i], model->centroids[c],
                                             model->config.algorithm.distance);
        result.final_loss += d * d;
    }

    // Copy assignments to result
    result.assignments = calloc(data->num_samples, sizeof(int));
    if (result.assignments) {
        memcpy(result.assignments, model->assignments, data->num_samples * sizeof(int));
    }

    model->is_trained = true;
    result.status = CLUSTERING_SUCCESS;

    return result;
}

int quantum_assign_cluster(quantum_clustering_t* model, quantum_state_t* state) {
    if (!model || !state || !model->is_trained) return -1;

    int best_cluster = 0;
    double min_dist = INFINITY;

    for (size_t c = 0; c < model->num_clusters; c++) {
        double d = calculate_quantum_distance(state, model->centroids[c],
                                             model->config.algorithm.distance);
        if (d < min_dist) {
            min_dist = d;
            best_cluster = (int)c;
        }
    }

    return best_cluster;
}

// =============================================================================
// Evaluation Implementation
// =============================================================================

clustering_eval_result_t quantum_evaluate_clustering(quantum_clustering_t* model,
                                                quantum_dataset_t* data) {
    clustering_eval_result_t result = {0};

    if (!model || !data || !model->is_trained || data->num_samples == 0) {
        return result;
    }

    // Calculate silhouette score
    // s(i) = (b(i) - a(i)) / max(a(i), b(i))
    // a(i) = average distance to same cluster
    // b(i) = minimum average distance to other clusters

    double total_silhouette = 0.0;

    for (size_t i = 0; i < data->num_samples; i++) {
        int cluster_i = model->assignments[i];

        // Calculate a(i) - average distance to points in same cluster
        double a_i = 0.0;
        size_t same_count = 0;
        for (size_t j = 0; j < data->num_samples; j++) {
            if (i != j && model->assignments[j] == cluster_i) {
                a_i += calculate_quantum_distance(data->states[i], data->states[j],
                                                 model->config.algorithm.distance);
                same_count++;
            }
        }
        a_i = (same_count > 0) ? a_i / same_count : 0.0;

        // Calculate b(i) - minimum average distance to other clusters
        double b_i = INFINITY;
        for (size_t c = 0; c < model->num_clusters; c++) {
            if ((int)c != cluster_i) {
                double avg_dist = 0.0;
                size_t other_count = 0;
                for (size_t j = 0; j < data->num_samples; j++) {
                    if (model->assignments[j] == (int)c) {
                        avg_dist += calculate_quantum_distance(data->states[i], data->states[j],
                                                              model->config.algorithm.distance);
                        other_count++;
                    }
                }
                if (other_count > 0) {
                    avg_dist /= other_count;
                    if (avg_dist < b_i) b_i = avg_dist;
                }
            }
        }
        if (b_i == INFINITY) b_i = 0.0;

        // Silhouette for point i
        double max_ab = fmax(a_i, b_i);
        double s_i = (max_ab > 1e-10) ? (b_i - a_i) / max_ab : 0.0;
        total_silhouette += s_i;
    }

    result.silhouette_score = total_silhouette / data->num_samples;

    // Calculate Davies-Bouldin index
    // DB = (1/k) * sum_i(max_j!=i((s_i + s_j) / d_ij))
    // where s_i is average distance from centroid i, d_ij is centroid distance

    double* cluster_spreads = calloc(model->num_clusters, sizeof(double));
    size_t* cluster_counts = calloc(model->num_clusters, sizeof(size_t));

    if (cluster_spreads && cluster_counts) {
        // Calculate cluster spreads
        for (size_t i = 0; i < data->num_samples; i++) {
            int c = model->assignments[i];
            cluster_spreads[c] += calculate_quantum_distance(data->states[i], model->centroids[c],
                                                            model->config.algorithm.distance);
            cluster_counts[c]++;
        }
        for (size_t c = 0; c < model->num_clusters; c++) {
            if (cluster_counts[c] > 0) {
                cluster_spreads[c] /= cluster_counts[c];
            }
        }

        // Calculate DB index
        double db_sum = 0.0;
        for (size_t i = 0; i < model->num_clusters; i++) {
            double max_ratio = 0.0;
            for (size_t j = 0; j < model->num_clusters; j++) {
                if (i != j) {
                    double d_ij = calculate_quantum_distance(model->centroids[i], model->centroids[j],
                                                            model->config.algorithm.distance);
                    if (d_ij > 1e-10) {
                        double ratio = (cluster_spreads[i] + cluster_spreads[j]) / d_ij;
                        if (ratio > max_ratio) max_ratio = ratio;
                    }
                }
            }
            db_sum += max_ratio;
        }
        result.davies_bouldin_index = db_sum / model->num_clusters;
    }

    free(cluster_spreads);
    free(cluster_counts);

    // Calculate quantum entropy (von Neumann entropy approximation)
    // Using cluster assignment probabilities
    result.quantum_entropy = 0.0;
    size_t* sizes = calloc(model->num_clusters, sizeof(size_t));
    if (sizes) {
        for (size_t i = 0; i < data->num_samples; i++) {
            sizes[model->assignments[i]]++;
        }
        for (size_t c = 0; c < model->num_clusters; c++) {
            if (sizes[c] > 0) {
                double p = sizes[c] / (double)data->num_samples;
                result.quantum_entropy -= p * log2(p);
            }
        }
        free(sizes);
    }

    // Calculate inertia
    result.inertia = 0.0;
    for (size_t i = 0; i < data->num_samples; i++) {
        int c = model->assignments[i];
        double d = calculate_quantum_distance(data->states[i], model->centroids[c],
                                             model->config.algorithm.distance);
        result.inertia += d * d;
    }

    return result;
}

cluster_stats_t quantum_calculate_cluster_stats(quantum_clustering_t* model,
                                                quantum_dataset_t* data) {
    cluster_stats_t stats = {0};

    if (!model || !data || !model->is_trained) {
        return stats;
    }

    stats.num_clusters = model->num_clusters;
    stats.cluster_sizes = calloc(model->num_clusters, sizeof(size_t));

    if (stats.cluster_sizes) {
        for (size_t i = 0; i < data->num_samples; i++) {
            if (model->assignments[i] >= 0 && (size_t)model->assignments[i] < model->num_clusters) {
                stats.cluster_sizes[model->assignments[i]]++;
            }
        }
    }

    return stats;
}

// =============================================================================
// Model Persistence Implementation
// =============================================================================

int quantum_save_clustering_model(quantum_clustering_t* model, const char* path) {
    if (!model || !path) return -1;

    FILE* f = fopen(path, "wb");
    if (!f) return -1;

    // Write header
    fwrite(&model->num_clusters, sizeof(size_t), 1, f);
    fwrite(&model->input_dim, sizeof(size_t), 1, f);
    fwrite(&model->quantum_depth, sizeof(size_t), 1, f);
    fwrite(&model->config, sizeof(quantum_clustering_config_t), 1, f);

    // Write centroids
    for (size_t c = 0; c < model->num_clusters; c++) {
        if (model->centroids[c]) {
            fwrite(&model->centroids[c]->dimension, sizeof(size_t), 1, f);
            fwrite(model->centroids[c]->coordinates,
                   sizeof(ComplexFloat), model->centroids[c]->dimension, f);
        }
    }

    fclose(f);
    return 0;
}

quantum_clustering_t* quantum_load_clustering_model(const char* path) {
    if (!path) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    size_t num_clusters, input_dim, quantum_depth;
    quantum_clustering_config_t config;

    fread(&num_clusters, sizeof(size_t), 1, f);
    fread(&input_dim, sizeof(size_t), 1, f);
    fread(&quantum_depth, sizeof(size_t), 1, f);
    fread(&config, sizeof(quantum_clustering_config_t), 1, f);

    quantum_clustering_t* model = quantum_clustering_create(&config);
    if (!model) {
        fclose(f);
        return NULL;
    }

    // Read centroids
    for (size_t c = 0; c < model->num_clusters; c++) {
        size_t dim;
        fread(&dim, sizeof(size_t), 1, f);
        if (model->centroids[c] && model->centroids[c]->dimension == dim) {
            fread(model->centroids[c]->coordinates, sizeof(ComplexFloat), dim, f);
        }
    }

    model->is_trained = true;
    fclose(f);
    return model;
}

bool clustering_models_equal(quantum_clustering_t* a, quantum_clustering_t* b) {
    if (!a || !b) return (!a && !b);
    if (a->num_clusters != b->num_clusters) return false;
    if (a->input_dim != b->input_dim) return false;
    if (a->quantum_depth != b->quantum_depth) return false;

    // Compare centroids
    for (size_t c = 0; c < a->num_clusters; c++) {
        if (!a->centroids[c] || !b->centroids[c]) return false;
        if (a->centroids[c]->dimension != b->centroids[c]->dimension) return false;

        for (size_t i = 0; i < a->centroids[c]->dimension; i++) {
            if (fabs(a->centroids[c]->coordinates[i].real - b->centroids[c]->coordinates[i].real) > 1e-6 ||
                fabs(a->centroids[c]->coordinates[i].imag - b->centroids[c]->coordinates[i].imag) > 1e-6) {
                return false;
            }
        }
    }

    return true;
}

// =============================================================================
// Utility Functions Implementation
// =============================================================================

bool quantum_is_valid_state(quantum_state_t* state) {
    if (!state || !state->coordinates || state->dimension == 0) return false;

    // Check normalization
    double norm = 0.0;
    for (size_t i = 0; i < state->dimension; i++) {
        norm += state->coordinates[i].real * state->coordinates[i].real +
                state->coordinates[i].imag * state->coordinates[i].imag;
    }

    return fabs(norm - 1.0) < 1e-6;
}

double quantum_trace_norm(quantum_state_t* state) {
    if (!state || !state->coordinates) return 0.0;

    // For pure states, trace norm is 1 if normalized
    double norm = 0.0;
    for (size_t i = 0; i < state->dimension; i++) {
        norm += state->coordinates[i].real * state->coordinates[i].real +
                state->coordinates[i].imag * state->coordinates[i].imag;
    }

    return sqrt(norm);
}

quantum_clustering_t* create_test_clustering_model(size_t input_dim,
                                                   size_t num_clusters,
                                                   size_t quantum_depth) {
    quantum_clustering_config_t config = {
        .num_clusters = num_clusters,
        .input_dim = input_dim,
        .quantum_depth = quantum_depth,
        .algorithm = {
            .type = CLUSTERING_QUANTUM_KMEANS,
            .distance = DISTANCE_QUANTUM_FIDELITY,
            .initialization = INIT_QUANTUM_KMEANS_PLUS_PLUS
        },
        .optimization = {
            .geometric_enhancement = true,
            .error_mitigation = false,
            .convergence = {
                .max_iterations = 100,
                .tolerance = 1e-6
            }
        }
    };

    return quantum_clustering_create(&config);
}

quantum_state_t* quantum_create_random_state(size_t num_qubits) {
    size_t dim = 1ULL << num_qubits;
    quantum_state_t* state = NULL;

    if (quantum_state_create(&state, QUANTUM_STATE_PURE, dim) != QGT_SUCCESS) {
        return NULL;
    }

    // Generate random amplitudes
    double norm = 0.0;
    for (size_t i = 0; i < dim; i++) {
        state->coordinates[i].real = (float)(rand() / (double)RAND_MAX - 0.5);
        state->coordinates[i].imag = (float)(rand() / (double)RAND_MAX - 0.5);
        norm += state->coordinates[i].real * state->coordinates[i].real +
                state->coordinates[i].imag * state->coordinates[i].imag;
    }

    // Normalize
    norm = sqrt(norm);
    for (size_t i = 0; i < dim; i++) {
        state->coordinates[i].real /= (float)norm;
        state->coordinates[i].imag /= (float)norm;
    }

    state->is_normalized = true;
    return state;
}

// quantum_state_destroy is provided by quantum_state_operations.c

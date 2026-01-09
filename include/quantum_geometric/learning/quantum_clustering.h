/**
 * @file quantum_clustering.h
 * @brief Quantum clustering module for the Quantum Geometric Tensor Library
 *
 * Provides quantum-enhanced clustering algorithms including quantum k-means,
 * quantum spectral clustering, and fidelity-based distance metrics.
 */

#ifndef QUANTUM_CLUSTERING_H
#define QUANTUM_CLUSTERING_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/error_codes.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct distributed_manager_t;

// =============================================================================
// Enumerations
// =============================================================================

/**
 * @brief Clustering algorithm types
 */
typedef enum {
    CLUSTERING_QUANTUM_KMEANS = 0,      ///< Quantum k-means clustering
    CLUSTERING_QUANTUM_SPECTRAL = 1,    ///< Quantum spectral clustering
    CLUSTERING_QUANTUM_DBSCAN = 2,      ///< Quantum DBSCAN clustering
    CLUSTERING_QUANTUM_HIERARCHICAL = 3 ///< Quantum hierarchical clustering
} ClusteringAlgorithmType;

/**
 * @brief Distance metric types for clustering
 */
typedef enum {
    DISTANCE_QUANTUM_FIDELITY = 0,  ///< Quantum state fidelity distance
    DISTANCE_EUCLIDEAN = 1,         ///< Classical Euclidean distance
    DISTANCE_TRACE = 2,             ///< Trace distance for quantum states
    DISTANCE_BURES = 3              ///< Bures distance for quantum states
} DistanceMetricType;

/**
 * @brief Initialization methods for clustering
 */
typedef enum {
    INIT_QUANTUM_KMEANS_PLUS_PLUS = 0,  ///< Quantum k-means++ initialization
    INIT_RANDOM = 1,                     ///< Random initialization
    INIT_FORGY = 2,                      ///< Forgy initialization
    INIT_QUANTUM_RANDOM = 3              ///< Quantum random state initialization
} InitializationType;

/**
 * @brief Data types for synthetic data generation
 */
typedef enum {
    DATA_TYPE_CLUSTERING = 0,      ///< Data for clustering tasks
    DATA_TYPE_CLASSIFICATION = 1,  ///< Data for classification tasks
    DATA_TYPE_REGRESSION = 2       ///< Data for regression tasks
} DataType;

/**
 * @brief Clustering operation status
 */
typedef enum {
    CLUSTERING_SUCCESS = 0,          ///< Operation succeeded
    CLUSTERING_ERROR = -1,           ///< General error
    CLUSTERING_NOT_CONVERGED = -2,   ///< Algorithm did not converge
    CLUSTERING_INVALID_INPUT = -3,   ///< Invalid input parameters
    CLUSTERING_MEMORY_ERROR = -4     ///< Memory allocation error
} ClusteringStatus;

// =============================================================================
// Configuration Structures
// =============================================================================

/**
 * @brief Algorithm configuration for clustering
 */
typedef struct {
    ClusteringAlgorithmType type;       ///< Clustering algorithm type
    DistanceMetricType distance;        ///< Distance metric type
    InitializationType initialization;  ///< Initialization method
} clustering_algorithm_config_t;

/**
 * @brief Convergence parameters for clustering
 */
typedef struct {
    size_t max_iterations;  ///< Maximum number of iterations
    double tolerance;       ///< Convergence tolerance
} convergence_config_t;

/**
 * @brief Optimization configuration for clustering
 */
typedef struct {
    bool geometric_enhancement;   ///< Enable geometric enhancement
    bool error_mitigation;        ///< Enable quantum error mitigation
    convergence_config_t convergence;  ///< Convergence parameters
} clustering_optimization_config_t;

/**
 * @brief Main configuration for quantum clustering
 */
typedef struct {
    size_t num_clusters;                     ///< Number of clusters
    size_t input_dim;                        ///< Input feature dimension
    size_t quantum_depth;                    ///< Quantum circuit depth
    clustering_algorithm_config_t algorithm; ///< Algorithm configuration
    clustering_optimization_config_t optimization; ///< Optimization configuration
} quantum_clustering_config_t;

// =============================================================================
// Data Structures
// =============================================================================

/**
 * @brief Classical dataset structure
 */
typedef struct {
    size_t num_samples;     ///< Number of samples
    size_t feature_dim;     ///< Feature dimension
    double** features;      ///< Feature matrix [num_samples][feature_dim]
    int* labels;            ///< Optional labels for supervised learning
    void* auxiliary_data;   ///< Additional data
} dataset_t;

/**
 * @brief Quantum dataset structure
 */
typedef struct {
    size_t num_samples;         ///< Number of quantum states
    size_t state_dim;           ///< Quantum state dimension
    quantum_state_t** states;   ///< Array of quantum states
    void* auxiliary_data;       ///< Additional data
} quantum_dataset_t;

/**
 * @brief Clustering result structure
 */
typedef struct {
    ClusteringStatus status;   ///< Operation status
    size_t iterations;         ///< Number of iterations performed
    double final_loss;         ///< Final clustering loss
    int* assignments;          ///< Cluster assignments for each sample
} clustering_result_t;

/**
 * @brief Clustering evaluation metrics
 */
typedef struct {
    double silhouette_score;      ///< Silhouette coefficient [-1, 1]
    double davies_bouldin_index;  ///< Davies-Bouldin index (lower is better)
    double quantum_entropy;       ///< Quantum entropy of clusters
    double inertia;               ///< Within-cluster sum of squares
} clustering_eval_result_t;

/**
 * @brief Cluster statistics
 */
typedef struct {
    size_t* cluster_sizes;    ///< Size of each cluster
    size_t num_clusters;      ///< Number of clusters
    double* cluster_purities; ///< Purity of each cluster (if labels available)
} cluster_stats_t;

/**
 * @brief Opaque quantum clustering model
 */
typedef struct quantum_clustering_t {
    size_t num_clusters;      ///< Number of clusters
    size_t input_dim;         ///< Input feature dimension
    size_t quantum_depth;     ///< Quantum circuit depth
    quantum_clustering_config_t config;  ///< Configuration
    quantum_state_t** centroids;  ///< Cluster centroid states
    int* assignments;             ///< Current cluster assignments
    size_t num_samples;           ///< Number of samples being clustered
    bool is_trained;              ///< Whether model has been trained
    void* internal_data;          ///< Internal implementation data
} quantum_clustering_t;

// =============================================================================
// Core API Functions
// =============================================================================

/**
 * @brief Create a quantum clustering model
 * @param config Configuration for the clustering model
 * @return Pointer to created model, or NULL on failure
 */
quantum_clustering_t* quantum_clustering_create(const quantum_clustering_config_t* config);

/**
 * @brief Destroy a quantum clustering model
 * @param model Model to destroy
 */
void quantum_clustering_destroy(quantum_clustering_t* model);

// =============================================================================
// Data Preparation Functions
// =============================================================================

/**
 * @brief Generate synthetic data for testing
 * @param num_samples Number of samples to generate
 * @param feature_dim Feature dimension
 * @param type Type of data to generate
 * @return Pointer to generated dataset, or NULL on failure
 */
dataset_t* quantum_generate_synthetic_data(size_t num_samples,
                                           size_t feature_dim,
                                           DataType type);

/**
 * @brief Destroy a classical dataset
 * @param data Dataset to destroy
 */
void quantum_destroy_dataset(dataset_t* data);

/**
 * @brief Prepare quantum states from classical data
 * @param classical_data Classical dataset
 * @param system Quantum system for state preparation
 * @return Pointer to quantum dataset, or NULL on failure
 */
quantum_dataset_t* quantum_prepare_states(dataset_t* classical_data,
                                          quantum_system_t* system);

/**
 * @brief Destroy a quantum dataset
 * @param data Quantum dataset to destroy
 */
void quantum_destroy_quantum_dataset(quantum_dataset_t* data);

// =============================================================================
// Clustering Operations
// =============================================================================

/**
 * @brief Perform distributed quantum clustering
 * @param model Clustering model
 * @param data Quantum dataset
 * @param manager Distributed training manager
 * @param options Optional clustering options
 * @return Clustering result
 */
clustering_result_t quantum_cluster_distributed(quantum_clustering_t* model,
                                                quantum_dataset_t* data,
                                                struct distributed_manager_t* manager,
                                                void* options);

/**
 * @brief Assign a single quantum state to a cluster
 * @param model Trained clustering model
 * @param state Quantum state to assign
 * @return Cluster index (0 to num_clusters-1), or -1 on error
 */
int quantum_assign_cluster(quantum_clustering_t* model, quantum_state_t* state);

// =============================================================================
// Evaluation Functions
// =============================================================================

/**
 * @brief Evaluate clustering quality
 * @param model Trained clustering model
 * @param data Quantum dataset used for clustering
 * @return Evaluation metrics
 */
clustering_eval_result_t quantum_evaluate_clustering(quantum_clustering_t* model,
                                                quantum_dataset_t* data);

/**
 * @brief Calculate cluster statistics
 * @param model Trained clustering model
 * @param data Quantum dataset used for clustering
 * @return Cluster statistics
 */
cluster_stats_t quantum_calculate_cluster_stats(quantum_clustering_t* model,
                                                quantum_dataset_t* data);

// =============================================================================
// Model Persistence
// =============================================================================

/**
 * @brief Save clustering model to file
 * @param model Model to save
 * @param path File path
 * @return 0 on success, non-zero on failure
 */
int quantum_save_clustering_model(quantum_clustering_t* model, const char* path);

/**
 * @brief Load clustering model from file
 * @param path File path
 * @return Loaded model, or NULL on failure
 */
quantum_clustering_t* quantum_load_clustering_model(const char* path);

/**
 * @brief Check if two clustering models are equal
 * @param a First model
 * @param b Second model
 * @return true if models are equal
 */
bool clustering_models_equal(quantum_clustering_t* a, quantum_clustering_t* b);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Check if a quantum state is valid (normalized)
 * @param state Quantum state to check
 * @return true if state is valid
 */
bool quantum_is_valid_state(quantum_state_t* state);

/**
 * @brief Calculate trace norm of a quantum state
 * @param state Quantum state
 * @return Trace norm value
 */
double quantum_trace_norm(quantum_state_t* state);

/**
 * @brief Create a test clustering model for unit tests
 * @param input_dim Input dimension
 * @param num_clusters Number of clusters
 * @param quantum_depth Quantum circuit depth
 * @return Test model
 */
quantum_clustering_t* create_test_clustering_model(size_t input_dim,
                                                   size_t num_clusters,
                                                   size_t quantum_depth);

/**
 * @brief Create a random quantum state
 * @param num_qubits Number of qubits
 * @return Random normalized quantum state
 */
quantum_state_t* quantum_create_random_state(size_t num_qubits);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_CLUSTERING_H

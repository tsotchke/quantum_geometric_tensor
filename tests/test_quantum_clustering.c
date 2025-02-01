/**
 * @file test_quantum_clustering.c
 * @brief Tests for the quantum clustering example
 */

#include <quantum_geometric/core/quantum_geometric_core.h>
#include <quantum_geometric/learning/quantum_stochastic_sampling.h>
#include <quantum_geometric/distributed/distributed_training_manager.h>
#include <quantum_geometric/hardware/quantum_hardware_abstraction.h>
#include "test_helpers.h"

// Test configurations
#define TEST_NUM_SAMPLES 50
#define TEST_INPUT_DIM 8
#define TEST_NUM_CLUSTERS 3
#define TEST_QUANTUM_DEPTH 2
#define TEST_BATCH_SIZE 4

void test_quantum_clustering_creation() {
    // Configure clustering
    quantum_clustering_config_t config = {
        .num_clusters = TEST_NUM_CLUSTERS,
        .input_dim = TEST_INPUT_DIM,
        .quantum_depth = TEST_QUANTUM_DEPTH,
        .algorithm = {
            .type = CLUSTERING_QUANTUM_KMEANS,
            .distance = DISTANCE_QUANTUM_FIDELITY,
            .initialization = INIT_QUANTUM_KMEANS_PLUS_PLUS
        },
        .optimization = {
            .geometric_enhancement = true,
            .error_mitigation = true,
            .convergence = {
                .max_iterations = 100,
                .tolerance = 1e-6
            }
        }
    };

    // Create clustering model
    quantum_clustering_t* model = quantum_clustering_create(&config);
    TEST_ASSERT(model != NULL, "Clustering model creation failed");

    // Verify model parameters
    TEST_ASSERT_EQUAL(model->num_clusters, TEST_NUM_CLUSTERS, "Incorrect number of clusters");
    TEST_ASSERT_EQUAL(model->input_dim, TEST_INPUT_DIM, "Incorrect input dimension");
    TEST_ASSERT_EQUAL(model->quantum_depth, TEST_QUANTUM_DEPTH, "Incorrect quantum depth");

    quantum_clustering_destroy(model);
}

void test_synthetic_data_generation() {
    // Generate synthetic dataset
    dataset_t* data = quantum_generate_synthetic_data(
        TEST_NUM_SAMPLES,
        TEST_INPUT_DIM,
        DATA_TYPE_CLUSTERING
    );
    TEST_ASSERT(data != NULL, "Dataset generation failed");

    // Verify dataset properties
    TEST_ASSERT_EQUAL(data->num_samples, TEST_NUM_SAMPLES, "Incorrect number of samples");
    TEST_ASSERT_EQUAL(data->feature_dim, TEST_INPUT_DIM, "Incorrect feature dimension");
    TEST_ASSERT(data->features != NULL, "Features array is NULL");

    // Verify data ranges and properties
    for (int i = 0; i < TEST_NUM_SAMPLES; i++) {
        for (int j = 0; j < TEST_INPUT_DIM; j++) {
            TEST_ASSERT(data->features[i][j] >= -1.0 && data->features[i][j] <= 1.0,
                       "Feature value out of range");
        }
    }

    quantum_destroy_dataset(data);
}

void test_quantum_state_preparation() {
    // Create quantum system
    quantum_hardware_config_t hw_config = {
        .backend = BACKEND_SIMULATOR,
        .num_qubits = TEST_INPUT_DIM,
        .optimization = {
            .circuit_optimization = true,
            .error_mitigation = true
        }
    };
    quantum_system_t* system = quantum_init_system(&hw_config);
    TEST_ASSERT(system != NULL, "System initialization failed");

    // Generate classical data
    dataset_t* classical_data = quantum_generate_synthetic_data(
        TEST_NUM_SAMPLES, TEST_INPUT_DIM, DATA_TYPE_CLUSTERING
    );
    TEST_ASSERT(classical_data != NULL, "Classical data generation failed");

    // Prepare quantum states
    quantum_dataset_t* quantum_data = quantum_prepare_states(classical_data, system);
    TEST_ASSERT(quantum_data != NULL, "Quantum state preparation failed");

    // Verify quantum dataset properties
    TEST_ASSERT_EQUAL(quantum_data->num_samples, TEST_NUM_SAMPLES,
                     "Incorrect number of quantum states");
    TEST_ASSERT_EQUAL(quantum_data->state_dim, TEST_INPUT_DIM,
                     "Incorrect quantum state dimension");

    // Verify quantum state properties
    for (int i = 0; i < TEST_NUM_SAMPLES; i++) {
        quantum_state_t* state = quantum_data->states[i];
        TEST_ASSERT(state != NULL, "Quantum state is NULL");
        TEST_ASSERT(quantum_is_valid_state(state), "Invalid quantum state");
        TEST_ASSERT_FLOAT_EQUAL(quantum_trace_norm(state), 1.0, 1e-6,
                               "State not normalized");
    }

    // Cleanup
    quantum_destroy_quantum_dataset(quantum_data);
    quantum_destroy_dataset(classical_data);
    quantum_system_destroy(system);
}

void test_distributed_clustering() {
    // Initialize MPI
    int rank = 0, size = 1;
    #ifdef USE_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    #endif

    // Create quantum system
    quantum_hardware_config_t hw_config = {
        .backend = BACKEND_SIMULATOR,
        .num_qubits = TEST_INPUT_DIM,
        .optimization = {
            .circuit_optimization = true,
            .error_mitigation = true
        }
    };
    quantum_system_t* system = quantum_init_system(&hw_config);
    TEST_ASSERT(system != NULL, "System initialization failed");

    // Create clustering model
    quantum_clustering_config_t model_config = {
        .num_clusters = TEST_NUM_CLUSTERS,
        .input_dim = TEST_INPUT_DIM,
        .quantum_depth = TEST_QUANTUM_DEPTH,
        .algorithm = {
            .type = CLUSTERING_QUANTUM_KMEANS,
            .distance = DISTANCE_QUANTUM_FIDELITY,
            .initialization = INIT_QUANTUM_KMEANS_PLUS_PLUS
        },
        .optimization = {
            .geometric_enhancement = true,
            .error_mitigation = true,
            .convergence = {
                .max_iterations = 100,
                .tolerance = 1e-6
            }
        }
    };
    quantum_clustering_t* model = quantum_clustering_create(&model_config);
    TEST_ASSERT(model != NULL, "Model creation failed");

    // Configure distributed computation
    distributed_config_t dist_config = {
        .world_size = size,
        .local_rank = rank,
        .batch_size = TEST_BATCH_SIZE,
        .checkpoint_dir = "/tmp/quantum_geometric/test_checkpoints"
    };
    distributed_manager_t* manager = distributed_manager_create(&dist_config);
    TEST_ASSERT(manager != NULL, "Distributed manager creation failed");

    // Generate and prepare data
    dataset_t* classical_data = quantum_generate_synthetic_data(
        TEST_NUM_SAMPLES, TEST_INPUT_DIM, DATA_TYPE_CLUSTERING
    );
    quantum_dataset_t* quantum_data = quantum_prepare_states(classical_data, system);
    TEST_ASSERT(classical_data != NULL && quantum_data != NULL, "Data preparation failed");

    // Perform clustering
    clustering_result_t result = quantum_cluster_distributed(
        model, quantum_data, manager, NULL
    );
    TEST_ASSERT(result.status == CLUSTERING_SUCCESS, "Clustering failed");

    // Verify clustering results
    if (rank == 0) {
        evaluation_result_t eval = quantum_evaluate_clustering(model, quantum_data);
        
        // Verify basic metrics
        TEST_ASSERT(eval.silhouette_score >= -1.0 && eval.silhouette_score <= 1.0,
                   "Invalid silhouette score");
        TEST_ASSERT(eval.davies_bouldin_index >= 0.0,
                   "Invalid Davies-Bouldin index");
        TEST_ASSERT(eval.quantum_entropy >= 0.0,
                   "Invalid quantum entropy");
        
        // Verify cluster assignments
        cluster_stats_t stats = quantum_calculate_cluster_stats(model, quantum_data);
        int total_assigned = 0;
        for (int i = 0; i < TEST_NUM_CLUSTERS; i++) {
            TEST_ASSERT(stats.cluster_sizes[i] >= 0,
                       "Invalid cluster size");
            total_assigned += stats.cluster_sizes[i];
        }
        TEST_ASSERT_EQUAL(total_assigned, TEST_NUM_SAMPLES,
                         "Not all samples assigned to clusters");
    }

    // Cleanup
    quantum_destroy_quantum_dataset(quantum_data);
    quantum_destroy_dataset(classical_data);
    distributed_manager_destroy(manager);
    quantum_clustering_destroy(model);
    quantum_system_destroy(system);

    #ifdef USE_MPI
    MPI_Finalize();
    #endif
}

void test_cluster_assignment() {
    // Create test model
    quantum_clustering_t* model = create_test_clustering_model(
        TEST_INPUT_DIM, TEST_NUM_CLUSTERS, TEST_QUANTUM_DEPTH
    );
    TEST_ASSERT(model != NULL, "Model creation failed");

    // Create test quantum state
    quantum_state_t* test_state = quantum_create_random_state(TEST_INPUT_DIM);
    TEST_ASSERT(test_state != NULL, "Test state creation failed");

    // Assign cluster
    int cluster_id = quantum_assign_cluster(model, test_state);
    TEST_ASSERT(cluster_id >= 0 && cluster_id < TEST_NUM_CLUSTERS,
                "Invalid cluster assignment");

    // Verify assignment consistency
    int second_assignment = quantum_assign_cluster(model, test_state);
    TEST_ASSERT_EQUAL(cluster_id, second_assignment,
                     "Inconsistent cluster assignment");

    // Cleanup
    quantum_destroy_state(test_state);
    quantum_clustering_destroy(model);
}

void test_clustering_save_load() {
    // Create and train a model
    quantum_clustering_t* model = create_test_clustering_model(
        TEST_INPUT_DIM, TEST_NUM_CLUSTERS, TEST_QUANTUM_DEPTH
    );
    TEST_ASSERT(model != NULL, "Model creation failed");

    // Save model
    const char* save_path = "/tmp/quantum_geometric/test_clustering.qg";
    TEST_ASSERT(quantum_save_model(model, save_path) == 0, "Model saving failed");

    // Load model
    quantum_clustering_t* loaded_model = quantum_load_model(save_path);
    TEST_ASSERT(loaded_model != NULL, "Model loading failed");

    // Compare models
    TEST_ASSERT(clustering_models_equal(model, loaded_model),
                "Loaded model differs from original");

    // Test assignment consistency
    quantum_state_t* test_state = quantum_create_random_state(TEST_INPUT_DIM);
    int original_cluster = quantum_assign_cluster(model, test_state);
    int loaded_cluster = quantum_assign_cluster(loaded_model, test_state);
    TEST_ASSERT_EQUAL(original_cluster, loaded_cluster,
                     "Inconsistent cluster assignment between original and loaded models");

    // Cleanup
    quantum_destroy_state(test_state);
    quantum_clustering_destroy(loaded_model);
    quantum_clustering_destroy(model);
}

int main() {
    // Register tests
    TEST_BEGIN();
    RUN_TEST(test_quantum_clustering_creation);
    RUN_TEST(test_synthetic_data_generation);
    RUN_TEST(test_quantum_state_preparation);
    RUN_TEST(test_distributed_clustering);
    RUN_TEST(test_cluster_assignment);
    RUN_TEST(test_clustering_save_load);
    TEST_END();

    return 0;
}

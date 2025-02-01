#include "test_config.h"
#include "test_matrix_helpers.h"
#include "quantum_geometric/learning/data_loader.h"

// Test helper functions
static void assert_dataset_valid(dataset_t* dataset, size_t expected_samples, 
                               size_t expected_features) {
    TEST_ASSERT(dataset != NULL);
    TEST_ASSERT(dataset->num_samples == expected_samples);
    TEST_ASSERT(dataset->feature_dim == expected_features);
    TEST_ASSERT(dataset->features != NULL);
    TEST_ASSERT(dataset->labels != NULL);
}

// Test cases
void test_csv_loading() {
    TEST_SETUP();
    
    // Create test CSV file
    FILE* fp = fopen("test_data.csv", "w");
    TEST_ASSERT(fp != NULL);
    fprintf(fp, "f1,f2,f3,label\n");
    for (int i = 0; i < 10; i++) {
        fprintf(fp, "%f,%f,%f,%d\n", 
                (float)i/10, (float)(i+1)/10, (float)(i+2)/10, i % 2);
    }
    fclose(fp);

    // Load dataset
    dataset_config_t config = {
        .format = DATA_FORMAT_CSV,
        .csv_config = {
            .delimiter = ",",
            .has_header = true,
            .skip_rows = 0,
            .skip_cols = 0
        },
        .normalize = false
    };

    dataset_t* dataset = quantum_load_dataset("test_data.csv", config);
    assert_dataset_valid(dataset, 10, 3);

    // Verify data
    for (int i = 0; i < 10; i++) {
        TEST_ASSERT_FLOAT_EQ(dataset->features[i][0], (float)i/10);
        TEST_ASSERT_FLOAT_EQ(dataset->features[i][1], (float)(i+1)/10);
        TEST_ASSERT_FLOAT_EQ(dataset->features[i][2], (float)(i+2)/10);
        TEST_ASSERT_FLOAT_EQ(dataset->labels[i], i % 2);
    }

    quantum_dataset_destroy(dataset);
    remove("test_data.csv");
    TEST_TEARDOWN();
}

void test_normalization() {
    TEST_SETUP();
    
    // Create test dataset
    dataset_t* dataset = quantum_create_synthetic_data(100, 5, 2, 0);
    assert_dataset_valid(dataset, 100, 5);

    // Test min-max normalization
    TEST_ASSERT(quantum_normalize_data(dataset, NORMALIZATION_MINMAX));
    
    // Verify normalization
    for (size_t j = 0; j < dataset->feature_dim; j++) {
        float min_val = dataset->features[0][j];
        float max_val = dataset->features[0][j];
        
        for (size_t i = 1; i < dataset->num_samples; i++) {
            if (dataset->features[i][j] < min_val) min_val = dataset->features[i][j];
            if (dataset->features[i][j] > max_val) max_val = dataset->features[i][j];
        }
        
        TEST_ASSERT_FLOAT_EQ(min_val, 0.0f);
        TEST_ASSERT_FLOAT_EQ(max_val, 1.0f);
    }

    quantum_dataset_destroy(dataset);
    TEST_TEARDOWN();
}

void test_dataset_split() {
    TEST_SETUP();
    
    // Create test dataset
    dataset_t* dataset = quantum_create_synthetic_data(1000, 10, 2, 0);
    assert_dataset_valid(dataset, 1000, 10);

    // Split dataset
    dataset_split_t split = quantum_split_dataset(dataset, 0.7f, 0.15f, true, true);
    
    // Verify split sizes
    assert_dataset_valid(split.train_data, 700, 10);
    assert_dataset_valid(split.val_data, 150, 10);
    assert_dataset_valid(split.test_data, 150, 10);

    quantum_dataset_split_destroy(&split);
    quantum_dataset_destroy(dataset);
    TEST_TEARDOWN();
}

void test_performance_monitoring() {
    TEST_SETUP();
    
    // Configure performance monitoring
    performance_config_t perf_config = {
        .num_workers = 4,
        .prefetch_size = 2,
        .pin_memory = true,
        .cache_size = 1024 * 1024,
        .profile = true
    };
    
    TEST_ASSERT(quantum_configure_performance(perf_config));

    // Create and process dataset
    dataset_t* dataset = quantum_create_synthetic_data(10000, 50, 2, 0);
    assert_dataset_valid(dataset, 10000, 50);

    // Get performance metrics
    performance_metrics_t metrics;
    TEST_ASSERT(quantum_get_performance_metrics(&metrics));
    
    // Verify metrics
    TEST_ASSERT(metrics.load_time > 0);
    TEST_ASSERT(metrics.memory_usage > 0);
    TEST_ASSERT(metrics.throughput > 0);

    quantum_dataset_destroy(dataset);
    TEST_TEARDOWN();
}

void test_memory_management() {
    TEST_SETUP();
    
    // Configure memory management
    memory_config_t mem_config = {
        .streaming = true,
        .chunk_size = 1024 * 1024,
        .max_memory = 1024 * 1024 * 1024,
        .gpu_cache = false,
        .compress = false
    };
    
    TEST_ASSERT(quantum_configure_memory(mem_config));

    // Create large dataset
    dataset_t* dataset = quantum_create_synthetic_data(100000, 100, 2, 0);
    assert_dataset_valid(dataset, 100000, 100);

    // Get performance metrics
    performance_metrics_t metrics;
    TEST_ASSERT(quantum_get_performance_metrics(&metrics));
    
    // Verify memory usage is within limits
    TEST_ASSERT(metrics.memory_usage <= mem_config.max_memory);

    quantum_dataset_destroy(dataset);
    TEST_TEARDOWN();
}

void test_quantum_matrix_loading() {
    TEST_SETUP();
    
    // Create a decomposable matrix for testing
    const int size = 32;
    float* matrix = (float*)malloc(size * size * sizeof(float));
    make_decomposable_matrix(matrix, size);

    // Create test dataset file
    FILE* fp = fopen("test_quantum_data.csv", "w");
    TEST_ASSERT(fp != NULL);
    fprintf(fp, "matrix_data\n");
    for (int i = 0; i < size * size; i++) {
        fprintf(fp, "%f\n", matrix[i]);
    }
    fclose(fp);

    // Load dataset with quantum matrix configuration
    dataset_config_t config = {
        .format = DATA_FORMAT_CSV,
        .csv_config = {
            .delimiter = ",",
            .has_header = true
        },
        .normalize = false
    };

    dataset_t* dataset = quantum_load_dataset("test_quantum_data.csv", config);
    assert_dataset_valid(dataset, size, size);

    // Verify matrix properties are preserved
    float* loaded_matrix = (float*)malloc(size * size * sizeof(float));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            loaded_matrix[i * size + j] = dataset->features[i][j];
        }
    }

    // Test matrix is still decomposable
    float* U = (float*)malloc(size * (size/16) * sizeof(float));
    float* V = (float*)malloc((size/16) * size * sizeof(float));
    TEST_ASSERT(quantum_decompose_matrix(loaded_matrix, size, U, V));

    free(matrix);
    free(loaded_matrix);
    free(U);
    free(V);
    quantum_dataset_destroy(dataset);
    remove("test_quantum_data.csv");
    TEST_TEARDOWN();
}

void test_well_conditioned_data() {
    TEST_SETUP();
    
    // Create a well-conditioned matrix
    const int size = 16;
    float* matrix = (float*)malloc(size * size * sizeof(float));
    make_well_conditioned_matrix(matrix, size);

    // Create dataset from matrix
    dataset_t* dataset = quantum_create_synthetic_data(size, size, 0, 0);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            dataset->features[i][j] = matrix[i * size + j];
        }
    }

    // Verify condition number is good
    float condition_number;
    TEST_ASSERT(quantum_compute_condition_number(dataset->features, size, &condition_number));
    TEST_ASSERT(condition_number < 100.0f);

    free(matrix);
    quantum_dataset_destroy(dataset);
    TEST_TEARDOWN();
}

void test_sparse_data_handling() {
    TEST_SETUP();
    
    // Create sparse matrix
    const int size = 64;
    float* matrix = (float*)malloc(size * size * sizeof(float));
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
    make_sparse_matrix(matrix, size, 0.95f); // 95% sparsity

    // Create dataset from sparse matrix
    dataset_t* dataset = quantum_create_synthetic_data(size, size, 0, 0);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            dataset->features[i][j] = matrix[i * size + j];
        }
    }

    // Verify sparsity is preserved
    int num_zeros = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (fabs(dataset->features[i][j]) < TEST_EPSILON) {
                num_zeros++;
            }
        }
    }
    float actual_sparsity = (float)num_zeros / (size * size);
    TEST_ASSERT_FLOAT_EQ(actual_sparsity, 0.95f);

    free(matrix);
    quantum_dataset_destroy(dataset);
    TEST_TEARDOWN();
}

// Register tests
REGISTER_TEST(test_csv_loading);
REGISTER_TEST(test_normalization);
REGISTER_TEST(test_dataset_split);
REGISTER_TEST(test_performance_monitoring);
REGISTER_TEST(test_memory_management);
REGISTER_TEST(test_quantum_matrix_loading);
REGISTER_TEST(test_well_conditioned_data);
REGISTER_TEST(test_sparse_data_handling);

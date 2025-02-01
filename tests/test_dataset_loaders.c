#include <quantum_geometric/learning/data_loader.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Test MNIST dataset loading
static void test_mnist_loading() {
    printf("Testing MNIST dataset loading...\n");
    
    dataset_config_t config = {
        .normalize = true,
        .normalization_method = NORMALIZATION_MINMAX
    };
    
    dataset_t* dataset = quantum_load_mnist(config);
    assert(dataset != NULL);
    
    // Verify dataset properties
    assert(dataset->num_samples == 60000);  // Training set size
    assert(dataset->feature_dim == 784);    // 28x28 pixels
    assert(dataset->num_classes == 10);     // 10 digits
    
    // Verify data ranges after normalization
    for (size_t i = 0; i < dataset->num_samples; i++) {
        // Check features
        for (size_t j = 0; j < dataset->feature_dim; j++) {
            assert(dataset->features[i][j] >= 0.0f);
            assert(dataset->features[i][j] <= 1.0f);
        }
        
        // Check labels
        assert(dataset->labels[i] >= 0.0f);
        assert(dataset->labels[i] <= 9.0f);
    }
    
    quantum_dataset_destroy(dataset);
    printf("MNIST dataset loading test passed\n");
}

// Test CIFAR-10 dataset loading
static void test_cifar10_loading() {
    printf("Testing CIFAR-10 dataset loading...\n");
    
    dataset_config_t config = {
        .normalize = true,
        .normalization_method = NORMALIZATION_MINMAX
    };
    
    dataset_t* dataset = quantum_load_cifar10(config);
    assert(dataset != NULL);
    
    // Verify dataset properties
    assert(dataset->num_samples == 50000);  // Training set size
    assert(dataset->feature_dim == 3072);   // 32x32x3 pixels
    assert(dataset->num_classes == 10);     // 10 classes
    
    // Verify data ranges after normalization
    for (size_t i = 0; i < dataset->num_samples; i++) {
        // Check features
        for (size_t j = 0; j < dataset->feature_dim; j++) {
            assert(dataset->features[i][j] >= 0.0f);
            assert(dataset->features[i][j] <= 1.0f);
        }
        
        // Check labels
        assert(dataset->labels[i] >= 0.0f);
        assert(dataset->labels[i] <= 9.0f);
    }
    
    quantum_dataset_destroy(dataset);
    printf("CIFAR-10 dataset loading test passed\n");
}

// Test UCI dataset loading
static void test_uci_loading() {
    printf("Testing UCI dataset loading...\n");
    
    dataset_config_t config = {
        .normalize = true,
        .normalization_method = NORMALIZATION_ZSCORE
    };
    
    // Test with Iris dataset
    dataset_t* dataset = quantum_load_uci("iris", config);
    assert(dataset != NULL);
    
    // Verify dataset properties
    assert(dataset->num_samples == 150);  // Iris dataset size
    assert(dataset->feature_dim == 4);    // 4 features
    
    // Verify data ranges after z-score normalization
    // For z-score normalized data, most values should be within [-3, 3]
    for (size_t i = 0; i < dataset->num_samples; i++) {
        // Check features
        for (size_t j = 0; j < dataset->feature_dim; j++) {
            assert(dataset->features[i][j] >= -4.0f);  // Allow some margin
            assert(dataset->features[i][j] <= 4.0f);   // Allow some margin
        }
        
        // Check labels (0-2 for Iris classes)
        assert(dataset->labels[i] >= 0.0f);
        assert(dataset->labels[i] <= 2.0f);
    }
    
    quantum_dataset_destroy(dataset);
    printf("UCI dataset loading test passed\n");
}

// Test dataset splitting
static void test_dataset_splitting() {
    printf("Testing dataset splitting...\n");
    
    // Create synthetic dataset
    dataset_t* dataset = quantum_create_synthetic_data(1000, 10, 2, SYNTHETIC_DATA_CLASSIFICATION);
    assert(dataset != NULL);
    
    // Split dataset
    dataset_split_t split = quantum_split_dataset(
        dataset,
        0.7f,   // train ratio
        0.15f,  // validation ratio
        true,   // shuffle
        true    // stratify
    );
    
    // Verify split properties
    assert(split.train_data != NULL);
    assert(split.val_data != NULL);
    assert(split.test_data != NULL);
    
    // Verify split sizes
    assert(split.train_data->num_samples == 700);
    assert(split.val_data->num_samples == 150);
    assert(split.test_data->num_samples == 150);
    
    // Verify feature dimensions
    assert(split.train_data->feature_dim == dataset->feature_dim);
    assert(split.val_data->feature_dim == dataset->feature_dim);
    assert(split.test_data->feature_dim == dataset->feature_dim);
    
    // Verify class distribution if stratified
    if (dataset->num_classes > 0) {
        float train_class_ratio = 0.0f;
        float val_class_ratio = 0.0f;
        float test_class_ratio = 0.0f;
        
        // Count class 0 instances in each split
        for (size_t i = 0; i < split.train_data->num_samples; i++) {
            if (split.train_data->labels[i] == 0) train_class_ratio += 1.0f;
        }
        for (size_t i = 0; i < split.val_data->num_samples; i++) {
            if (split.val_data->labels[i] == 0) val_class_ratio += 1.0f;
        }
        for (size_t i = 0; i < split.test_data->num_samples; i++) {
            if (split.test_data->labels[i] == 0) test_class_ratio += 1.0f;
        }
        
        train_class_ratio /= split.train_data->num_samples;
        val_class_ratio /= split.val_data->num_samples;
        test_class_ratio /= split.test_data->num_samples;
        
        // Class ratios should be approximately equal
        assert(fabsf(train_class_ratio - val_class_ratio) < 0.1f);
        assert(fabsf(train_class_ratio - test_class_ratio) < 0.1f);
    }
    
    quantum_dataset_split_destroy(&split);
    quantum_dataset_destroy(dataset);
    printf("Dataset splitting test passed\n");
}

int main() {
    printf("Running dataset loader tests...\n\n");
    
    // Run all tests
    test_mnist_loading();
    test_cifar10_loading();
    test_uci_loading();
    test_dataset_splitting();
    
    printf("\nAll dataset loader tests passed successfully!\n");
    return 0;
}

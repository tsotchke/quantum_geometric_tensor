/**
 * @file test_error_prediction.c
 * @brief Tests for error prediction functionality
 */

#include "quantum_geometric/physics/error_prediction.h"
#include "quantum_geometric/physics/error_patterns.h"
#include "quantum_geometric/physics/error_correlation.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

// Test helper functions
static void test_initialization(void);
static void test_prediction_generation(void);
static void test_prediction_verification(void);
static void test_model_updates(void);
static void test_error_cases(void);
static void test_performance_requirements(void);

// Mock functions and data
static quantum_state* create_test_state(void);
static ErrorPattern* create_test_patterns(size_t* num_patterns);
static ErrorCorrelation* create_test_correlations(size_t* num_correlations);
static void cleanup_test_data(quantum_state* state,
                            ErrorPattern* patterns,
                            ErrorCorrelation* correlations);

int main(void) {
    printf("Running error prediction tests...\n");

    // Run all tests
    test_initialization();
    test_prediction_generation();
    test_prediction_verification();
    test_model_updates();
    test_error_cases();
    test_performance_requirements();

    printf("All error prediction tests passed!\n");
    return 0;
}

static void test_initialization(void) {
    printf("Testing initialization...\n");

    // Test valid initialization
    PredictionState state;
    PredictionConfig config = {
        .max_predictions = 100,
        .history_length = 1000,
        .confidence_threshold = 0.7,
        .temporal_weight = 1.0,
        .min_success_rate = 0.8
    };

    bool success = init_error_prediction(&state, &config);
    assert(success);
    assert(state.predictions != NULL);
    assert(state.history != NULL);
    assert(state.num_predictions == 0);
    assert(state.max_predictions == config.max_predictions);
    assert(state.history_length == config.history_length);

    // Test cleanup
    cleanup_error_prediction(&state);

    // Test invalid parameters
    success = init_error_prediction(NULL, &config);
    assert(!success);
    success = init_error_prediction(&state, NULL);
    assert(!success);

    printf("Initialization tests passed\n");
}

static void test_prediction_generation(void) {
    printf("Testing prediction generation...\n");

    // Initialize prediction system
    PredictionState state;
    PredictionConfig config = {
        .max_predictions = 100,
        .history_length = 1000,
        .confidence_threshold = 0.7,
        .temporal_weight = 1.0,
        .min_success_rate = 0.8
    };
    bool success = init_error_prediction(&state, &config);
    assert(success);

    // Create test data
    size_t num_patterns, num_correlations;
    ErrorPattern* patterns = create_test_patterns(&num_patterns);
    ErrorCorrelation* correlations = create_test_correlations(&num_correlations);

    // Generate predictions
    size_t num_predictions = predict_errors(&state, patterns, num_patterns,
                                          correlations, num_correlations);
    assert(num_predictions > 0);
    assert(num_predictions <= state.max_predictions);

    // Verify prediction properties
    for (size_t i = 0; i < num_predictions; i++) {
        const ErrorPrediction* prediction = get_prediction(&state, i);
        assert(prediction != NULL);
        assert(prediction->confidence >= config.confidence_threshold);
        assert(prediction->pattern_id < num_patterns);
        assert(prediction->correlation_id < num_correlations);
    }

    // Cleanup
    cleanup_error_prediction(&state);
    free(patterns);
    free(correlations);

    printf("Prediction generation tests passed\n");
}

static void test_prediction_verification(void) {
    printf("Testing prediction verification...\n");

    // Initialize prediction system
    PredictionState state;
    PredictionConfig config = {
        .max_predictions = 100,
        .history_length = 1000,
        .confidence_threshold = 0.7,
        .temporal_weight = 1.0,
        .min_success_rate = 0.8
    };
    bool success = init_error_prediction(&state, &config);
    assert(success);

    // Create test data
    quantum_state* test_state = create_test_state();
    size_t num_patterns, num_correlations;
    ErrorPattern* patterns = create_test_patterns(&num_patterns);
    ErrorCorrelation* correlations = create_test_correlations(&num_correlations);

    // Generate and verify predictions
    size_t num_predictions = predict_errors(&state, patterns, num_patterns,
                                          correlations, num_correlations);
    assert(num_predictions > 0);

    success = verify_predictions(&state, test_state);
    assert(success);

    // Check success rate is within reasonable bounds
    double success_rate = get_prediction_success_rate(&state);
    assert(success_rate >= 0.0 && success_rate <= 1.0);

    // Cleanup
    cleanup_error_prediction(&state);
    cleanup_test_data(test_state, patterns, correlations);

    printf("Prediction verification tests passed\n");
}

static void test_model_updates(void) {
    printf("Testing model updates...\n");

    // Initialize prediction system
    PredictionState state;
    PredictionConfig config = {
        .max_predictions = 100,
        .history_length = 1000,
        .confidence_threshold = 0.7,
        .temporal_weight = 1.0,
        .min_success_rate = 0.8
    };
    bool success = init_error_prediction(&state, &config);
    assert(success);

    // Create test data
    quantum_state* test_state = create_test_state();
    size_t num_patterns, num_correlations;
    ErrorPattern* patterns = create_test_patterns(&num_patterns);
    ErrorCorrelation* correlations = create_test_correlations(&num_correlations);

    // Run multiple prediction cycles
    for (size_t i = 0; i < 10; i++) {
        size_t num_predictions = predict_errors(&state, patterns, num_patterns,
                                              correlations, num_correlations);
        assert(num_predictions > 0);

        success = verify_predictions(&state, test_state);
        assert(success);

        success = update_prediction_model(&state);
        assert(success);

        // Verify model parameters remain in valid ranges
        assert(state.config.confidence_threshold >= 0.5);
        assert(state.config.confidence_threshold <= 0.95);
        assert(state.config.temporal_weight >= 0.5);
        assert(state.config.temporal_weight <= 2.0);
    }

    // Cleanup
    cleanup_error_prediction(&state);
    cleanup_test_data(test_state, patterns, correlations);

    printf("Model update tests passed\n");
}

static void test_error_cases(void) {
    printf("Testing error cases...\n");

    PredictionState state;
    PredictionConfig config = {
        .max_predictions = 100,
        .history_length = 1000,
        .confidence_threshold = 0.7,
        .temporal_weight = 1.0,
        .min_success_rate = 0.8
    };

    // Test NULL parameters
    size_t num_patterns = 1, num_correlations = 1;
    ErrorPattern patterns[1] = {0};
    ErrorCorrelation correlations[1] = {0};
    quantum_state test_state = {0};

    size_t result = predict_errors(NULL, patterns, num_patterns,
                                 correlations, num_correlations);
    assert(result == 0);

    result = predict_errors(&state, NULL, num_patterns,
                          correlations, num_correlations);
    assert(result == 0);

    bool success = verify_predictions(NULL, &test_state);
    assert(!success);

    success = verify_predictions(&state, NULL);
    assert(!success);

    success = update_prediction_model(NULL);
    assert(!success);

    // Test invalid indices
    assert(get_prediction(&state, 1000) == NULL);

    printf("Error case tests passed\n");
}

static void test_performance_requirements(void) {
    printf("Testing performance requirements...\n");

    // Initialize prediction system
    PredictionState state;
    PredictionConfig config = {
        .max_predictions = 1000,  // Large number for stress testing
        .history_length = 10000,
        .confidence_threshold = 0.7,
        .temporal_weight = 1.0,
        .min_success_rate = 0.8
    };
    bool success = init_error_prediction(&state, &config);
    assert(success);

    // Create large test dataset
    quantum_state* test_state = create_test_state();
    size_t num_patterns = 1000, num_correlations = 1000;
    ErrorPattern* patterns = malloc(num_patterns * sizeof(ErrorPattern));
    ErrorCorrelation* correlations = malloc(num_correlations * sizeof(ErrorCorrelation));
    assert(patterns && correlations);

    // Measure prediction time
    clock_t start = clock();
    size_t num_predictions = predict_errors(&state, patterns, num_patterns,
                                          correlations, num_correlations);
    clock_t end = clock();
    double prediction_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Verify meets latency requirement (10μs per prediction)
    assert(prediction_time / num_predictions < 0.00001);

    // Measure verification time
    start = clock();
    success = verify_predictions(&state, test_state);
    end = clock();
    double verification_time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Verify meets latency requirement (100μs total)
    assert(verification_time < 0.0001);

    // Cleanup
    cleanup_error_prediction(&state);
    free(test_state);
    free(patterns);
    free(correlations);

    printf("Performance requirement tests passed\n");
}

// Mock implementation of test helpers
static quantum_state* create_test_state(void) {
    quantum_state* state = malloc(sizeof(quantum_state));
    // Initialize with test data
    return state;
}

static ErrorPattern* create_test_patterns(size_t* num_patterns) {
    *num_patterns = 10;
    ErrorPattern* patterns = malloc(*num_patterns * sizeof(ErrorPattern));
    // Initialize with test patterns
    return patterns;
}

static ErrorCorrelation* create_test_correlations(size_t* num_correlations) {
    *num_correlations = 10;
    ErrorCorrelation* correlations = malloc(*num_correlations * sizeof(ErrorCorrelation));
    // Initialize with test correlations
    return correlations;
}

static void cleanup_test_data(quantum_state* state,
                            ErrorPattern* patterns,
                            ErrorCorrelation* correlations) {
    free(state);
    free(patterns);
    free(correlations);
}

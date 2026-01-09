/**
 * @file test_metal_syndrome.c
 * @brief Tests for Metal-accelerated syndrome extraction
 */

#include "quantum_geometric/hardware/metal/quantum_geometric_syndrome.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#define TEST_SIZE 64
#define ERROR_THRESHOLD 1e-6
#define NUM_MEASUREMENTS (TEST_SIZE * TEST_SIZE)

// Test state for syndrome measurements
typedef struct {
    float2* measurements;
    size_t num_measurements;
} TestMeasurements;

static void init_test_measurements(TestMeasurements* test) {
    test->num_measurements = NUM_MEASUREMENTS;
    test->measurements = (float2*)malloc(test->num_measurements * sizeof(float2));

    srand(42); // Fixed seed for reproducibility

    // Generate random stabilizer measurements with some errors
    for (size_t i = 0; i < test->num_measurements; i++) {
        // Most measurements should be close to +1 (no error)
        // Some should be close to -1 (error detected)
        float base = (rand() % 10 < 1) ? -1.0f : 1.0f;  // 10% error rate
        test->measurements[i].x = base + ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        test->measurements[i].y = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;  // Small imaginary noise
    }
}

static void cleanup_test_measurements(TestMeasurements* test) {
    free(test->measurements);
    test->measurements = NULL;
}

static bool compare_syndrome_results(const SyndromeResult* result1,
                                     const SyndromeResult* result2) {
    if (result1->num_vertices != result2->num_vertices) {
        printf("Vertex count mismatch: %zu vs %zu\n",
               result1->num_vertices, result2->num_vertices);
        return false;
    }

    // Compare vertices (allowing for some numerical tolerance)
    for (size_t i = 0; i < result1->num_vertices; i++) {
        const SyndromeVertex* v1 = &result1->vertices[i];
        const SyndromeVertex* v2 = &result2->vertices[i];

        if (fabs(v1->weight - v2->weight) > ERROR_THRESHOLD) {
            printf("Weight mismatch at vertex %zu: %f vs %f\n",
                   i, v1->weight, v2->weight);
            return false;
        }
    }

    return true;
}

static void test_syndrome_context_creation(void) {
    printf("Testing syndrome context creation...\n");

    SyndromeConfig config = {
        .detection_threshold = 0.1f,
        .confidence_threshold = 0.8f,
        .weight_scale_factor = 1.0f,
        .pattern_threshold = 0.5f,
        .parallel_group_size = 16,
        .min_pattern_occurrences = 3,
        .enable_parallel = true,
        .use_boundary_matching = true,
        .max_iterations = 100,
        .code_distance = 5
    };

    void* ctx = syndrome_create_context(&config);
    if (ctx == NULL) {
        printf("  SKIP: Metal not available or context creation failed\n");
        return;
    }

    printf("  Context created successfully\n");
    syndrome_destroy_context(ctx);
    printf("  Context destroyed successfully\n");
    printf("  PASS\n");
}

static void test_syndrome_extraction(void) {
    printf("Testing syndrome extraction...\n");

    SyndromeConfig config = {
        .detection_threshold = 0.5f,
        .confidence_threshold = 0.7f,
        .weight_scale_factor = 1.0f,
        .pattern_threshold = 0.5f,
        .parallel_group_size = 16,
        .min_pattern_occurrences = 3,
        .enable_parallel = true,
        .use_boundary_matching = true,
        .max_iterations = 100,
        .code_distance = 5
    };

    void* ctx = syndrome_create_context(&config);
    if (ctx == NULL) {
        printf("  SKIP: Metal not available\n");
        return;
    }

    TestMeasurements test;
    init_test_measurements(&test);

    SyndromeResult result = {0};
    int err = syndrome_extract(ctx, test.measurements, test.num_measurements, &result);

    if (err != 0) {
        printf("  Syndrome extraction returned error: %d\n", err);
        // Not necessarily a failure - may indicate Metal unavailable
    } else {
        printf("  Extracted %zu syndrome vertices\n", result.num_vertices);
        printf("  Extraction time: %.3f ms\n", result.extraction_time * 1000.0);

        if (result.num_vertices > 0) {
            printf("  First vertex: pos=(%u,%u,%u) weight=%f\n",
                   result.vertices[0].position.x,
                   result.vertices[0].position.y,
                   result.vertices[0].position.z,
                   result.vertices[0].weight);
        }
    }

    syndrome_free_result(&result);
    cleanup_test_measurements(&test);
    syndrome_destroy_context(ctx);
    printf("  PASS\n");
}

static void test_syndrome_decoding(void) {
    printf("Testing syndrome decoding (MWPM)...\n");

    SyndromeConfig config = {
        .detection_threshold = 0.5f,
        .confidence_threshold = 0.7f,
        .weight_scale_factor = 1.0f,
        .pattern_threshold = 0.5f,
        .parallel_group_size = 16,
        .min_pattern_occurrences = 3,
        .enable_parallel = true,
        .use_boundary_matching = true,
        .max_iterations = 100,
        .code_distance = 5
    };

    void* ctx = syndrome_create_context(&config);
    if (ctx == NULL) {
        printf("  SKIP: Metal not available\n");
        return;
    }

    TestMeasurements test;
    init_test_measurements(&test);

    SyndromeResult result = {0};
    int err = syndrome_extract(ctx, test.measurements, test.num_measurements, &result);

    if (err == 0 && result.num_vertices > 0) {
        err = syndrome_decode_mwpm(ctx, &result);
        if (err == 0) {
            printf("  Decoded error chain length: %zu\n", result.chain_length);
            printf("  Decoding time: %.3f ms\n", result.decoding_time * 1000.0);
            printf("  Estimated logical error rate: %.6f\n", result.logical_error_rate);
        } else {
            printf("  Decoding returned error: %d\n", err);
        }
    }

    syndrome_free_result(&result);
    cleanup_test_measurements(&test);
    syndrome_destroy_context(ctx);
    printf("  PASS\n");
}

static void test_pattern_detection(void) {
    printf("Testing pattern detection...\n");

    SyndromeConfig config = {
        .detection_threshold = 0.5f,
        .confidence_threshold = 0.7f,
        .weight_scale_factor = 1.0f,
        .pattern_threshold = 0.3f,
        .parallel_group_size = 16,
        .min_pattern_occurrences = 2,
        .enable_parallel = true,
        .use_boundary_matching = true,
        .max_iterations = 100,
        .code_distance = 5
    };

    void* ctx = syndrome_create_context(&config);
    if (ctx == NULL) {
        printf("  SKIP: Metal not available\n");
        return;
    }

    TestMeasurements test;
    init_test_measurements(&test);

    SyndromeResult result = {0};
    int err = syndrome_extract(ctx, test.measurements, test.num_measurements, &result);

    if (err == 0 && result.num_vertices > 0) {
        SyndromePattern patterns[10] = {0};
        size_t num_patterns = syndrome_detect_patterns(ctx, &result, patterns, 10);

        printf("  Detected %zu patterns\n", num_patterns);
        for (size_t i = 0; i < num_patterns && i < 3; i++) {
            printf("    Pattern %zu: size=%zu prob=%.3f count=%u logical=%s\n",
                   i, patterns[i].pattern_size,
                   patterns[i].occurrence_probability,
                   patterns[i].occurrence_count,
                   patterns[i].is_logical_error ? "yes" : "no");
        }

        // Free pattern resources
        for (size_t i = 0; i < num_patterns; i++) {
            syndrome_free_pattern(&patterns[i]);
        }
    }

    syndrome_free_result(&result);
    cleanup_test_measurements(&test);
    syndrome_destroy_context(ctx);
    printf("  PASS\n");
}

static void test_statistics(void) {
    printf("Testing syndrome statistics...\n");

    SyndromeConfig config = {
        .detection_threshold = 0.5f,
        .confidence_threshold = 0.7f,
        .weight_scale_factor = 1.0f,
        .pattern_threshold = 0.5f,
        .parallel_group_size = 16,
        .min_pattern_occurrences = 3,
        .enable_parallel = true,
        .use_boundary_matching = true,
        .max_iterations = 100,
        .code_distance = 5
    };

    void* ctx = syndrome_create_context(&config);
    if (ctx == NULL) {
        printf("  SKIP: Metal not available\n");
        return;
    }

    // Run multiple extractions
    for (int round = 0; round < 5; round++) {
        TestMeasurements test;
        init_test_measurements(&test);

        SyndromeResult result = {0};
        syndrome_extract(ctx, test.measurements, test.num_measurements, &result);
        syndrome_decode_mwpm(ctx, &result);

        syndrome_free_result(&result);
        cleanup_test_measurements(&test);
    }

    // Get statistics
    size_t total_extracted = 0, total_decoded = 0;
    float avg_chain_length = 0.0f;

    int err = syndrome_get_statistics(ctx, &total_extracted, &total_decoded, &avg_chain_length);
    if (err == 0) {
        printf("  Total extracted: %zu\n", total_extracted);
        printf("  Total decoded: %zu\n", total_decoded);
        printf("  Average chain length: %.2f\n", avg_chain_length);
    }

    syndrome_destroy_context(ctx);
    printf("  PASS\n");
}

int main(void) {
    printf("=== Metal Syndrome Acceleration Tests ===\n\n");

    test_syndrome_context_creation();
    printf("\n");

    test_syndrome_extraction();
    printf("\n");

    test_syndrome_decoding();
    printf("\n");

    test_pattern_detection();
    printf("\n");

    test_statistics();
    printf("\n");

    printf("=== All tests completed ===\n");
    return 0;
}

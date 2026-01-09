#include "quantum_geometric/hardware/metal/quantum_geometric_metal.h"
#include "test_helpers.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Test fixtures
static void* metal_ctx;
static StabilizerQubit* test_qubits;
static uint32_t* stabilizer_indices;
static StabilizerConfig config;
static CorrectionConfig correction_config;
static float2* results;

void setUp(void) {
    // Initialize Metal context
    metal_initialize();
    metal_ctx = metal_create_context(0);
    assert(metal_ctx != NULL);
    
    // Initialize test qubits (4x4 lattice)
    test_qubits = malloc(16 * sizeof(StabilizerQubit));
    for (int i = 0; i < 16; i++) {
        test_qubits[i].amplitude = (float2){1.0f, 0.0f};  // Initialize to |0⟩
        test_qubits[i].error_rate = 0.01f;
        test_qubits[i].flags = 0;
    }
    
    // Initialize stabilizer config
    config.type = 2;  // Z stabilizer
    config.num_qubits = 4;  // Each stabilizer acts on 4 qubits
    config.weight = 1.0f;
    config.confidence = 0.9f;

    // Initialize correction config
    correction_config.decoder_type = 0;  // MWPM decoder
    correction_config.max_iterations = 100;
    correction_config.success_threshold = 0.9f;
    correction_config.adaptive = true;
    
    // Initialize stabilizer indices (3x3 stabilizers)
    stabilizer_indices = malloc(9 * 4 * sizeof(uint32_t));
    for (int i = 0; i < 9; i++) {
        // Create plaquette indices
        int x = i % 3;
        int y = i / 3;
        stabilizer_indices[i * 4 + 0] = (y * 4 + x);
        stabilizer_indices[i * 4 + 1] = (y * 4 + x + 1);
        stabilizer_indices[i * 4 + 2] = ((y + 1) * 4 + x);
        stabilizer_indices[i * 4 + 3] = ((y + 1) * 4 + x + 1);
    }
    
    // Initialize results buffer
    results = malloc(9 * sizeof(float2));
    memset(results, 0, 9 * sizeof(float2));
}

void tearDown(void) {
    free(test_qubits);
    free(stabilizer_indices);
    free(results);
    metal_destroy_context(metal_ctx);
    metal_cleanup();
}

// Test Metal stabilizer measurement
void test_metal_stabilizer_measure(void) {
    printf("Testing Metal stabilizer measurement...\n");

    // Inject test errors
    test_qubits[2].amplitude = (float2){-1.0f, 0.0f};  // Flip qubit 2
    test_qubits[6].amplitude = (float2){-1.0f, 0.0f};  // Flip qubit 6

    // Perform Metal-accelerated measurement
    assert(metal_measure_stabilizers(metal_ctx, test_qubits,
                                     stabilizer_indices, &config,
                                     results, 9) == 0);

    // Verify error detection
    assert(fabs(results[0].x - (-1.0f)) <= 0.1f);  // Should detect error
    assert(fabs(results[4].x - 1.0f) <= 0.1f);     // Should be error-free

    printf("Metal stabilizer measurement test passed\n");
}

// Test Metal error correction
void test_metal_error_correction(void) {
    printf("Testing Metal error correction...\n");

    // Create test state with phase errors
    for (int i = 0; i < 16; i++) {
        float phase = (float)i * M_PI / 8.0f;
        test_qubits[i].amplitude = (float2){cosf(phase), sinf(phase)};
        test_qubits[i].error_rate = 0.05f;
    }

    // Measure stabilizers
    assert(metal_measure_stabilizers(metal_ctx, test_qubits,
                                     stabilizer_indices, &config,
                                     results, 9) == 0);

    // Apply error correction
    assert(metal_apply_correction(metal_ctx, test_qubits,
                                  results, 9,
                                  &correction_config) == 0);

    // Verify error rates reduced
    for (int i = 0; i < 16; i++) {
        assert(fabs(test_qubits[i].error_rate - 0.0f) <= 0.01f);
    }

    printf("Metal error correction test passed\n");
}

// Test Metal stabilizer correlation
void test_metal_stabilizer_correlation(void) {
    printf("Testing Metal stabilizer correlation...\n");

    // Create correlated errors
    test_qubits[2].amplitude = (float2){-1.0f, 0.0f};
    test_qubits[3].amplitude = (float2){-1.0f, 0.0f};

    // Allocate correlation matrix
    float* correlations = malloc(9 * 9 * sizeof(float));

    // Compute correlations
    assert(metal_compute_correlations(metal_ctx, test_qubits,
                                      stabilizer_indices, 9,
                                      correlations) == 0);

    // Verify correlation between adjacent stabilizers
    int idx1 = 0;  // First stabilizer affected by error
    int idx2 = 1;  // Second stabilizer affected by error
    assert(fabs(correlations[idx1 * 9 + idx2] - 0.8f) <= 0.2f);

    free(correlations);

    printf("Metal stabilizer correlation test passed\n");
}

// Test parallel measurement groups
void test_metal_parallel_measurement(void) {
    printf("Testing Metal parallel measurement...\n");

    // Initialize test qubits
    for (int i = 0; i < 16; i++) {
        test_qubits[i].amplitude = (float2){1.0f, 0.0f};
        test_qubits[i].error_rate = 0.01f;
    }

    // Perform measurements in parallel
    assert(metal_measure_stabilizers(metal_ctx, test_qubits,
                                     stabilizer_indices, &config,
                                     results, 9) == 0);

    // Verify non-interference between parallel measurements
    for (int i = 0; i < 9; i += 4) {  // Groups of 4
        for (int j = 0; j < 4 && (i + j) < 9; j++) {
            assert(fabs(results[i + j].x - 1.0f) <= 0.1f);
        }
    }

    printf("Metal parallel measurement test passed\n");
}

int main(void) {
    printf("Running Metal stabilizer tests...\n\n");

    setUp();

    test_metal_stabilizer_measure();
    test_metal_error_correction();
    test_metal_stabilizer_correlation();
    test_metal_parallel_measurement();

    tearDown();

    printf("\nAll Metal stabilizer tests passed!\n");
    return 0;
}

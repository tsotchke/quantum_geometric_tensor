#include "quantum_geometric/hardware/metal/quantum_geometric_syndrome.h"
#include "quantum_geometric/physics/error_syndrome.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include "quantum_geometric/core/performance_monitor.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TEST_SIZE 64
#define ERROR_THRESHOLD 1e-6

static void init_test_state(quantum_state* state) {
    state->size = TEST_SIZE * TEST_SIZE * TEST_SIZE;
    state->amplitudes = (float2*)malloc(state->size * sizeof(float2));
    state->indices = (uint32_t*)malloc(state->size * sizeof(uint32_t));
    
    srand(42); // Fixed seed for reproducibility
    
    for (size_t i = 0; i < state->size; i++) {
        state->amplitudes[i].x = (float)rand() / RAND_MAX;
        state->amplitudes[i].y = (float)rand() / RAND_MAX;
        state->indices[i] = i;
    }
}

static void cleanup_test_state(quantum_state* state) {
    free(state->amplitudes);
    free(state->indices);
}

static bool compare_results(const MatchingGraph* metal_graph,
                          const MatchingGraph* cpu_graph) {
    if (metal_graph->num_vertices != cpu_graph->num_vertices) {
        printf("Vertex count mismatch: Metal=%zu, CPU=%zu\n",
               metal_graph->num_vertices, cpu_graph->num_vertices);
        return false;
    }
    
    // Compare vertices
    for (size_t i = 0; i < metal_graph->num_vertices; i++) {
        const SyndromeVertex* mv = &metal_graph->vertices[i];
        const SyndromeVertex* cv = &cpu_graph->vertices[i];
        
        if (fabs(mv->weight - cv->weight) > ERROR_THRESHOLD) {
            printf("Weight mismatch at vertex %zu: Metal=%f, CPU=%f\n",
                   i, mv->weight, cv->weight);
            return false;
        }
        
        if (mv->x != cv->x || mv->y != cv->y || mv->z != cv->z) {
            printf("Coordinate mismatch at vertex %zu\n", i);
            return false;
        }
    }
    
    // Compare correlation matrix
    for (size_t i = 0; i < metal_graph->num_vertices; i++) {
        for (size_t j = 0; j < metal_graph->num_vertices; j++) {
            float metal_corr = metal_graph->correlation_matrix[i * metal_graph->num_vertices + j];
            float cpu_corr = cpu_graph->correlation_matrix[i * cpu_graph->num_vertices + j];
            
            if (fabs(metal_corr - cpu_corr) > ERROR_THRESHOLD) {
                printf("Correlation mismatch at (%zu,%zu): Metal=%f, CPU=%f\n",
                       i, j, metal_corr, cpu_corr);
                return false;
            }
        }
    }
    
    return true;
}

static void print_performance_comparison(const char* operation,
                                      double metal_time,
                                      double cpu_time) {
    printf("%s Performance:\n", operation);
    printf("  Metal: %.3f ms\n", metal_time * 1000.0);
    printf("  CPU:   %.3f ms\n", cpu_time * 1000.0);
    printf("  Speedup: %.2fx\n", cpu_time / metal_time);
}

int main() {
    printf("Testing Metal Syndrome Acceleration...\n");
    
    // Initialize Metal
    if (!init_metal_syndrome_resources()) {
        printf("Failed to initialize Metal resources\n");
        return 1;
    }
    
    // Create test state
    quantum_state state;
    init_test_state(&state);
    
    // Create test config
    SyndromeConfig config = {
        .detection_threshold = 0.1f,
        .confidence_threshold = 0.8f,
        .weight_scale_factor = 1.0f,
        .pattern_threshold = 0.5f,
        .parallel_group_size = 16,
        .min_pattern_occurrences = 3,
        .enable_parallel = true,
        .use_boundary_matching = true
    };
    
    // Create graphs for Metal and CPU
    MatchingGraph* metal_graph = init_matching_graph(TEST_SIZE * TEST_SIZE, TEST_SIZE * TEST_SIZE);
    MatchingGraph* cpu_graph = init_matching_graph(TEST_SIZE * TEST_SIZE, TEST_SIZE * TEST_SIZE);
    
    if (!metal_graph || !cpu_graph) {
        printf("Failed to create matching graphs\n");
        return 1;
    }
    
    // Test syndrome extraction
    printf("\nTesting syndrome extraction...\n");
    
    uint64_t metal_start = get_performance_counter();
    size_t metal_syndromes = extract_syndromes_metal(&state, &config, metal_graph);
    double metal_time = get_performance_elapsed(metal_start);
    
    uint64_t cpu_start = get_performance_counter();
    size_t cpu_syndromes = extract_error_syndromes(&state, &config, cpu_graph);
    double cpu_time = get_performance_elapsed(cpu_start);
    
    printf("Syndromes found: Metal=%zu, CPU=%zu\n", metal_syndromes, cpu_syndromes);
    print_performance_comparison("Syndrome Extraction", metal_time, cpu_time);
    
    // Test correlation computation
    printf("\nTesting correlation computation...\n");
    
    metal_start = get_performance_counter();
    bool metal_corr = compute_syndrome_correlations_metal(metal_graph, &config);
    metal_time = get_performance_elapsed(metal_start);
    
    cpu_start = get_performance_counter();
    bool cpu_corr = find_minimum_weight_matching(cpu_graph, &config);
    cpu_time = get_performance_elapsed(cpu_start);
    
    printf("Correlation computation: Metal=%s, CPU=%s\n",
           metal_corr ? "success" : "failed",
           cpu_corr ? "success" : "failed");
    print_performance_comparison("Correlation Computation", metal_time, cpu_time);
    
    // Compare results
    printf("\nComparing results...\n");
    if (compare_results(metal_graph, cpu_graph)) {
        printf("Results match between Metal and CPU implementations\n");
    } else {
        printf("Results differ between Metal and CPU implementations\n");
    }
    
    // Print performance metrics
    printf("\nPerformance Metrics:\n");
    printf("Metal Syndrome Count: %zu\n", get_metal_syndrome_count());
    printf("Metal Resource Usage: %.1f%%\n", get_metal_resource_usage() * 100.0f);
    printf("Metal Extraction Time: %.3f ms\n", get_metal_syndrome_extraction_time() * 1000.0);
    printf("Metal Correlation Time: %.3f ms\n", get_metal_correlation_time() * 1000.0);
    
    // Cleanup
    cleanup_test_state(&state);
    cleanup_matching_graph(metal_graph);
    cleanup_matching_graph(cpu_graph);
    cleanup_metal_syndrome_resources();
    
    printf("\nTest complete\n");
    return 0;
}

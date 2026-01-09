/**
 * @file quantum_geometric_syndrome.h
 * @brief Metal-accelerated syndrome extraction and error decoding
 *
 * Provides GPU-accelerated syndrome extraction, pattern detection,
 * and minimum weight perfect matching for quantum error correction.
 */

#ifndef QUANTUM_GEOMETRIC_SYNDROME_H
#define QUANTUM_GEOMETRIC_SYNDROME_H

#include "quantum_geometric/hardware/metal/quantum_geometric_metal.h"
#include "quantum_geometric/core/error_codes.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===========================================================================
// Syndrome Types and Structures
// ===========================================================================

/**
 * @brief 3D coordinate for syndrome vertex position
 */
typedef struct SyndromeCoord {
    uint32_t x;
    uint32_t y;
    uint32_t z;  // Time layer for measurement rounds
} SyndromeCoord;

/**
 * @brief Syndrome vertex for error correction graphs
 */
typedef struct SyndromeVertex {
    SyndromeCoord position;         // Position in syndrome graph
    float weight;                   // Vertex weight for matching
    bool is_boundary;               // Whether this is a boundary vertex
    uint32_t timestamp;             // Measurement round
    float confidence;               // Detection confidence
    float correlation_weight;       // Weight from error correlations
    bool part_of_chain;             // Part of error chain
    uint32_t matched_to;            // Index of matched vertex (-1 if unmatched)
} SyndromeVertex;

/**
 * @brief Edge in syndrome matching graph
 */
typedef struct SyndromeEdge {
    uint32_t vertex1;               // First vertex index
    uint32_t vertex2;               // Second vertex index
    float weight;                   // Edge weight (error probability)
    bool is_boundary_edge;          // Connects to boundary
    float correlation;              // Error correlation coefficient
} SyndromeEdge;

/**
 * @brief Configuration for syndrome extraction
 */
typedef struct SyndromeConfig {
    float detection_threshold;      // Threshold for syndrome detection
    float confidence_threshold;     // Minimum confidence for valid syndrome
    float weight_scale_factor;      // Scaling factor for edge weights
    float pattern_threshold;        // Threshold for pattern detection
    uint32_t parallel_group_size;   // Number of syndromes per GPU group
    uint32_t min_pattern_occurrences; // Minimum occurrences for pattern
    bool enable_parallel;           // Enable parallel extraction
    bool use_boundary_matching;     // Use boundary for unmatched defects
    uint32_t max_iterations;        // Maximum decoder iterations
    uint32_t code_distance;         // Code distance for surface code
} SyndromeConfig;

/**
 * @brief Result of syndrome extraction
 */
typedef struct SyndromeResult {
    SyndromeVertex* vertices;       // Extracted syndrome vertices
    size_t num_vertices;            // Number of vertices
    SyndromeEdge* edges;            // Syndrome graph edges
    size_t num_edges;               // Number of edges
    uint32_t* error_chain;          // Decoded error chain
    size_t chain_length;            // Length of error chain
    float logical_error_rate;       // Estimated logical error rate
    double extraction_time;         // Time for extraction (seconds)
    double decoding_time;           // Time for decoding (seconds)
} SyndromeResult;

/**
 * @brief Pattern detected in syndrome measurements
 */
typedef struct SyndromePattern {
    uint32_t* vertex_indices;       // Indices of vertices in pattern
    size_t pattern_size;            // Number of vertices
    float occurrence_probability;   // Probability of this pattern
    uint32_t occurrence_count;      // Number of occurrences seen
    bool is_logical_error;          // Whether this indicates logical error
} SyndromePattern;

// ===========================================================================
// Syndrome Extraction Functions
// ===========================================================================

/**
 * Create syndrome extraction context
 * @param config Configuration for syndrome extraction
 * @return Context pointer or NULL on failure
 */
void* syndrome_create_context(const SyndromeConfig* config);

/**
 * Destroy syndrome extraction context
 * @param ctx Context to destroy
 */
void syndrome_destroy_context(void* ctx);

/**
 * Extract syndromes from stabilizer measurements using Metal GPU
 * @param ctx Syndrome context
 * @param measurements Stabilizer measurement results
 * @param num_measurements Number of measurements
 * @param result Output syndrome result (caller must free)
 * @return 0 on success, error code on failure
 */
int syndrome_extract(void* ctx,
                     const float2* measurements,
                     size_t num_measurements,
                     SyndromeResult* result);

/**
 * Extract syndromes from multiple measurement rounds
 * @param ctx Syndrome context
 * @param measurements Array of measurement results per round
 * @param num_rounds Number of measurement rounds
 * @param measurements_per_round Measurements per round
 * @param result Output syndrome result
 * @return 0 on success, error code on failure
 */
int syndrome_extract_rounds(void* ctx,
                            const float2** measurements,
                            size_t num_rounds,
                            size_t measurements_per_round,
                            SyndromeResult* result);

// ===========================================================================
// Syndrome Decoding Functions
// ===========================================================================

/**
 * Decode syndromes using minimum weight perfect matching
 * @param ctx Syndrome context
 * @param result Syndrome result with vertices and edges
 * @return 0 on success, error code on failure
 */
int syndrome_decode_mwpm(void* ctx, SyndromeResult* result);

/**
 * Decode syndromes using Union-Find algorithm
 * @param ctx Syndrome context
 * @param result Syndrome result with vertices and edges
 * @return 0 on success, error code on failure
 */
int syndrome_decode_union_find(void* ctx, SyndromeResult* result);

/**
 * Decode syndromes using belief propagation
 * @param ctx Syndrome context
 * @param result Syndrome result with vertices and edges
 * @param max_iterations Maximum BP iterations
 * @return 0 on success, error code on failure
 */
int syndrome_decode_belief_propagation(void* ctx,
                                       SyndromeResult* result,
                                       uint32_t max_iterations);

// ===========================================================================
// Pattern Detection Functions
// ===========================================================================

/**
 * Detect error patterns in syndrome data
 * @param ctx Syndrome context
 * @param result Syndrome result to analyze
 * @param patterns Output array of detected patterns
 * @param max_patterns Maximum patterns to detect
 * @return Number of patterns detected
 */
size_t syndrome_detect_patterns(void* ctx,
                                const SyndromeResult* result,
                                SyndromePattern* patterns,
                                size_t max_patterns);

/**
 * Apply learned patterns for error correction
 * @param ctx Syndrome context
 * @param result Syndrome result to correct
 * @param patterns Array of known patterns
 * @param num_patterns Number of patterns
 * @return 0 on success, error code on failure
 */
int syndrome_apply_patterns(void* ctx,
                            SyndromeResult* result,
                            const SyndromePattern* patterns,
                            size_t num_patterns);

// ===========================================================================
// Graph Construction Functions
// ===========================================================================

/**
 * Build syndrome matching graph for decoding
 * @param ctx Syndrome context
 * @param vertices Syndrome vertices
 * @param num_vertices Number of vertices
 * @param config Syndrome configuration
 * @param edges Output edge array (caller must free)
 * @param num_edges Output number of edges
 * @return 0 on success, error code on failure
 */
int syndrome_build_graph(void* ctx,
                         const SyndromeVertex* vertices,
                         size_t num_vertices,
                         const SyndromeConfig* config,
                         SyndromeEdge** edges,
                         size_t* num_edges);

/**
 * Compute edge weights based on error model
 * @param ctx Syndrome context
 * @param edges Edge array to update
 * @param num_edges Number of edges
 * @param error_rates Per-qubit error rates
 * @param num_qubits Number of qubits
 * @return 0 on success, error code on failure
 */
int syndrome_compute_weights(void* ctx,
                             SyndromeEdge* edges,
                             size_t num_edges,
                             const float* error_rates,
                             size_t num_qubits);

// ===========================================================================
// Utility Functions
// ===========================================================================

/**
 * Free syndrome result resources
 * @param result Result to free
 */
void syndrome_free_result(SyndromeResult* result);

/**
 * Free syndrome pattern resources
 * @param pattern Pattern to free
 */
void syndrome_free_pattern(SyndromePattern* pattern);

/**
 * Estimate logical error rate from syndrome statistics
 * @param ctx Syndrome context
 * @param results Array of syndrome results
 * @param num_results Number of results
 * @return Estimated logical error rate
 */
float syndrome_estimate_logical_error_rate(void* ctx,
                                           const SyndromeResult* results,
                                           size_t num_results);

/**
 * Get syndrome extraction statistics
 * @param ctx Syndrome context
 * @param total_extracted Total syndromes extracted (output)
 * @param total_decoded Total syndromes decoded (output)
 * @param avg_chain_length Average error chain length (output)
 * @return 0 on success, error code on failure
 */
int syndrome_get_statistics(void* ctx,
                            size_t* total_extracted,
                            size_t* total_decoded,
                            float* avg_chain_length);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GEOMETRIC_SYNDROME_H

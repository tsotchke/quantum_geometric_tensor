/**
 * @file error_patterns.h
 * @brief Error pattern detection and analysis types
 *
 * Provides types and functions for detecting, classifying, and tracking
 * error patterns in quantum error correction systems.
 */

#ifndef ERROR_PATTERNS_H
#define ERROR_PATTERNS_H

#include <stddef.h>
#include <stdbool.h>
#include "quantum_geometric/physics/error_correlation.h"
#include "quantum_geometric/physics/error_syndrome.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define MAX_ERROR_PATTERNS 256
#define MAX_PATTERN_SIZE 32

// ============================================================================
// Pattern Type Enumeration
// ============================================================================

/**
 * Classification of error patterns
 */
typedef enum PatternType {
    PATTERN_UNKNOWN = 0,    // Unknown or unclassified pattern
    PATTERN_POINT,          // Single-site error
    PATTERN_LINE,           // Two-site linear error
    PATTERN_CHAIN,          // Multi-site linear chain
    PATTERN_CLUSTER,        // Densely connected errors
    PATTERN_CYCLE,          // Closed loop of errors
    PATTERN_BRAID,          // Intertwined error chains
    PATTERN_BOUNDARY,       // Boundary-related errors
    PATTERN_CORNER          // Corner or junction errors
} PatternType;

// ============================================================================
// Pattern Timing Structure
// ============================================================================

/**
 * Timing information for pattern tracking
 */
typedef struct PatternTiming {
    size_t first_seen;      // First occurrence timestamp
    size_t last_seen;       // Most recent occurrence timestamp
    size_t occurrences;     // Total number of occurrences
    double frequency;       // Pattern frequency (occurrences per time)
    double avg_duration;    // Average duration of pattern
    double stability;       // Stability metric [0, 1]
} PatternTiming;

// ============================================================================
// Error Pattern Structure
// ============================================================================

/**
 * Complete error pattern representation
 */
typedef struct ErrorPattern {
    PatternType type;                       // Pattern classification
    size_t size;                            // Number of vertices in pattern
    SyndromeVertex vertices[MAX_PATTERN_SIZE]; // Pattern vertices
    double weight;                          // Pattern weight/importance
    bool is_active;                         // Whether pattern is currently active
    ErrorCorrelation correlation;           // Associated correlations
    PatternTiming timing;                   // Timing information
    size_t pattern_id;                      // Unique pattern identifier
    double confidence;                      // Detection confidence [0, 1]
} ErrorPattern;

// ============================================================================
// Pattern Configuration
// ============================================================================

/**
 * Configuration for pattern detection
 */
typedef struct PatternConfig {
    size_t max_patterns;            // Maximum patterns to track
    size_t min_occurrences;         // Minimum occurrences to consider pattern
    double similarity_threshold;    // Threshold for pattern matching [0, 1]
    double detection_threshold;     // Threshold for pattern detection
    bool track_timing;              // Enable timing information tracking
    double decay_rate;              // Rate at which old patterns decay
    size_t max_pattern_size;        // Maximum size of a single pattern
    bool enable_merging;            // Enable similar pattern merging
    double merge_threshold;         // Threshold for pattern merging
} PatternConfig;

// ============================================================================
// Pattern State
// ============================================================================

/**
 * Global state for pattern detection system
 */
typedef struct PatternState {
    ErrorPattern patterns[MAX_ERROR_PATTERNS]; // Pattern storage
    size_t num_patterns;                        // Current number of patterns
    PatternConfig config;                       // Configuration
    size_t current_timestamp;                   // Current timestamp
    double total_weight;                        // Sum of pattern weights
    size_t active_count;                        // Number of active patterns
} PatternState;

// ============================================================================
// Initialization and Cleanup
// ============================================================================

/**
 * Initialize error pattern detection
 * @param config Configuration parameters
 * @return true on success, false on failure
 */
bool init_error_patterns(const PatternConfig* config);

/**
 * Clean up error pattern resources
 */
void cleanup_error_patterns(void);

// ============================================================================
// Pattern Detection Functions
// ============================================================================

/**
 * Detect error patterns in a matching graph
 * @param graph Matching graph to analyze
 * @param correlation Current correlation data
 * @return Number of patterns detected
 */
size_t detect_error_patterns(const MatchingGraph* graph,
                            const ErrorCorrelation* correlation);

/**
 * Update pattern database with new measurements
 * @param graph Current matching graph
 * @return Number of active patterns
 */
size_t update_pattern_database(const MatchingGraph* graph);

// ============================================================================
// Pattern Classification Functions
// ============================================================================

/**
 * Classify pattern type based on vertex arrangement
 * @param vertices Array of syndrome vertices
 * @param size Number of vertices
 * @return Pattern type classification
 */
PatternType classify_pattern_type(const SyndromeVertex* vertices, size_t size);

/**
 * Calculate similarity between two patterns
 * @param pattern1 First pattern
 * @param pattern2 Second pattern
 * @return Similarity score [0, 1]
 */
double calculate_pattern_similarity(const ErrorPattern* pattern1,
                                   const ErrorPattern* pattern2);

// ============================================================================
// Pattern Matching Functions
// ============================================================================

/**
 * Check if vertices match a known pattern
 * @param vertices Array of syndrome vertices
 * @param size Number of vertices
 * @param pattern Pattern to match against
 * @return true if match found
 */
bool match_pattern(const SyndromeVertex* vertices,
                  size_t size,
                  const ErrorPattern* pattern);

/**
 * Calculate pattern weight
 * @param pattern Pattern to calculate weight for
 * @return Pattern weight
 */
double calculate_pattern_weight(const ErrorPattern* pattern);

// ============================================================================
// Pattern Management Functions
// ============================================================================

/**
 * Update timing information for a pattern
 * @param pattern Pattern to update
 * @param current_time Current timestamp
 */
void update_pattern_timing(ErrorPattern* pattern, size_t current_time);

/**
 * Merge two similar patterns
 * @param pattern1 Target pattern (will be updated)
 * @param pattern2 Source pattern
 * @return true on success
 */
bool merge_similar_patterns(ErrorPattern* pattern1, ErrorPattern* pattern2);

/**
 * Remove inactive patterns from database
 */
void prune_inactive_patterns(void);

// ============================================================================
// Pattern Query Functions
// ============================================================================

/**
 * Get pattern statistics
 * @param pattern_idx Pattern index
 * @param timing Output timing information
 * @return true if pattern exists
 */
bool get_pattern_statistics(size_t pattern_idx, PatternTiming* timing);

/**
 * Check if pattern is currently active
 * @param pattern_idx Pattern index
 * @return true if active
 */
bool is_pattern_active(size_t pattern_idx);

/**
 * Get total number of patterns
 * @return Pattern count
 */
size_t get_pattern_count(void);

/**
 * Get pattern by index
 * @param pattern_idx Pattern index
 * @return Pattern pointer or NULL
 */
const ErrorPattern* get_pattern(size_t pattern_idx);

#ifdef __cplusplus
}
#endif

#endif // ERROR_PATTERNS_H

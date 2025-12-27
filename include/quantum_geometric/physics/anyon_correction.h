/**
 * @file anyon_correction.h
 * @brief Anyon-based error correction system
 *
 * Provides infrastructure for determining and applying error corrections
 * based on anyon fusion outcomes. Uses fusion channel analysis to identify
 * required corrections and tracks correction success rates over time.
 */

#ifndef ANYON_CORRECTION_H
#define ANYON_CORRECTION_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Maximum number of correction operations for static allocation
#define MAX_CORRECTION_OPERATIONS 1024

// Maximum number of fusion channels
#ifndef MAX_FUSION_CHANNELS
#define MAX_FUSION_CHANNELS 1024
#endif

// Maximum correction history length
#define MAX_CORRECTION_HISTORY 128

/**
 * Correction operation types
 */
typedef enum {
    CORRECTION_NONE,    // No correction needed
    CORRECTION_X,       // X-type (bit flip) correction
    CORRECTION_Z,       // Z-type (phase flip) correction
    CORRECTION_Y        // Y-type (combined) correction
} correction_type_t;

/**
 * Configuration for the error correction system
 */
typedef struct CorrectionConfig {
    size_t lattice_width;           // Width of the lattice
    size_t lattice_height;          // Height of the lattice
    size_t history_length;          // Number of historical corrections to track
    double error_threshold;         // Threshold for successful correction rate
    bool track_success;             // Enable correction success tracking
} CorrectionConfig;

/**
 * Cache for correction computation data
 */
typedef struct CorrectionCache {
    int* correction_map;            // Current correction at each lattice site
    double* success_map;            // Success probability at each site
    int* correction_history;        // Historical correction values for pattern detection
    double* correlation_matrix;     // Correction-correction correlation matrix
    size_t cache_size;              // Size of cached data
} CorrectionCache;

/**
 * State of the error correction system
 */
typedef struct CorrectionState {
    CorrectionConfig config;        // Configuration parameters
    CorrectionCache* cache;         // Cached computation data
    size_t total_corrections;       // Total number of correction rounds performed
    size_t total_successes;         // Total number of successful corrections
    double success_rate;            // Current correction success rate [0,1]
} CorrectionState;

/**
 * Fusion channel information for correction determination
 * Represents the outcomes of anyon fusion processes
 */
#ifndef FUSION_CHANNELS_DEFINED
#define FUSION_CHANNELS_DEFINED
typedef struct FusionChannels {
    size_t num_channels;                            // Number of active fusion channels
    size_t locations[MAX_FUSION_CHANNELS];          // Lattice site indices for each channel
    int input_charges[MAX_FUSION_CHANNELS][2];      // Input anyon charges for each fusion
    int output_charge[MAX_FUSION_CHANNELS];         // Resulting charge from fusion
    double probabilities[MAX_FUSION_CHANNELS];      // Fusion outcome probabilities
} FusionChannels;
#endif

/**
 * Collection of correction operations to be applied
 */
typedef struct CorrectionOperations {
    size_t num_corrections;                         // Number of corrections to apply
    size_t locations[MAX_CORRECTION_OPERATIONS];    // Lattice site indices
    correction_type_t types[MAX_CORRECTION_OPERATIONS]; // Correction types
    double weights[MAX_CORRECTION_OPERATIONS];      // Correction weights/priorities
} CorrectionOperations;

/**
 * Initialize error correction system
 * @param state Pointer to state structure to initialize
 * @param config Configuration parameters
 * @return true if initialization successful, false otherwise
 */
bool init_error_correction(CorrectionState* state,
                          const CorrectionConfig* config);

/**
 * Clean up error correction system
 * @param state State structure to clean up
 */
void cleanup_error_correction(CorrectionState* state);

/**
 * Determine corrections based on fusion channel outcomes
 * @param state Correction system state
 * @param channels Fusion channel information
 * @param operations Output correction operations
 * @return true if corrections determined successfully, false otherwise
 */
bool determine_corrections(CorrectionState* state,
                          const FusionChannels* channels,
                          CorrectionOperations* operations);

/**
 * Update correction metrics
 * @param state Correction system state
 * @return true if update successful, false otherwise
 */
bool update_correction_metrics(CorrectionState* state);

/**
 * Get current correction success rate
 * @param state Correction system state
 * @return Success rate in range [0,1], higher is better
 */
static inline double get_correction_success_rate(const CorrectionState* state) {
    return state ? state->success_rate : 0.0;
}

/**
 * Get total number of corrections performed
 * @param state Correction system state
 * @return Total correction count
 */
static inline size_t get_correction_count(const CorrectionState* state) {
    return state ? state->total_corrections : 0;
}

/**
 * Get total number of successful corrections
 * @param state Correction system state
 * @return Successful correction count
 */
static inline size_t get_success_count(const CorrectionState* state) {
    return state ? state->total_successes : 0;
}

/**
 * Check if correction success rate is above threshold
 * @param state Correction system state
 * @param threshold Acceptable success rate threshold
 * @return true if success rate is above threshold
 */
static inline bool is_correction_effective(const CorrectionState* state, double threshold) {
    return state && state->success_rate >= threshold;
}

#ifdef __cplusplus
}
#endif

#endif // ANYON_CORRECTION_H

/**
 * @file versatility_analyzer.h
 * @brief System Versatility Scoring for Quantum Operations
 *
 * Provides versatility analysis including:
 * - Algorithm coverage assessment
 * - Hardware compatibility scoring
 * - Feature utilization tracking
 * - Adaptability metrics
 * - Configuration flexibility
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef VERSATILITY_ANALYZER_H
#define VERSATILITY_ANALYZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define VERSATILITY_MAX_CAPABILITIES 256
#define VERSATILITY_MAX_NAME_LENGTH 128

// ============================================================================
// Enumerations
// ============================================================================

typedef enum {
    VERSATILITY_CATEGORY_ALGORITHM,
    VERSATILITY_CATEGORY_HARDWARE,
    VERSATILITY_CATEGORY_FEATURE,
    VERSATILITY_CATEGORY_PROTOCOL,
    VERSATILITY_CATEGORY_FORMAT
} versatility_category_t;

typedef enum {
    VERSATILITY_LEVEL_LIMITED,
    VERSATILITY_LEVEL_BASIC,
    VERSATILITY_LEVEL_MODERATE,
    VERSATILITY_LEVEL_ADVANCED,
    VERSATILITY_LEVEL_COMPREHENSIVE
} versatility_level_t;

// ============================================================================
// Data Structures
// ============================================================================

typedef struct {
    char name[VERSATILITY_MAX_NAME_LENGTH];
    versatility_category_t category;
    bool supported;
    double performance_score;         // 0.0 to 1.0
    uint64_t usage_count;
} versatility_capability_t;

typedef struct {
    double overall_score;             // 0.0 to 1.0
    double algorithm_coverage;
    double hardware_compatibility;
    double feature_completeness;
    double adaptability_score;
    versatility_level_t level;
    size_t capabilities_supported;
    size_t capabilities_total;
} versatility_metrics_t;

typedef struct {
    bool track_usage;
    double min_performance_threshold;
} versatility_analyzer_config_t;

typedef struct versatility_analyzer versatility_analyzer_t;

// ============================================================================
// API Functions
// ============================================================================

versatility_analyzer_t* versatility_analyzer_create(void);
versatility_analyzer_t* versatility_analyzer_create_with_config(
    const versatility_analyzer_config_t* config);
versatility_analyzer_config_t versatility_analyzer_default_config(void);
void versatility_analyzer_destroy(versatility_analyzer_t* analyzer);
bool versatility_analyzer_reset(versatility_analyzer_t* analyzer);

bool versatility_register_capability(versatility_analyzer_t* analyzer,
                                      const char* name,
                                      versatility_category_t category,
                                      bool supported);

bool versatility_record_usage(versatility_analyzer_t* analyzer,
                               const char* capability_name,
                               double performance);

bool versatility_calculate_metrics(versatility_analyzer_t* analyzer,
                                    versatility_metrics_t* metrics);

bool versatility_get_capabilities(versatility_analyzer_t* analyzer,
                                   versatility_capability_t** capabilities,
                                   size_t* count);

bool versatility_get_by_category(versatility_analyzer_t* analyzer,
                                  versatility_category_t category,
                                  versatility_capability_t** capabilities,
                                  size_t* count);

char* versatility_generate_report(versatility_analyzer_t* analyzer);
char* versatility_export_json(versatility_analyzer_t* analyzer);

const char* versatility_category_name(versatility_category_t category);
const char* versatility_level_name(versatility_level_t level);
void versatility_free_capabilities(versatility_capability_t* caps, size_t count);
const char* versatility_get_last_error(versatility_analyzer_t* analyzer);

#ifdef __cplusplus
}
#endif

#endif // VERSATILITY_ANALYZER_H

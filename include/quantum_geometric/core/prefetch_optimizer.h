/**
 * @file prefetch_optimizer.h
 * @brief Memory Prefetch Optimization for Quantum Operations
 *
 * Provides prefetch optimization including:
 * - Access pattern analysis
 * - Prefetch hint generation
 * - Cache line optimization
 * - Memory access prediction
 * - Bandwidth utilization
 *
 * Part of the QGTL Monitoring Framework.
 */

#ifndef PREFETCH_OPTIMIZER_H
#define PREFETCH_OPTIMIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Constants
// ============================================================================

#define PREFETCH_MAX_PATTERNS 128
#define PREFETCH_CACHE_LINE_SIZE 64
#define PREFETCH_DEFAULT_LOOKAHEAD 4

// ============================================================================
// Enumerations
// ============================================================================

typedef enum {
    PREFETCH_PATTERN_SEQUENTIAL,
    PREFETCH_PATTERN_STRIDED,
    PREFETCH_PATTERN_RANDOM,
    PREFETCH_PATTERN_BLOCKED,
    PREFETCH_PATTERN_RECURSIVE,
    PREFETCH_PATTERN_UNKNOWN
} prefetch_pattern_t;

typedef enum {
    PREFETCH_LOCALITY_TEMPORAL,       // Keep in cache
    PREFETCH_LOCALITY_NON_TEMPORAL,   // Streaming, don't pollute cache
    PREFETCH_LOCALITY_L1,
    PREFETCH_LOCALITY_L2,
    PREFETCH_LOCALITY_L3
} prefetch_locality_t;

typedef enum {
    PREFETCH_STRATEGY_NONE,
    PREFETCH_STRATEGY_SOFTWARE,
    PREFETCH_STRATEGY_HARDWARE,
    PREFETCH_STRATEGY_HYBRID
} prefetch_strategy_t;

// ============================================================================
// Data Structures
// ============================================================================

typedef struct {
    prefetch_pattern_t pattern;
    size_t stride;
    size_t block_size;
    double confidence;
    uint64_t access_count;
} prefetch_pattern_info_t;

typedef struct {
    void* base_address;
    size_t size;
    prefetch_locality_t locality;
    int lookahead;
} prefetch_hint_t;

typedef struct {
    double hit_rate;
    double miss_rate;
    double bandwidth_utilization;
    uint64_t prefetches_issued;
    uint64_t prefetches_useful;
    uint64_t cache_misses_avoided;
} prefetch_stats_t;

typedef struct {
    bool enable_auto_prefetch;
    prefetch_strategy_t strategy;
    int default_lookahead;
    size_t min_pattern_samples;
    double pattern_confidence_threshold;
} prefetch_optimizer_config_t;

typedef struct prefetch_optimizer prefetch_optimizer_t;

// ============================================================================
// API Functions
// ============================================================================

prefetch_optimizer_t* prefetch_optimizer_create(void);
prefetch_optimizer_t* prefetch_optimizer_create_with_config(
    const prefetch_optimizer_config_t* config);
prefetch_optimizer_config_t prefetch_optimizer_default_config(void);
void prefetch_optimizer_destroy(prefetch_optimizer_t* optimizer);
bool prefetch_optimizer_reset(prefetch_optimizer_t* optimizer);

bool prefetch_record_access(prefetch_optimizer_t* optimizer,
                             const void* address,
                             size_t size);

bool prefetch_analyze_pattern(prefetch_optimizer_t* optimizer,
                               const char* region_name,
                               prefetch_pattern_info_t* pattern);

bool prefetch_generate_hints(prefetch_optimizer_t* optimizer,
                              const char* region_name,
                              prefetch_hint_t** hints,
                              size_t* count);

bool prefetch_issue_hint(prefetch_optimizer_t* optimizer,
                          const prefetch_hint_t* hint);

bool prefetch_get_stats(prefetch_optimizer_t* optimizer,
                         prefetch_stats_t* stats);

char* prefetch_generate_report(prefetch_optimizer_t* optimizer);
char* prefetch_export_json(prefetch_optimizer_t* optimizer);

const char* prefetch_pattern_name(prefetch_pattern_t pattern);
const char* prefetch_locality_name(prefetch_locality_t locality);
const char* prefetch_strategy_name(prefetch_strategy_t strategy);
void prefetch_free_hints(prefetch_hint_t* hints, size_t count);
const char* prefetch_get_last_error(prefetch_optimizer_t* optimizer);

#ifdef __cplusplus
}
#endif

#endif // PREFETCH_OPTIMIZER_H

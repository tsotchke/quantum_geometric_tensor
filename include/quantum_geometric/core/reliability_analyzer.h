/**
 * @file reliability_analyzer.h
 * @brief Reliability analysis for quantum systems
 */

#ifndef RELIABILITY_ANALYZER_H
#define RELIABILITY_ANALYZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { REL_HIGH, REL_MEDIUM, REL_LOW, REL_CRITICAL } ReliabilityLevel;
typedef struct { double mtbf; double mttr; double availability; double failure_rate; size_t total_ops; size_t failed_ops; } ReliabilityMetrics;
typedef struct { char* id; ReliabilityLevel level; double score; double failure_prob; uint64_t last_failure; size_t failures; } ComponentReliability;
typedef struct { double target_availability; double acceptable_failure_rate; size_t window_ms; bool enable_prediction; } ReliabilityConfig;
typedef struct { ComponentReliability* components; size_t num; ReliabilityConfig config; ReliabilityMetrics metrics; uint64_t start; } ReliabilityContext;

int reliability_context_create(ReliabilityContext** ctx, ReliabilityConfig* config);
void reliability_context_destroy(ReliabilityContext* ctx);
int reliability_register_component(ReliabilityContext* ctx, const char* id);
int reliability_record_success(ReliabilityContext* ctx, const char* id);
int reliability_record_failure(ReliabilityContext* ctx, const char* id);
int reliability_get_metrics(ReliabilityContext* ctx, ReliabilityMetrics* m);
ReliabilityLevel reliability_assess(ReliabilityContext* ctx);
void reliability_reset(ReliabilityContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // RELIABILITY_ANALYZER_H

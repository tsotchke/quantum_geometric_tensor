/**
 * @file resilience_analyzer.h
 * @brief Resilience analysis for quantum systems
 */

#ifndef RESILIENCE_ANALYZER_H
#define RESILIENCE_ANALYZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RESILIENCE_FRAGILE,
    RESILIENCE_WEAK,
    RESILIENCE_MODERATE,
    RESILIENCE_STRONG,
    RESILIENCE_ANTIFRAGILE
} ResilienceLevel;

typedef enum {
    STRESS_TYPE_NOISE,
    STRESS_TYPE_LOAD,
    STRESS_TYPE_FAULT,
    STRESS_TYPE_ATTACK,
    STRESS_TYPE_ENVIRONMENTAL
} StressType;

typedef struct StressTest {
    StressType type;
    double intensity;
    uint64_t duration_ms;
    double impact;
    bool system_survived;
} StressTest;

typedef struct ResilienceMetrics {
    ResilienceLevel level;
    double resilience_score;
    double recovery_speed;
    double adaptation_rate;
    size_t stress_tests_passed;
    size_t stress_tests_failed;
} ResilienceMetrics;

typedef struct ResilienceConfig {
    double stress_threshold;
    size_t test_iterations;
    bool adaptive_testing;
    double target_resilience;
} ResilienceConfig;

typedef struct ResilienceContext {
    StressTest* tests;
    size_t num_tests;
    size_t capacity;
    ResilienceConfig config;
    ResilienceMetrics metrics;
} ResilienceContext;

int resilience_context_create(ResilienceContext** ctx, ResilienceConfig* config);
void resilience_context_destroy(ResilienceContext* ctx);
int resilience_run_stress_test(ResilienceContext* ctx, StressType type, double intensity, StressTest* result);
int resilience_analyze(ResilienceContext* ctx, ResilienceMetrics* metrics);
ResilienceLevel resilience_assess(ResilienceContext* ctx);
int resilience_get_weaknesses(ResilienceContext* ctx, StressType** types, size_t* count);
int resilience_recommend_improvements(ResilienceContext* ctx, char*** recommendations, size_t* count);
void resilience_reset(ResilienceContext* ctx);

#ifdef __cplusplus
}
#endif

#endif // RESILIENCE_ANALYZER_H

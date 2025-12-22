#ifndef INSIGHT_GENERATOR_H
#define INSIGHT_GENERATOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

// Insight parameters
#define MAX_INSIGHTS 100
#define MIN_CONFIDENCE 0.8
#define MAX_PRIORITY 10
#define HISTORY_WINDOW 1000

// Forward declarations
typedef struct InsightGeneratorImpl InsightGenerator;
typedef struct InsightImpl Insight;
typedef struct InsightHistoryImpl InsightHistory;
typedef struct ActionPlanImpl ActionPlan;
typedef struct PriorityQueueImpl PriorityQueue;

// Insight type
typedef enum {
    PATTERN_INSIGHT,
    CORRELATION_INSIGHT,
    TREND_INSIGHT,
    REGRESSION_INSIGHT,
    OPTIMIZATION_INSIGHT
} InsightType;

// Insight configuration
typedef struct {
    size_t max_insights;
    double min_confidence;
    bool enable_action_plans;
    bool enable_trend_analysis;
} InsightConfig;

// Create and destroy
InsightGenerator* init_insight_generator(const InsightConfig* config);
void cleanup_insight_generator(InsightGenerator* generator);

// Insight history functions
InsightHistory* create_insight_history(void);
void cleanup_insight_history(InsightHistory* history);

// Priority queue functions
PriorityQueue* create_priority_queue(void);
void cleanup_priority_queue(PriorityQueue* queue);
void clear_priority_queue(PriorityQueue* queue);
void add_to_priority_queue(PriorityQueue* queue, Insight* insight, double score);

// Insight functions
void cleanup_insight(Insight* insight);
void store_insight(InsightGenerator* generator, Insight* insight);

#ifdef __cplusplus
}
#endif

#endif // INSIGHT_GENERATOR_H

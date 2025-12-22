/**
 * @file insight_generator.c
 * @brief Insight generation for distributed training optimization
 */

#include "quantum_geometric/distributed/insight_generator.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Performance insight - internal structure
struct InsightImpl {
    InsightType type;
    char* description;
    double confidence;
    int priority;
    time_t timestamp;
    bool is_actionable;
    ActionPlan* action_plan;
};

// Action plan - internal structure
struct ActionPlanImpl {
    char* description;
    double priority;
    double expected_improvement;
};

// Insight history - internal structure
struct InsightHistoryImpl {
    Insight** insights;
    size_t num_insights;
    size_t capacity;
    double* effectiveness;
    time_t* timestamps;
};

// Priority queue entry
typedef struct {
    Insight* insight;
    double score;
} PriorityEntry;

// Priority queue - internal structure
struct PriorityQueueImpl {
    PriorityEntry* entries;
    size_t num_entries;
    size_t capacity;
};

// Insight generator - internal structure
struct InsightGeneratorImpl {
    Insight** active_insights;
    size_t num_active;
    size_t capacity;
    InsightHistory* history;
    PriorityQueue* priority_queue;
    InsightConfig config;
};

// Forward declarations
static Insight* create_insight(InsightType type, const char* description, double confidence);
static ActionPlan* create_action_plan_from_insight(const Insight* insight);
static double compute_insight_priority(const Insight* insight, const InsightHistory* history);

// Create insight history
InsightHistory* create_insight_history(void) {
    InsightHistory* history = calloc(1, sizeof(InsightHistory));
    if (!history) return NULL;

    history->capacity = HISTORY_WINDOW;
    history->insights = calloc(history->capacity, sizeof(Insight*));
    history->effectiveness = calloc(history->capacity, sizeof(double));
    history->timestamps = calloc(history->capacity, sizeof(time_t));
    history->num_insights = 0;

    return history;
}

// Cleanup insight history
void cleanup_insight_history(InsightHistory* history) {
    if (!history) return;

    for (size_t i = 0; i < history->num_insights; i++) {
        cleanup_insight(history->insights[i]);
    }
    free(history->insights);
    free(history->effectiveness);
    free(history->timestamps);
    free(history);
}

// Create priority queue
PriorityQueue* create_priority_queue(void) {
    PriorityQueue* queue = calloc(1, sizeof(PriorityQueue));
    if (!queue) return NULL;

    queue->capacity = MAX_INSIGHTS;
    queue->entries = calloc(queue->capacity, sizeof(PriorityEntry));
    queue->num_entries = 0;

    return queue;
}

// Cleanup priority queue
void cleanup_priority_queue(PriorityQueue* queue) {
    if (!queue) return;
    free(queue->entries);
    free(queue);
}

// Clear priority queue
void clear_priority_queue(PriorityQueue* queue) {
    if (!queue) return;
    queue->num_entries = 0;
}

// Add to priority queue (maintains sorted order by score descending)
void add_to_priority_queue(PriorityQueue* queue, Insight* insight, double score) {
    if (!queue || !insight || queue->num_entries >= queue->capacity) return;

    // Find insertion point
    size_t insert_pos = queue->num_entries;
    for (size_t i = 0; i < queue->num_entries; i++) {
        if (score > queue->entries[i].score) {
            insert_pos = i;
            break;
        }
    }

    // Shift entries to make room
    if (insert_pos < queue->num_entries) {
        memmove(&queue->entries[insert_pos + 1],
                &queue->entries[insert_pos],
                (queue->num_entries - insert_pos) * sizeof(PriorityEntry));
    }

    queue->entries[insert_pos].insight = insight;
    queue->entries[insert_pos].score = score;
    queue->num_entries++;
}

// Create insight
static Insight* create_insight(InsightType type, const char* description, double confidence) {
    Insight* insight = calloc(1, sizeof(Insight));
    if (!insight) return NULL;

    insight->type = type;
    insight->description = description ? strdup(description) : NULL;
    insight->confidence = confidence;
    insight->priority = 0;
    insight->timestamp = time(NULL);
    insight->is_actionable = (confidence >= MIN_CONFIDENCE);
    insight->action_plan = NULL;

    return insight;
}

// Cleanup insight
void cleanup_insight(Insight* insight) {
    if (!insight) return;

    free(insight->description);
    if (insight->action_plan) {
        free(insight->action_plan->description);
        free(insight->action_plan);
    }
    free(insight);
}

// Create action plan based on insight
static ActionPlan* create_action_plan_from_insight(const Insight* insight) {
    if (!insight || !insight->is_actionable) return NULL;

    ActionPlan* plan = calloc(1, sizeof(ActionPlan));
    if (!plan) return NULL;

    switch (insight->type) {
        case PATTERN_INSIGHT:
            plan->description = strdup("Optimize based on detected pattern");
            plan->expected_improvement = 0.1;
            break;
        case CORRELATION_INSIGHT:
            plan->description = strdup("Adjust correlated parameters together");
            plan->expected_improvement = 0.15;
            break;
        case TREND_INSIGHT:
            plan->description = strdup("Proactively adjust for trend");
            plan->expected_improvement = 0.2;
            break;
        case REGRESSION_INSIGHT:
            plan->description = strdup("Investigate and mitigate regression");
            plan->expected_improvement = 0.25;
            break;
        case OPTIMIZATION_INSIGHT:
            plan->description = strdup("Apply optimization recommendation");
            plan->expected_improvement = 0.3;
            break;
    }

    plan->priority = insight->confidence * plan->expected_improvement;
    return plan;
}

// Store insight in generator
void store_insight(InsightGenerator* generator, Insight* insight) {
    if (!generator || !insight) return;

    if (generator->num_active >= generator->capacity) {
        cleanup_insight(generator->active_insights[0]);
        memmove(generator->active_insights, generator->active_insights + 1,
                (generator->capacity - 1) * sizeof(Insight*));
        generator->num_active--;
    }

    generator->active_insights[generator->num_active++] = insight;
}

// Compute insight priority based on history
static double compute_insight_priority(const Insight* insight, const InsightHistory* history) {
    if (!insight) return 0.0;

    double base_priority = insight->confidence;

    if (insight->is_actionable) {
        base_priority *= 1.5;
    }

    if (history && history->num_insights > 0) {
        double avg_effectiveness = 0.0;
        size_t count = 0;

        for (size_t i = 0; i < history->num_insights; i++) {
            if (history->insights[i] && history->insights[i]->type == insight->type) {
                avg_effectiveness += history->effectiveness[i];
                count++;
            }
        }

        if (count > 0) {
            avg_effectiveness /= count;
            base_priority *= (1.0 + avg_effectiveness);
        }
    }

    return base_priority;
}

// Initialize insight generator
InsightGenerator* init_insight_generator(const InsightConfig* config) {
    InsightGenerator* generator = calloc(1, sizeof(InsightGenerator));
    if (!generator) return NULL;

    if (config) {
        generator->config = *config;
    } else {
        generator->config.max_insights = MAX_INSIGHTS;
        generator->config.min_confidence = MIN_CONFIDENCE;
        generator->config.enable_action_plans = true;
        generator->config.enable_trend_analysis = true;
    }

    generator->capacity = generator->config.max_insights;
    generator->active_insights = calloc(generator->capacity, sizeof(Insight*));
    generator->num_active = 0;

    generator->history = create_insight_history();
    generator->priority_queue = create_priority_queue();

    return generator;
}

// Cleanup insight generator
void cleanup_insight_generator(InsightGenerator* generator) {
    if (!generator) return;

    for (size_t i = 0; i < generator->num_active; i++) {
        cleanup_insight(generator->active_insights[i]);
    }
    free(generator->active_insights);

    cleanup_insight_history(generator->history);
    cleanup_priority_queue(generator->priority_queue);

    free(generator);
}

/**
 * @file optimization_suggester.c
 * @brief Intelligent optimization suggestion system implementation
 *
 * Analyzes performance bottlenecks and generates prioritized optimization
 * suggestions using ML-based impact prediction and historical tracking.
 */

#include "quantum_geometric/distributed/optimization_suggester.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Number of suggestion types
#define NUM_SUGGESTION_TYPES 6

// Impact prediction weights
static const double IMPACT_WEIGHTS[NUM_SUGGESTION_TYPES] = {
    0.25,  // SUGGEST_COMPUTE_OPTIMIZATION
    0.20,  // SUGGEST_MEMORY_OPTIMIZATION
    0.20,  // SUGGEST_COMMUNICATION_OPTIMIZATION
    0.15,  // SUGGEST_QUANTUM_OPTIMIZATION
    0.10,  // SUGGEST_HARDWARE_OPTIMIZATION
    0.10   // SUGGEST_WORKLOAD_OPTIMIZATION
};

// Suggestion history internal structure
struct SuggestionHistoryImpl {
    SuggestionHistoryEntry* entries;
    size_t num_entries;
    size_t capacity;
    double type_effectiveness[NUM_SUGGESTION_TYPES];
    size_t type_counts[NUM_SUGGESTION_TYPES];
};

// Optimization suggester internal structure
struct OptimizationSuggesterImpl {
    // Active suggestions
    Suggestion* suggestions;
    size_t num_suggestions;
    size_t capacity;

    // History tracking
    SuggestionHistory* history;

    // ML model for impact prediction
    MLModel* impact_model;

    // Configuration
    SuggesterConfig config;

    // Statistics
    size_t total_suggestions_generated;
    size_t total_suggestions_implemented;
    double average_effectiveness;
};

// Forward declarations - use prefixed names to avoid conflict with bottleneck_detector.h
static void add_suggestion(OptimizationSuggester* suggester,
                          const char* description,
                          SuggestionType type,
                          double base_impact);
static void suggester_gen_compute(OptimizationSuggester* suggester,
                                  const BottleneckInfo* bottleneck);
static void suggester_gen_memory(OptimizationSuggester* suggester,
                                 const BottleneckInfo* bottleneck);
static void suggester_gen_communication(OptimizationSuggester* suggester,
                                        const BottleneckInfo* bottleneck);
static void suggester_gen_quantum(OptimizationSuggester* suggester,
                                  const BottleneckInfo* bottleneck);
static void suggester_gen_workload(OptimizationSuggester* suggester,
                                   const BottleneckInfo* bottleneck);
static void score_and_sort_suggestions(OptimizationSuggester* suggester);
static void filter_low_impact_suggestions(OptimizationSuggester* suggester);
static double predict_impact(OptimizationSuggester* suggester,
                            const Suggestion* suggestion,
                            const BottleneckInfo* bottleneck);
static SuggestionHistory* create_suggestion_history(size_t capacity);
static void cleanup_suggestion_history(SuggestionHistory* history);
static void cleanup_suggestion(Suggestion* suggestion);

// Initialize optimization suggester
OptimizationSuggester* init_optimization_suggester(const SuggesterConfig* config) {
    OptimizationSuggester* suggester = calloc(1, sizeof(OptimizationSuggester));
    if (!suggester) return NULL;

    // Apply configuration
    if (config) {
        suggester->config = *config;
    } else {
        suggester->config.max_suggestions = SUGGESTER_MAX_SUGGESTIONS;
        suggester->config.min_impact_threshold = SUGGESTER_MIN_IMPACT;
        suggester->config.enable_ml_prediction = true;
        suggester->config.enable_history_tracking = true;
        suggester->config.learning_rate = 0.1;
    }

    // Allocate suggestions array
    suggester->capacity = suggester->config.max_suggestions;
    suggester->suggestions = calloc(suggester->capacity, sizeof(Suggestion));
    if (!suggester->suggestions) {
        free(suggester);
        return NULL;
    }
    suggester->num_suggestions = 0;

    // Create history tracker
    if (suggester->config.enable_history_tracking) {
        suggester->history = create_suggestion_history(SUGGESTER_MAX_HISTORY);
    }

    // Create ML model for impact prediction
    if (suggester->config.enable_ml_prediction) {
        suggester->impact_model = init_ml_model();
    }

    // Initialize statistics
    suggester->total_suggestions_generated = 0;
    suggester->total_suggestions_implemented = 0;
    suggester->average_effectiveness = 0.0;

    return suggester;
}

// Create suggestion history
static SuggestionHistory* create_suggestion_history(size_t capacity) {
    SuggestionHistory* history = calloc(1, sizeof(SuggestionHistory));
    if (!history) return NULL;

    history->entries = calloc(capacity, sizeof(SuggestionHistoryEntry));
    if (!history->entries) {
        free(history);
        return NULL;
    }

    history->capacity = capacity;
    history->num_entries = 0;

    // Initialize type effectiveness to neutral
    for (int i = 0; i < NUM_SUGGESTION_TYPES; i++) {
        history->type_effectiveness[i] = 0.5;
        history->type_counts[i] = 0;
    }

    return history;
}

// Generate suggestions based on bottleneck analysis
void suggester_generate(OptimizationSuggester* suggester,
                        const BottleneckInfo* bottleneck) {
    if (!suggester || !bottleneck) return;

    // Clear expired suggestions
    time_t now = time(NULL);
    for (size_t i = 0; i < suggester->num_suggestions; ) {
        if (now - suggester->suggestions[i].timestamp > SUGGESTER_TIMEOUT_SECS) {
            cleanup_suggestion(&suggester->suggestions[i]);
            // Shift remaining suggestions
            memmove(&suggester->suggestions[i],
                   &suggester->suggestions[i + 1],
                   (suggester->num_suggestions - i - 1) * sizeof(Suggestion));
            suggester->num_suggestions--;
        } else {
            i++;
        }
    }

    // Generate type-specific suggestions based on bottleneck
    switch (bottleneck->type) {
        case COMPUTE_BOUND:
            suggester_gen_compute(suggester, bottleneck);
            break;

        case MEMORY_BOUND:
            suggester_gen_memory(suggester, bottleneck);
            break;

        case COMMUNICATION_BOUND:
            suggester_gen_communication(suggester, bottleneck);
            break;

        case QUANTUM_BOUND:
            suggester_gen_quantum(suggester, bottleneck);
            break;

        case IO_BOUND:
            suggester_gen_workload(suggester, bottleneck);
            break;

        case NO_BOTTLENECK:
        default:
            // No specific suggestions needed
            break;
    }

    // Score and sort suggestions
    score_and_sort_suggestions(suggester);

    // Filter low-impact suggestions
    filter_low_impact_suggestions(suggester);
}

// Generate compute optimization suggestions
static void suggester_gen_compute(OptimizationSuggester* suggester,
                                  const BottleneckInfo* bottleneck) {
    if (bottleneck->cpu_utilization > 0.9) {
        add_suggestion(suggester,
            "Enable SIMD vectorization for compute-intensive loops",
            SUGGEST_COMPUTE_OPTIMIZATION, 0.3);

        add_suggestion(suggester,
            "Increase thread-level parallelism with work stealing",
            SUGGEST_COMPUTE_OPTIMIZATION, 0.25);

        add_suggestion(suggester,
            "Apply loop tiling for better cache utilization",
            SUGGEST_COMPUTE_OPTIMIZATION, 0.2);
    }

    if (bottleneck->gpu_utilization > 0.9) {
        add_suggestion(suggester,
            "Optimize GPU kernel launch configurations",
            SUGGEST_COMPUTE_OPTIMIZATION, 0.35);

        add_suggestion(suggester,
            "Use tensor cores for matrix operations",
            SUGGEST_COMPUTE_OPTIMIZATION, 0.4);

        add_suggestion(suggester,
            "Implement kernel fusion to reduce memory transfers",
            SUGGEST_COMPUTE_OPTIMIZATION, 0.3);
    }

    if (bottleneck->cpu_utilization < 0.3 && bottleneck->gpu_utilization < 0.3) {
        add_suggestion(suggester,
            "Workload may be I/O or memory bound - profile carefully",
            SUGGEST_COMPUTE_OPTIMIZATION, 0.15);
    }
}

// Generate memory optimization suggestions
static void suggester_gen_memory(OptimizationSuggester* suggester,
                                 const BottleneckInfo* bottleneck) {
    if (bottleneck->has_memory_pressure) {
        add_suggestion(suggester,
            "Implement gradient checkpointing to reduce memory footprint",
            SUGGEST_MEMORY_OPTIMIZATION, 0.35);

        add_suggestion(suggester,
            "Use mixed precision (FP16/BF16) for intermediate computations",
            SUGGEST_MEMORY_OPTIMIZATION, 0.3);

        add_suggestion(suggester,
            "Enable memory pooling with pre-allocation",
            SUGGEST_MEMORY_OPTIMIZATION, 0.25);
    }

    if (bottleneck->memory_utilization > 0.8) {
        add_suggestion(suggester,
            "Optimize data layout for sequential memory access",
            SUGGEST_MEMORY_OPTIMIZATION, 0.2);

        add_suggestion(suggester,
            "Implement software prefetching for predictable access patterns",
            SUGGEST_MEMORY_OPTIMIZATION, 0.15);

        add_suggestion(suggester,
            "Use compression for large intermediate tensors",
            SUGGEST_MEMORY_OPTIMIZATION, 0.25);
    }
}

// Generate communication optimization suggestions
static void suggester_gen_communication(OptimizationSuggester* suggester,
                                        const BottleneckInfo* bottleneck) {
    if (bottleneck->has_communication_overhead) {
        add_suggestion(suggester,
            "Implement gradient compression with error feedback",
            SUGGEST_COMMUNICATION_OPTIMIZATION, 0.35);

        add_suggestion(suggester,
            "Use ring-allreduce for large tensor synchronization",
            SUGGEST_COMMUNICATION_OPTIMIZATION, 0.3);

        add_suggestion(suggester,
            "Overlap communication with computation",
            SUGGEST_COMMUNICATION_OPTIMIZATION, 0.4);
    }

    if (bottleneck->network_utilization > 0.7) {
        add_suggestion(suggester,
            "Increase batch size to improve communication efficiency",
            SUGGEST_COMMUNICATION_OPTIMIZATION, 0.2);

        add_suggestion(suggester,
            "Implement hierarchical allreduce for multi-node training",
            SUGGEST_COMMUNICATION_OPTIMIZATION, 0.25);
    }
}

// Generate quantum optimization suggestions
static void suggester_gen_quantum(OptimizationSuggester* suggester,
                                  const BottleneckInfo* bottleneck) {
    if (bottleneck->quantum_utilization > 0.8) {
        add_suggestion(suggester,
            "Reduce quantum circuit depth through gate optimization",
            SUGGEST_QUANTUM_OPTIMIZATION, 0.3);

        add_suggestion(suggester,
            "Implement error mitigation with zero-noise extrapolation",
            SUGGEST_QUANTUM_OPTIMIZATION, 0.25);

        add_suggestion(suggester,
            "Use variational circuit compilation for hardware-efficient ansatz",
            SUGGEST_QUANTUM_OPTIMIZATION, 0.35);
    }

    add_suggestion(suggester,
        "Consider hybrid quantum-classical partitioning",
        SUGGEST_QUANTUM_OPTIMIZATION, 0.2);
}

// Generate workload optimization suggestions
static void suggester_gen_workload(OptimizationSuggester* suggester,
                                   const BottleneckInfo* bottleneck) {
    add_suggestion(suggester,
        "Enable asynchronous data loading with prefetch queue",
        SUGGEST_WORKLOAD_OPTIMIZATION, 0.25);

    add_suggestion(suggester,
        "Implement dynamic batching based on sequence lengths",
        SUGGEST_WORKLOAD_OPTIMIZATION, 0.2);

    if (bottleneck->severity > 0.5) {
        add_suggestion(suggester,
            "Consider data sharding across distributed workers",
            SUGGEST_WORKLOAD_OPTIMIZATION, 0.3);
    }
}

// Add a suggestion to the suggester
static void add_suggestion(OptimizationSuggester* suggester,
                          const char* description,
                          SuggestionType type,
                          double base_impact) {
    if (!suggester || !description) return;
    if (suggester->num_suggestions >= suggester->capacity) return;

    // Check for duplicate suggestions
    for (size_t i = 0; i < suggester->num_suggestions; i++) {
        if (suggester->suggestions[i].type == type &&
            strcmp(suggester->suggestions[i].description, description) == 0) {
            return;  // Already exists
        }
    }

    Suggestion* s = &suggester->suggestions[suggester->num_suggestions];
    s->description = strdup(description);
    s->type = type;
    s->impact_score = base_impact;
    s->confidence = 0.5;
    s->is_implemented = false;
    s->timestamp = time(NULL);

    suggester->num_suggestions++;
    suggester->total_suggestions_generated++;
}

// Predict impact using ML model and history
static double predict_impact(OptimizationSuggester* suggester,
                            const Suggestion* suggestion,
                            const BottleneckInfo* bottleneck) {
    double base_impact = suggestion->impact_score;

    // Adjust based on historical effectiveness
    if (suggester->history && suggester->config.enable_history_tracking) {
        int type_idx = (int)suggestion->type;
        if (type_idx < NUM_SUGGESTION_TYPES) {
            double effectiveness = suggester->history->type_effectiveness[type_idx];
            base_impact *= (0.5 + effectiveness);  // Scale 0.5x to 1.5x
        }
    }

    // Adjust based on bottleneck severity
    if (bottleneck) {
        base_impact *= (0.5 + bottleneck->severity);
    }

    return fmin(1.0, fmax(0.0, base_impact));
}

// Score and sort suggestions by predicted impact
static void score_and_sort_suggestions(OptimizationSuggester* suggester) {
    if (!suggester || suggester->num_suggestions == 0) return;

    // Simple bubble sort (small array)
    for (size_t i = 0; i < suggester->num_suggestions - 1; i++) {
        for (size_t j = 0; j < suggester->num_suggestions - i - 1; j++) {
            if (suggester->suggestions[j].impact_score <
                suggester->suggestions[j + 1].impact_score) {
                Suggestion temp = suggester->suggestions[j];
                suggester->suggestions[j] = suggester->suggestions[j + 1];
                suggester->suggestions[j + 1] = temp;
            }
        }
    }
}

// Filter out low-impact suggestions
static void filter_low_impact_suggestions(OptimizationSuggester* suggester) {
    if (!suggester) return;

    size_t write_idx = 0;
    for (size_t read_idx = 0; read_idx < suggester->num_suggestions; read_idx++) {
        if (suggester->suggestions[read_idx].impact_score >=
            suggester->config.min_impact_threshold) {
            if (write_idx != read_idx) {
                suggester->suggestions[write_idx] = suggester->suggestions[read_idx];
            }
            write_idx++;
        } else {
            cleanup_suggestion(&suggester->suggestions[read_idx]);
        }
    }
    suggester->num_suggestions = write_idx;
}

// Get active suggestions
const Suggestion* suggester_get_suggestions(const OptimizationSuggester* suggester,
                                            size_t* num_suggestions) {
    if (!suggester) {
        if (num_suggestions) *num_suggestions = 0;
        return NULL;
    }

    if (num_suggestions) *num_suggestions = suggester->num_suggestions;
    return suggester->suggestions;
}

// Get top-priority suggestion
const Suggestion* suggester_get_top_suggestion(const OptimizationSuggester* suggester) {
    if (!suggester || suggester->num_suggestions == 0) return NULL;
    return &suggester->suggestions[0];
}

// Mark suggestion as implemented
void suggester_mark_implemented(OptimizationSuggester* suggester, size_t index) {
    if (!suggester || index >= suggester->num_suggestions) return;

    suggester->suggestions[index].is_implemented = true;
    suggester->total_suggestions_implemented++;
}

// Track effectiveness of implemented suggestion
void suggester_track_effectiveness(OptimizationSuggester* suggester,
                                   const Suggestion* suggestion,
                                   double actual_improvement) {
    if (!suggester || !suggestion) return;
    if (!suggester->history || !suggester->config.enable_history_tracking) return;

    // Add to history
    SuggestionHistory* h = suggester->history;
    if (h->num_entries >= h->capacity) {
        // Shift entries to make room
        memmove(h->entries, h->entries + 1,
               (h->capacity - 1) * sizeof(SuggestionHistoryEntry));
        h->num_entries = h->capacity - 1;
    }

    SuggestionHistoryEntry* entry = &h->entries[h->num_entries++];
    entry->type = suggestion->type;
    entry->impact_predicted = suggestion->impact_score;
    entry->impact_actual = actual_improvement;
    entry->timestamp = time(NULL);
    entry->was_effective = actual_improvement > 0.0;

    // Update type effectiveness with exponential moving average
    int type_idx = (int)suggestion->type;
    if (type_idx < NUM_SUGGESTION_TYPES) {
        h->type_counts[type_idx]++;
        double alpha = suggester->config.learning_rate;
        h->type_effectiveness[type_idx] = (1.0 - alpha) * h->type_effectiveness[type_idx] +
                                          alpha * (actual_improvement > 0 ? 1.0 : 0.0);
    }

    // Update global effectiveness
    suggester->average_effectiveness =
        0.95 * suggester->average_effectiveness + 0.05 * actual_improvement;
}

// Get historical effectiveness for suggestion type
double suggester_get_type_effectiveness(const OptimizationSuggester* suggester,
                                        SuggestionType type) {
    if (!suggester || !suggester->history) return 0.5;

    int type_idx = (int)type;
    if (type_idx >= NUM_SUGGESTION_TYPES) return 0.5;

    return suggester->history->type_effectiveness[type_idx];
}

// Clear all suggestions
void suggester_clear(OptimizationSuggester* suggester) {
    if (!suggester) return;

    for (size_t i = 0; i < suggester->num_suggestions; i++) {
        cleanup_suggestion(&suggester->suggestions[i]);
    }
    suggester->num_suggestions = 0;
}

// Cleanup a single suggestion
static void cleanup_suggestion(Suggestion* suggestion) {
    if (!suggestion) return;
    free(suggestion->description);
    suggestion->description = NULL;
}

// Cleanup suggestion history
static void cleanup_suggestion_history(SuggestionHistory* history) {
    if (!history) return;
    free(history->entries);
    free(history);
}

// Clean up optimization suggester
void cleanup_optimization_suggester(OptimizationSuggester* suggester) {
    if (!suggester) return;

    // Cleanup suggestions
    for (size_t i = 0; i < suggester->num_suggestions; i++) {
        cleanup_suggestion(&suggester->suggestions[i]);
    }
    free(suggester->suggestions);

    // Cleanup history
    cleanup_suggestion_history(suggester->history);

    // Cleanup ML model
    if (suggester->impact_model) {
        cleanup_ml_model(suggester->impact_model);
    }

    free(suggester);
}

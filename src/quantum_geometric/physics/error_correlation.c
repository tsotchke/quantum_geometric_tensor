/**
 * @file error_correlation.c
 * @brief Implementation of error correlation analysis
 */

#include "quantum_geometric/physics/error_correlation.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Internal state for correlation tracking
static struct {
    CorrelationConfig config;
    SyndromeVertex** history;
    size_t* history_counts;
    size_t history_index;
    bool initialized;
} correlation_state = {0};

bool init_error_correlation(const CorrelationConfig* config) {
    if (!config || config->history_length == 0) {
        return false;
    }

    // Allocate history buffers
    correlation_state.history = (SyndromeVertex**)malloc(
        config->history_length * sizeof(SyndromeVertex*));
    correlation_state.history_counts = (size_t*)malloc(
        config->history_length * sizeof(size_t));

    if (!correlation_state.history || !correlation_state.history_counts) {
        cleanup_error_correlation();
        return false;
    }

    // Initialize history tracking
    for (size_t i = 0; i < config->history_length; i++) {
        correlation_state.history[i] = NULL;
        correlation_state.history_counts[i] = 0;
    }

    // Store configuration
    memcpy(&correlation_state.config, config, sizeof(CorrelationConfig));
    correlation_state.history_index = 0;
    correlation_state.initialized = true;

    return true;
}

void cleanup_error_correlation(void) {
    if (correlation_state.history) {
        for (size_t i = 0; i < correlation_state.config.history_length; i++) {
            free(correlation_state.history[i]);
        }
        free(correlation_state.history);
        correlation_state.history = NULL;
    }

    free(correlation_state.history_counts);
    correlation_state.history_counts = NULL;
    correlation_state.initialized = false;
}

ErrorCorrelation analyze_error_correlations(const MatchingGraph* graph,
                                          const quantum_state* state) {
    ErrorCorrelation correlation = {0};
    if (!graph || !state || !correlation_state.initialized) {
        return correlation;
    }

    // Store current syndromes in history
    size_t hist_idx = correlation_state.history_index;
    free(correlation_state.history[hist_idx]);
    correlation_state.history[hist_idx] = (SyndromeVertex*)malloc(
        graph->num_vertices * sizeof(SyndromeVertex));
    
    if (!correlation_state.history[hist_idx]) {
        return correlation;
    }

    memcpy(correlation_state.history[hist_idx], graph->vertices,
           graph->num_vertices * sizeof(SyndromeVertex));
    correlation_state.history_counts[hist_idx] = graph->num_vertices;

    // Calculate hardware-aware spatial correlations
    double total_spatial = 0.0;
    size_t spatial_count = 0;
    double hw_factor = get_hardware_reliability_factor();
    double noise_factor = get_noise_factor();
    
    // Get hardware-specific thresholds
    double spatial_threshold = correlation_state.config.spatial_threshold * 
                             (1.0 - noise_factor) * hw_factor;
                             
    for (size_t i = 0; i < graph->num_vertices; i++) {
        // Get hardware-specific qubit reliability
        double qubit_reliability = get_qubit_reliability(i);
        
        for (size_t j = i + 1; j < graph->num_vertices; j++) {
            // Check correlation with hardware factors
            if (is_spatially_correlated(&graph->vertices[i],
                                      &graph->vertices[j],
                                      spatial_threshold)) {
                                      
                // Calculate correlation with hardware weighting
                double correlation = calculate_spatial_correlation(
                    &graph->vertices[i],
                    &graph->vertices[j]
                );
                
                // Apply hardware-specific adjustments
                correlation *= qubit_reliability * get_qubit_reliability(j);
                correlation *= get_gate_fidelity();
                correlation *= (1.0 - get_crosstalk_factor(i, j));
                
                total_spatial += correlation;
                spatial_count++;
            }
        }
    }
    
    // Calculate final spatial correlation with hardware factors
    correlation.spatial_correlation = spatial_count > 0 ?
        total_spatial / spatial_count * hw_factor : 0.0;

    // Calculate hardware-aware temporal correlations with fast feedback
    double total_temporal = 0.0;
    size_t temporal_count = 0;
    size_t prev_idx = (hist_idx + correlation_state.config.history_length - 1) %
                      correlation_state.config.history_length;
                      
    // Get hardware-specific temporal factors
    double temporal_threshold = correlation_state.config.temporal_threshold *
                              (1.0 - get_temporal_noise_factor());
    double coherence_factor = get_coherence_time_factor();
                              
    if (correlation_state.history[prev_idx]) {
        for (size_t i = 0; i < graph->num_vertices; i++) {
            // Get vertex-specific hardware factors
            double vertex_stability = get_vertex_stability(i);
            
            for (size_t j = 0; j < correlation_state.history_counts[prev_idx]; j++) {
                // Check temporal correlation with hardware factors
                if (is_temporally_correlated(&graph->vertices[i],
                                           &correlation_state.history[prev_idx][j],
                                           temporal_threshold)) {
                                           
                    // Calculate correlation with hardware weighting
                    double correlation = calculate_temporal_correlation(
                        &graph->vertices[i],
                        &correlation_state.history[prev_idx][j]
                    );
                    
                    // Apply hardware-specific temporal adjustments
                    correlation *= vertex_stability;
                    correlation *= coherence_factor;
                    correlation *= get_measurement_stability(i);
                    
                    // Apply fast feedback adjustment
                    correlation *= get_feedback_factor(i, j);
                    
                    total_temporal += correlation;
                    temporal_count++;
                    
                    // Trigger fast feedback if needed
                    if (should_trigger_feedback(correlation)) {
                        trigger_correlation_feedback(i, j, correlation);
                    }
                }
            }
        }
    }
    
    // Calculate final temporal correlation with hardware factors
    correlation.temporal_correlation = temporal_count > 0 ?
        total_temporal / temporal_count * hw_factor : 0.0;

    // Calculate cross-correlations if enabled
    if (correlation_state.config.enable_cross_correlation) {
        correlation.cross_correlation = sqrt(
            correlation.spatial_correlation * correlation.temporal_correlation);
    }

    // Estimate characteristic scales
    correlation.correlation_length = estimate_correlation_length(graph);
    correlation.correlation_time = estimate_correlation_time(graph);

    // Update history index
    correlation_state.history_index = (hist_idx + 1) %
                                    correlation_state.config.history_length;

    return correlation;
}

ErrorCorrelation update_correlation_model(const MatchingGraph* graph,
                                        const ErrorCorrelation* prev_correlation) {
    ErrorCorrelation new_correlation = {0};
    if (!graph || !prev_correlation) {
        return new_correlation;
    }

    // Analyze current correlations
    new_correlation = analyze_error_correlations(graph, NULL);

    // Apply exponential moving average update
    const double alpha = 0.3; // Update weight
    new_correlation.spatial_correlation = alpha * new_correlation.spatial_correlation +
        (1 - alpha) * prev_correlation->spatial_correlation;
    new_correlation.temporal_correlation = alpha * new_correlation.temporal_correlation +
        (1 - alpha) * prev_correlation->temporal_correlation;
    new_correlation.cross_correlation = alpha * new_correlation.cross_correlation +
        (1 - alpha) * prev_correlation->cross_correlation;
    
    // Update characteristic scales with smoothing
    new_correlation.correlation_length = (size_t)(
        alpha * new_correlation.correlation_length +
        (1 - alpha) * prev_correlation->correlation_length);
    new_correlation.correlation_time = (size_t)(
        alpha * new_correlation.correlation_time +
        (1 - alpha) * prev_correlation->correlation_time);

    return new_correlation;
}

CorrelationType detect_correlation_type(const SyndromeVertex* vertex1,
                                      const SyndromeVertex* vertex2) {
    if (!vertex1 || !vertex2) {
        return CORRELATION_NONE;
    }

    bool spatial = is_spatially_correlated(
        vertex1, vertex2, correlation_state.config.spatial_threshold);
    bool temporal = is_temporally_correlated(
        vertex1, vertex2, correlation_state.config.temporal_threshold);

    if (spatial && temporal) {
        return CORRELATION_SPATIOTEMPORAL;
    } else if (spatial) {
        return CORRELATION_SPATIAL;
    } else if (temporal) {
        return CORRELATION_TEMPORAL;
    }
    return CORRELATION_NONE;
}

double calculate_spatial_correlation(const SyndromeVertex* vertex1,
                                  const SyndromeVertex* vertex2) {
    if (!vertex1 || !vertex2) {
        return 0.0;
    }

    // Calculate distance-based correlation
    double dx = (double)vertex1->x - (double)vertex2->x;
    double dy = (double)vertex1->y - (double)vertex2->y;
    double dz = (double)vertex1->z - (double)vertex2->z;
    double distance = sqrt(dx*dx + dy*dy + dz*dz);

    // Correlation decays exponentially with distance
    return exp(-distance / correlation_state.config.max_correlation_dist);
}

double calculate_temporal_correlation(const SyndromeVertex* vertex1,
                                   const SyndromeVertex* vertex2) {
    if (!vertex1 || !vertex2) {
        return 0.0;
    }

    // Calculate time-based correlation
    double dt = (double)abs((int)vertex1->timestamp - (int)vertex2->timestamp);
    
    // Correlation decays exponentially with time difference
    return exp(-dt / correlation_state.config.history_length);
}

bool is_spatially_correlated(const SyndromeVertex* v1,
                           const SyndromeVertex* v2,
                           double threshold) {
    return calculate_spatial_correlation(v1, v2) > threshold;
}

bool is_temporally_correlated(const SyndromeVertex* v1,
                            const SyndromeVertex* v2,
                            double threshold) {
    return calculate_temporal_correlation(v1, v2) > threshold;
}

double calculate_correlation_strength(const SyndromeVertex* v1,
                                   const SyndromeVertex* v2) {
    double spatial = calculate_spatial_correlation(v1, v2);
    double temporal = calculate_temporal_correlation(v1, v2);
    return sqrt(spatial * spatial + temporal * temporal);
}

size_t estimate_correlation_length(const MatchingGraph* graph) {
    if (!graph || graph->num_vertices < 2) {
        return 0;
    }

    // Find maximum distance between correlated syndromes
    size_t max_dist = 0;
    for (size_t i = 0; i < graph->num_vertices; i++) {
        for (size_t j = i + 1; j < graph->num_vertices; j++) {
            if (is_spatially_correlated(&graph->vertices[i],
                                      &graph->vertices[j],
                                      correlation_state.config.spatial_threshold)) {
                size_t dist = get_correction_chain_length(&graph->vertices[i],
                                                        &graph->vertices[j]);
                if (dist > max_dist) {
                    max_dist = dist;
                }
            }
        }
    }
    return max_dist;
}

size_t estimate_correlation_time(const MatchingGraph* graph) {
    if (!graph || !correlation_state.initialized) {
        return 0;
    }

    // Find maximum time difference between correlated syndromes
    size_t max_dt = 0;
    for (size_t i = 0; i < correlation_state.config.history_length; i++) {
        if (!correlation_state.history[i]) {
            continue;
        }
        
        for (size_t j = 0; j < graph->num_vertices; j++) {
            for (size_t k = 0; k < correlation_state.history_counts[i]; k++) {
                if (is_temporally_correlated(&graph->vertices[j],
                                           &correlation_state.history[i][k],
                                           correlation_state.config.temporal_threshold)) {
                    size_t dt = abs((int)graph->vertices[j].timestamp -
                                  (int)correlation_state.history[i][k].timestamp);
                    if (dt > max_dt) {
                        max_dt = dt;
                    }
                }
            }
        }
    }
    return max_dt;
}

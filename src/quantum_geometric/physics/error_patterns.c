/**
 * @file error_patterns.c
 * @brief Implementation of error pattern detection and analysis
 */

#include "quantum_geometric/physics/error_patterns.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Internal pattern detection state
static PatternState pattern_state = {0};

bool init_error_patterns(const PatternConfig* config) {
    if (!config || config->max_patterns > MAX_ERROR_PATTERNS) {
        return false;
    }

    memset(&pattern_state, 0, sizeof(PatternState));
    memcpy(&pattern_state.config, config, sizeof(PatternConfig));
    pattern_state.current_timestamp = 0;
    return true;
}

void cleanup_error_patterns(void) {
    memset(&pattern_state, 0, sizeof(PatternState));
}

size_t detect_error_patterns(const MatchingGraph* graph,
                           const ErrorCorrelation* correlation) {
    if (!graph || !correlation || !pattern_state.config.max_patterns) {
        return 0;
    }

    size_t patterns_found = 0;
    size_t vertices_remaining = graph->num_vertices;
    size_t current_vertex = 0;

    while (vertices_remaining > 0 && 
           patterns_found < pattern_state.config.max_patterns) {
        
        // Find connected components that could form patterns
        size_t pattern_size = 0;
        SyndromeVertex pattern_vertices[MAX_PATTERN_SIZE];
        
        // Start with current vertex
        pattern_vertices[pattern_size++] = graph->vertices[current_vertex];
        bool* visited = calloc(graph->num_vertices, sizeof(bool));
        visited[current_vertex] = true;
        vertices_remaining--;

        // Find connected vertices that could be part of the pattern
        for (size_t i = 0; i < graph->num_vertices && pattern_size < MAX_PATTERN_SIZE; i++) {
            if (!visited[i]) {
                double corr = calculate_correlation_strength(&graph->vertices[current_vertex],
                                                          &graph->vertices[i]);
                if (corr > pattern_state.config.similarity_threshold) {
                    pattern_vertices[pattern_size++] = graph->vertices[i];
                    visited[i] = true;
                    vertices_remaining--;
                }
            }
        }

        // Classify and store pattern if large enough
        if (pattern_size >= pattern_state.config.min_occurrences) {
            ErrorPattern new_pattern = {
                .type = classify_pattern_type(pattern_vertices, pattern_size),
                .size = pattern_size,
                .weight = calculate_pattern_weight(&pattern_state.patterns[patterns_found]),
                .is_active = true,
                .correlation = *correlation
            };

            // Copy vertices
            memcpy(new_pattern.vertices, pattern_vertices,
                   pattern_size * sizeof(SyndromeVertex));

            // Initialize timing
            if (pattern_state.config.track_timing) {
                new_pattern.timing.first_seen = pattern_state.current_timestamp;
                new_pattern.timing.last_seen = pattern_state.current_timestamp;
                new_pattern.timing.occurrences = 1;
                new_pattern.timing.frequency = 1.0;
            }

            // Try to merge with existing similar pattern
            bool merged = false;
            for (size_t i = 0; i < pattern_state.num_patterns; i++) {
                if (pattern_state.patterns[i].is_active) {
                    double similarity = calculate_pattern_similarity(
                        &new_pattern, &pattern_state.patterns[i]);
                    if (similarity > pattern_state.config.similarity_threshold) {
                        merge_similar_patterns(&pattern_state.patterns[i], &new_pattern);
                        merged = true;
                        break;
                    }
                }
            }

            // Add as new pattern if not merged
            if (!merged && pattern_state.num_patterns < pattern_state.config.max_patterns) {
                pattern_state.patterns[pattern_state.num_patterns++] = new_pattern;
                patterns_found++;
            }
        }

        // Move to next unvisited vertex
        while (current_vertex < graph->num_vertices && visited[current_vertex]) {
            current_vertex++;
        }

        free(visited);
    }

    pattern_state.current_timestamp++;
    return patterns_found;
}

size_t update_pattern_database(const MatchingGraph* graph) {
    if (!graph) {
        return 0;
    }

    // Update existing patterns
    for (size_t i = 0; i < pattern_state.num_patterns; i++) {
        if (pattern_state.patterns[i].is_active) {
            bool pattern_found = false;
            
            // Check if pattern still exists in current measurements
            for (size_t j = 0; j < graph->num_vertices; j++) {
                if (match_pattern(&graph->vertices[j], 1, &pattern_state.patterns[i])) {
                    pattern_found = true;
                    update_pattern_timing(&pattern_state.patterns[i],
                                       pattern_state.current_timestamp);
                    break;
                }
            }

            // Deactivate patterns that haven't been seen recently
            if (!pattern_found && pattern_state.config.track_timing) {
                size_t time_since_last = pattern_state.current_timestamp - 
                                       pattern_state.patterns[i].timing.last_seen;
                if (time_since_last > pattern_state.config.min_occurrences) {
                    pattern_state.patterns[i].is_active = false;
                }
            }
        }
    }

    // Prune inactive patterns periodically
    if (pattern_state.current_timestamp % 100 == 0) {
        prune_inactive_patterns();
    }

    // Count active patterns
    size_t active_count = 0;
    for (size_t i = 0; i < pattern_state.num_patterns; i++) {
        if (pattern_state.patterns[i].is_active) {
            active_count++;
        }
    }

    return active_count;
}

PatternType classify_pattern_type(const SyndromeVertex* vertices, size_t size) {
    if (!vertices || size == 0) {
        return PATTERN_UNKNOWN;
    }

    if (size == 1) {
        return PATTERN_POINT;
    }

    // Check if vertices form a line
    if (size == 2) {
        return PATTERN_LINE;
    }

    // Check for clusters (densely connected vertices)
    double avg_distance = 0;
    size_t connections = 0;
    for (size_t i = 0; i < size; i++) {
        for (size_t j = i + 1; j < size; j++) {
            double dx = vertices[i].x - vertices[j].x;
            double dy = vertices[i].y - vertices[j].y;
            double dz = vertices[i].z - vertices[j].z;
            avg_distance += sqrt(dx*dx + dy*dy + dz*dz);
            connections++;
        }
    }
    avg_distance /= connections;

    if (avg_distance < 2.0) {
        return PATTERN_CLUSTER;
    }

    // Check for chains (linear sequence of vertices)
    bool is_chain = true;
    for (size_t i = 1; i < size - 1; i++) {
        double prev_dist = sqrt(
            pow(vertices[i].x - vertices[i-1].x, 2) +
            pow(vertices[i].y - vertices[i-1].y, 2) +
            pow(vertices[i].z - vertices[i-1].z, 2));
        double next_dist = sqrt(
            pow(vertices[i].x - vertices[i+1].x, 2) +
            pow(vertices[i].y - vertices[i+1].y, 2) +
            pow(vertices[i].z - vertices[i+1].z, 2));
        if (fabs(prev_dist - next_dist) > 0.5) {
            is_chain = false;
            break;
        }
    }
    if (is_chain) {
        return PATTERN_CHAIN;
    }

    // Check for cycles (closed chains)
    double first_last_dist = sqrt(
        pow(vertices[0].x - vertices[size-1].x, 2) +
        pow(vertices[0].y - vertices[size-1].y, 2) +
        pow(vertices[0].z - vertices[size-1].z, 2));
    if (is_chain && first_last_dist < 2.0) {
        return PATTERN_CYCLE;
    }

    // Check for braids (intertwined chains)
    bool has_crossings = false;
    for (size_t i = 0; i < size - 2; i++) {
        for (size_t j = i + 2; j < size; j++) {
            double dx = vertices[i].x - vertices[j].x;
            double dy = vertices[i].y - vertices[j].y;
            double dz = vertices[i].z - vertices[j].z;
            double dist = sqrt(dx*dx + dy*dy + dz*dz);
            if (dist < 1.0) {
                has_crossings = true;
                break;
            }
        }
    }
    if (has_crossings) {
        return PATTERN_BRAID;
    }

    return PATTERN_UNKNOWN;
}

double calculate_pattern_similarity(const ErrorPattern* pattern1,
                                 const ErrorPattern* pattern2) {
    if (!pattern1 || !pattern2) {
        return 0.0;
    }

    // Compare pattern types
    if (pattern1->type != pattern2->type) {
        return 0.0;
    }

    // Compare sizes
    if (abs((int)pattern1->size - (int)pattern2->size) > 2) {
        return 0.0;
    }

    // Calculate average vertex distance
    double total_similarity = 0.0;
    size_t comparisons = 0;

    for (size_t i = 0; i < pattern1->size; i++) {
        double best_match = 0.0;
        for (size_t j = 0; j < pattern2->size; j++) {
            double dx = pattern1->vertices[i].x - pattern2->vertices[j].x;
            double dy = pattern1->vertices[i].y - pattern2->vertices[j].y;
            double dz = pattern1->vertices[i].z - pattern2->vertices[j].z;
            double dist = sqrt(dx*dx + dy*dy + dz*dz);
            double similarity = exp(-dist);
            if (similarity > best_match) {
                best_match = similarity;
            }
        }
        total_similarity += best_match;
        comparisons++;
    }

    return comparisons > 0 ? total_similarity / comparisons : 0.0;
}

bool match_pattern(const SyndromeVertex* vertices,
                  size_t size,
                  const ErrorPattern* pattern) {
    if (!vertices || !pattern || size == 0) {
        return false;
    }

    // For single vertex matching
    if (size == 1) {
        for (size_t i = 0; i < pattern->size; i++) {
            double dx = vertices[0].x - pattern->vertices[i].x;
            double dy = vertices[0].y - pattern->vertices[i].y;
            double dz = vertices[0].z - pattern->vertices[i].z;
            double dist = sqrt(dx*dx + dy*dy + dz*dz);
            if (dist < 1.0) {
                return true;
            }
        }
        return false;
    }

    // For multi-vertex matching
    double similarity = calculate_pattern_similarity(pattern, (ErrorPattern*)&vertices);
    return similarity > pattern_state.config.similarity_threshold;
}

void update_pattern_timing(ErrorPattern* pattern, size_t current_time) {
    if (!pattern || !pattern_state.config.track_timing) {
        return;
    }

    pattern->timing.last_seen = current_time;
    pattern->timing.occurrences++;
    
    double time_span = current_time - pattern->timing.first_seen + 1;
    pattern->timing.frequency = pattern->timing.occurrences / time_span;
}

double calculate_pattern_weight(const ErrorPattern* pattern) {
    if (!pattern) {
        return 0.0;
    }

    double base_weight = pattern->size * 0.5;
    
    if (pattern_state.config.track_timing) {
        base_weight *= pattern->timing.frequency;
    }

    // Adjust weight based on pattern type
    switch (pattern->type) {
        case PATTERN_BRAID:
            base_weight *= 2.0;
            break;
        case PATTERN_CYCLE:
            base_weight *= 1.5;
            break;
        case PATTERN_CHAIN:
            base_weight *= 1.2;
            break;
        case PATTERN_CLUSTER:
            base_weight *= 1.1;
            break;
        default:
            break;
    }

    return base_weight;
}

bool merge_similar_patterns(ErrorPattern* pattern1, ErrorPattern* pattern2) {
    if (!pattern1 || !pattern2) {
        return false;
    }

    // Update timing information
    if (pattern_state.config.track_timing) {
        pattern1->timing.first_seen = fmin(pattern1->timing.first_seen,
                                         pattern2->timing.first_seen);
        pattern1->timing.last_seen = fmax(pattern1->timing.last_seen,
                                        pattern2->timing.last_seen);
        pattern1->timing.occurrences += pattern2->timing.occurrences;
        
        double time_span = pattern1->timing.last_seen - 
                          pattern1->timing.first_seen + 1;
        pattern1->timing.frequency = pattern1->timing.occurrences / time_span;
    }

    // Update weight
    pattern1->weight = fmax(pattern1->weight, pattern2->weight);

    // Merge correlation data
    pattern1->correlation.spatial_correlation = 
        (pattern1->correlation.spatial_correlation +
         pattern2->correlation.spatial_correlation) * 0.5;
    pattern1->correlation.temporal_correlation =
        (pattern1->correlation.temporal_correlation +
         pattern2->correlation.temporal_correlation) * 0.5;

    return true;
}

void prune_inactive_patterns(void) {
    size_t write_idx = 0;
    for (size_t read_idx = 0; read_idx < pattern_state.num_patterns; read_idx++) {
        if (pattern_state.patterns[read_idx].is_active) {
            if (write_idx != read_idx) {
                pattern_state.patterns[write_idx] = pattern_state.patterns[read_idx];
            }
            write_idx++;
        }
    }
    pattern_state.num_patterns = write_idx;
}

bool get_pattern_statistics(size_t pattern_idx, PatternTiming* timing) {
    if (pattern_idx >= pattern_state.num_patterns || !timing) {
        return false;
    }

    if (!pattern_state.config.track_timing) {
        return false;
    }

    *timing = pattern_state.patterns[pattern_idx].timing;
    return true;
}

bool is_pattern_active(size_t pattern_idx) {
    if (pattern_idx >= pattern_state.num_patterns) {
        return false;
    }
    return pattern_state.patterns[pattern_idx].is_active;
}

size_t get_pattern_count(void) {
    return pattern_state.num_patterns;
}

const ErrorPattern* get_pattern(size_t pattern_idx) {
    if (pattern_idx >= pattern_state.num_patterns) {
        return NULL;
    }
    return &pattern_state.patterns[pattern_idx];
}

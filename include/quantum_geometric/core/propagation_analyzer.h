/**
 * @file propagation_analyzer.h
 * @brief Error and state propagation analysis for quantum systems
 */

#ifndef PROPAGATION_ANALYZER_H
#define PROPAGATION_ANALYZER_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { PROP_ERROR, PROP_STATE, PROP_CORRELATION, PROP_ENTANGLEMENT, PROP_NOISE } PropagationType;
typedef enum { PROP_FORWARD, PROP_BACKWARD, PROP_BIDIRECTIONAL } PropagationDirection;

typedef struct { size_t node_id; size_t* sources; size_t num_sources; size_t* targets; size_t num_targets; double* weights; double value; } PropagationNode;
typedef struct { PropagationNode* nodes; size_t num_nodes; PropagationType type; PropagationDirection direction; } PropagationGraph;
typedef struct { double* values; size_t num_nodes; double total; double max_value; size_t max_node; } PropagationResult;
typedef struct { PropagationType type; PropagationDirection direction; size_t max_iterations; double threshold; double decay; } PropagationConfig;

int propagation_graph_create(PropagationGraph** graph, PropagationType type);
void propagation_graph_destroy(PropagationGraph* graph);
int propagation_graph_add_node(PropagationGraph* graph, size_t id);
int propagation_graph_add_edge(PropagationGraph* graph, size_t src, size_t dst, double weight);
int propagation_analyze(PropagationGraph* graph, double* initial, PropagationConfig* cfg, PropagationResult** result);
int propagation_compute_influence(PropagationGraph* graph, size_t source, double* influence, size_t* count);
void propagation_result_free(PropagationResult* result);

#ifdef __cplusplus
}
#endif

#endif // PROPAGATION_ANALYZER_H

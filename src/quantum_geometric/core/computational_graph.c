#include "quantum_geometric/core/computational_graph.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include <stdlib.h>
#include <string.h>

// Forward declarations of static functions
static bool resize_node_array(computational_graph_t* graph);
static bool has_cycle(computational_graph_t* graph, size_t node_idx,
                     bool* visited, bool* in_stack);
static void topological_sort(computational_graph_t* graph, size_t node_idx,
                           bool* visited, size_t* sorted, size_t* count);
static size_t calculate_node_level(computation_node_t* node);

#define INITIAL_CAPACITY 16
#define GROWTH_FACTOR 2

// Graph creation and destruction
computational_graph_t* create_computational_graph(geometric_processor_t* processor) {
    if (!processor) return NULL;
    
    computational_graph_t* graph = malloc(sizeof(computational_graph_t));
    if (!graph) return NULL;
    
    graph->nodes = malloc(INITIAL_CAPACITY * sizeof(computation_node_t*));
    if (!graph->nodes) {
        free(graph);
        return NULL;
    }
    
    graph->num_nodes = 0;
    graph->capacity = INITIAL_CAPACITY;
    graph->processor = processor;
    graph->inputs = NULL;
    graph->outputs = NULL;
    graph->num_inputs = 0;
    graph->num_outputs = 0;
    
    return graph;
}

void destroy_computational_graph(computational_graph_t* graph) {
    if (!graph) return;
    
    // Free all nodes
    for (size_t i = 0; i < graph->num_nodes; i++) {
        computation_node_t* node = graph->nodes[i];
        if (node) {
            free(node->inputs);
            free(node->outputs);
            free(node->data);
            free(node);
        }
    }
    
    free(graph->nodes);
    free(graph->inputs);
    free(graph->outputs);
    free(graph);
}

// Node management
computation_node_t* add_node(computational_graph_t* graph,
                           node_type_t type,
                           operation_type_t op_type,
                           void* data) {
    if (!graph) return NULL;
    
    // Resize if needed
    if (graph->num_nodes >= graph->capacity) {
        if (!resize_node_array(graph)) return NULL;
    }
    
    // Create new node
    computation_node_t* node = malloc(sizeof(computation_node_t));
    if (!node) return NULL;
    
    node->type = type;
    node->op_type = op_type;
    node->data = data;
    node->num_inputs = 0;
    node->num_outputs = 0;
    node->inputs = NULL;
    node->outputs = NULL;
    node->forward = NULL;
    node->backward = NULL;
    node->gradient = NULL;
    
    // Add to graph
    graph->nodes[graph->num_nodes++] = node;
    
    // Update inputs/outputs arrays if needed
    if (type == NODE_INPUT) {
        size_t new_size = (graph->num_inputs + 1) * sizeof(computation_node_t*);
        computation_node_t** new_inputs = realloc(graph->inputs, new_size);
        if (new_inputs) {
            graph->inputs = new_inputs;
            graph->inputs[graph->num_inputs++] = node;
        }
    } else if (type == NODE_OUTPUT) {
        size_t new_size = (graph->num_outputs + 1) * sizeof(computation_node_t*);
        computation_node_t** new_outputs = realloc(graph->outputs, new_size);
        if (new_outputs) {
            graph->outputs = new_outputs;
            graph->outputs[graph->num_outputs++] = node;
        }
    }
    
    return node;
}

bool connect_nodes(computation_node_t* source, computation_node_t* target) {
    if (!source || !target) return false;
    
    // Add output connection to source
    size_t new_size = (source->num_outputs + 1) * sizeof(computation_node_t*);
    computation_node_t** new_outputs = realloc(source->outputs, new_size);
    if (!new_outputs) return false;
    source->outputs = new_outputs;
    source->outputs[source->num_outputs++] = target;
    
    // Add input connection to target
    new_size = (target->num_inputs + 1) * sizeof(computation_node_t*);
    computation_node_t** new_inputs = realloc(target->inputs, new_size);
    if (!new_inputs) return false;
    target->inputs = new_inputs;
    target->inputs[target->num_inputs++] = source;
    
    return true;
}

bool disconnect_nodes(computation_node_t* source, computation_node_t* target) {
    if (!source || !target) return false;
    
    // Remove output connection from source
    for (size_t i = 0; i < source->num_outputs; i++) {
        if (source->outputs[i] == target) {
            memmove(&source->outputs[i], &source->outputs[i + 1],
                   (source->num_outputs - i - 1) * sizeof(computation_node_t*));
            source->num_outputs--;
            break;
        }
    }
    
    // Remove input connection from target
    for (size_t i = 0; i < target->num_inputs; i++) {
        if (target->inputs[i] == source) {
            memmove(&target->inputs[i], &target->inputs[i + 1],
                   (target->num_inputs - i - 1) * sizeof(computation_node_t*));
            target->num_inputs--;
            break;
        }
    }
    
    return true;
}

// Graph validation and execution
bool validate_graph(computational_graph_t* graph) {
    if (!graph || !graph->nodes || graph->num_nodes == 0) return false;
    
    // Check for cycles using DFS
    bool* visited = calloc(graph->num_nodes, sizeof(bool));
    bool* in_stack = calloc(graph->num_nodes, sizeof(bool));
    if (!visited || !in_stack) {
        free(visited);
        free(in_stack);
        return false;
    }
    
    bool is_valid = true;
    for (size_t i = 0; i < graph->num_inputs; i++) {
        computation_node_t* start = graph->inputs[i];
        for (size_t j = 0; j < graph->num_nodes; j++) {
            if (graph->nodes[j] == start) {
                if (!is_valid) break;
                memset(visited, 0, graph->num_nodes * sizeof(bool));
                memset(in_stack, 0, graph->num_nodes * sizeof(bool));
                // Check for cycles starting from this input
                is_valid = !has_cycle(graph, j, visited, in_stack);
                break;
            }
        }
    }
    
    free(visited);
    free(in_stack);
    return is_valid;
}

bool execute_graph(computational_graph_t* graph) {
    if (!validate_graph(graph)) return false;
    
    // Topological sort for execution order
    size_t* sorted = malloc(graph->num_nodes * sizeof(size_t));
    size_t sorted_count = 0;
    bool* visited = calloc(graph->num_nodes, sizeof(bool));
    
    if (!sorted || !visited) {
        free(sorted);
        free(visited);
        return false;
    }
    
    // Perform topological sort
    for (size_t i = 0; i < graph->num_inputs; i++) {
        computation_node_t* start = graph->inputs[i];
        for (size_t j = 0; j < graph->num_nodes; j++) {
            if (graph->nodes[j] == start) {
                topological_sort(graph, j, visited, sorted, &sorted_count);
                break;
            }
        }
    }
    
    // Execute nodes in topological order
    bool success = true;
    for (size_t i = 0; i < sorted_count && success; i++) {
        computation_node_t* node = graph->nodes[sorted[i]];
        if (node->forward) {
            node->forward(node);
        }
    }
    
    free(sorted);
    free(visited);
    return success;
}

// Graph analysis
bool analyze_graph(computational_graph_t* graph, graph_metrics_t* metrics) {
    if (!graph || !metrics) return false;
    
    metrics->total_nodes = graph->num_nodes;
    metrics->total_edges = 0;
    metrics->depth = 0;
    metrics->width = 0;
    metrics->complexity = 0;
    metrics->memory_usage = sizeof(computational_graph_t);
    
    // Calculate edges and memory usage
    size_t max_width = 0;
    size_t* level_width = calloc(graph->num_nodes, sizeof(size_t));
    if (!level_width) return false;
    
    for (size_t i = 0; i < graph->num_nodes; i++) {
        computation_node_t* node = graph->nodes[i];
        metrics->total_edges += node->num_outputs;
        metrics->memory_usage += sizeof(computation_node_t);
        metrics->memory_usage += (node->num_inputs + node->num_outputs) * 
                               sizeof(computation_node_t*);
        
        // Calculate level (depth) for each node
        size_t level = calculate_node_level(node);
        level_width[level]++;
        if (level_width[level] > max_width) {
            max_width = level_width[level];
        }
        if (level + 1 > metrics->depth) {
            metrics->depth = level + 1;
        }
    }
    
    metrics->width = max_width;
    
    // Calculate complexity (O(nodes * edges))
    metrics->complexity = (double)metrics->total_nodes * 
                        (double)metrics->total_edges;
    
    free(level_width);
    return true;
}

// Static helper functions
static bool resize_node_array(computational_graph_t* graph) {
    size_t new_capacity = graph->capacity * GROWTH_FACTOR;
    computation_node_t** new_nodes = realloc(graph->nodes, 
                                           new_capacity * sizeof(computation_node_t*));
    if (!new_nodes) return false;
    
    graph->nodes = new_nodes;
    graph->capacity = new_capacity;
    return true;
}

static bool has_cycle(computational_graph_t* graph, size_t node_idx,
                     bool* visited, bool* in_stack) {
    if (in_stack[node_idx]) return true;
    if (visited[node_idx]) return false;
    
    visited[node_idx] = true;
    in_stack[node_idx] = true;
    
    computation_node_t* node = graph->nodes[node_idx];
    for (size_t i = 0; i < node->num_outputs; i++) {
        computation_node_t* next = node->outputs[i];
        for (size_t j = 0; j < graph->num_nodes; j++) {
            if (graph->nodes[j] == next) {
                if (has_cycle(graph, j, visited, in_stack)) return true;
                break;
            }
        }
    }
    
    in_stack[node_idx] = false;
    return false;
}

static void topological_sort(computational_graph_t* graph, size_t node_idx,
                           bool* visited, size_t* sorted, size_t* count) {
    if (visited[node_idx]) return;
    
    visited[node_idx] = true;
    computation_node_t* node = graph->nodes[node_idx];
    
    for (size_t i = 0; i < node->num_outputs; i++) {
        computation_node_t* next = node->outputs[i];
        for (size_t j = 0; j < graph->num_nodes; j++) {
            if (graph->nodes[j] == next) {
                topological_sort(graph, j, visited, sorted, count);
                break;
            }
        }
    }
    
    sorted[(*count)++] = node_idx;
}

static size_t calculate_node_level(computation_node_t* node) {
    if (!node || node->num_inputs == 0) return 0;
    
    size_t max_level = 0;
    for (size_t i = 0; i < node->num_inputs; i++) {
        size_t level = calculate_node_level(node->inputs[i]);
        if (level > max_level) {
            max_level = level;
        }
    }
    
    return max_level + 1;
}

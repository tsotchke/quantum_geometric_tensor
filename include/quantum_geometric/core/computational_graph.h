#ifndef COMPUTATIONAL_GRAPH_H
#define COMPUTATIONAL_GRAPH_H

#include <stdbool.h>
#include <stddef.h>
#include <complex.h>
#include "geometric_processor.h"

// Node types for computational graph
typedef enum {
    NODE_INPUT,           // Input node
    NODE_OPERATION,       // Operation node
    NODE_PARAMETER,       // Parameter node
    NODE_CONSTANT,       // Constant node
    NODE_OUTPUT          // Output node
} node_type_t;

// Operation types
typedef enum {
    OP_UNARY,            // Unary operation
    OP_BINARY,           // Binary operation
    OP_REDUCTION,        // Reduction operation
    OP_TRANSFORM,        // Transform operation
    OP_QUANTUM,          // Quantum operation
    OP_CUSTOM           // Custom operation
} operation_type_t;

// Node structure
typedef struct computation_node_t {
    node_type_t type;                    // Node type
    operation_type_t op_type;            // Operation type
    void* data;                          // Node data
    size_t num_inputs;                   // Number of input edges
    size_t num_outputs;                  // Number of output edges
    struct computation_node_t** inputs;   // Input nodes
    struct computation_node_t** outputs;  // Output nodes
    void (*forward)(struct computation_node_t*);   // Forward computation
    void (*backward)(struct computation_node_t*);  // Backward computation
    void (*gradient)(struct computation_node_t*);  // Gradient computation
} computation_node_t;

// Graph structure
typedef struct {
    computation_node_t** nodes;          // Array of nodes
    size_t num_nodes;                    // Number of nodes
    size_t capacity;                     // Node array capacity
    computation_node_t** inputs;         // Input nodes
    computation_node_t** outputs;        // Output nodes
    size_t num_inputs;                   // Number of input nodes
    size_t num_outputs;                  // Number of output nodes
    geometric_processor_t* processor;     // Associated geometric processor
} computational_graph_t;

// Graph creation and destruction
computational_graph_t* create_computational_graph(geometric_processor_t* processor);
void destroy_computational_graph(computational_graph_t* graph);

// Node management
computation_node_t* add_node(computational_graph_t* graph, 
                           node_type_t type,
                           operation_type_t op_type,
                           void* data);
bool connect_nodes(computation_node_t* source, 
                  computation_node_t* target);
bool disconnect_nodes(computation_node_t* source,
                     computation_node_t* target);

// Graph operations
bool validate_graph(computational_graph_t* graph);
bool optimize_graph(computational_graph_t* graph);
bool execute_graph(computational_graph_t* graph);
bool compute_gradients(computational_graph_t* graph);

// Operation registration
typedef struct {
    void (*forward)(void* data, void** inputs, void* output);
    void (*backward)(void* data, void** inputs, void** gradients);
    void (*gradient)(void* data, void** inputs, void* gradient);
} operation_functions_t;

bool register_operation(computational_graph_t* graph,
                       const char* name,
                       operation_functions_t functions);

// Utility functions
bool export_graph(computational_graph_t* graph, const char* filename);
bool import_graph(computational_graph_t* graph, const char* filename);
void print_graph(computational_graph_t* graph);

// Graph analysis
typedef struct {
    size_t total_nodes;
    size_t total_edges;
    size_t depth;
    size_t width;
    double complexity;
    double memory_usage;
} graph_metrics_t;

bool analyze_graph(computational_graph_t* graph, graph_metrics_t* metrics);

#endif // COMPUTATIONAL_GRAPH_H

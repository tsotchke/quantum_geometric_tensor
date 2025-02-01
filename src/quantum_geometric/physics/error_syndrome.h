#ifndef QUANTUM_GEOMETRIC_ERROR_SYNDROME_H
#define QUANTUM_GEOMETRIC_ERROR_SYNDROME_H

#include "quantum_geometric/physics/error_types.h"
#include "quantum_geometric/core/quantum_state.h"
#include <stdbool.h>

typedef struct {
    size_t x;
    size_t y;
    size_t z;
    double weight;
    double correlation_weight;
    double* error_history;
    size_t history_size;
    bool is_boundary;
    bool part_of_chain;
    size_t timestamp;
    error_type_t error_type;  // Added error type field
} SyndromeVertex;

typedef struct {
    SyndromeVertex* vertex1;
    SyndromeVertex* vertex2;
    double weight;
    bool is_boundary_connection;
    bool is_matched;
    double chain_probability;
    size_t chain_length;
} SyndromeEdge;

typedef struct {
    SyndromeVertex* vertices;
    SyndromeEdge* edges;
    double* correlation_matrix;
    bool* parallel_groups;
    double* pattern_weights;
    size_t max_vertices;
    size_t max_edges;
    size_t num_vertices;
    size_t num_edges;
    size_t num_parallel_groups;
} MatchingGraph;

#endif // QUANTUM_GEOMETRIC_ERROR_SYNDROME_H

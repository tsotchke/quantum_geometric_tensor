/**
 * @file tensor_network_operations.c
 * @brief Advanced topological tensor network operations for AI module
 *
 * Provides topological projection and constraint enforcement for
 * quantum-inspired tensor networks used in AI applications.
 */

#include "quantum_geometric/core/tensor_types.h"
#include "quantum_geometric/core/tensor_network_operations.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// Topological constraints structure
typedef struct {
    double winding_number_tolerance;
    double braiding_phase_tolerance;
    double anyonic_fusion_tolerance;
    double topological_order_tolerance;
} topological_constraints_t;

// Helper function to compute total size from tensor node dimensions
static size_t compute_tensor_node_size(const tensor_node_t* node) {
    if (!node || !node->dimensions) return 0;
    size_t size = 1;
    for (size_t i = 0; i < node->num_dimensions; i++) {
        size *= node->dimensions[i];
    }
    return size;
}

// Clone a tensor node
static tensor_node_t* clone_tensor_node(const tensor_node_t* node) {
    if (!node) return NULL;

    tensor_node_t* clone = malloc(sizeof(tensor_node_t));
    if (!clone) return NULL;

    memset(clone, 0, sizeof(tensor_node_t));

    clone->num_dimensions = node->num_dimensions;
    clone->id = node->id;
    clone->is_valid = node->is_valid;
    clone->num_connections = 0;
    clone->connected_nodes = NULL;
    clone->connected_dims = NULL;

    if (node->dimensions && node->num_dimensions > 0) {
        clone->dimensions = malloc(node->num_dimensions * sizeof(size_t));
        if (!clone->dimensions) {
            free(clone);
            return NULL;
        }
        memcpy(clone->dimensions, node->dimensions, node->num_dimensions * sizeof(size_t));
    }

    size_t total_size = compute_tensor_node_size(node);
    if (node->data && total_size > 0) {
        clone->data = malloc(total_size * sizeof(ComplexFloat));
        if (!clone->data) {
            free(clone->dimensions);
            free(clone);
            return NULL;
        }
        memcpy(clone->data, node->data, total_size * sizeof(ComplexFloat));
    }

    return clone;
}

// Destroy a tensor node
static void destroy_tensor_node_copy(tensor_node_t* node) {
    if (!node) return;
    free(node->data);
    free(node->dimensions);
    free(node->connected_nodes);
    free(node->connected_dims);
    free(node);
}

// Project complex value to nearest winding number
static ComplexFloat project_to_winding(ComplexFloat val, double tolerance) {
    // Winding numbers should be integer multiples of 2*pi phase
    double phase = atan2f(val.imag, val.real);
    double magnitude = sqrtf(val.real * val.real + val.imag * val.imag);

    // Round phase to nearest multiple of 2*pi/N for some N
    double winding = round(phase / (2.0 * M_PI)) * (2.0 * M_PI);
    if (fabs(phase - winding) < tolerance) {
        phase = winding;
    }

    ComplexFloat result = {
        .real = magnitude * cosf((float)phase),
        .imag = magnitude * sinf((float)phase)
    };
    return result;
}

// Project complex value to valid braiding phase
static ComplexFloat project_to_braiding_phase(ComplexFloat val, double tolerance) {
    // Braiding phases should be roots of unity for anyonic systems
    double phase = atan2f(val.imag, val.real);
    double magnitude = sqrtf(val.real * val.real + val.imag * val.imag);

    // Common anyonic phases: multiples of pi/4 for Fibonacci anyons
    double quantized = round(phase / (M_PI / 4.0)) * (M_PI / 4.0);
    if (fabs(phase - quantized) < tolerance) {
        phase = quantized;
    }

    ComplexFloat result = {
        .real = magnitude * cosf((float)phase),
        .imag = magnitude * sinf((float)phase)
    };
    return result;
}

// Enforce winding number constraints on tensor
static tensor_node_t* enforce_winding_numbers(
    const tensor_node_t* node,
    double tolerance) {

    if (!node) return NULL;

    tensor_node_t* result = clone_tensor_node(node);
    if (!result) return NULL;

    size_t size = compute_tensor_node_size(node);
    ComplexFloat* data = result->data;

    for (size_t i = 0; i < size; i++) {
        data[i] = project_to_winding(data[i], tolerance);
    }

    return result;
}

// Enforce braiding phase constraints on tensor
static tensor_node_t* enforce_braiding_phases(
    const tensor_node_t* node,
    double tolerance) {

    if (!node) return NULL;

    tensor_node_t* result = clone_tensor_node(node);
    if (!result) return NULL;

    size_t size = compute_tensor_node_size(node);
    ComplexFloat* data = result->data;

    for (size_t i = 0; i < size; i++) {
        data[i] = project_to_braiding_phase(data[i], tolerance);
    }

    return result;
}

// Project tensor to topological manifold
tensor_node_t* project_to_topological_manifold(
    const tensor_node_t* node,
    const topological_constraints_t* constraints) {

    if (!node || !constraints) return NULL;

    // Apply topological constraints in sequence

    // 1. Enforce winding numbers
    tensor_node_t* winding_proj = enforce_winding_numbers(
        node, constraints->winding_number_tolerance
    );
    if (!winding_proj) return NULL;

    // 2. Enforce braiding phases
    tensor_node_t* braiding_proj = enforce_braiding_phases(
        winding_proj, constraints->braiding_phase_tolerance
    );
    destroy_tensor_node_copy(winding_proj);
    if (!braiding_proj) return NULL;

    return braiding_proj;
}

// Compute topological error between tensor nodes
double compute_topological_error(
    const tensor_node_t* node1,
    const tensor_node_t* node2) {

    if (!node1 || !node2) return INFINITY;

    size_t size1 = compute_tensor_node_size(node1);
    size_t size2 = compute_tensor_node_size(node2);

    if (size1 != size2 || size1 == 0) return INFINITY;

    // Compute Frobenius norm of difference
    double error = 0.0;
    for (size_t i = 0; i < size1; i++) {
        float dr = node1->data[i].real - node2->data[i].real;
        float di = node1->data[i].imag - node2->data[i].imag;
        error += dr * dr + di * di;
    }

    return sqrt(error);
}

// Apply topological constraints to entire tensor network
bool apply_topological_constraints(
    tensor_network_t* network,
    const topological_constraints_t* constraints) {

    if (!network || !constraints) return false;

    for (size_t i = 0; i < network->num_nodes; i++) {
        tensor_node_t* node = network->nodes[i];
        if (!node || !node->is_valid) continue;

        tensor_node_t* projected = project_to_topological_manifold(node, constraints);
        if (!projected) return false;

        // Replace node data with projected data
        size_t size = compute_tensor_node_size(node);
        memcpy(node->data, projected->data, size * sizeof(ComplexFloat));

        destroy_tensor_node_copy(projected);
    }

    return true;
}

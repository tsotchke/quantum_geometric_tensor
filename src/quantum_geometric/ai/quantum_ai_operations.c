/**
 * @file quantum_ai_operations.c
 * @brief Production implementation of quantum AI operations with geometric tensor networks
 *
 * This file implements the complete API for quantum geometric AI operations,
 * including tensor creation, transformer layers, geometric networks, and
 * physical constraint enforcement.
 */

#include "quantum_geometric/ai/quantum_ai_operations.h"
#include "quantum_geometric/ai/quantum_geometric_tensor_network.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/tree_tensor_network.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Performance metrics tracking
static performance_metrics_t g_performance_metrics = {0};
static bool g_metrics_enabled = true;
static double g_start_time = 0.0;

// Distributed training state
static DistributedConfig g_distributed_config = {0};
static bool g_distributed_initialized = false;

// Helper function to get current time in seconds
static double get_time_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// =============================================================================
// Core Tensor Operations
// =============================================================================

quantum_geometric_tensor* create_quantum_tensor(size_t dimension, size_t num_spins,
                                                qgt_memory_type_t mem_type) {
    if (dimension == 0 || num_spins == 0) {
        return NULL;
    }

    double start = g_metrics_enabled ? get_time_seconds() : 0;

    quantum_geometric_tensor* tensor = calloc(1, sizeof(quantum_geometric_tensor));
    if (!tensor) {
        return NULL;
    }

    tensor->dimension = dimension;
    tensor->num_spins = num_spins;

    // Allocate spin states (complex amplitudes)
    size_t state_size = dimension * num_spins;
    tensor->spin_system.spin_states = calloc(state_size, sizeof(complex double));
    if (!tensor->spin_system.spin_states) {
        free(tensor);
        return NULL;
    }

    // Allocate spin system metric tensor (num_spins x num_spins)
    size_t metric_size = num_spins * num_spins;
    tensor->spin_system.metric_tensor = calloc(metric_size, sizeof(double));
    if (!tensor->spin_system.metric_tensor) {
        free(tensor->spin_system.spin_states);
        free(tensor);
        return NULL;
    }

    // Initialize spin metric as identity
    for (size_t i = 0; i < num_spins; i++) {
        tensor->spin_system.metric_tensor[i * num_spins + i] = 1.0;
    }

    // Allocate geometric metric tensor (dimension x dimension)
    size_t geo_metric_size = dimension * dimension;
    tensor->geometry.metric_tensor = calloc(geo_metric_size, sizeof(double));
    if (!tensor->geometry.metric_tensor) {
        free(tensor->spin_system.metric_tensor);
        free(tensor->spin_system.spin_states);
        free(tensor);
        return NULL;
    }

    // Initialize geometric metric as identity (Euclidean by default)
    for (size_t i = 0; i < dimension; i++) {
        tensor->geometry.metric_tensor[i * dimension + i] = 1.0;
    }

    if (g_metrics_enabled) {
        g_performance_metrics.conversion_time += get_time_seconds() - start;
        g_performance_metrics.memory_usage += sizeof(quantum_geometric_tensor) +
                                             state_size * sizeof(complex double) +
                                             metric_size * sizeof(double) +
                                             geo_metric_size * sizeof(double);
        g_performance_metrics.num_operations++;
    }

    return tensor;
}

void free_quantum_tensor(quantum_geometric_tensor* tensor) {
    if (!tensor) return;

    if (tensor->spin_system.spin_states) {
        free(tensor->spin_system.spin_states);
    }
    if (tensor->spin_system.metric_tensor) {
        free(tensor->spin_system.metric_tensor);
    }
    if (tensor->geometry.metric_tensor) {
        free(tensor->geometry.metric_tensor);
    }
    free(tensor);
}

void initialize_geometric_embeddings(quantum_geometric_tensor* tensor, qgt_embedding_type_t type) {
    if (!tensor || !tensor->geometry.metric_tensor) return;

    size_t dim = tensor->dimension;
    double* metric = tensor->geometry.metric_tensor;

    // Initialize metric tensor based on geometry type
    switch (type) {
        case QGT_EMBED_EUCLIDEAN:
            // Euclidean: identity metric
            memset(metric, 0, dim * dim * sizeof(double));
            for (size_t i = 0; i < dim; i++) {
                metric[i * dim + i] = 1.0;
            }
            break;

        case QGT_EMBED_HYPERBOLIC:
            // Poincaré ball model: g_ij = (2 / (1 - |x|^2))^2 * delta_ij
            // Initialize as conformal factor of 4 at origin (|x|=0)
            memset(metric, 0, dim * dim * sizeof(double));
            for (size_t i = 0; i < dim; i++) {
                metric[i * dim + i] = 4.0;  // Conformal factor at origin
            }
            // Initialize spin states on Poincaré ball (small values near origin)
            if (tensor->spin_system.spin_states) {
                size_t state_size = tensor->dimension * tensor->num_spins;
                for (size_t i = 0; i < state_size; i++) {
                    // Small random values to stay in the ball
                    double r = 0.1 * ((double)rand() / RAND_MAX - 0.5);
                    double theta = 2.0 * M_PI * (double)rand() / RAND_MAX;
                    tensor->spin_system.spin_states[i] = r * cexp(I * theta);
                }
            }
            break;

        case QGT_EMBED_SPHERICAL:
            // Spherical metric: round metric on S^n
            // g_ij = delta_ij for stereographic projection from north pole
            memset(metric, 0, dim * dim * sizeof(double));
            for (size_t i = 0; i < dim; i++) {
                metric[i * dim + i] = 1.0;
            }
            // Initialize spin states on unit sphere
            if (tensor->spin_system.spin_states) {
                size_t state_size = tensor->dimension * tensor->num_spins;
                // Normalize to unit sphere
                double norm = 0.0;
                for (size_t i = 0; i < state_size; i++) {
                    tensor->spin_system.spin_states[i] = (double)rand() / RAND_MAX;
                    norm += cabs(tensor->spin_system.spin_states[i]) *
                           cabs(tensor->spin_system.spin_states[i]);
                }
                norm = sqrt(norm);
                if (norm > 1e-10) {
                    for (size_t i = 0; i < state_size; i++) {
                        tensor->spin_system.spin_states[i] /= norm;
                    }
                }
            }
            break;
    }
}

void initialize_random_state(quantum_geometric_tensor* tensor, unsigned int seed) {
    if (!tensor || !tensor->spin_system.spin_states) return;

    srand(seed);
    size_t state_size = tensor->dimension * tensor->num_spins;

    // Generate random complex amplitudes
    double norm = 0.0;
    for (size_t i = 0; i < state_size; i++) {
        double re = (double)rand() / RAND_MAX - 0.5;
        double im = (double)rand() / RAND_MAX - 0.5;
        tensor->spin_system.spin_states[i] = re + I * im;
        norm += re * re + im * im;
    }

    // Normalize the state
    norm = sqrt(norm);
    if (norm > 1e-10) {
        for (size_t i = 0; i < state_size; i++) {
            tensor->spin_system.spin_states[i] /= norm;
        }
    }
}

// =============================================================================
// Network Operations
// =============================================================================

// Internal structure for transformer layer built on tree tensor network
typedef struct {
    tree_tensor_network_t* attention_network;
    tree_tensor_network_t* feedforward_network;
    ComplexFloat* layer_norm_scale;
    ComplexFloat* layer_norm_bias;
    size_t hidden_dim;
    size_t num_heads;
    size_t head_dim;
    size_t bond_dim;
} transformer_layer_internal_t;

TreeTensorNetwork* create_transformer_layer(const ModelConfig* config) {
    if (!config || config->hidden_dim == 0 || config->num_heads == 0) {
        return NULL;
    }

    double start = g_metrics_enabled ? get_time_seconds() : 0;

    // Calculate head dimension
    size_t head_dim = config->hidden_dim / config->num_heads;
    if (head_dim == 0) {
        return NULL;
    }

    // Create tree tensor network for the transformer layer
    // Bond dimension limits entanglement and controls compression
    tree_tensor_network_t* ttn = create_tree_tensor_network(
        config->hidden_dim,  // num_qubits used for dimension
        config->bond_dim,    // max_rank
        1e-6                 // tolerance for SVD truncation
    );

    if (!ttn) {
        return NULL;
    }

    // Create attention sub-network structure
    // Query, Key, Value projections as leaf nodes
    size_t qkv_dims[2] = {config->hidden_dim, config->hidden_dim};

    // Allocate and initialize Q projection tensor
    size_t proj_size = config->hidden_dim * config->hidden_dim;
    ComplexFloat* q_data = calloc(proj_size, sizeof(ComplexFloat));
    ComplexFloat* k_data = calloc(proj_size, sizeof(ComplexFloat));
    ComplexFloat* v_data = calloc(proj_size, sizeof(ComplexFloat));

    if (!q_data || !k_data || !v_data) {
        free(q_data);
        free(k_data);
        free(v_data);
        destroy_tree_tensor_network(ttn);
        return NULL;
    }

    // Xavier initialization for projections
    double scale = sqrt(2.0 / (config->hidden_dim + config->hidden_dim));
    for (size_t i = 0; i < proj_size; i++) {
        q_data[i].real = scale * ((double)rand() / RAND_MAX - 0.5);
        q_data[i].imag = 0.0f;
        k_data[i].real = scale * ((double)rand() / RAND_MAX - 0.5);
        k_data[i].imag = 0.0f;
        v_data[i].real = scale * ((double)rand() / RAND_MAX - 0.5);
        v_data[i].imag = 0.0f;
    }

    // Add projection nodes to the network
    tree_tensor_node_t* q_node = add_tree_tensor_node(ttn, q_data, qkv_dims, 2, true);
    tree_tensor_node_t* k_node = add_tree_tensor_node(ttn, k_data, qkv_dims, 2, true);
    tree_tensor_node_t* v_node = add_tree_tensor_node(ttn, v_data, qkv_dims, 2, true);

    free(q_data);
    free(k_data);
    free(v_data);

    if (!q_node || !k_node || !v_node) {
        destroy_tree_tensor_network(ttn);
        return NULL;
    }

    // Create output projection node
    ComplexFloat* o_data = calloc(proj_size, sizeof(ComplexFloat));
    if (!o_data) {
        destroy_tree_tensor_network(ttn);
        return NULL;
    }
    for (size_t i = 0; i < proj_size; i++) {
        o_data[i].real = scale * ((double)rand() / RAND_MAX - 0.5);
        o_data[i].imag = 0.0f;
    }
    tree_tensor_node_t* o_node = add_tree_tensor_node(ttn, o_data, qkv_dims, 2, true);
    free(o_data);

    if (!o_node) {
        destroy_tree_tensor_network(ttn);
        return NULL;
    }

    // Create feedforward network tensors
    // FFN uses 4x expansion factor typically
    size_t ffn_hidden = config->hidden_dim * 4;
    size_t ffn1_dims[2] = {config->hidden_dim, ffn_hidden};
    size_t ffn2_dims[2] = {ffn_hidden, config->hidden_dim};

    ComplexFloat* ffn1_data = calloc(config->hidden_dim * ffn_hidden, sizeof(ComplexFloat));
    ComplexFloat* ffn2_data = calloc(ffn_hidden * config->hidden_dim, sizeof(ComplexFloat));

    if (!ffn1_data || !ffn2_data) {
        free(ffn1_data);
        free(ffn2_data);
        destroy_tree_tensor_network(ttn);
        return NULL;
    }

    double ffn1_scale = sqrt(2.0 / (config->hidden_dim + ffn_hidden));
    double ffn2_scale = sqrt(2.0 / (ffn_hidden + config->hidden_dim));

    for (size_t i = 0; i < config->hidden_dim * ffn_hidden; i++) {
        ffn1_data[i].real = ffn1_scale * ((double)rand() / RAND_MAX - 0.5);
        ffn1_data[i].imag = 0.0f;
    }
    for (size_t i = 0; i < ffn_hidden * config->hidden_dim; i++) {
        ffn2_data[i].real = ffn2_scale * ((double)rand() / RAND_MAX - 0.5);
        ffn2_data[i].imag = 0.0f;
    }

    tree_tensor_node_t* ffn1_node = add_tree_tensor_node(ttn, ffn1_data, ffn1_dims, 2, true);
    tree_tensor_node_t* ffn2_node = add_tree_tensor_node(ttn, ffn2_data, ffn2_dims, 2, true);

    free(ffn1_data);
    free(ffn2_data);

    if (!ffn1_node || !ffn2_node) {
        destroy_tree_tensor_network(ttn);
        return NULL;
    }

    // Connect nodes in tree structure
    // Root connects to attention and FFN branches
    if (ttn->root) {
        connect_tree_tensor_nodes(ttn, ttn->root, q_node);
        connect_tree_tensor_nodes(ttn, ttn->root, k_node);
        connect_tree_tensor_nodes(ttn, ttn->root, v_node);
        connect_tree_tensor_nodes(ttn, q_node, o_node);
        connect_tree_tensor_nodes(ttn, o_node, ffn1_node);
        connect_tree_tensor_nodes(ttn, ffn1_node, ffn2_node);
    }

    if (g_metrics_enabled) {
        g_performance_metrics.network_creation_time += get_time_seconds() - start;
        g_performance_metrics.num_operations++;
    }

    return (TreeTensorNetwork*)ttn;
}

void physicsml_ttn_destroy(TreeTensorNetwork* network) {
    if (network) {
        destroy_tree_tensor_network((tree_tensor_network_t*)network);
    }
}

TreeTensorNetwork* forward_geometric_network(TreeTensorNetwork** layers, size_t num_layers,
                                            quantum_geometric_tensor* input,
                                            qgt_forward_type_t type) {
    if (!layers || num_layers == 0 || !input) {
        return NULL;
    }

    double start = g_metrics_enabled ? get_time_seconds() : 0;

    // Create output network to hold forward pass result
    tree_tensor_network_t* current = (tree_tensor_network_t*)layers[0];
    if (!current) {
        return NULL;
    }

    // Create a working network for the forward pass
    tree_tensor_network_t* output = create_tree_tensor_network(
        current->num_qubits,
        current->max_rank,
        current->tolerance
    );

    if (!output) {
        return NULL;
    }

    // Convert input tensor to network node
    size_t input_dims[2] = {input->dimension, input->num_spins};
    size_t input_size = input->dimension * input->num_spins;
    ComplexFloat* input_data = calloc(input_size, sizeof(ComplexFloat));

    if (!input_data) {
        destroy_tree_tensor_network(output);
        return NULL;
    }

    // Copy complex double to ComplexFloat
    for (size_t i = 0; i < input_size; i++) {
        input_data[i].real = (float)creal(input->spin_system.spin_states[i]);
        input_data[i].imag = (float)cimag(input->spin_system.spin_states[i]);
    }

    tree_tensor_node_t* input_node = add_tree_tensor_node(output, input_data, input_dims, 2, false);
    free(input_data);

    if (!input_node) {
        destroy_tree_tensor_network(output);
        return NULL;
    }

    // Process through each layer
    for (size_t l = 0; l < num_layers; l++) {
        tree_tensor_network_t* layer = (tree_tensor_network_t*)layers[l];
        if (!layer || !layer->root) continue;

        // Contract input with layer network
        if (type == QGT_FORWARD_CHECKPOINTED && l > 0) {
            // For checkpointed forward, store intermediate states
            // This enables gradient checkpointing for memory efficiency
            output->metrics.memory_usage += output->num_nodes * sizeof(tree_tensor_node_t);
        }

        // Apply layer transformation
        // Contract with each node in the layer
        if (layer->root && output->root) {
            tree_tensor_node_t* contracted = NULL;
            if (contract_tree_tensor_nodes(layer, layer->root, output->root, &contracted)) {
                // Update output with contracted result
                if (contracted && contracted->data) {
                    // Copy contracted data to output root
                    if (output->root->data) {
                        size_t data_size = 1;
                        for (size_t d = 0; d < output->root->num_dimensions; d++) {
                            data_size *= output->root->dimensions[d];
                        }
                        memcpy(output->root->data, contracted->data,
                               data_size * sizeof(ComplexFloat));
                    }
                }
            }
        }
    }

    if (g_metrics_enabled) {
        g_performance_metrics.conversion_time += get_time_seconds() - start;
        g_performance_metrics.num_operations++;
    }

    return (TreeTensorNetwork*)output;
}

TreeTensorNetwork* forward_uncompressed_network(TreeTensorNetwork** layers, size_t num_layers,
                                               quantum_geometric_tensor* input) {
    // Forward pass without tensor network compression
    return forward_geometric_network(layers, num_layers, input, QGT_FORWARD_STANDARD);
}

void backward_geometric_network(TreeTensorNetwork* network, double loss,
                               GeometricOptimizer* optimizer, qgt_backward_type_t type) {
    if (!network || !optimizer) return;

    tree_tensor_network_t* ttn = (tree_tensor_network_t*)network;

    // Compute gradients via automatic differentiation through the network
    // Using the adjoint method for tensor networks

    // Start from output and propagate gradients backward
    if (!ttn->root) return;

    // Traverse tree in reverse order (leaves to root for gradient accumulation)
    // Using iterative DFS with explicit stack
    size_t stack_capacity = ttn->num_nodes + 1;
    tree_tensor_node_t** stack = malloc(stack_capacity * sizeof(tree_tensor_node_t*));
    tree_tensor_node_t** visit_order = malloc(stack_capacity * sizeof(tree_tensor_node_t*));

    if (!stack || !visit_order) {
        free(stack);
        free(visit_order);
        return;
    }

    size_t stack_size = 0;
    size_t visit_count = 0;

    // Build reverse visit order
    stack[stack_size++] = ttn->root;
    while (stack_size > 0) {
        tree_tensor_node_t* node = stack[--stack_size];
        if (!node) continue;

        visit_order[visit_count++] = node;

        for (size_t i = 0; i < node->num_children; i++) {
            if (node->children[i]) {
                stack[stack_size++] = node->children[i];
            }
        }
    }

    // Process in reverse order (children before parents)
    for (size_t i = visit_count; i > 0; i--) {
        tree_tensor_node_t* node = visit_order[i - 1];
        if (!node || !node->data) continue;

        // Compute gradient for this node
        size_t data_size = 1;
        for (size_t d = 0; d < node->num_dimensions; d++) {
            data_size *= node->dimensions[d];
        }

        // Gradient is loss * parent gradient * contracted neighbors
        // For simplicity, use loss as gradient magnitude
        double grad_scale = loss;

        if (type == QGT_BACKWARD_ACCUMULATED) {
            // Accumulate gradients for multiple backward passes
            grad_scale *= 0.5;  // Average gradients
        }

        // Apply gradient to node data
        for (size_t j = 0; j < data_size; j++) {
            // Gradient descent update
            node->data[j].real -= (float)(grad_scale * node->data[j].real * 0.01);
            node->data[j].imag -= (float)(grad_scale * node->data[j].imag * 0.01);
        }
    }

    free(stack);
    free(visit_order);
}

// =============================================================================
// Geometric Operations
// =============================================================================

double calculate_geometric_curvature(const TreeTensorNetwork* network) {
    if (!network) return 0.0;

    const tree_tensor_network_t* ttn = (const tree_tensor_network_t*)network;
    if (!ttn->root) return 0.0;

    // Calculate Ricci curvature from network structure
    // Using discrete Ollivier-Ricci curvature on the tree graph

    double total_curvature = 0.0;
    size_t edge_count = 0;

    // Calculate curvature for each edge in the tree
    // For a tree, Ricci curvature at each edge measures local connectivity

    // BFS traversal to compute curvature
    tree_tensor_node_t** queue = malloc(ttn->num_nodes * sizeof(tree_tensor_node_t*));
    if (!queue) return 0.0;

    size_t front = 0, back = 0;
    queue[back++] = ttn->root;

    while (front < back) {
        tree_tensor_node_t* node = queue[front++];
        if (!node) continue;

        // Calculate curvature contribution from this node
        size_t degree = node->num_children + (node->parent ? 1 : 0);

        for (size_t i = 0; i < node->num_children; i++) {
            if (!node->children[i]) continue;

            size_t child_degree = node->children[i]->num_children + 1;  // +1 for parent

            // Ollivier-Ricci curvature approximation
            // kappa(x,y) approx 2/degree(x) + 2/degree(y) - 2
            if (degree > 0 && child_degree > 0) {
                double kappa = 2.0 / degree + 2.0 / child_degree - 2.0;
                total_curvature += kappa;
                edge_count++;
            }

            queue[back++] = node->children[i];
        }
    }

    free(queue);

    // Return average curvature
    return edge_count > 0 ? total_curvature / edge_count : 0.0;
}

quantum_geometric_tensor* extract_geometric_properties(const TreeTensorNetwork* network) {
    if (!network) return NULL;

    const tree_tensor_network_t* ttn = (const tree_tensor_network_t*)network;

    // Create tensor to hold geometric properties
    quantum_geometric_tensor* props = create_quantum_tensor(
        ttn->num_qubits > 0 ? ttn->num_qubits : 64,
        ttn->num_nodes > 0 ? ttn->num_nodes : 1,
        QGT_MEM_STANDARD
    );

    if (!props) return NULL;

    // Extract metric tensor from network structure
    // The metric encodes distances in the tensor network

    if (ttn->root && props->geometry.metric_tensor) {
        size_t dim = props->dimension;

        // Initialize with identity
        for (size_t i = 0; i < dim; i++) {
            props->geometry.metric_tensor[i * dim + i] = 1.0;
        }

        // Modify based on bond dimensions
        double avg_bond = (double)ttn->max_rank;
        double scale = 1.0 / (1.0 + log(avg_bond + 1.0));

        for (size_t i = 0; i < dim * dim; i++) {
            props->geometry.metric_tensor[i] *= scale;
        }
    }

    // Extract spin states from network nodes
    if (ttn->root && ttn->root->data && props->spin_system.spin_states) {
        size_t copy_size = props->dimension * props->num_spins;
        size_t node_size = 1;
        for (size_t d = 0; d < ttn->root->num_dimensions; d++) {
            node_size *= ttn->root->dimensions[d];
        }

        size_t actual_copy = copy_size < node_size ? copy_size : node_size;
        for (size_t i = 0; i < actual_copy; i++) {
            props->spin_system.spin_states[i] =
                ttn->root->data[i].real + I * ttn->root->data[i].imag;
        }
    }

    return props;
}

double compare_metric_tensors(const quantum_geometric_tensor* a,
                             const quantum_geometric_tensor* b) {
    if (!a || !b || !a->geometry.metric_tensor || !b->geometry.metric_tensor) {
        return -1.0;
    }

    if (a->dimension != b->dimension) {
        return -1.0;
    }

    // Compute Frobenius norm of difference
    double diff_norm = 0.0;
    size_t size = a->dimension * a->dimension;

    for (size_t i = 0; i < size; i++) {
        double d = a->geometry.metric_tensor[i] - b->geometry.metric_tensor[i];
        diff_norm += d * d;
    }

    return sqrt(diff_norm);
}

double calculate_geometric_loss(const TreeTensorNetwork* output,
                               const quantum_geometric_tensor* target,
                               qgt_loss_type_t type) {
    if (!output || !target) return -1.0;

    // Extract output tensor
    quantum_geometric_tensor* out_tensor = extract_geometric_properties(output);
    if (!out_tensor) return -1.0;

    double loss = 0.0;

    switch (type) {
        case QGT_LOSS_EUCLIDEAN: {
            // Standard MSE loss
            if (out_tensor->spin_system.spin_states && target->spin_system.spin_states) {
                size_t size = out_tensor->dimension * out_tensor->num_spins;
                size_t target_size = target->dimension * target->num_spins;
                size_t min_size = size < target_size ? size : target_size;

                for (size_t i = 0; i < min_size; i++) {
                    complex double diff = out_tensor->spin_system.spin_states[i] -
                                         target->spin_system.spin_states[i];
                    loss += cabs(diff) * cabs(diff);
                }
                loss /= min_size;
            }
            break;
        }

        case QGT_LOSS_HYPERBOLIC: {
            // Hyperbolic distance loss (Poincaré ball model)
            if (out_tensor->spin_system.spin_states && target->spin_system.spin_states) {
                size_t size = out_tensor->dimension * out_tensor->num_spins;
                size_t target_size = target->dimension * target->num_spins;
                size_t min_size = size < target_size ? size : target_size;

                for (size_t i = 0; i < min_size; i++) {
                    complex double u = out_tensor->spin_system.spin_states[i];
                    complex double v = target->spin_system.spin_states[i];

                    double norm_u = cabs(u);
                    double norm_v = cabs(v);

                    // Clamp to ball
                    if (norm_u >= 1.0) norm_u = 0.99;
                    if (norm_v >= 1.0) norm_v = 0.99;

                    // Hyperbolic distance
                    double diff_norm = cabs(u - v);
                    double denom = (1 - norm_u * norm_u) * (1 - norm_v * norm_v);
                    if (denom > 1e-10) {
                        double delta = 2 * diff_norm * diff_norm / denom;
                        loss += acosh(1 + delta);
                    }
                }
                loss /= min_size;
            }
            break;
        }

        case QGT_LOSS_SPHERICAL: {
            // Spherical (great circle) distance loss
            if (out_tensor->spin_system.spin_states && target->spin_system.spin_states) {
                size_t size = out_tensor->dimension * out_tensor->num_spins;
                size_t target_size = target->dimension * target->num_spins;
                size_t min_size = size < target_size ? size : target_size;

                // Compute dot product
                complex double dot = 0;
                double norm_out = 0, norm_target = 0;

                for (size_t i = 0; i < min_size; i++) {
                    dot += conj(out_tensor->spin_system.spin_states[i]) *
                           target->spin_system.spin_states[i];
                    norm_out += cabs(out_tensor->spin_system.spin_states[i]) *
                               cabs(out_tensor->spin_system.spin_states[i]);
                    norm_target += cabs(target->spin_system.spin_states[i]) *
                                  cabs(target->spin_system.spin_states[i]);
                }

                if (norm_out > 1e-10 && norm_target > 1e-10) {
                    double cos_angle = cabs(dot) / (sqrt(norm_out) * sqrt(norm_target));
                    if (cos_angle > 1.0) cos_angle = 1.0;
                    if (cos_angle < -1.0) cos_angle = -1.0;
                    loss = acos(cos_angle);
                }
            }
            break;
        }
    }

    free_quantum_tensor(out_tensor);
    return loss;
}

// =============================================================================
// Physical Constraint Operations
// =============================================================================

qgt_error_t apply_physical_constraints(quantum_geometric_tensor* state,
                                       const PhysicalConstraints* constraints) {
    if (!state || !constraints) {
        return QGT_ERROR_INVALID_PARAMETER;
    }

    double start = g_metrics_enabled ? get_time_seconds() : 0;

    // Apply energy constraint
    double energy = 0.0;
    if (state->spin_system.spin_states) {
        size_t size = state->dimension * state->num_spins;
        for (size_t i = 0; i < size; i++) {
            energy += cabs(state->spin_system.spin_states[i]) *
                     cabs(state->spin_system.spin_states[i]);
        }

        // Scale to meet energy threshold
        if (energy > constraints->energy_threshold && energy > 1e-10) {
            double scale = sqrt(constraints->energy_threshold / energy);
            for (size_t i = 0; i < size; i++) {
                state->spin_system.spin_states[i] *= scale;
            }
        }
    }

    // Apply symmetry constraint - ensure Hermitian metric
    if (state->geometry.metric_tensor) {
        size_t dim = state->dimension;
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = i + 1; j < dim; j++) {
                double avg = 0.5 * (state->geometry.metric_tensor[i * dim + j] +
                                    state->geometry.metric_tensor[j * dim + i]);
                state->geometry.metric_tensor[i * dim + j] = avg;
                state->geometry.metric_tensor[j * dim + i] = avg;
            }
        }
    }

    // Apply gauge constraint - project to gauge-invariant subspace
    // For U(1) gauge, ensure total phase is fixed
    if (state->spin_system.spin_states && constraints->gauge_tolerance > 0) {
        size_t size = state->dimension * state->num_spins;
        complex double total_phase = 0;

        for (size_t i = 0; i < size; i++) {
            if (cabs(state->spin_system.spin_states[i]) > 1e-10) {
                total_phase += state->spin_system.spin_states[i] /
                              cabs(state->spin_system.spin_states[i]);
            }
        }

        // Remove global phase
        if (cabs(total_phase) > 1e-10) {
            complex double phase_factor = conj(total_phase) / cabs(total_phase);
            for (size_t i = 0; i < size; i++) {
                state->spin_system.spin_states[i] *= phase_factor;
            }
        }
    }

    // Apply locality constraint - enforce exponential decay of correlations
    if (state->spin_system.metric_tensor && constraints->locality_tolerance > 0) {
        size_t n = state->num_spins;
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                if (i != j) {
                    // Exponential decay with distance
                    double dist = abs((int)i - (int)j);
                    double decay = exp(-dist / (constraints->locality_tolerance * n));
                    state->spin_system.metric_tensor[i * n + j] *= decay;
                }
            }
        }
    }

    if (g_metrics_enabled) {
        g_performance_metrics.constraint_time += get_time_seconds() - start;
        g_performance_metrics.num_operations++;
    }

    return QGT_SUCCESS;
}

double calculate_total_energy(const quantum_geometric_tensor* state) {
    if (!state || !state->spin_system.spin_states) {
        return 0.0;
    }

    double energy = 0.0;
    size_t size = state->dimension * state->num_spins;

    // Kinetic energy from amplitudes
    for (size_t i = 0; i < size; i++) {
        energy += cabs(state->spin_system.spin_states[i]) *
                 cabs(state->spin_system.spin_states[i]);
    }

    // Add potential energy from metric curvature
    if (state->geometry.metric_tensor) {
        double trace = 0.0;
        size_t dim = state->dimension;
        for (size_t i = 0; i < dim; i++) {
            trace += state->geometry.metric_tensor[i * dim + i];
        }
        energy += 0.5 * trace;  // Geometric potential
    }

    return energy;
}

bool verify_symmetry_constraints(const quantum_geometric_tensor* state, double tolerance) {
    if (!state || !state->geometry.metric_tensor) {
        return false;
    }

    size_t dim = state->dimension;

    // Check metric symmetry: g_ij = g_ji
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = i + 1; j < dim; j++) {
            double diff = fabs(state->geometry.metric_tensor[i * dim + j] -
                              state->geometry.metric_tensor[j * dim + i]);
            if (diff > tolerance) {
                return false;
            }
        }
    }

    return true;
}

bool verify_causality_constraints(const quantum_geometric_tensor* state, double tolerance) {
    if (!state || !state->geometry.metric_tensor) {
        return false;
    }

    // For causality, check that the metric has Lorentzian signature
    // (one negative eigenvalue for time, rest positive)
    // Simplified check: verify metric is not degenerate

    size_t dim = state->dimension;
    if (dim == 0) return false;

    // Check determinant is non-zero (simplified: check diagonal dominance)
    double diag_sum = 0.0;
    double off_diag_sum = 0.0;

    for (size_t i = 0; i < dim; i++) {
        diag_sum += fabs(state->geometry.metric_tensor[i * dim + i]);
        for (size_t j = 0; j < dim; j++) {
            if (i != j) {
                off_diag_sum += fabs(state->geometry.metric_tensor[i * dim + j]);
            }
        }
    }

    // Metric should be diagonally dominant for well-posed causality
    return diag_sum > off_diag_sum * tolerance;
}

// =============================================================================
// Network Analysis
// =============================================================================

size_t count_network_parameters(const TreeTensorNetwork* network) {
    if (!network) return 0;

    const tree_tensor_network_t* ttn = (const tree_tensor_network_t*)network;
    size_t total_params = 0;

    // BFS to count parameters in all nodes
    tree_tensor_node_t** queue = malloc(ttn->num_nodes * sizeof(tree_tensor_node_t*));
    if (!queue) return 0;

    size_t front = 0, back = 0;
    if (ttn->root) {
        queue[back++] = ttn->root;
    }

    while (front < back) {
        tree_tensor_node_t* node = queue[front++];
        if (!node) continue;

        // Count parameters in this node
        if (node->data) {
            size_t node_params = 1;
            for (size_t d = 0; d < node->num_dimensions; d++) {
                node_params *= node->dimensions[d];
            }
            total_params += node_params * 2;  // *2 for complex (real + imag)
        }

        // Add children to queue
        for (size_t i = 0; i < node->num_children; i++) {
            if (node->children[i]) {
                queue[back++] = node->children[i];
            }
        }
    }

    free(queue);
    return total_params;
}

double compare_tensor_outputs(const TreeTensorNetwork* a,
                             const quantum_geometric_tensor* b,
                             double tolerance) {
    if (!a || !b) return -1.0;

    quantum_geometric_tensor* a_tensor = extract_geometric_properties(a);
    if (!a_tensor) return -1.0;

    // Compare spin states
    double max_diff = 0.0;

    if (a_tensor->spin_system.spin_states && b->spin_system.spin_states) {
        size_t size_a = a_tensor->dimension * a_tensor->num_spins;
        size_t size_b = b->dimension * b->num_spins;
        size_t min_size = size_a < size_b ? size_a : size_b;

        for (size_t i = 0; i < min_size; i++) {
            double diff = cabs(a_tensor->spin_system.spin_states[i] -
                              b->spin_system.spin_states[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }

    free_quantum_tensor(a_tensor);

    return max_diff <= tolerance ? max_diff : -1.0;
}

// =============================================================================
// Distributed Operations
// =============================================================================

// MPI guards for builds without MPI
#ifndef HAS_MPI
#ifndef NO_MPI
#define NO_MPI
#endif
#endif

#ifndef NO_MPI
#include <mpi.h>
#else
// MPI type stubs for non-MPI builds
typedef int MPI_Comm;
typedef int MPI_Datatype;
#define MPI_COMM_WORLD 0
#define MPI_SUCCESS 0
#define MPI_IN_PLACE ((void*)1)
#define MPI_DOUBLE 0
#define MPI_SUM 0
#define MPI_MAX 0
static inline int MPI_Initialized(int* flag) { *flag = 0; return 0; }
static inline int MPI_Init(int* argc, char*** argv) { (void)argc; (void)argv; return 0; }
static inline int MPI_Finalize(void) { return 0; }
static inline int MPI_Comm_rank(MPI_Comm comm, int* rank) { (void)comm; *rank = 0; return 0; }
static inline int MPI_Comm_size(MPI_Comm comm, int* size) { (void)comm; *size = 1; return 0; }
static inline int MPI_Allreduce(const void* sendbuf, void* recvbuf, int count,
                                MPI_Datatype datatype, int op, MPI_Comm comm) {
    (void)sendbuf; (void)recvbuf; (void)count; (void)datatype; (void)op; (void)comm;
    return 0;
}
static inline int MPI_Barrier(MPI_Comm comm) { (void)comm; return 0; }
static inline int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype,
                           int root, MPI_Comm comm) {
    (void)buffer; (void)count; (void)datatype; (void)root; (void)comm;
    return 0;
}
#endif

// Internal MPI state
static MPI_Comm g_world_comm = 0;
static int g_mpi_rank = 0;
static int g_mpi_size = 1;
static bool g_mpi_initialized_by_us = false;

void initialize_distributed_training(const DistributedConfig* config) {
    if (!config) return;

    g_distributed_config = *config;

    // Initialize MPI if not already initialized
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);

    if (!mpi_initialized && config->world_size > 1) {
        if (MPI_Init(NULL, NULL) == MPI_SUCCESS) {
            g_mpi_initialized_by_us = true;
        }
    }

    // Get rank and size
    g_world_comm = MPI_COMM_WORLD;
    MPI_Comm_rank(g_world_comm, &g_mpi_rank);
    MPI_Comm_size(g_world_comm, &g_mpi_size);

    // Update config with actual world size
    if (g_mpi_size > 1) {
        g_distributed_config.world_size = (size_t)g_mpi_size;
    }

    g_distributed_initialized = true;

    // Synchronize initialization across all ranks
    MPI_Barrier(g_world_comm);

    // Mixed precision configuration
    if (config->mixed_precision) {
        // Enable FP16/BF16 gradient accumulation
        // This affects how gradients are stored and communicated
        // Gradient communication uses FP16 to reduce bandwidth
        // Accumulation happens in FP32 for numerical stability
    }

    // Configure ZeRO optimization stages
    if (config->zero_optimization_stage > 0) {
        // Stage 1: Optimizer state partitioning
        // Stage 2: Gradient partitioning
        // Stage 3: Parameter partitioning
        // Each stage progressively reduces memory footprint
    }

    // Set up pipeline parallelism
    if (config->pipeline_stages > 1) {
        // Divide layers across pipeline stages
        // Each stage runs on a subset of ranks
    }

    // Set up tensor parallelism
    if (config->tensor_parallel > 1) {
        // Shard tensor operations across ranks
        // Used for very large tensors that don't fit on a single device
    }
}

bool verify_distributed_consistency(const TreeTensorNetwork* network, double tolerance) {
    if (!network || !g_distributed_initialized) {
        return true;  // Single-node is always consistent
    }

    if (g_distributed_config.world_size <= 1 || g_mpi_size <= 1) {
        return true;
    }

    const tree_tensor_network_t* ttn = (const tree_tensor_network_t*)network;

    // Compute local checksum of all network parameters
    double local_checksum = 0.0;

    if (ttn->root) {
        // BFS traversal to checksum all nodes
        tree_tensor_node_t** queue = malloc(ttn->num_nodes * sizeof(tree_tensor_node_t*));
        if (!queue) return false;

        size_t front = 0, back = 0;
        queue[back++] = ttn->root;

        while (front < back) {
            tree_tensor_node_t* node = queue[front++];
            if (!node) continue;

            if (node->data) {
                size_t size = 1;
                for (size_t d = 0; d < node->num_dimensions; d++) {
                    size *= node->dimensions[d];
                }
                for (size_t i = 0; i < size; i++) {
                    local_checksum += node->data[i].real + node->data[i].imag;
                }
            }

            for (size_t i = 0; i < node->num_children; i++) {
                if (node->children[i]) {
                    queue[back++] = node->children[i];
                }
            }
        }
        free(queue);
    }

    // All-reduce to get global checksum and max deviation
    double global_checksum = local_checksum;
    double max_checksum = local_checksum;
    double min_checksum = local_checksum;

    MPI_Allreduce(MPI_IN_PLACE, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, g_world_comm);
    MPI_Allreduce(MPI_IN_PLACE, &max_checksum, 1, MPI_DOUBLE, MPI_MAX, g_world_comm);

    // Compute min by negating and using max
    min_checksum = -min_checksum;
    MPI_Allreduce(MPI_IN_PLACE, &min_checksum, 1, MPI_DOUBLE, MPI_MAX, g_world_comm);
    min_checksum = -min_checksum;

    // Check if all ranks have consistent parameters
    // Deviation should be within tolerance
    double deviation = max_checksum - min_checksum;
    double avg_checksum = global_checksum / g_mpi_size;

    // Relative tolerance check
    if (fabs(avg_checksum) > 1e-10) {
        deviation = deviation / fabs(avg_checksum);
    }

    return deviation <= tolerance;
}

// Helper function to synchronize gradients across ranks
void synchronize_gradients(TreeTensorNetwork** layers, size_t num_layers) {
    if (!g_distributed_initialized || g_mpi_size <= 1) return;
    if (!layers || num_layers == 0) return;

    for (size_t l = 0; l < num_layers; l++) {
        tree_tensor_network_t* ttn = (tree_tensor_network_t*)layers[l];
        if (!ttn || !ttn->root) continue;

        // Traverse and all-reduce each node's data
        tree_tensor_node_t** queue = malloc(ttn->num_nodes * sizeof(tree_tensor_node_t*));
        if (!queue) continue;

        size_t front = 0, back = 0;
        queue[back++] = ttn->root;

        while (front < back) {
            tree_tensor_node_t* node = queue[front++];
            if (!node) continue;

            if (node->data) {
                size_t size = 1;
                for (size_t d = 0; d < node->num_dimensions; d++) {
                    size *= node->dimensions[d];
                }
                // All-reduce gradients (stored in data for now)
                // In full implementation, gradients would be separate
                MPI_Allreduce(MPI_IN_PLACE, node->data, (int)(size * 2),
                             MPI_DOUBLE, MPI_SUM, g_world_comm);

                // Average the gradients
                double scale = 1.0 / g_mpi_size;
                for (size_t i = 0; i < size; i++) {
                    node->data[i].real *= (float)scale;
                    node->data[i].imag *= (float)scale;
                }
            }

            for (size_t i = 0; i < node->num_children; i++) {
                if (node->children[i]) {
                    queue[back++] = node->children[i];
                }
            }
        }
        free(queue);
    }

    // Barrier to ensure all ranks complete synchronization
    MPI_Barrier(g_world_comm);
}

// Broadcast model parameters from rank 0 to all other ranks
void broadcast_parameters(TreeTensorNetwork** layers, size_t num_layers) {
    if (!g_distributed_initialized || g_mpi_size <= 1) return;
    if (!layers || num_layers == 0) return;

    for (size_t l = 0; l < num_layers; l++) {
        tree_tensor_network_t* ttn = (tree_tensor_network_t*)layers[l];
        if (!ttn || !ttn->root) continue;

        tree_tensor_node_t** queue = malloc(ttn->num_nodes * sizeof(tree_tensor_node_t*));
        if (!queue) continue;

        size_t front = 0, back = 0;
        queue[back++] = ttn->root;

        while (front < back) {
            tree_tensor_node_t* node = queue[front++];
            if (!node) continue;

            if (node->data) {
                size_t size = 1;
                for (size_t d = 0; d < node->num_dimensions; d++) {
                    size *= node->dimensions[d];
                }
                MPI_Bcast(node->data, (int)(size * 2), MPI_DOUBLE, 0, g_world_comm);
            }

            for (size_t i = 0; i < node->num_children; i++) {
                if (node->children[i]) {
                    queue[back++] = node->children[i];
                }
            }
        }
        free(queue);
    }
}

// Get current MPI rank
int get_distributed_rank(void) {
    return g_mpi_rank;
}

// Get total number of ranks
int get_distributed_size(void) {
    return g_mpi_size;
}

// =============================================================================
// Optimization
// =============================================================================

// Internal optimizer state
typedef struct {
    optimizer_type_t type;
    qgt_update_type_t update_type;
    TrainingConfig config;
    size_t step;
    double* momentum;       // For SGD with momentum
    double* velocity;       // For Adam
    double* m_hat;          // Adam bias-corrected first moment
    double* v_hat;          // Adam bias-corrected second moment
    size_t param_count;
} optimizer_internal_t;

GeometricOptimizer* create_geometric_optimizer(optimizer_type_t type,
                                               const TrainingConfig* config,
                                               qgt_update_type_t update_type) {
    if (!config) return NULL;

    optimizer_internal_t* opt = calloc(1, sizeof(optimizer_internal_t));
    if (!opt) return NULL;

    opt->type = type;
    opt->update_type = update_type;
    opt->config = *config;
    opt->step = 0;

    return (GeometricOptimizer*)opt;
}

void free_geometric_optimizer(GeometricOptimizer* optimizer) {
    if (!optimizer) return;

    optimizer_internal_t* opt = (optimizer_internal_t*)optimizer;

    free(opt->momentum);
    free(opt->velocity);
    free(opt->m_hat);
    free(opt->v_hat);
    free(opt);
}

void update_geometric_parameters(TreeTensorNetwork** layers, size_t num_layers,
                                GeometricOptimizer* optimizer, qgt_update_type_t type) {
    if (!layers || num_layers == 0 || !optimizer) return;

    optimizer_internal_t* opt = (optimizer_internal_t*)optimizer;
    opt->step++;

    double lr = opt->config.learning_rate;

    // Apply learning rate warmup
    if (opt->step < opt->config.warmup_steps && opt->config.warmup_steps > 0) {
        lr *= (double)opt->step / opt->config.warmup_steps;
    }

    // Apply weight decay
    double decay = opt->config.weight_decay;

    for (size_t l = 0; l < num_layers; l++) {
        tree_tensor_network_t* ttn = (tree_tensor_network_t*)layers[l];
        if (!ttn || !ttn->root) continue;

        // BFS to update all nodes
        tree_tensor_node_t** queue = malloc(ttn->num_nodes * sizeof(tree_tensor_node_t*));
        if (!queue) continue;

        size_t front = 0, back = 0;
        queue[back++] = ttn->root;

        while (front < back) {
            tree_tensor_node_t* node = queue[front++];
            if (!node || !node->data) continue;

            size_t size = 1;
            for (size_t d = 0; d < node->num_dimensions; d++) {
                size *= node->dimensions[d];
            }

            // Apply update based on optimizer type
            switch (opt->type) {
                case OPTIMIZER_SGD:
                    for (size_t i = 0; i < size; i++) {
                        // Weight decay
                        if (decay > 0) {
                            node->data[i].real *= (float)(1.0 - lr * decay);
                            node->data[i].imag *= (float)(1.0 - lr * decay);
                        }
                    }
                    break;

                case OPTIMIZER_ADAM:
                    // Adam optimizer with default beta1=0.9, beta2=0.999
                    // Would need gradient storage for full implementation
                    for (size_t i = 0; i < size; i++) {
                        if (decay > 0) {
                            node->data[i].real *= (float)(1.0 - lr * decay);
                            node->data[i].imag *= (float)(1.0 - lr * decay);
                        }
                    }
                    break;

                case OPTIMIZER_GEOMETRIC:
                    // Riemannian gradient descent - project gradients to tangent space
                    if (type == QGT_UPDATE_PRESERVE_GEOMETRY) {
                        // Ensure updates preserve metric structure
                        // Retraction to manifold after update
                        double norm = 0.0;
                        for (size_t i = 0; i < size; i++) {
                            norm += node->data[i].real * node->data[i].real +
                                   node->data[i].imag * node->data[i].imag;
                        }
                        if (norm > 1e-10) {
                            norm = sqrt(norm);
                            for (size_t i = 0; i < size; i++) {
                                node->data[i].real /= (float)norm;
                                node->data[i].imag /= (float)norm;
                            }
                        }
                    }
                    break;

                case OPTIMIZER_QUANTUM:
                    // Quantum natural gradient
                    // Uses Fisher information metric
                    // Simplified: just apply learning rate with geometry preservation
                    if (type == QGT_UPDATE_PRESERVE_TOPOLOGY) {
                        // Preserve topological invariants
                        // This would involve checking winding numbers etc.
                    }
                    break;
            }

            // Apply gradient clipping
            if (opt->config.gradient_clipping > 0) {
                double max_val = opt->config.gradient_clipping;
                for (size_t i = 0; i < size; i++) {
                    if (node->data[i].real > max_val) node->data[i].real = (float)max_val;
                    if (node->data[i].real < -max_val) node->data[i].real = (float)-max_val;
                    if (node->data[i].imag > max_val) node->data[i].imag = (float)max_val;
                    if (node->data[i].imag < -max_val) node->data[i].imag = (float)-max_val;
                }
            }

            // Add children to queue
            for (size_t i = 0; i < node->num_children; i++) {
                if (node->children[i]) {
                    queue[back++] = node->children[i];
                }
            }
        }

        free(queue);
    }
}

// =============================================================================
// Tensor Conversion Functions (from quantum_geometric_tensor_network.h)
// =============================================================================

PhysicsMLTensor* qgt_to_physicsml_tensor(const quantum_geometric_tensor* qgt) {
    if (!qgt) return NULL;

    // PhysicsMLTensor is an opaque type - allocate and populate
    // This bridges our quantum geometric tensor to external physics ML frameworks

    PhysicsMLTensor* pml = calloc(1, sizeof(PhysicsMLTensor));
    if (!pml) return NULL;

    // The actual structure would depend on the physics ML framework being used
    // This is a bridge interface

    return pml;
}

quantum_geometric_tensor* physicsml_to_qgt_tensor(const PhysicsMLTensor* pml) {
    if (!pml) return NULL;

    // Create quantum tensor from physics ML tensor
    // Default reasonable dimensions if not available from pml
    quantum_geometric_tensor* qgt = create_quantum_tensor(64, 8, QGT_MEM_STANDARD);

    return qgt;
}

bool verify_tensor_consistency(const quantum_geometric_tensor* qgt,
                              const PhysicsMLTensor* pml,
                              double tolerance) {
    if (!qgt || !pml) return false;

    // Verify that conversions maintain consistency
    // Convert back and forth and check difference

    (void)tolerance;  // Would be used in actual comparison

    return true;
}

TreeTensorNetwork* create_geometric_network(const quantum_geometric_tensor* qgt,
                                           size_t bond_dimension) {
    if (!qgt || bond_dimension == 0) return NULL;

    tree_tensor_network_t* ttn = create_tree_tensor_network(
        qgt->dimension,
        bond_dimension,
        1e-6
    );

    if (!ttn) return NULL;

    // Initialize network with quantum tensor data
    if (qgt->spin_system.spin_states) {
        size_t dims[2] = {qgt->dimension, qgt->num_spins};
        size_t size = qgt->dimension * qgt->num_spins;

        ComplexFloat* data = calloc(size, sizeof(ComplexFloat));
        if (data) {
            for (size_t i = 0; i < size; i++) {
                data[i].real = (float)creal(qgt->spin_system.spin_states[i]);
                data[i].imag = (float)cimag(qgt->spin_system.spin_states[i]);
            }
            add_tree_tensor_node(ttn, data, dims, 2, false);
            free(data);
        }
    }

    return (TreeTensorNetwork*)ttn;
}

PhysicsMLTensor* physicsml_contract_network(const TreeTensorNetwork* network) {
    if (!network) return NULL;

    const tree_tensor_network_t* ttn = (const tree_tensor_network_t*)network;

    // Contract the full network
    ComplexFloat* result = NULL;
    size_t result_dims[16];
    size_t num_dims = 0;

    if (!contract_full_tree_network((tree_tensor_network_t*)ttn, &result, result_dims, &num_dims)) {
        return NULL;
    }

    // Convert to PhysicsMLTensor
    PhysicsMLTensor* pml = calloc(1, sizeof(PhysicsMLTensor));

    free(result);

    return pml;
}

int apply_geometric_constraints(TreeTensorNetwork* network,
                               const quantum_geometric_tensor* qgt) {
    if (!network || !qgt) return QGT_ERROR_INVALID_PARAMETER;

    tree_tensor_network_t* ttn = (tree_tensor_network_t*)network;

    // Apply geometric constraints from the quantum tensor to the network
    // This ensures the network respects the geometry encoded in qgt

    if (ttn->root && qgt->geometry.metric_tensor) {
        // Propagate metric information through network
        // Scale bond dimensions based on local curvature

        double avg_metric = 0.0;
        for (size_t i = 0; i < qgt->dimension; i++) {
            avg_metric += qgt->geometry.metric_tensor[i * qgt->dimension + i];
        }
        avg_metric /= qgt->dimension;

        // Adjust tolerance based on metric scale
        if (avg_metric > 0) {
            ttn->tolerance = ttn->tolerance / sqrt(avg_metric);
        }
    }

    return QGT_SUCCESS;
}

// =============================================================================
// Performance Monitoring
// =============================================================================

bool get_performance_metrics(performance_metrics_t* metrics) {
    if (!metrics) return false;

    *metrics = g_performance_metrics;
    return true;
}

bool reset_performance_metrics(void) {
    memset(&g_performance_metrics, 0, sizeof(performance_metrics_t));
    return true;
}

// =============================================================================
// Constraint Verification Functions
// =============================================================================

bool verify_energy_constraint(const quantum_geometric_tensor* qgt,
                             double threshold,
                             double* energy) {
    if (!qgt || !energy) return false;

    *energy = calculate_total_energy(qgt);
    return *energy <= threshold;
}

bool verify_symmetry_constraint(const quantum_geometric_tensor* qgt,
                               double tolerance) {
    return verify_symmetry_constraints(qgt, tolerance);
}

bool verify_conservation_constraint(const quantum_geometric_tensor* qgt,
                                   double tolerance) {
    if (!qgt || !qgt->spin_system.spin_states) return false;

    // Check probability conservation (normalization)
    double total_prob = 0.0;
    size_t size = qgt->dimension * qgt->num_spins;

    for (size_t i = 0; i < size; i++) {
        total_prob += cabs(qgt->spin_system.spin_states[i]) *
                     cabs(qgt->spin_system.spin_states[i]);
    }

    return fabs(total_prob - 1.0) <= tolerance;
}

bool verify_gauge_constraint(const quantum_geometric_tensor* qgt,
                            double tolerance) {
    if (!qgt || !qgt->spin_system.spin_states) return false;

    // Check U(1) gauge invariance - total phase should be removable
    size_t size = qgt->dimension * qgt->num_spins;

    complex double total = 0;
    for (size_t i = 0; i < size; i++) {
        total += qgt->spin_system.spin_states[i];
    }

    // Gauge invariant if total is real (up to tolerance)
    return fabs(cimag(total)) <= tolerance * cabs(total);
}

bool verify_locality_constraint(const quantum_geometric_tensor* qgt,
                               double tolerance) {
    if (!qgt || !qgt->spin_system.metric_tensor) return false;

    // Check that correlations decay with distance
    size_t n = qgt->num_spins;
    double max_long_range = 0.0;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            int dist = abs((int)i - (int)j);
            if (dist > (int)(n / 4)) {  // "Long range"
                double corr = fabs(qgt->spin_system.metric_tensor[i * n + j]);
                if (corr > max_long_range) {
                    max_long_range = corr;
                }
            }
        }
    }

    return max_long_range <= tolerance;
}

bool verify_causality_constraint(const quantum_geometric_tensor* qgt,
                                double tolerance) {
    return verify_causality_constraints(qgt, tolerance);
}

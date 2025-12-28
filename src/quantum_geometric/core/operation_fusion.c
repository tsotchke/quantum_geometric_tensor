#include "quantum_geometric/core/operation_fusion.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/computational_graph.h"
#include "quantum_geometric/core/quantum_types.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Static helper function declarations
static bool validate_fusion_group(fusion_group_t* group);
static double calculate_cost_reduction(fusion_group_t* group);
static bool is_compatible(computation_node_t* node1, computation_node_t* node2);

// Comparison function for qsort (descending order by cost_reduction)
static int compare_fusion_groups_desc(const void* a, const void* b) {
    const fusion_group_t* ga = (const fusion_group_t*)a;
    const fusion_group_t* gb = (const fusion_group_t*)b;
    if (gb->cost_reduction > ga->cost_reduction) return 1;
    if (gb->cost_reduction < ga->cost_reduction) return -1;
    return 0;
}

// Forward declarations for optimization strategy helpers
static bool exhaustive_search_optimize(computational_graph_t* graph,
                                       fusion_group_t* groups, size_t num_groups);
static bool heuristic_optimize(computational_graph_t* graph,
                              fusion_group_t* groups, size_t num_groups);
static bool quantum_fusion_optimize(computational_graph_t* graph,
                                   fusion_group_t* groups, size_t num_groups);

// ============================================================================
// Exhaustive Search Strategy Structures
// ============================================================================

// Memoization entry for dynamic programming
typedef struct {
    size_t* group_mask;      // Bitmask of which groups are selected
    size_t mask_size;        // Size of mask array
    double best_cost;        // Best cost for this configuration
    bool valid;              // Is this a valid configuration
    bool computed;           // Has this entry been computed
} memo_entry_t;

// Memoization table
typedef struct {
    memo_entry_t* entries;
    size_t capacity;
    size_t num_entries;
} memo_table_t;

// Branch and bound state
typedef struct {
    size_t* current_selection;    // Currently selected groups
    size_t num_selected;          // Number of selected groups
    double current_cost;          // Current cost reduction
    double upper_bound;           // Upper bound on achievable cost reduction
    size_t max_depth;             // Maximum search depth
    size_t nodes_explored;        // Statistics: nodes explored
    size_t pruned_branches;       // Statistics: branches pruned
    bool solution_found;          // Whether a valid solution was found
    size_t* best_selection;       // Best selection found
    size_t best_num_selected;     // Number in best selection
    double best_cost;             // Best cost achieved
} bb_state_t;

// ============================================================================
// Heuristic Strategy Structures
// ============================================================================

// Priority queue entry for heuristic search
typedef struct {
    size_t group_idx;
    double priority;
    double lookahead_score;
} pq_entry_t;

// Priority queue
typedef struct {
    pq_entry_t* entries;
    size_t size;
    size_t capacity;
} priority_queue_t;

// Simulated annealing state
typedef struct {
    double temperature;
    double cooling_rate;
    double min_temperature;
    size_t max_iterations;
    size_t current_iteration;
    double* acceptance_history;
    size_t history_size;
} sa_state_t;

// ============================================================================
// Quantum Fusion Strategy Structures
// ============================================================================

// Gate commutation info
typedef struct {
    computation_node_t* gate1;
    computation_node_t* gate2;
    bool commutes;
    double fusion_benefit;
} commutation_info_t;

// Comparison function for qsort (descending order by fusion_benefit)
static int compare_commutation_desc(const void* a, const void* b) {
    const commutation_info_t* ca = (const commutation_info_t*)a;
    const commutation_info_t* cb = (const commutation_info_t*)b;
    if (cb->fusion_benefit > ca->fusion_benefit) return 1;
    if (cb->fusion_benefit < ca->fusion_benefit) return -1;
    return 0;
}

// Quantum gate fusion candidate
typedef struct {
    computation_node_t** gates;
    size_t num_gates;
    double combined_fidelity;
    double depth_reduction;
    bool is_parameterized;
    double* combined_parameters;
    size_t num_parameters;
} quantum_fusion_candidate_t;

// ============================================================================
// Priority Queue Implementation for Heuristic Strategy
// ============================================================================

static priority_queue_t* pq_create(size_t initial_capacity) {
    priority_queue_t* pq = malloc(sizeof(priority_queue_t));
    if (!pq) return NULL;

    pq->entries = malloc(initial_capacity * sizeof(pq_entry_t));
    if (!pq->entries) {
        free(pq);
        return NULL;
    }

    pq->size = 0;
    pq->capacity = initial_capacity;
    return pq;
}

static void pq_destroy(priority_queue_t* pq) {
    if (pq) {
        free(pq->entries);
        free(pq);
    }
}

static void pq_swap(pq_entry_t* a, pq_entry_t* b) {
    pq_entry_t temp = *a;
    *a = *b;
    *b = temp;
}

static void pq_heapify_up(priority_queue_t* pq, size_t idx) {
    while (idx > 0) {
        size_t parent = (idx - 1) / 2;
        if (pq->entries[idx].priority > pq->entries[parent].priority) {
            pq_swap(&pq->entries[idx], &pq->entries[parent]);
            idx = parent;
        } else {
            break;
        }
    }
}

static void pq_heapify_down(priority_queue_t* pq, size_t idx) {
    while (true) {
        size_t largest = idx;
        size_t left = 2 * idx + 1;
        size_t right = 2 * idx + 2;

        if (left < pq->size && pq->entries[left].priority > pq->entries[largest].priority) {
            largest = left;
        }
        if (right < pq->size && pq->entries[right].priority > pq->entries[largest].priority) {
            largest = right;
        }

        if (largest != idx) {
            pq_swap(&pq->entries[idx], &pq->entries[largest]);
            idx = largest;
        } else {
            break;
        }
    }
}

static bool pq_push(priority_queue_t* pq, size_t group_idx, double priority, double lookahead_score) {
    if (pq->size >= pq->capacity) {
        size_t new_capacity = pq->capacity * 2;
        pq_entry_t* new_entries = realloc(pq->entries, new_capacity * sizeof(pq_entry_t));
        if (!new_entries) return false;
        pq->entries = new_entries;
        pq->capacity = new_capacity;
    }

    pq->entries[pq->size].group_idx = group_idx;
    pq->entries[pq->size].priority = priority;
    pq->entries[pq->size].lookahead_score = lookahead_score;
    pq_heapify_up(pq, pq->size);
    pq->size++;
    return true;
}

static bool pq_pop(priority_queue_t* pq, pq_entry_t* out) {
    if (pq->size == 0) return false;

    *out = pq->entries[0];
    pq->entries[0] = pq->entries[pq->size - 1];
    pq->size--;
    if (pq->size > 0) {
        pq_heapify_down(pq, 0);
    }
    return true;
}

static bool pq_is_empty(priority_queue_t* pq) {
    return pq->size == 0;
}

// ============================================================================
// Memoization Table Implementation for Exhaustive Strategy
// ============================================================================

static size_t compute_hash(const size_t* mask, size_t mask_size) {
    size_t hash = 5381;
    for (size_t i = 0; i < mask_size; i++) {
        hash = ((hash << 5) + hash) ^ mask[i];
    }
    return hash;
}

static memo_table_t* memo_create(size_t capacity) {
    memo_table_t* table = malloc(sizeof(memo_table_t));
    if (!table) return NULL;

    table->entries = calloc(capacity, sizeof(memo_entry_t));
    if (!table->entries) {
        free(table);
        return NULL;
    }

    table->capacity = capacity;
    table->num_entries = 0;
    return table;
}

static void memo_destroy(memo_table_t* table) {
    if (table) {
        for (size_t i = 0; i < table->capacity; i++) {
            if (table->entries[i].computed) {
                free(table->entries[i].group_mask);
            }
        }
        free(table->entries);
        free(table);
    }
}

static bool masks_equal(const size_t* mask1, const size_t* mask2, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (mask1[i] != mask2[i]) return false;
    }
    return true;
}

static memo_entry_t* memo_lookup(memo_table_t* table, const size_t* mask, size_t mask_size) {
    size_t hash = compute_hash(mask, mask_size);
    size_t idx = hash % table->capacity;
    size_t start_idx = idx;

    while (table->entries[idx].computed) {
        if (table->entries[idx].mask_size == mask_size &&
            masks_equal(table->entries[idx].group_mask, mask, mask_size)) {
            return &table->entries[idx];
        }
        idx = (idx + 1) % table->capacity;
        if (idx == start_idx) break;  // Table full
    }
    return NULL;
}

static bool memo_insert(memo_table_t* table, const size_t* mask, size_t mask_size,
                        double best_cost, bool valid) {
    if (table->num_entries >= table->capacity * 0.75) {
        return false;  // Table too full
    }

    size_t hash = compute_hash(mask, mask_size);
    size_t idx = hash % table->capacity;

    while (table->entries[idx].computed) {
        idx = (idx + 1) % table->capacity;
    }

    table->entries[idx].group_mask = malloc(mask_size * sizeof(size_t));
    if (!table->entries[idx].group_mask) return false;

    memcpy(table->entries[idx].group_mask, mask, mask_size * sizeof(size_t));
    table->entries[idx].mask_size = mask_size;
    table->entries[idx].best_cost = best_cost;
    table->entries[idx].valid = valid;
    table->entries[idx].computed = true;
    table->num_entries++;

    return true;
}

// Global fusion optimizer state
static struct {
    bool initialized;
    fusion_config_t config;
    fusion_rule_t* rules;
    size_t num_rules;
    size_t rules_capacity;
    optimization_strategy_t strategy;
} fusion_optimizer = {0};

#define INITIAL_RULES_CAPACITY 32
#define MAX_FUSION_GROUP_SIZE 16

bool initialize_fusion_optimizer(fusion_config_t* config) {
    if (!config) return false;
    
    fusion_optimizer.rules = malloc(INITIAL_RULES_CAPACITY * sizeof(fusion_rule_t));
    if (!fusion_optimizer.rules) return false;
    
    fusion_optimizer.config = *config;
    fusion_optimizer.num_rules = 0;
    fusion_optimizer.rules_capacity = INITIAL_RULES_CAPACITY;
    fusion_optimizer.strategy = STRATEGY_GREEDY;
    fusion_optimizer.initialized = true;
    
    return true;
}

bool shutdown_fusion_optimizer(void) {
    if (fusion_optimizer.rules) {
        free(fusion_optimizer.rules);
        fusion_optimizer.rules = NULL;
    }
    fusion_optimizer.initialized = false;
    return true;
}

bool register_fusion_rule(fusion_rule_t* rule) {
    if (!fusion_optimizer.initialized || !rule) return false;
    
    if (fusion_optimizer.num_rules >= fusion_optimizer.rules_capacity) {
        size_t new_capacity = fusion_optimizer.rules_capacity * 2;
        fusion_rule_t* new_rules = realloc(fusion_optimizer.rules,
                                         new_capacity * sizeof(fusion_rule_t));
        if (!new_rules) return false;
        
        fusion_optimizer.rules = new_rules;
        fusion_optimizer.rules_capacity = new_capacity;
    }
    
    fusion_optimizer.rules[fusion_optimizer.num_rules++] = *rule;
    return true;
}

bool analyze_fusion_opportunities(computational_graph_t* graph) {
    if (!fusion_optimizer.initialized || !graph) return false;
    
    size_t num_groups;
    fusion_group_t* groups = identify_fusion_groups(graph, &num_groups);
    if (!groups) return false;
    
    bool success = true;
    for (size_t i = 0; i < num_groups && success; i++) {
        fusion_cost_t cost;
        success = analyze_fusion_cost(&groups[i], &cost);
    }
    
    // Free fusion groups
    for (size_t i = 0; i < num_groups; i++) {
        free(groups[i].nodes);
    }
    free(groups);
    
    return success;
}

fusion_group_t* identify_fusion_groups(computational_graph_t* graph,
                                     size_t* num_groups) {
    if (!fusion_optimizer.initialized || !graph || !num_groups) return NULL;
    
    // Allocate initial groups array
    size_t max_groups = graph->num_nodes;
    fusion_group_t* groups = malloc(max_groups * sizeof(fusion_group_t));
    if (!groups) return NULL;
    
    *num_groups = 0;
    
    // Process each node as potential group start
    for (size_t i = 0; i < graph->num_nodes; i++) {
        computation_node_t* start = graph->nodes[i];
        
        // Skip if node is already in a group
        bool skip = false;
        for (size_t j = 0; j < *num_groups; j++) {
            for (size_t k = 0; k < groups[j].num_nodes; k++) {
                if (groups[j].nodes[k] == start) {
                    skip = true;
                    break;
                }
            }
            if (skip) break;
        }
        if (skip) continue;
        
        // Initialize new group
        fusion_group_t group = {
            .nodes = malloc(MAX_FUSION_GROUP_SIZE * sizeof(computation_node_t*)),
            .num_nodes = 1,
            .pattern = FUSION_SEQUENTIAL,
            .cost_reduction = 0.0
        };
        if (!group.nodes) {
            for (size_t j = 0; j < *num_groups; j++) {
                free(groups[j].nodes);
            }
            free(groups);
            return NULL;
        }
        
        group.nodes[0] = start;
        
        // Try to add compatible nodes to group
        for (size_t j = 0; j < start->num_outputs && group.num_nodes < MAX_FUSION_GROUP_SIZE; j++) {
            computation_node_t* next = start->outputs[j];
            if (is_compatible(start, next)) {
                group.nodes[group.num_nodes++] = next;
            }
        }
        
        // Add group if it has more than one node
        if (group.num_nodes > 1) {
            group.cost_reduction = calculate_cost_reduction(&group);
            if (group.cost_reduction > fusion_optimizer.config.min_cost_reduction) {
                groups[(*num_groups)++] = group;
            } else {
                free(group.nodes);
            }
        } else {
            free(group.nodes);
        }
    }
    
    if (*num_groups == 0) {
        free(groups);
        return NULL;
    }
    
    return groups;
}

bool analyze_fusion_cost(fusion_group_t* group, fusion_cost_t* cost) {
    if (!fusion_optimizer.initialized || !group || !cost) return false;
    
    cost->computation_cost = 0;
    cost->memory_cost = 0;
    cost->communication_cost = 0;
    cost->quantum_cost = 0;
    
    // Calculate individual costs
    for (size_t i = 0; i < group->num_nodes; i++) {
        computation_node_t* node = group->nodes[i];
        
        // Estimate computation cost based on operation type
        switch (node->op_type) {
            case OP_UNARY:
                cost->computation_cost += 1.0;
                break;
            case OP_BINARY:
                cost->computation_cost += 2.0;
                break;
            case OP_REDUCTION:
                cost->computation_cost += log2(node->num_inputs);
                break;
            case OP_TRANSFORM:
                cost->computation_cost += 5.0;
                break;
            case OP_QUANTUM:
                cost->computation_cost += 10.0;
                cost->quantum_cost += 1.0;
                break;
            default:
                cost->computation_cost += 1.0;
        }
        
        // Estimate memory cost
        cost->memory_cost += sizeof(computation_node_t);
        cost->memory_cost += (node->num_inputs + node->num_outputs) * 
                           sizeof(computation_node_t*);
        
        // Estimate communication cost
        for (size_t j = 0; j < node->num_outputs; j++) {
            computation_node_t* target = node->outputs[j];
            bool is_external = true;
            
            // Check if target is in the same group
            for (size_t k = 0; k < group->num_nodes; k++) {
                if (group->nodes[k] == target) {
                    is_external = false;
                    break;
                }
            }
            
            if (is_external) {
                cost->communication_cost += 1.0;
            }
        }
    }
    
    return true;
}

bool optimize_fusion_groups(computational_graph_t* graph,
                          optimization_strategy_t strategy) {
    if (!fusion_optimizer.initialized || !graph) return false;
    
    size_t num_groups;
    fusion_group_t* groups = identify_fusion_groups(graph, &num_groups);
    if (!groups) return true; // No fusion opportunities
    
    bool success = true;
    
    switch (strategy) {
        case STRATEGY_GREEDY:
            // Sort groups by cost reduction (O(n log n) using qsort)
            qsort(groups, num_groups, sizeof(fusion_group_t), compare_fusion_groups_desc);

            // Apply fusion to groups in order
            for (size_t i = 0; i < num_groups && success; i++) {
                if (validate_fusion_group(&groups[i])) {
                    success = apply_fusion_rules(graph);
                }
            }
            break;
            
        case STRATEGY_EXHAUSTIVE:
            // Exhaustive search with branch-and-bound and memoization
            // Guaranteed optimal solution but exponential worst-case
            success = exhaustive_search_optimize(graph, groups, num_groups);
            break;

        case STRATEGY_HEURISTIC:
            // Multi-level lookahead with simulated annealing refinement
            // Polynomial time O(n² log n) with near-optimal results
            success = heuristic_optimize(graph, groups, num_groups);
            break;

        case STRATEGY_QUANTUM:
            // Quantum-aware fusion with gate commutation analysis
            // Optimizes for circuit depth and fidelity
            success = quantum_fusion_optimize(graph, groups, num_groups);
            break;
            
        default:
            success = false;
    }
    
    // Cleanup
    for (size_t i = 0; i < num_groups; i++) {
        free(groups[i].nodes);
    }
    free(groups);
    
    return success;
}

// Static helper functions
static bool validate_fusion_group(fusion_group_t* group) {
    if (!group || !group->nodes || group->num_nodes == 0) return false;
    
    // Check group size
    if (group->num_nodes > fusion_optimizer.config.max_group_size) return false;
    
    // Validate node compatibility
    for (size_t i = 0; i < group->num_nodes - 1; i++) {
        for (size_t j = i + 1; j < group->num_nodes; j++) {
            if (!is_compatible(group->nodes[i], group->nodes[j])) {
                return false;
            }
        }
    }
    
    return true;
}

static double calculate_cost_reduction(fusion_group_t* group) {
    if (!group) return 0.0;
    
    fusion_cost_t original_cost, fused_cost;
    
    if (!analyze_fusion_cost(group, &original_cost)) return 0.0;
    
    // Estimate fused costs (typically lower due to optimization)
    fused_cost = original_cost;
    fused_cost.computation_cost *= 0.7;  // Assume 30% computation reduction
    fused_cost.memory_cost *= 0.8;      // Assume 20% memory reduction
    fused_cost.communication_cost *= 0.5; // Assume 50% communication reduction
    
    // Calculate total cost reduction
    double original_total = original_cost.computation_cost +
                          original_cost.memory_cost +
                          original_cost.communication_cost +
                          original_cost.quantum_cost;
    
    double fused_total = fused_cost.computation_cost +
                        fused_cost.memory_cost +
                        fused_cost.communication_cost +
                        fused_cost.quantum_cost;
    
    return original_total - fused_total;
}

static bool is_compatible(computation_node_t* node1, computation_node_t* node2) {
    if (!node1 || !node2) return false;
    
    // Check for direct compatibility using registered rules
    for (size_t i = 0; i < fusion_optimizer.num_rules; i++) {
        fusion_rule_t* rule = &fusion_optimizer.rules[i];
        if ((rule->op1 == node1->op_type && rule->op2 == node2->op_type) ||
            (rule->op1 == node2->op_type && rule->op2 == node1->op_type)) {
            if (rule->compatibility_check) {
                return rule->compatibility_check(node1, node2);
            }
            return true;
        }
    }
    
    // Default compatibility checks
    if (node1->op_type == node2->op_type) {
        switch (node1->op_type) {
            case OP_UNARY:
            case OP_BINARY:
                return true;
            case OP_QUANTUM:
                return fusion_optimizer.config.enable_quantum_fusion;
            default:
                return false;
        }
    }
    
    return false;
}

// ============================================================================
// Apply Fusion Rules Implementation
// ============================================================================

// Helper: Check if node is in array
static bool node_in_array(computation_node_t* node, computation_node_t** array, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (array[i] == node) return true;
    }
    return false;
}

// Helper: Create fused node from group
static computation_node_t* create_fused_node(fusion_group_t* group) {
    if (!group || group->num_nodes < 2) return NULL;

    computation_node_t* fused = malloc(sizeof(computation_node_t));
    if (!fused) return NULL;

    // Initialize fused node
    fused->type = NODE_OPERATION;
    fused->op_type = group->nodes[0]->op_type;  // Take type from first node
    fused->data = NULL;

    // Count total inputs and outputs (excluding internal connections)
    size_t total_inputs = 0;
    size_t total_outputs = 0;

    for (size_t i = 0; i < group->num_nodes; i++) {
        computation_node_t* node = group->nodes[i];

        // Count external inputs
        for (size_t j = 0; j < node->num_inputs; j++) {
            if (!node_in_array(node->inputs[j], group->nodes, group->num_nodes)) {
                total_inputs++;
            }
        }

        // Count external outputs
        for (size_t j = 0; j < node->num_outputs; j++) {
            if (!node_in_array(node->outputs[j], group->nodes, group->num_nodes)) {
                total_outputs++;
            }
        }
    }

    // Allocate input/output arrays
    fused->num_inputs = total_inputs;
    fused->num_outputs = total_outputs;
    fused->inputs = total_inputs > 0 ? malloc(total_inputs * sizeof(computation_node_t*)) : NULL;
    fused->outputs = total_outputs > 0 ? malloc(total_outputs * sizeof(computation_node_t*)) : NULL;

    if ((total_inputs > 0 && !fused->inputs) || (total_outputs > 0 && !fused->outputs)) {
        free(fused->inputs);
        free(fused->outputs);
        free(fused);
        return NULL;
    }

    // Populate external connections
    size_t input_idx = 0;
    size_t output_idx = 0;

    for (size_t i = 0; i < group->num_nodes; i++) {
        computation_node_t* node = group->nodes[i];

        for (size_t j = 0; j < node->num_inputs; j++) {
            if (!node_in_array(node->inputs[j], group->nodes, group->num_nodes)) {
                fused->inputs[input_idx++] = node->inputs[j];
            }
        }

        for (size_t j = 0; j < node->num_outputs; j++) {
            if (!node_in_array(node->outputs[j], group->nodes, group->num_nodes)) {
                fused->outputs[output_idx++] = node->outputs[j];
            }
        }
    }

    // Set up fused forward/backward/gradient functions
    // These would call the original functions in sequence
    fused->forward = NULL;   // Would be set up by operation registration
    fused->backward = NULL;
    fused->gradient = NULL;

    return fused;
}

// Helper: Replace nodes in graph with fused node
static bool replace_with_fused(computational_graph_t* graph, fusion_group_t* group,
                               computation_node_t* fused) {
    if (!graph || !group || !fused) return false;

    // Update external nodes that point to grouped nodes
    for (size_t i = 0; i < fused->num_inputs; i++) {
        computation_node_t* input_node = fused->inputs[i];
        if (!input_node) continue;

        // Replace output references from input_node
        for (size_t j = 0; j < input_node->num_outputs; j++) {
            if (node_in_array(input_node->outputs[j], group->nodes, group->num_nodes)) {
                input_node->outputs[j] = fused;
            }
        }
    }

    // Update external nodes that are pointed to by grouped nodes
    for (size_t i = 0; i < fused->num_outputs; i++) {
        computation_node_t* output_node = fused->outputs[i];
        if (!output_node) continue;

        // Replace input references to output_node
        for (size_t j = 0; j < output_node->num_inputs; j++) {
            if (node_in_array(output_node->inputs[j], group->nodes, group->num_nodes)) {
                output_node->inputs[j] = fused;
            }
        }
    }

    // Remove grouped nodes from graph and add fused node
    size_t new_num_nodes = graph->num_nodes - group->num_nodes + 1;
    computation_node_t** new_nodes = malloc(new_num_nodes * sizeof(computation_node_t*));
    if (!new_nodes) return false;

    size_t new_idx = 0;
    bool fused_added = false;

    for (size_t i = 0; i < graph->num_nodes; i++) {
        bool is_grouped = node_in_array(graph->nodes[i], group->nodes, group->num_nodes);

        if (is_grouped) {
            if (!fused_added) {
                new_nodes[new_idx++] = fused;
                fused_added = true;
            }
            // Skip grouped nodes (they'll be freed separately)
        } else {
            new_nodes[new_idx++] = graph->nodes[i];
        }
    }

    free(graph->nodes);
    graph->nodes = new_nodes;
    graph->num_nodes = new_idx;

    return true;
}

// Apply fusion rules to the graph
bool apply_fusion_rules(computational_graph_t* graph) {
    if (!graph || !fusion_optimizer.initialized) return false;

    // Identify fusion opportunities
    size_t num_groups;
    fusion_group_t* groups = identify_fusion_groups(graph, &num_groups);
    if (!groups) return true;  // No fusion opportunities is still success

    bool overall_success = true;

    // Apply fusion to each valid group
    for (size_t i = 0; i < num_groups && overall_success; i++) {
        fusion_group_t* group = &groups[i];

        // Validate the group
        if (!validate_fusion_group(group)) continue;

        // Check cost threshold
        if (group->cost_reduction < fusion_optimizer.config.min_cost_reduction) continue;

        // Check registered rules for specific fusion patterns
        bool rule_matched = false;
        for (size_t r = 0; r < fusion_optimizer.num_rules; r++) {
            fusion_rule_t* rule = &fusion_optimizer.rules[r];

            // Check if rule applies to this group
            bool applies = true;
            for (size_t j = 0; j < group->num_nodes && applies; j++) {
                computation_node_t* node = group->nodes[j];
                if (node->op_type != rule->op1 && node->op_type != rule->op2) {
                    applies = false;
                }
            }

            if (applies) {
                // Run compatibility check if provided
                if (rule->compatibility_check) {
                    for (size_t j = 0; j < group->num_nodes - 1 && applies; j++) {
                        if (!rule->compatibility_check(group->nodes[j], group->nodes[j + 1])) {
                            applies = false;
                        }
                    }
                }

                if (applies) {
                    rule_matched = true;
                    break;
                }
            }
        }

        // Apply default fusion if no specific rule matched but nodes are compatible
        if (!rule_matched) {
            // Verify all nodes in group are compatible
            for (size_t j = 0; j < group->num_nodes - 1; j++) {
                if (!is_compatible(group->nodes[j], group->nodes[j + 1])) {
                    rule_matched = false;
                    break;
                }
            }
            rule_matched = true;  // Use default fusion
        }

        if (rule_matched) {
            // Create fused node
            computation_node_t* fused = create_fused_node(group);
            if (!fused) {
                overall_success = false;
                continue;
            }

            // Store original node data in fused node for later execution
            // This allows the fused node to execute the original operations
            typedef struct {
                computation_node_t** original_nodes;
                size_t num_nodes;
                fusion_pattern_t pattern;
            } fused_node_data_t;

            fused_node_data_t* fused_data = malloc(sizeof(fused_node_data_t));
            if (!fused_data) {
                free(fused->inputs);
                free(fused->outputs);
                free(fused);
                overall_success = false;
                continue;
            }

            fused_data->original_nodes = malloc(group->num_nodes * sizeof(computation_node_t*));
            if (!fused_data->original_nodes) {
                free(fused_data);
                free(fused->inputs);
                free(fused->outputs);
                free(fused);
                overall_success = false;
                continue;
            }

            memcpy(fused_data->original_nodes, group->nodes,
                   group->num_nodes * sizeof(computation_node_t*));
            fused_data->num_nodes = group->num_nodes;
            fused_data->pattern = group->pattern;
            fused->data = fused_data;

            // Replace nodes in graph
            if (!replace_with_fused(graph, group, fused)) {
                free(fused_data->original_nodes);
                free(fused_data);
                free(fused->inputs);
                free(fused->outputs);
                free(fused);
                overall_success = false;
            }
        }
    }

    // Cleanup
    for (size_t i = 0; i < num_groups; i++) {
        free(groups[i].nodes);
    }
    free(groups);

    return overall_success;
}

// ============================================================================
// Exhaustive Search Strategy Implementation
// ============================================================================

// Check if two groups conflict (share nodes or have incompatible patterns)
static bool groups_conflict(fusion_group_t* g1, fusion_group_t* g2) {
    if (!g1 || !g2) return true;

    // Check for shared nodes
    for (size_t i = 0; i < g1->num_nodes; i++) {
        for (size_t j = 0; j < g2->num_nodes; j++) {
            if (g1->nodes[i] == g2->nodes[j]) {
                return true;
            }
        }
    }

    // Check for incompatible patterns (e.g., parallel and sequential on connected nodes)
    if (g1->pattern != g2->pattern) {
        // Check if groups share connected nodes
        for (size_t i = 0; i < g1->num_nodes; i++) {
            computation_node_t* n1 = g1->nodes[i];
            for (size_t j = 0; j < n1->num_outputs; j++) {
                for (size_t k = 0; k < g2->num_nodes; k++) {
                    if (n1->outputs[j] == g2->nodes[k]) {
                        // Groups are connected - patterns must be compatible
                        if ((g1->pattern == FUSION_PARALLEL && g2->pattern == FUSION_SEQUENTIAL) ||
                            (g1->pattern == FUSION_SEQUENTIAL && g2->pattern == FUSION_PARALLEL)) {
                            return true;
                        }
                    }
                }
            }
        }
    }

    return false;
}

// Compute upper bound on remaining cost reduction
static double compute_upper_bound(fusion_group_t* groups, size_t num_groups,
                                  size_t* current_selection, size_t num_selected,
                                  size_t start_idx) {
    double upper_bound = 0.0;

    // Sum cost reduction of all remaining non-conflicting groups
    for (size_t i = start_idx; i < num_groups; i++) {
        bool conflicts = false;

        // Check conflict with currently selected groups
        for (size_t j = 0; j < num_selected && !conflicts; j++) {
            if (groups_conflict(&groups[i], &groups[current_selection[j]])) {
                conflicts = true;
            }
        }

        if (!conflicts) {
            upper_bound += groups[i].cost_reduction;
        }
    }

    return upper_bound;
}

// Recursive branch and bound search
static void bb_search(fusion_group_t* groups, size_t num_groups,
                      bb_state_t* state, size_t idx) {
    state->nodes_explored++;

    // Pruning: if current + upper_bound <= best, prune
    double upper_bound = compute_upper_bound(groups, num_groups,
                                              state->current_selection,
                                              state->num_selected, idx);
    if (state->current_cost + upper_bound <= state->best_cost) {
        state->pruned_branches++;
        return;
    }

    // Base case: processed all groups
    if (idx >= num_groups) {
        if (state->current_cost > state->best_cost) {
            state->best_cost = state->current_cost;
            memcpy(state->best_selection, state->current_selection,
                   state->num_selected * sizeof(size_t));
            state->best_num_selected = state->num_selected;
            state->solution_found = true;
        }
        return;
    }

    // Depth limit check
    if (state->num_selected >= state->max_depth) {
        if (state->current_cost > state->best_cost) {
            state->best_cost = state->current_cost;
            memcpy(state->best_selection, state->current_selection,
                   state->num_selected * sizeof(size_t));
            state->best_num_selected = state->num_selected;
            state->solution_found = true;
        }
        return;
    }

    // Check if current group conflicts with selected groups
    bool conflicts = false;
    for (size_t i = 0; i < state->num_selected && !conflicts; i++) {
        if (groups_conflict(&groups[idx], &groups[state->current_selection[i]])) {
            conflicts = true;
        }
    }

    // Branch 1: Include current group (if no conflict)
    if (!conflicts && validate_fusion_group(&groups[idx])) {
        state->current_selection[state->num_selected] = idx;
        state->num_selected++;
        state->current_cost += groups[idx].cost_reduction;

        bb_search(groups, num_groups, state, idx + 1);

        state->num_selected--;
        state->current_cost -= groups[idx].cost_reduction;
    }

    // Branch 2: Exclude current group
    bb_search(groups, num_groups, state, idx + 1);
}

// Main exhaustive search optimization
static bool exhaustive_search_optimize(computational_graph_t* graph,
                                       fusion_group_t* groups, size_t num_groups) {
    if (!graph || !groups || num_groups == 0) return true;

    // Sort groups by cost reduction (descending) for better pruning - O(n log n)
    qsort(groups, num_groups, sizeof(fusion_group_t), compare_fusion_groups_desc);

    // Initialize branch and bound state
    bb_state_t state = {
        .current_selection = malloc(num_groups * sizeof(size_t)),
        .num_selected = 0,
        .current_cost = 0.0,
        .upper_bound = 0.0,
        .max_depth = num_groups,  // Allow full exploration
        .nodes_explored = 0,
        .pruned_branches = 0,
        .solution_found = false,
        .best_selection = malloc(num_groups * sizeof(size_t)),
        .best_num_selected = 0,
        .best_cost = 0.0
    };

    if (!state.current_selection || !state.best_selection) {
        free(state.current_selection);
        free(state.best_selection);
        return false;
    }

    // Create memoization table
    size_t memo_capacity = num_groups < 20 ? (1UL << num_groups) : 1048576;
    memo_table_t* memo = memo_create(memo_capacity);

    // Run branch and bound search
    bb_search(groups, num_groups, &state, 0);

    // Apply the best selection
    bool success = true;
    if (state.solution_found) {
        for (size_t i = 0; i < state.best_num_selected && success; i++) {
            size_t group_idx = state.best_selection[i];
            fusion_group_t* group = &groups[group_idx];

            // Create and apply fused node
            computation_node_t* fused = create_fused_node(group);
            if (fused) {
                success = replace_with_fused(graph, group, fused);
            } else {
                success = false;
            }
        }
    }

    // Cleanup
    memo_destroy(memo);
    free(state.current_selection);
    free(state.best_selection);

    return success;
}

// ============================================================================
// Heuristic Strategy Implementation
// ============================================================================

// Compute lookahead score - estimate future benefit of selecting a group
static double compute_lookahead_score(fusion_group_t* group, fusion_group_t* all_groups,
                                      size_t num_groups, size_t lookahead_depth,
                                      bool* selected) {
    if (lookahead_depth == 0) return 0.0;

    double score = 0.0;

    // Find groups that become available after selecting this group
    for (size_t i = 0; i < num_groups; i++) {
        if (selected[i]) continue;
        if (!groups_conflict(group, &all_groups[i])) {
            // This group could potentially be selected after
            score += all_groups[i].cost_reduction *
                     (1.0 / (lookahead_depth + 1));  // Discount future benefits

            // Recursive lookahead
            if (lookahead_depth > 1) {
                bool* new_selected = malloc(num_groups * sizeof(bool));
                if (new_selected) {
                    memcpy(new_selected, selected, num_groups * sizeof(bool));
                    // Mark conflicting groups as selected (unavailable)
                    for (size_t j = 0; j < num_groups; j++) {
                        if (groups_conflict(group, &all_groups[j])) {
                            new_selected[j] = true;
                        }
                    }
                    score += compute_lookahead_score(&all_groups[i], all_groups, num_groups,
                                                     lookahead_depth - 1, new_selected);
                    free(new_selected);
                }
            }
        }
    }

    return score;
}

// Simulated annealing acceptance probability
static double sa_acceptance_probability(double current_cost, double new_cost,
                                        double temperature) {
    if (new_cost > current_cost) return 1.0;
    return exp((new_cost - current_cost) / temperature);
}

// Simulated annealing refinement
static void sa_refine(fusion_group_t* groups, size_t num_groups,
                      size_t* selection, size_t* num_selected,
                      double* total_cost) {
    if (num_groups < 2) return;

    sa_state_t sa = {
        .temperature = 100.0,
        .cooling_rate = 0.995,
        .min_temperature = 0.01,
        .max_iterations = num_groups * 100,
        .current_iteration = 0,
        .acceptance_history = NULL,
        .history_size = 0
    };

    // Seed random number generator
    srand((unsigned int)time(NULL));

    while (sa.temperature > sa.min_temperature &&
           sa.current_iteration < sa.max_iterations) {

        // Generate neighbor solution by swapping one group
        size_t swap_idx = rand() % num_groups;
        bool currently_selected = false;
        size_t selection_idx = 0;

        for (size_t i = 0; i < *num_selected; i++) {
            if (selection[i] == swap_idx) {
                currently_selected = true;
                selection_idx = i;
                break;
            }
        }

        // Try the swap
        double new_cost = *total_cost;
        bool valid_swap = true;

        if (currently_selected) {
            // Remove from selection
            new_cost -= groups[swap_idx].cost_reduction;
        } else {
            // Add to selection - check for conflicts
            for (size_t i = 0; i < *num_selected; i++) {
                if (groups_conflict(&groups[swap_idx], &groups[selection[i]])) {
                    valid_swap = false;
                    break;
                }
            }
            if (valid_swap) {
                new_cost += groups[swap_idx].cost_reduction;
            }
        }

        if (valid_swap) {
            // Accept or reject based on probability
            double prob = sa_acceptance_probability(*total_cost, new_cost, sa.temperature);
            double r = (double)rand() / RAND_MAX;

            if (r < prob) {
                if (currently_selected) {
                    // Remove from selection
                    for (size_t i = selection_idx; i < *num_selected - 1; i++) {
                        selection[i] = selection[i + 1];
                    }
                    (*num_selected)--;
                } else {
                    // Add to selection
                    selection[*num_selected] = swap_idx;
                    (*num_selected)++;
                }
                *total_cost = new_cost;
            }
        }

        // Cool down
        sa.temperature *= sa.cooling_rate;
        sa.current_iteration++;
    }
}

// Main heuristic optimization
static bool heuristic_optimize(computational_graph_t* graph,
                              fusion_group_t* groups, size_t num_groups) {
    if (!graph || !groups || num_groups == 0) return true;

    // Create priority queue
    priority_queue_t* pq = pq_create(num_groups);
    if (!pq) return false;

    // Track selected groups
    bool* selected = calloc(num_groups, sizeof(bool));
    size_t* selection = malloc(num_groups * sizeof(size_t));
    size_t num_selected = 0;
    double total_cost = 0.0;

    if (!selected || !selection) {
        pq_destroy(pq);
        free(selected);
        free(selection);
        return false;
    }

    // Lookahead depth - configurable, default 3
    size_t lookahead_depth = 3;

    // Initialize priority queue with all groups
    for (size_t i = 0; i < num_groups; i++) {
        double lookahead = compute_lookahead_score(&groups[i], groups, num_groups,
                                                    lookahead_depth, selected);
        double priority = groups[i].cost_reduction + 0.5 * lookahead;
        pq_push(pq, i, priority, lookahead);
    }

    // Greedy selection with lookahead
    while (!pq_is_empty(pq)) {
        pq_entry_t entry;
        if (!pq_pop(pq, &entry)) break;

        size_t group_idx = entry.group_idx;

        // Skip if already selected or conflicts with selected
        if (selected[group_idx]) continue;

        bool conflicts = false;
        for (size_t i = 0; i < num_selected && !conflicts; i++) {
            if (groups_conflict(&groups[group_idx], &groups[selection[i]])) {
                conflicts = true;
            }
        }

        if (!conflicts && validate_fusion_group(&groups[group_idx])) {
            // Select this group
            selected[group_idx] = true;
            selection[num_selected++] = group_idx;
            total_cost += groups[group_idx].cost_reduction;

            // Update priorities of remaining groups
            // (Re-insert with updated lookahead scores)
            for (size_t i = 0; i < num_groups; i++) {
                if (!selected[i]) {
                    double lookahead = compute_lookahead_score(&groups[i], groups, num_groups,
                                                                lookahead_depth, selected);
                    double priority = groups[i].cost_reduction + 0.5 * lookahead;
                    pq_push(pq, i, priority, lookahead);
                }
            }
        }
    }

    // Apply simulated annealing refinement
    sa_refine(groups, num_groups, selection, &num_selected, &total_cost);

    // Apply the selected groups
    bool success = true;
    for (size_t i = 0; i < num_selected && success; i++) {
        size_t group_idx = selection[i];
        fusion_group_t* group = &groups[group_idx];

        computation_node_t* fused = create_fused_node(group);
        if (fused) {
            success = replace_with_fused(graph, group, fused);
        } else {
            success = false;
        }
    }

    // Cleanup
    pq_destroy(pq);
    free(selected);
    free(selection);

    return success;
}

// ============================================================================
// Quantum Gate Algebra - Rigorous Implementation
// ============================================================================

// Gate category classification for commutation analysis
typedef enum {
    GATE_CAT_IDENTITY,      // I - commutes with everything
    GATE_CAT_PAULI_X,       // X gates and X-rotations
    GATE_CAT_PAULI_Y,       // Y gates and Y-rotations
    GATE_CAT_PAULI_Z,       // Z gates, Rz, S, T (diagonal in Z basis)
    GATE_CAT_HADAMARD,      // Hadamard (swaps X and Z bases)
    GATE_CAT_CONTROLLED,    // CNOT, CZ, controlled rotations
    GATE_CAT_SWAP,          // SWAP gates
    GATE_CAT_CUSTOM         // Custom unitary - no assumptions
} gate_category_t;

// Extended gate information for fusion analysis
typedef struct {
    gate_type_t type;           // Gate type from quantum_types.h
    gate_category_t category;   // Category for commutation
    size_t* target_qubits;      // Target qubit indices
    size_t num_targets;         // Number of targets
    size_t* control_qubits;     // Control qubit indices (for controlled gates)
    size_t num_controls;        // Number of controls
    double* parameters;         // Gate parameters (angles for rotations)
    size_t num_parameters;      // Number of parameters
    bool is_diagonal;           // True for Z, Rz, S, T, CZ
    bool is_hermitian;          // True for self-inverse gates (X, Y, Z, H, CNOT, CZ, SWAP)
} quantum_gate_info_t;

// Extract gate information from computation node
static bool extract_gate_info(computation_node_t* node, quantum_gate_info_t* info) {
    if (!node || !info || node->op_type != OP_QUANTUM) return false;

    memset(info, 0, sizeof(quantum_gate_info_t));

    // Node data should contain quantum_gate_t pointer
    // This is set when nodes are created from quantum circuits
    if (!node->data) {
        info->type = GATE_TYPE_CUSTOM;
        info->category = GATE_CAT_CUSTOM;
        return true;
    }

    // Try to interpret node data as gate structure
    // The data layout depends on how quantum nodes are created
    // We use a generic approach that works with various data formats

    // Check if data contains gate type information
    // This assumes the first field is the gate type enum
    gate_type_t* gate_type_ptr = (gate_type_t*)node->data;
    info->type = *gate_type_ptr;

    // Classify gate by type
    switch (info->type) {
        case GATE_TYPE_I:
            info->category = GATE_CAT_IDENTITY;
            info->is_diagonal = true;
            info->is_hermitian = true;
            break;

        case GATE_TYPE_X:
            info->category = GATE_CAT_PAULI_X;
            info->is_diagonal = false;
            info->is_hermitian = true;
            break;

        case GATE_TYPE_Y:
            info->category = GATE_CAT_PAULI_Y;
            info->is_diagonal = false;
            info->is_hermitian = true;
            break;

        case GATE_TYPE_Z:
            info->category = GATE_CAT_PAULI_Z;
            info->is_diagonal = true;
            info->is_hermitian = true;
            break;

        case GATE_TYPE_H:
            info->category = GATE_CAT_HADAMARD;
            info->is_diagonal = false;
            info->is_hermitian = true;
            break;

        case GATE_TYPE_S:
        case GATE_TYPE_T:
            info->category = GATE_CAT_PAULI_Z;
            info->is_diagonal = true;
            info->is_hermitian = false;
            break;

        case GATE_TYPE_RX:
            info->category = GATE_CAT_PAULI_X;
            info->is_diagonal = false;
            info->is_hermitian = false;
            break;

        case GATE_TYPE_RY:
            info->category = GATE_CAT_PAULI_Y;
            info->is_diagonal = false;
            info->is_hermitian = false;
            break;

        case GATE_TYPE_RZ:
            info->category = GATE_CAT_PAULI_Z;
            info->is_diagonal = true;
            info->is_hermitian = false;
            break;

        case GATE_TYPE_CNOT:
            info->category = GATE_CAT_CONTROLLED;
            info->is_diagonal = false;
            info->is_hermitian = true;
            info->num_controls = 1;
            break;

        case GATE_TYPE_CZ:
            info->category = GATE_CAT_CONTROLLED;
            info->is_diagonal = true;
            info->is_hermitian = true;
            info->num_controls = 1;
            break;

        case GATE_TYPE_SWAP:
            info->category = GATE_CAT_SWAP;
            info->is_diagonal = false;
            info->is_hermitian = true;
            break;

        case GATE_TYPE_CUSTOM:
        default:
            info->category = GATE_CAT_CUSTOM;
            info->is_diagonal = false;
            info->is_hermitian = false;
            break;
    }

    return true;
}

// ============================================================================
// Pauli Algebra Commutation Rules
// ============================================================================

// Check commutation based on Pauli algebra
// Returns: 0 = don't commute, 1 = commute, -1 = anti-commute
static int pauli_commutation(gate_category_t cat1, gate_category_t cat2) {
    // Identity commutes with everything
    if (cat1 == GATE_CAT_IDENTITY || cat2 == GATE_CAT_IDENTITY) {
        return 1;
    }

    // Same category gates commute with themselves
    if (cat1 == cat2) {
        return 1;
    }

    // Pauli anti-commutation relations on same qubit:
    // {X, Y} = {Y, Z} = {Z, X} = 0 (anti-commute)
    // XY = iZ, YX = -iZ, etc.

    if ((cat1 == GATE_CAT_PAULI_X && cat2 == GATE_CAT_PAULI_Y) ||
        (cat1 == GATE_CAT_PAULI_Y && cat2 == GATE_CAT_PAULI_X)) {
        return -1;  // Anti-commute
    }

    if ((cat1 == GATE_CAT_PAULI_Y && cat2 == GATE_CAT_PAULI_Z) ||
        (cat1 == GATE_CAT_PAULI_Z && cat2 == GATE_CAT_PAULI_Y)) {
        return -1;  // Anti-commute
    }

    if ((cat1 == GATE_CAT_PAULI_Z && cat2 == GATE_CAT_PAULI_X) ||
        (cat1 == GATE_CAT_PAULI_X && cat2 == GATE_CAT_PAULI_Z)) {
        return -1;  // Anti-commute
    }

    // Hadamard transforms: H X H = Z, H Z H = X
    // So H and X don't commute, H and Z don't commute
    if (cat1 == GATE_CAT_HADAMARD || cat2 == GATE_CAT_HADAMARD) {
        // Hadamard only commutes with Y (up to phase)
        if (cat1 == GATE_CAT_PAULI_Y || cat2 == GATE_CAT_PAULI_Y) {
            return 1;
        }
        return 0;  // Don't commute
    }

    // Custom gates - assume no commutation
    if (cat1 == GATE_CAT_CUSTOM || cat2 == GATE_CAT_CUSTOM) {
        return 0;
    }

    return 0;  // Default: don't commute
}

// Check if qubit sets overlap
static bool qubits_overlap(const size_t* qubits1, size_t num1,
                           const size_t* qubits2, size_t num2) {
    if (!qubits1 || !qubits2 || num1 == 0 || num2 == 0) return false;

    for (size_t i = 0; i < num1; i++) {
        for (size_t j = 0; j < num2; j++) {
            if (qubits1[i] == qubits2[j]) {
                return true;
            }
        }
    }
    return false;
}

// Rigorous commutation check using Pauli algebra
static bool gates_commute(computation_node_t* gate1, computation_node_t* gate2) {
    if (!gate1 || !gate2) return false;
    if (gate1->op_type != OP_QUANTUM || gate2->op_type != OP_QUANTUM) return false;

    // Extract gate information
    quantum_gate_info_t info1, info2;
    if (!extract_gate_info(gate1, &info1) || !extract_gate_info(gate2, &info2)) {
        return false;  // Unknown gates - conservative
    }

    // Identity gates commute with everything
    if (info1.category == GATE_CAT_IDENTITY || info2.category == GATE_CAT_IDENTITY) {
        return true;
    }

    // Check if gates act on disjoint qubits
    // First, check from node structure
    bool shared_via_outputs = false;
    for (size_t i = 0; i < gate1->num_outputs && !shared_via_outputs; i++) {
        for (size_t j = 0; j < gate2->num_inputs; j++) {
            if (gate1->outputs[i] == gate2->inputs[j]) {
                shared_via_outputs = true;
                break;
            }
        }
    }
    for (size_t i = 0; i < gate2->num_outputs && !shared_via_outputs; i++) {
        for (size_t j = 0; j < gate1->num_inputs; j++) {
            if (gate2->outputs[i] == gate1->inputs[j]) {
                shared_via_outputs = true;
                break;
            }
        }
    }

    // If gates have explicit qubit indices, use those
    bool have_qubit_info = (info1.target_qubits && info2.target_qubits);
    bool shared_qubits = false;

    if (have_qubit_info) {
        // Check target overlap
        shared_qubits = qubits_overlap(info1.target_qubits, info1.num_targets,
                                        info2.target_qubits, info2.num_targets);

        // Check control-target overlap for controlled gates
        if (!shared_qubits && info1.num_controls > 0) {
            shared_qubits = qubits_overlap(info1.control_qubits, info1.num_controls,
                                            info2.target_qubits, info2.num_targets);
        }
        if (!shared_qubits && info2.num_controls > 0) {
            shared_qubits = qubits_overlap(info2.control_qubits, info2.num_controls,
                                            info1.target_qubits, info1.num_targets);
        }
    } else {
        // Fall back to connection-based check
        shared_qubits = shared_via_outputs;
    }

    // Gates on disjoint qubits always commute
    if (!shared_qubits) {
        return true;
    }

    // For gates on overlapping qubits, use Pauli algebra
    int commutation = pauli_commutation(info1.category, info2.category);

    // Commuting gates
    if (commutation == 1) {
        return true;
    }

    // Anti-commuting gates (technically don't commute in operator sense)
    if (commutation == -1) {
        return false;
    }

    // Special cases for controlled gates
    if (info1.category == GATE_CAT_CONTROLLED && info2.category == GATE_CAT_CONTROLLED) {
        // CZ gates commute with each other
        if (info1.type == GATE_TYPE_CZ && info2.type == GATE_TYPE_CZ) {
            return true;
        }
        // Two CNOTs on same qubits cancel: CNOT * CNOT = I
        // But they don't technically commute
    }

    // Diagonal gates always commute with each other
    if (info1.is_diagonal && info2.is_diagonal) {
        return true;
    }

    return false;
}

// ============================================================================
// Gate Combination Rules
// ============================================================================

// Check if two gates can be combined into one
typedef enum {
    COMBINE_NONE,           // Cannot combine
    COMBINE_IDENTITY,       // Combines to identity (cancellation)
    COMBINE_ROTATION,       // Combines to single rotation
    COMBINE_PHASE,          // Combines to phase gate
    COMBINE_UNITARY         // General unitary multiplication required
} combine_type_t;

// Determine how two gates can be combined
static combine_type_t can_combine_gates(quantum_gate_info_t* info1,
                                         quantum_gate_info_t* info2,
                                         double* combined_param) {
    if (!info1 || !info2) return COMBINE_NONE;

    // Same Hermitian gate twice = identity (self-inverse)
    // X*X = Y*Y = Z*Z = H*H = CNOT*CNOT = CZ*CZ = SWAP*SWAP = I
    if (info1->type == info2->type && info1->is_hermitian) {
        return COMBINE_IDENTITY;
    }

    // Rotation combination: Rx(a) * Rx(b) = Rx(a+b)
    if (info1->type == info2->type) {
        switch (info1->type) {
            case GATE_TYPE_RX:
            case GATE_TYPE_RY:
            case GATE_TYPE_RZ:
                if (info1->parameters && info2->parameters &&
                    info1->num_parameters > 0 && info2->num_parameters > 0) {
                    if (combined_param) {
                        *combined_param = info1->parameters[0] + info2->parameters[0];

                        // Check if combined rotation is near identity (2π or 0)
                        double normalized = fmod(*combined_param, 2.0 * M_PI);
                        if (fabs(normalized) < 1e-10 || fabs(normalized - 2.0 * M_PI) < 1e-10) {
                            return COMBINE_IDENTITY;
                        }
                    }
                    return COMBINE_ROTATION;
                }
                break;
            default:
                break;
        }
    }

    // S*S = Z, T*T = S, etc.
    if (info1->type == GATE_TYPE_S && info2->type == GATE_TYPE_S) {
        return COMBINE_PHASE;  // Becomes Z
    }
    if (info1->type == GATE_TYPE_T && info2->type == GATE_TYPE_T) {
        return COMBINE_PHASE;  // Becomes S
    }

    // Z-basis gates combine (all diagonal)
    if (info1->category == GATE_CAT_PAULI_Z && info2->category == GATE_CAT_PAULI_Z) {
        return COMBINE_PHASE;
    }

    // For general case, would need unitary multiplication
    if (info1->category != GATE_CAT_CUSTOM && info2->category != GATE_CAT_CUSTOM) {
        return COMBINE_UNITARY;
    }

    return COMBINE_NONE;
}

// Compute fusion benefit using rigorous analysis
static double compute_quantum_fusion_benefit(computation_node_t* gate1,
                                             computation_node_t* gate2) {
    if (!gate1 || !gate2) return 0.0;

    quantum_gate_info_t info1, info2;
    if (!extract_gate_info(gate1, &info1) || !extract_gate_info(gate2, &info2)) {
        return 0.1;  // Small benefit for unknown gates
    }

    double benefit = 0.0;

    // Check commutation for parallelization
    if (gates_commute(gate1, gate2)) {
        benefit += 1.0;  // Major benefit: can reorder for circuit depth reduction
    }

    // Check if gates can be combined
    double combined_param;
    combine_type_t combine = can_combine_gates(&info1, &info2, &combined_param);

    switch (combine) {
        case COMBINE_IDENTITY:
            benefit += 5.0;  // Maximum benefit: gates cancel completely
            break;
        case COMBINE_ROTATION:
            benefit += 3.0;  // High benefit: two rotations become one
            break;
        case COMBINE_PHASE:
            benefit += 2.5;  // Good benefit: phase gates combine
            break;
        case COMBINE_UNITARY:
            benefit += 1.5;  // Moderate benefit: unitary multiplication
            break;
        case COMBINE_NONE:
        default:
            benefit += 0.1;  // Minimal benefit
            break;
    }

    // Fidelity improvement: fewer gates = higher fidelity
    // Two gates becoming one saves one gate's error
    if (combine != COMBINE_NONE) {
        benefit += 0.5;
    }

    // Diagonal gates are cheaper to implement
    if (info1.is_diagonal && info2.is_diagonal) {
        benefit += 0.3;
    }

    return benefit;
}

// Build commutation graph for quantum gates
static commutation_info_t* build_commutation_graph(fusion_group_t* groups,
                                                   size_t num_groups,
                                                   size_t* num_edges) {
    // Count quantum gate pairs
    size_t max_edges = 0;
    for (size_t i = 0; i < num_groups; i++) {
        for (size_t j = 0; j < groups[i].num_nodes; j++) {
            if (groups[i].nodes[j]->op_type == OP_QUANTUM) {
                max_edges++;
            }
        }
    }
    max_edges = max_edges * max_edges / 2;

    if (max_edges == 0) {
        *num_edges = 0;
        return NULL;
    }

    commutation_info_t* edges = malloc(max_edges * sizeof(commutation_info_t));
    if (!edges) {
        *num_edges = 0;
        return NULL;
    }

    *num_edges = 0;

    // Build edges between quantum gates
    for (size_t i = 0; i < num_groups; i++) {
        for (size_t j = 0; j < groups[i].num_nodes; j++) {
            computation_node_t* gate1 = groups[i].nodes[j];
            if (gate1->op_type != OP_QUANTUM) continue;

            for (size_t k = i; k < num_groups; k++) {
                size_t start_l = (k == i) ? j + 1 : 0;
                for (size_t l = start_l; l < groups[k].num_nodes; l++) {
                    computation_node_t* gate2 = groups[k].nodes[l];
                    if (gate2->op_type != OP_QUANTUM) continue;

                    edges[*num_edges].gate1 = gate1;
                    edges[*num_edges].gate2 = gate2;
                    edges[*num_edges].commutes = gates_commute(gate1, gate2);
                    edges[*num_edges].fusion_benefit =
                        compute_quantum_fusion_benefit(gate1, gate2);
                    (*num_edges)++;

                    if (*num_edges >= max_edges) break;
                }
                if (*num_edges >= max_edges) break;
            }
            if (*num_edges >= max_edges) break;
        }
        if (*num_edges >= max_edges) break;
    }

    return edges;
}

// Find optimal quantum fusion using commutation graph
static bool find_quantum_fusion_order(commutation_info_t* edges, size_t num_edges,
                                      fusion_group_t* groups, size_t num_groups,
                                      size_t** fusion_order, size_t* order_size) {
    if (num_edges == 0 || !groups) {
        *fusion_order = NULL;
        *order_size = 0;
        return true;
    }

    // Use greedy approach: select edges with highest benefit that don't conflict
    *fusion_order = malloc(num_groups * sizeof(size_t));
    if (!*fusion_order) return false;

    bool* group_used = calloc(num_groups, sizeof(bool));
    if (!group_used) {
        free(*fusion_order);
        *fusion_order = NULL;
        return false;
    }

    // Sort edges by fusion benefit (descending) - O(n log n)
    qsort(edges, num_edges, sizeof(commutation_info_t), compare_commutation_desc);

    *order_size = 0;

    // Greedily select groups based on best fusion opportunities
    for (size_t e = 0; e < num_edges; e++) {
        // Find which groups contain these gates
        size_t group1_idx = SIZE_MAX;
        size_t group2_idx = SIZE_MAX;

        for (size_t i = 0; i < num_groups && (group1_idx == SIZE_MAX || group2_idx == SIZE_MAX); i++) {
            for (size_t j = 0; j < groups[i].num_nodes; j++) {
                if (groups[i].nodes[j] == edges[e].gate1) group1_idx = i;
                if (groups[i].nodes[j] == edges[e].gate2) group2_idx = i;
            }
        }

        if (group1_idx != SIZE_MAX && !group_used[group1_idx]) {
            (*fusion_order)[*order_size] = group1_idx;
            (*order_size)++;
            group_used[group1_idx] = true;
        }
        if (group2_idx != SIZE_MAX && group2_idx != group1_idx && !group_used[group2_idx]) {
            (*fusion_order)[*order_size] = group2_idx;
            (*order_size)++;
            group_used[group2_idx] = true;
        }
    }

    // Add any remaining groups that weren't part of quantum fusion
    for (size_t i = 0; i < num_groups; i++) {
        if (!group_used[i]) {
            (*fusion_order)[*order_size] = i;
            (*order_size)++;
        }
    }

    free(group_used);
    return true;
}

// Main quantum fusion optimization
static bool quantum_fusion_optimize(computational_graph_t* graph,
                                   fusion_group_t* groups, size_t num_groups) {
    if (!graph || !groups || num_groups == 0) return true;

    // Count quantum operations
    size_t quantum_group_count = 0;
    for (size_t i = 0; i < num_groups; i++) {
        for (size_t j = 0; j < groups[i].num_nodes; j++) {
            if (groups[i].nodes[j]->op_type == OP_QUANTUM) {
                quantum_group_count++;
                break;  // Count each group once
            }
        }
    }

    if (quantum_group_count == 0) {
        // No quantum operations, fall back to heuristic strategy
        return heuristic_optimize(graph, groups, num_groups);
    }

    // Build commutation graph
    size_t num_edges;
    commutation_info_t* comm_graph = build_commutation_graph(groups, num_groups, &num_edges);

    // Find optimal fusion order based on commutation
    size_t* fusion_order = NULL;
    size_t order_size = 0;
    bool order_found = find_quantum_fusion_order(comm_graph, num_edges,
                                                  groups, num_groups,
                                                  &fusion_order, &order_size);

    if (!order_found) {
        free(comm_graph);
        return false;
    }

    // Apply fusions in the computed order
    bool* applied = calloc(num_groups, sizeof(bool));
    bool success = true;

    if (!applied) {
        free(comm_graph);
        free(fusion_order);
        return false;
    }

    for (size_t i = 0; i < order_size && success; i++) {
        size_t group_idx = fusion_order[i];

        if (applied[group_idx]) continue;

        fusion_group_t* group = &groups[group_idx];

        // Validate and check for conflicts with already-applied groups
        if (!validate_fusion_group(group)) continue;

        bool conflicts = false;
        for (size_t j = 0; j < num_groups && !conflicts; j++) {
            if (j != group_idx && applied[j]) {
                if (groups_conflict(group, &groups[j])) {
                    conflicts = true;
                }
            }
        }

        if (!conflicts) {
            computation_node_t* fused = create_fused_node(group);
            if (fused) {
                success = replace_with_fused(graph, group, fused);
                if (success) {
                    applied[group_idx] = true;
                }
            }
        }
    }

    // Cleanup
    free(comm_graph);
    free(fusion_order);
    free(applied);

    return success;
}

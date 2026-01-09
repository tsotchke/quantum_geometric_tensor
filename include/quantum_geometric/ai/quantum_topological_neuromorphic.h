/**
 * @file quantum_topological_neuromorphic.h
 * @brief Topological Neuromorphic Computing API
 *
 * This header provides types and functions for quantum-classical hybrid computing
 * with topological protection and neuromorphic learning capabilities.
 *
 * Key features:
 * - Topological quantum memory with anyon-based encoding
 * - Neuromorphic computing units with persistent homology
 * - Quantum-classical interface with error correction
 * - Braiding operations for fault-tolerant computation
 */

#ifndef QUANTUM_TOPOLOGICAL_NEUROMORPHIC_H
#define QUANTUM_TOPOLOGICAL_NEUROMORPHIC_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/error_codes.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Constants and Enumerations
// =============================================================================

/**
 * @brief Anyon types for topological encoding
 */
typedef enum {
    ISING_ANYONS = 0,           ///< Ising anyons (simplest non-abelian)
    FIBONACCI_ANYONS = 1,       ///< Fibonacci anyons (universal)
    SU2_ANYONS = 2,            ///< SU(2)_k anyons
    MAJORANA_FERMIONS = 3       ///< Majorana zero modes
} anyon_type_t;

/**
 * @brief Topological protection levels
 */
typedef enum {
    TOPOLOGICAL_PROTECTION_NONE = 0,    ///< No protection
    TOPOLOGICAL_PROTECTION_LOW = 1,     ///< Basic protection
    TOPOLOGICAL_PROTECTION_MEDIUM = 2,  ///< Standard protection
    TOPOLOGICAL_PROTECTION_HIGH = 3     ///< Maximum protection
} protection_level_t;

/**
 * @brief Neuromorphic topology types
 */
typedef enum {
    FEEDFORWARD = 0,            ///< Standard feedforward
    RECURRENT = 1,              ///< Recurrent connections
    PERSISTENT_HOMOLOGY = 2,    ///< Topology-aware structure
    RESERVOIR = 3               ///< Reservoir computing
} neuromorphic_topology_t;

/**
 * @brief Protection schemes for quantum-classical interface
 */
typedef enum {
    NO_ERROR_CORRECTION = 0,        ///< No error correction
    CLASSICAL_ECC = 1,              ///< Classical error correction
    QUANTUM_ECC = 2,                ///< Quantum error correction
    TOPOLOGICAL_ERROR_CORRECTION = 3 ///< Topological protection
} protection_scheme_t;

/**
 * @brief Field types for persistent homology
 */
typedef enum {
    FIELD_Z2 = 0,       ///< Z/2Z coefficients
    FIELD_Z3 = 1,       ///< Z/3Z coefficients
    FIELD_Q = 2,        ///< Rational coefficients
    FIELD_R = 3         ///< Real coefficients
} field_type_t;

// =============================================================================
// Core Type Definitions
// =============================================================================

/**
 * @brief Parameters for topological state creation
 */
typedef struct topology_params {
    anyon_type_t anyon_type;            ///< Type of anyons to use
    protection_level_t protection_level; ///< Protection level
    size_t dimension;                    ///< Logical qubit dimension
    size_t num_anyons;                   ///< Number of anyons
    double decoherence_rate;             ///< Expected decoherence rate
    double error_threshold;              ///< Error correction threshold
} topology_params_t;

/**
 * @brief Anyonic quasiparticle state
 */
typedef struct anyon_state {
    size_t position;            ///< Position on lattice
    int charge;                 ///< Topological charge
    double phase;               ///< Accumulated phase
    ComplexDouble amplitude;    ///< State amplitude
    bool is_virtual;            ///< Virtual vs real anyon
} anyon_state_t;

/**
 * @brief Fusion tree for anyon states
 */
typedef struct fusion_tree {
    size_t* vertices;           ///< Vertex indices
    size_t num_vertices;        ///< Number of vertices
    int* fusion_channels;       ///< Fusion channel outcomes
    ComplexDouble* coefficients; ///< Fusion coefficients
    size_t depth;               ///< Tree depth
} fusion_tree_t;

/**
 * @brief Topological quantum memory
 */
typedef struct topological_memory {
    // Anyon configuration
    anyon_type_t anyon_type;            ///< Type of anyons
    protection_level_t protection_level; ///< Protection level
    anyon_state_t* anyons;              ///< Array of anyons
    size_t num_anyons;                  ///< Number of anyons
    size_t capacity;                    ///< Allocated capacity

    // Fusion space
    fusion_tree_t* fusion_tree;         ///< Current fusion tree
    ComplexDouble* fusion_space;        ///< Fusion space amplitudes
    size_t fusion_dim;                  ///< Fusion space dimension

    // Lattice structure
    size_t* lattice_dims;               ///< Lattice dimensions
    size_t num_dims;                    ///< Number of dimensions
    double* coupling_matrix;            ///< Coupling strengths

    // Protection metrics
    double topological_entropy;         ///< Topological entanglement entropy
    double gap;                         ///< Energy gap
    double fidelity;                    ///< State fidelity
    bool needs_correction;              ///< Error correction flag

    // Threading
    void* mutex;                        ///< Thread synchronization
} topological_memory_t;

/**
 * @brief Braiding operation
 */
typedef struct braid_operation {
    size_t anyon_i;             ///< First anyon index
    size_t anyon_j;             ///< Second anyon index
    bool clockwise;             ///< Braiding direction
    double phase_shift;         ///< Accumulated phase
    ComplexDouble r_matrix;     ///< R-matrix element
} braid_operation_t;

/**
 * @brief Braiding sequence for topological operations
 */
typedef struct braid_sequence {
    braid_operation_t* operations;  ///< Array of operations
    size_t num_operations;          ///< Number of operations
    size_t capacity;                ///< Allocated capacity
    ComplexDouble total_unitary;    ///< Cumulative unitary
    double total_phase;             ///< Total accumulated phase
    bool verified;                  ///< Verification status
} braid_sequence_t;

/**
 * @brief Neural connection
 */
typedef struct neural_connection {
    size_t from;                ///< Source neuron
    size_t to;                  ///< Target neuron
    double weight;              ///< Connection weight
    double gradient;            ///< Current gradient
    double momentum;            ///< Momentum term
} neural_connection_t;

/**
 * @brief Neural layer
 */
typedef struct neural_layer {
    size_t num_neurons;         ///< Number of neurons in layer
    double* activations;        ///< Neuron activations
    double* biases;             ///< Neuron biases
    double* gradients;          ///< Gradients
    double* pre_activations;    ///< Pre-activation values
} neural_layer_t;

/**
 * @brief Neural network structure
 */
typedef struct neural_network {
    // Layers
    neural_layer_t* layers;     ///< Array of layers
    size_t num_layers;          ///< Number of layers

    // Connections
    neural_connection_t** connections; ///< Connections between layers
    size_t* num_connections;    ///< Connections per layer pair

    // Data
    double* input_data;         ///< Input data buffer
    double* output_data;        ///< Output data buffer
    double* target_data;        ///< Target data buffer
    size_t data_dim;            ///< Data dimension
    size_t batch_size;          ///< Current batch size

    // Training state
    double learning_rate;       ///< Current learning rate
    double loss;                ///< Current loss value
    size_t epoch;               ///< Current epoch
    bool training;              ///< Training mode flag
} neural_network_t;

/**
 * @brief Parameters for neuromorphic unit
 */
typedef struct unit_params {
    size_t num_neurons;                 ///< Number of neurons
    neuromorphic_topology_t topology;   ///< Network topology
    double learning_rate;               ///< Learning rate
    size_t num_layers;                  ///< Number of layers
    size_t* layer_sizes;                ///< Neurons per layer
    double dropout_rate;                ///< Dropout rate
    double weight_decay;                ///< L2 regularization
} unit_params_t;

/**
 * @brief Neuromorphic computing unit
 */
typedef struct neuromorphic_unit {
    // Network
    neural_network_t* network;          ///< Neural network
    neuromorphic_topology_t topology;   ///< Topology type

    // Topology-aware features
    double* persistence_features;       ///< Persistent homology features
    size_t num_features;                ///< Number of features
    double* betti_numbers;              ///< Betti numbers
    size_t max_homology_dim;            ///< Maximum homology dimension

    // Learning state
    double learning_rate;               ///< Current learning rate
    double loss;                        ///< Current loss
    double best_loss;                   ///< Best achieved loss
    size_t patience_counter;            ///< Early stopping counter
    bool converged;                     ///< Convergence flag

    // Threading
    void* mutex;                        ///< Thread synchronization
} neuromorphic_unit_t;

/**
 * @brief Parameters for quantum-classical interface
 */
typedef struct interface_params {
    double coupling_strength;           ///< Q-C coupling strength
    double noise_threshold;             ///< Noise threshold
    protection_scheme_t protection_scheme; ///< Protection scheme
    size_t measurement_shots;           ///< Measurement repetitions
    double measurement_error_rate;      ///< Measurement error rate
} interface_params_t;

/**
 * @brief Quantum-classical interface
 */
typedef struct interface {
    // Configuration
    double coupling_strength;           ///< Coupling strength
    double noise_threshold;             ///< Noise threshold
    protection_scheme_t protection_scheme; ///< Protection scheme

    // Buffers
    ComplexDouble* quantum_buffer;      ///< Quantum state buffer
    double* classical_buffer;           ///< Classical data buffer
    size_t buffer_size;                 ///< Buffer size

    // Measurement
    size_t measurement_shots;           ///< Number of shots
    double* measurement_results;        ///< Measurement outcomes
    double measurement_error_rate;      ///< Error rate

    // Statistics
    size_t num_conversions;             ///< Conversion count
    double total_error;                 ///< Cumulative error
    double max_error;                   ///< Maximum observed error

    // Threading
    void* mutex;                        ///< Thread synchronization
} interface_t;

/**
 * @brief Parameters for persistent homology computation
 */
typedef struct persistence_params {
    size_t max_dimension;       ///< Maximum homology dimension
    double threshold;           ///< Filtration threshold
    field_type_t field;         ///< Coefficient field
    double persistence_threshold; ///< Minimum persistence
    bool use_reduced;           ///< Use reduced homology
} persistence_params_t;

/**
 * @brief Persistence pair (birth-death)
 */
typedef struct persistence_pair {
    double birth;               ///< Birth time
    double death;               ///< Death time
    size_t dimension;           ///< Homology dimension
    size_t generator_idx;       ///< Generator index
} persistence_pair_t;

/**
 * @brief Persistence diagram
 */
typedef struct persistence_diagram {
    persistence_pair_t* pairs;  ///< Persistence pairs
    size_t num_pairs;           ///< Number of pairs
    size_t capacity;            ///< Allocated capacity
    size_t max_dimension;       ///< Maximum dimension
    double total_persistence;   ///< Total persistence
} persistence_diagram_t;

/**
 * @brief Topological features extracted from data
 */
typedef struct topological_features {
    double* betti_numbers;      ///< Betti numbers
    size_t num_dimensions;      ///< Number of dimensions
    double* persistence_entropy; ///< Persistence entropy per dim
    double* barcode_statistics; ///< Barcode statistics
    double total_persistence;   ///< Total persistence
    double max_persistence;     ///< Maximum persistence
    double avg_persistence;     ///< Average persistence
    bool validated;             ///< Validation flag
} topological_features_t;

/**
 * @brief Quantum data representation
 */
typedef struct quantum_data {
    ComplexDouble* amplitudes;  ///< State amplitudes
    size_t dimension;           ///< State dimension
    double* probabilities;      ///< Measurement probabilities
    double entropy;             ///< Von Neumann entropy
    double purity;              ///< State purity
    bool is_mixed;              ///< Mixed vs pure state
} quantum_data_t;

/**
 * @brief Classical data representation
 */
typedef struct classical_data {
    double* values;             ///< Data values
    size_t dimension;           ///< Data dimension
    double* statistics;         ///< Statistical measures
    double mean;                ///< Mean value
    double variance;            ///< Variance
    bool normalized;            ///< Normalization flag
} classical_data_t;

/**
 * @brief Topological invariants
 */
typedef struct topological_invariants {
    double* betti_numbers;      ///< Betti numbers
    size_t num_dimensions;      ///< Number of dimensions
    double euler_characteristic; ///< Euler characteristic
    double topological_entropy; ///< Topological entropy
    ComplexDouble* wilson_loops; ///< Wilson loop expectation values
    size_t num_wilson_loops;    ///< Number of Wilson loops
    double chern_number;        ///< Chern number
    bool validated;             ///< Validation flag
} topological_invariants_t;

/**
 * @brief Network state snapshot
 */
typedef struct network_state {
    double** weights;           ///< Weight matrices
    double** biases;            ///< Bias vectors
    size_t num_layers;          ///< Number of layers
    size_t* layer_sizes;        ///< Layer sizes
    double loss;                ///< Current loss
    double* gradients;          ///< Gradient norms
    double learning_rate;       ///< Learning rate at snapshot
    size_t epoch;               ///< Epoch number
} network_state_t;

// =============================================================================
// Topological Memory Functions
// =============================================================================

/**
 * @brief Create topological quantum memory
 * @param params Configuration parameters
 * @return Allocated topological memory, NULL on failure
 */
topological_memory_t* create_topological_state(const topology_params_t* params);

/**
 * @brief Free topological memory
 * @param memory Memory to free
 */
void free_topological_memory(topological_memory_t* memory);

/**
 * @brief Create anyonic pairs in memory
 * @param memory Topological memory
 * @return QGT_SUCCESS on success
 */
qgt_error_t create_anyonic_pairs(topological_memory_t* memory);

/**
 * @brief Verify anyonic state consistency
 * @param memory Topological memory
 * @return true if states are valid
 */
bool verify_anyonic_states(const topological_memory_t* memory);

// =============================================================================
// Braiding Operations
// =============================================================================

/**
 * @brief Generate optimal braiding sequence
 * @return Allocated braiding sequence
 */
braid_sequence_t* generate_braid_sequence(void);

/**
 * @brief Perform braiding sequence on memory
 * @param memory Topological memory
 * @param sequence Braiding sequence
 * @return QGT_SUCCESS on success
 */
qgt_error_t perform_braiding_sequence(topological_memory_t* memory,
                                      const braid_sequence_t* sequence);

/**
 * @brief Verify braiding result
 * @param memory Topological memory
 * @param sequence Applied sequence
 * @return true if result is correct
 */
bool verify_braiding_result(const topological_memory_t* memory,
                           const braid_sequence_t* sequence);

/**
 * @brief Free braiding sequence
 * @param sequence Sequence to free
 */
void free_braid_sequence(braid_sequence_t* sequence);

// =============================================================================
// Neuromorphic Unit Functions
// =============================================================================

/**
 * @brief Initialize neuromorphic computing unit
 * @param params Unit parameters
 * @return Allocated unit, NULL on failure
 */
neuromorphic_unit_t* init_neuromorphic_unit(const unit_params_t* params);

/**
 * @brief Free neuromorphic unit
 * @param unit Unit to free
 */
void free_neuromorphic_unit(neuromorphic_unit_t* unit);

/**
 * @brief Update neuromorphic unit with data
 * @param unit Neuromorphic unit
 * @param data Classical training data
 * @return QGT_SUCCESS on success
 */
qgt_error_t update_neuromorphic_unit(neuromorphic_unit_t* unit,
                                     const classical_data_t* data);

/**
 * @brief Compute current loss
 * @param unit Neuromorphic unit
 * @return Loss value
 */
double compute_loss(const neuromorphic_unit_t* unit);

// =============================================================================
// Quantum-Classical Interface Functions
// =============================================================================

/**
 * @brief Create quantum-classical interface
 * @param params Interface parameters
 * @return Allocated interface, NULL on failure
 */
interface_t* create_quantum_classical_interface(const interface_params_t* params);

/**
 * @brief Free interface
 * @param iface Interface to free
 */
void free_interface(interface_t* iface);

/**
 * @brief Process quantum state for classical consumption
 * @param memory Topological memory
 * @return Quantum data representation
 */
quantum_data_t* process_quantum_state(const topological_memory_t* memory);

/**
 * @brief Convert quantum data to classical
 * @param iface Interface
 * @param q_data Quantum data
 * @return Classical data representation
 */
classical_data_t* quantum_to_classical(interface_t* iface,
                                       const quantum_data_t* q_data);

/**
 * @brief Verify data conversion accuracy
 * @param q_data Original quantum data
 * @param c_data Converted classical data
 * @return true if conversion is accurate
 */
bool verify_data_conversion(const quantum_data_t* q_data,
                           const classical_data_t* c_data);

/**
 * @brief Free quantum data
 * @param data Data to free
 */
void free_quantum_data(quantum_data_t* data);

/**
 * @brief Free classical data
 * @param data Data to free
 */
void free_classical_data(classical_data_t* data);

// =============================================================================
// Persistent Homology Functions
// =============================================================================

/**
 * @brief Analyze topology of data
 * @param data Input data
 * @param params Homology parameters
 * @return Persistence diagram
 */
persistence_diagram_t* analyze_data_topology(const double* data,
                                            const persistence_params_t* params);

/**
 * @brief Analyze network topology
 * @param network Neural network
 * @return Persistence diagram
 */
persistence_diagram_t* analyze_network_topology(const neural_network_t* network);

/**
 * @brief Extract topological features from persistence diagram
 * @param diagram Persistence diagram
 * @return Topological features
 */
topological_features_t* topo_neuro_extract_features(const persistence_diagram_t* diagram);

/**
 * @brief Verify topological features
 * @param features Features to verify
 * @return true if features are valid
 */
bool verify_topological_features(const topological_features_t* features);

/**
 * @brief Free persistence diagram
 * @param diagram Diagram to free
 */
void free_persistence_diagram(persistence_diagram_t* diagram);

/**
 * @brief Free topological features
 * @param features Features to free
 */
void free_topological_features(topological_features_t* features);

// =============================================================================
// Error Correction Functions
// =============================================================================

/**
 * @brief Introduce test error for verification
 * @param memory Topological memory
 */
void introduce_test_error(topological_memory_t* memory);

/**
 * @brief Check if error correction is needed
 * @param memory Topological memory
 * @return true if correction needed
 */
bool needs_correction(const topological_memory_t* memory);

/**
 * @brief Apply topological error correction
 * @param memory Topological memory
 * @return QGT_SUCCESS on success
 */
qgt_error_t apply_topological_error_correction(topological_memory_t* memory);

/**
 * @brief Verify state fidelity after correction
 * @param memory Topological memory
 * @return true if fidelity is acceptable
 */
bool verify_state_fidelity(const topological_memory_t* memory);

// =============================================================================
// Topological Protection Functions
// =============================================================================

/**
 * @brief Measure topological invariants
 * @param memory Topological memory
 * @return Topological invariants
 */
topological_invariants_t* measure_topological_invariants(const topological_memory_t* memory);

/**
 * @brief Apply test noise to memory
 * @param memory Topological memory
 */
void apply_test_noise(topological_memory_t* memory);

/**
 * @brief Perform test operations
 * @param memory Topological memory
 */
void perform_test_operations(topological_memory_t* memory);

/**
 * @brief Compare topological invariants
 * @param inv1 First invariants
 * @param inv2 Second invariants
 * @return true if invariants match
 */
bool compare_topological_invariants(const topological_invariants_t* inv1,
                                    const topological_invariants_t* inv2);

/**
 * @brief Free topological invariants
 * @param invariants Invariants to free
 */
void free_topological_invariants(topological_invariants_t* invariants);

/**
 * @brief Verify topological protection of memory
 * @param memory Topological memory
 * @return true if protected
 */
bool topo_neuro_verify_protection(const topological_memory_t* memory);

// =============================================================================
// Network State Functions
// =============================================================================

/**
 * @brief Capture network state snapshot
 * @param network Neural network
 * @return Network state snapshot
 */
network_state_t* capture_network_state(const neural_network_t* network);

/**
 * @brief Update weights based on topology
 * @param network Neural network
 * @param diagram Persistence diagram
 * @return QGT_SUCCESS on success
 */
qgt_error_t update_topological_weights(neural_network_t* network,
                                       const persistence_diagram_t* diagram);

/**
 * @brief Verify topological constraints
 * @param state Network state
 * @return true if constraints satisfied
 */
bool verify_topological_constraints(const network_state_t* state);

/**
 * @brief Verify weight update validity
 * @param before State before update
 * @param after State after update
 * @return true if update is valid
 */
bool verify_weight_update(const network_state_t* before,
                         const network_state_t* after);

/**
 * @brief Free network state
 * @param state State to free
 */
void free_network_state(network_state_t* state);

// =============================================================================
// Integration Functions
// =============================================================================

/**
 * @brief Verify learning convergence
 * @param unit Neuromorphic unit
 * @return true if converged
 */
bool verify_learning_convergence(const neuromorphic_unit_t* unit);

/**
 * @brief Verify system integration
 * @param memory Topological memory
 * @param unit Neuromorphic unit
 * @param iface Interface
 * @return true if integration is valid
 */
bool verify_system_integration(const topological_memory_t* memory,
                               const neuromorphic_unit_t* unit,
                               const interface_t* iface);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_TOPOLOGICAL_NEUROMORPHIC_H

#ifndef QUANTUM_TOPOLOGICAL_OPERATIONS_H
#define QUANTUM_TOPOLOGICAL_OPERATIONS_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/quantum_types.h"
#include "quantum_geometric/core/quantum_state_types.h"
#include "quantum_geometric/core/quantum_system.h"
#include "quantum_geometric/core/quantum_register.h"
#include "quantum_geometric/core/quantum_circuit_operations.h"
#include "quantum_geometric/core/quantum_phase_estimation.h"
#include "quantum_geometric/core/error_codes.h"
#include "quantum_geometric/core/numeric_utils.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <complex.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Constants for topological operations
// =============================================================================

#ifndef QG_QUANTUM_PRECISION
#define QG_QUANTUM_PRECISION 1e-10
#endif

#ifndef QG_SUCCESS_PROBABILITY
#define QG_SUCCESS_PROBABILITY 0.99
#endif

#ifndef QG_QUANTUM_CHUNK_SIZE
#define QG_QUANTUM_CHUNK_SIZE 256
#endif

#ifndef QG_CORRELATION_THRESHOLD
#define QG_CORRELATION_THRESHOLD 0.5
#endif

#ifndef QG_MAX_POWER_ITERATIONS
#define QG_MAX_POWER_ITERATIONS 100
#endif

#ifndef QGT_CACHE_LINE
#define QGT_CACHE_LINE 64
#endif

// Optimization flags for topological operations
#define TOPO_OPTIMIZE_AGGRESSIVE  (1 << 0)
#define TOPO_USE_TOPOLOGY         (1 << 1)

// Error codes for topological operations
#define TOPO_ERROR_OUT_OF_MEMORY  QGT_ERROR_MEMORY_ALLOCATION

// =============================================================================
// Mutex types for thread safety
// =============================================================================

typedef struct qgt_mutex_t {
    pthread_rwlock_t rwlock;
    pthread_mutex_t mutex;
} qgt_mutex_t;

// =============================================================================
// Simplicial complex types
// =============================================================================

typedef struct simplex_t {
    size_t dim;
    size_t* vertices;
    double weight;
    uint32_t flags;
} simplex_t;

typedef struct simplicial_complex_t {
    simplex_t** simplices;
    size_t num_simplices;
    size_t max_simplices;
    size_t max_dim;
    qgt_mutex_t* mutex;
} simplicial_complex_t;

typedef struct homology_data_t {
    double* betti_numbers;
    double** persistence_diagram;
    size_t num_features;
    size_t max_dim;
} homology_data_t;

typedef struct topology_data_t {
    simplicial_complex_t* simplicial_complex;
    homology_data_t* homology;
    double* singular_values;
    size_t num_singular_values;
} topology_data_t;

// =============================================================================
// Spin system and geometry types
// =============================================================================

typedef struct spin_system_t {
    complex double* spin_states;
    size_t num_spins;
    double coupling_strength;
} spin_system_t;

typedef struct geometry_data_t {
    double* metric_tensor;
    double* parallel_transport;
    double* christoffel_symbols;
    size_t dimension;
} geometry_data_t;

// =============================================================================
// Extended tensor for topological operations
// =============================================================================

typedef struct quantum_topological_tensor_t {
    size_t dimension;
    size_t rank;
    ComplexFloat* components;
    topology_data_t topology;
    spin_system_t spin_system;
    geometry_data_t geometry;
    qgt_mutex_t* mutex;
    size_t num_spins;
    HardwareType hardware;
} quantum_topological_tensor_t;

// =============================================================================
// Quantum workspace for parallel operations
// =============================================================================

typedef struct QuantumWorkspace {
    void* scratch_memory;
    size_t scratch_size;
    void* circuit_cache;
} QuantumWorkspace;

// Legacy alias
typedef quantum_circuit_t QuantumCircuit;

// =============================================================================
// Memory allocation (use platform_intrinsics.h for qgt_aligned_alloc/free)
// =============================================================================

#include "quantum_geometric/core/platform_intrinsics.h"

// =============================================================================
// Additional config types for topological operations
// =============================================================================

typedef struct quantum_amplitude_config_t {
    double precision;
    double success_probability;
    bool use_quantum_memory;
    int error_correction;
    int optimization_level;
} quantum_amplitude_config_t;

typedef struct quantum_topology_config_t {
    double precision;
    double success_probability;
    bool use_quantum_memory;
    int error_correction;
    int optimization_level;
    int topology_type;
} quantum_topology_config_t;

// Topology type constants
#define QUANTUM_TOPOLOGY_OPTIMAL 0

// Optimization level - use QUANTUM_OPT_AGGRESSIVE from quantum_geometric_constants.h enum
// Do NOT redefine QUANTUM_OPT_AGGRESSIVE as it conflicts with the enum value

// Region identifiers for entropy calculations
#define REGION_ABC 0
#define REGION_AB  1
#define REGION_BC  2
#define REGION_B   3

// =============================================================================
// Tree tensor network for topological calculations
// =============================================================================

// TreeTensorNetwork is defined in tree_tensor_network.h (included via quantum_geometric_core.h)
// It contains: num_sites, bond_dim, site_tensors, connectivity, entanglement_entropy
// as well as the full tree tensor network structure
#include "quantum_geometric/core/tree_tensor_network.h"

// Legacy type alias for backward compatibility
typedef qgt_error_t ErrorCode;
#define ERROR_INVALID_STATE QGT_ERROR_INVALID_STATE
#define ERROR_SUCCESS QGT_SUCCESS
#define NO_ERROR QGT_SUCCESS
#define ERROR_DETECTED QGT_ERROR_VALIDATION_FAILED

// Additional threshold constants
#ifndef QG_ERROR_THRESHOLD
#define QG_ERROR_THRESHOLD 0.01
#endif

#ifndef QG_COHERENCE_THRESHOLD
#define QG_COHERENCE_THRESHOLD 0.5
#endif

#ifndef QG_GAP_THRESHOLD
#define QG_GAP_THRESHOLD 0.1
#endif

#ifndef QG_GLOBAL_TEE_THRESHOLD
#define QG_GLOBAL_TEE_THRESHOLD 0.5
#endif

// Quantum optimization flags
#ifndef QUANTUM_OPTIMIZE_AGGRESSIVE
#define QUANTUM_OPTIMIZE_AGGRESSIVE (1 << 0)
#endif

#ifndef QUANTUM_USE_ESTIMATION
#define QUANTUM_USE_ESTIMATION (1 << 1)
#endif

#ifndef QUANTUM_USE_CIRCUITS
#define QUANTUM_USE_CIRCUITS (1 << 2)
#endif

#ifndef QUANTUM_ANNEAL_OPTIMAL
#define QUANTUM_ANNEAL_OPTIMAL (1 << 0)
#endif

#ifndef QUANTUM_ANNEAL_ADAPTIVE
#define QUANTUM_ANNEAL_ADAPTIVE (1 << 1)
#endif

#ifndef QUANTUM_SCHEDULE_ADAPTIVE
#define QUANTUM_SCHEDULE_ADAPTIVE 1
#endif

// =============================================================================
// Anyon and braiding types for topological error correction
// =============================================================================

typedef struct AnyonExcitation {
    size_t position;            // Position in the lattice
    int charge;                 // Topological charge
    double fusion_channel;      // Fusion channel probability
    bool is_paired;             // Whether paired with another anyon
} AnyonExcitation;

typedef struct AnyonGroup {
    AnyonExcitation* anyons;    // Array of anyons in this group
    size_t num_anyons;          // Number of anyons
    int total_charge;           // Total topological charge
} AnyonGroup;

typedef struct AnyonPair {
    AnyonExcitation* anyon1;    // First anyon
    AnyonExcitation* anyon2;    // Second anyon
    double distance;            // Distance between anyons
    double fusion_probability;  // Probability of fusion
} AnyonPair;

typedef struct BraidingPattern {
    size_t* path;               // Braiding path indices
    size_t path_length;         // Length of braiding path
    double phase;               // Accumulated phase
    int winding_number;         // Topological winding number
} BraidingPattern;

// =============================================================================
// Network partition types for distributed state protection
// =============================================================================

typedef struct NetworkPartition {
    size_t start_index;         // Start index in global state
    size_t end_index;           // End index in global state
    double* local_state;        // Local state data
    double boundary_entropy;    // Boundary entanglement entropy
    bool needs_sync;            // Whether synchronization needed
} NetworkPartition;

// =============================================================================
// Attention and monitoring configuration
// =============================================================================

typedef struct AttentionConfig {
    size_t num_heads;           // Number of attention heads
    size_t head_dim;            // Dimension per head
    double dropout_rate;        // Dropout rate
    bool use_causal_mask;       // Use causal attention mask
    double temperature;         // Softmax temperature
} AttentionConfig;

typedef struct MonitorConfig {
    double check_interval;      // Interval between checks (seconds)
    double order_threshold;     // Threshold for topological order
    double tee_threshold;       // Threshold for TEE
    double braiding_threshold;  // Threshold for braiding verification
    bool auto_correct;          // Automatically apply corrections
} MonitorConfig;

typedef struct TopologicalMonitor {
    MonitorConfig config;       // Configuration
    double current_order;       // Current topological order measure
    double current_tee;         // Current TEE
    double current_braiding;    // Current braiding verification
    bool active;                // Whether monitor is active
    bool needs_correction;      // Whether correction is needed
    uint64_t last_check_time;   // Timestamp of last check
} TopologicalMonitor;

// =============================================================================
// Entanglement spectrum types
// =============================================================================

typedef struct EntanglementSpectrum {
    double* eigenvalues;        // Entanglement spectrum eigenvalues
    size_t num_eigenvalues;     // Number of eigenvalues
    double gap;                 // Spectral gap
    double entropy;             // Entanglement entropy
} EntanglementSpectrum;

// =============================================================================
// Quantum annealing types
// =============================================================================

typedef struct quantum_annealing_t {
    size_t num_qubits;          // Number of qubits
    double* coupling_matrix;    // Coupling matrix
    double* local_fields;       // Local field terms
    double temperature;         // Current temperature
    double schedule_param;      // Annealing schedule parameter
    int flags;                  // Configuration flags
} quantum_annealing_t;

typedef struct quantum_annealing_config_t {
    double precision;           // Precision requirement
    int schedule_type;          // Type of annealing schedule
    bool use_quantum_memory;    // Use quantum memory
    int error_correction;       // Error correction level
    int optimization_level;     // Optimization level
} quantum_annealing_config_t;

typedef struct quantum_circuit_config_t {
    double precision;           // Precision requirement
    double success_probability; // Target success probability
    bool use_quantum_memory;    // Use quantum memory
    int error_correction;       // Error correction level
    int optimization_level;     // Optimization level
} quantum_circuit_config_t;

// =============================================================================
// Quantum state type for attention operations
// =============================================================================

// QuantumState is defined in quantum_state_types.h (included via quantum_geometric_types.h)
// We use that definition for compatibility

// =============================================================================
// Workspace management
// =============================================================================

QuantumWorkspace* init_quantum_workspace(size_t chunk_size);
void cleanup_quantum_workspace(QuantumWorkspace* qws);

QuantumCircuit* init_quantum_simplicial_circuit(size_t num_spins);
QuantumCircuit* init_quantum_correlation_circuit(size_t dimension);
void cleanup_topological_circuit(QuantumCircuit* circuit);

// Circuit creation wrappers for topological operations
quantum_circuit_t* quantum_create_boundary_circuit(size_t num_qubits, int flags);
quantum_circuit_t* quantum_create_smith_circuit(size_t num_qubits, int flags);
quantum_circuit_t* quantum_create_topology_circuit(size_t num_qubits, int flags);
quantum_circuit_t* quantum_create_entropy_circuit(size_t num_qubits, int flags);
quantum_circuit_t* quantum_create_error_circuit(size_t num_qubits, int flags);

// Region-based quantum register operations for entropy calculations
quantum_register_t* quantum_register_create_region(TreeTensorNetwork* network,
                                                   int region_id,
                                                   quantum_system_t* system);

// Quantum entropy estimation functions
double quantum_estimate_entropy(quantum_register_t* reg,
                               quantum_system_t* system,
                               quantum_circuit_t* circuit,
                               const quantum_phase_config_t* config,
                               QuantumWorkspace* qws);

double quantum_combine_entropies(double* entropies,
                                quantum_system_t* system,
                                quantum_circuit_t* circuit,
                                const quantum_phase_config_t* config);

// =============================================================================
// Quantum topological operations
// =============================================================================

void quantum_build_simplices(quantum_register_t* reg, simplicial_complex_t* sc,
                             size_t chunk, size_t chunk_size,
                             quantum_system_t* system, quantum_circuit_t* circuit,
                             void* config, QuantumWorkspace* qws);

void quantum_detect_correlations(complex double* spin_states, QuantumCircuit* circuit,
                                 QuantumWorkspace* qws, size_t chunk_size, size_t dim);

// =============================================================================
// Public topological operation functions
// =============================================================================

qgt_error_t build_simplicial_complex(quantum_topological_tensor_t* tensor, uint32_t flags);
qgt_error_t calculate_persistent_homology(quantum_topological_tensor_t* tensor, uint32_t flags);
qgt_error_t analyze_singular_spectrum(quantum_topological_tensor_t* tensor, uint32_t flags);
qgt_error_t update_learning_coefficients(quantum_topological_tensor_t* tensor, uint32_t flags);

// =============================================================================
// Topological protection functions
// =============================================================================

// Topological entropy calculation
double calculate_topological_entropy(TreeTensorNetwork* network);

// Topological error detection and correction
ErrorCode detect_topological_errors(quantum_topological_tensor_t* qgt);
void correct_topological_errors(quantum_topological_tensor_t* qgt);

// Long-range coherence maintenance
void maintain_long_range_coherence(TreeTensorNetwork* network);

// Distributed state protection
void protect_distributed_state(NetworkPartition* partitions, size_t num_parts);

// Topological attention
void apply_topological_attention(quantum_topological_tensor_t* qgt,
                                const AttentionConfig* config);

// Topological monitoring
void monitor_topological_order(quantum_topological_tensor_t* qgt,
                              const MonitorConfig* config);

// =============================================================================
// Anyon operations for topological error correction
// =============================================================================

QuantumCircuit* init_quantum_anyon_circuit(size_t dimension);
AnyonExcitation* quantum_identify_anyons(quantum_topological_tensor_t* qgt, QuantumCircuit* qc);
size_t quantum_count_anyon_types(AnyonExcitation* anyons, QuantumCircuit* qc);
AnyonGroup* quantum_group_anyons(AnyonExcitation* anyons, size_t num_types, QuantumCircuit* qc);
AnyonPair* quantum_find_anyon_pairs(AnyonGroup* group, size_t chunk_size, QuantumCircuit* qc, QuantumWorkspace* qws);
BraidingPattern* quantum_optimize_braiding(AnyonPair* pairs, QuantumCircuit* qc, QuantumWorkspace* qws);
void quantum_apply_braiding(quantum_topological_tensor_t* qgt, BraidingPattern* pattern, QuantumCircuit* qc, QuantumWorkspace* qws);
bool verify_topological_order(quantum_topological_tensor_t* qgt);
void update_ground_state(quantum_topological_tensor_t* qgt);
void free_braiding_pattern(BraidingPattern* pattern);
void free_anyon_pairs(AnyonPair* pairs);
void free_anyon_groups(AnyonGroup* groups, size_t num_types);
void free_anyon_excitations(AnyonExcitation* anyons);

// =============================================================================
// Quantum annealing and coherence operations
// =============================================================================

quantum_annealing_t* quantum_annealing_create(int flags);
void quantum_annealing_destroy(quantum_annealing_t* annealer);
quantum_circuit_t* quantum_create_coherence_circuit(size_t num_qubits, int flags);
double quantum_estimate_correlation(TreeTensorNetwork* network, quantum_annealing_t* annealer,
                                   quantum_circuit_t* circuit, quantum_annealing_config_t* config,
                                   QuantumWorkspace* qws);
size_t quantum_optimize_bond_dimension(TreeTensorNetwork* network, double xi, quantum_annealing_t* annealer,
                                      quantum_circuit_t* circuit, quantum_annealing_config_t* config,
                                      QuantumWorkspace* qws);
void quantum_increase_bond_dimension(TreeTensorNetwork* network, size_t new_bond_dim,
                                    quantum_annealing_t* annealer, quantum_circuit_t* circuit,
                                    quantum_annealing_config_t* config, QuantumWorkspace* qws);
EntanglementSpectrum* quantum_calculate_spectrum(TreeTensorNetwork* network, quantum_annealing_t* annealer,
                                                quantum_circuit_t* circuit, quantum_annealing_config_t* config,
                                                QuantumWorkspace* qws);
double quantum_calculate_gap(EntanglementSpectrum* spectrum, quantum_annealing_t* annealer,
                            quantum_circuit_t* circuit, quantum_annealing_config_t* config,
                            QuantumWorkspace* qws);
void quantum_apply_spectral_flow(TreeTensorNetwork* network, EntanglementSpectrum* spectrum,
                                quantum_annealing_t* annealer, quantum_circuit_t* circuit,
                                quantum_annealing_config_t* config, QuantumWorkspace* qws);
void quantum_update_protection(TreeTensorNetwork* network, EntanglementSpectrum* spectrum,
                              quantum_annealing_t* annealer, quantum_circuit_t* circuit,
                              quantum_annealing_config_t* config, QuantumWorkspace* qws);
void quantum_free_spectrum(EntanglementSpectrum* spectrum);

// =============================================================================
// Distributed protection operations
// =============================================================================

quantum_circuit_t* quantum_create_protection_circuit(size_t num_qubits, int flags);
double quantum_estimate_partition_tee(NetworkPartition* partitions, size_t chunk_size,
                                     quantum_system_t* system, quantum_circuit_t* circuit,
                                     quantum_circuit_config_t* config, QuantumWorkspace* qws);
void quantum_protect_partitions(NetworkPartition* partitions, size_t chunk_size,
                               quantum_system_t* system, quantum_circuit_t* circuit,
                               quantum_circuit_config_t* config, QuantumWorkspace* qws);
void quantum_synchronize_boundary(NetworkPartition* part1, NetworkPartition* part2,
                                 quantum_system_t* system, quantum_circuit_t* circuit,
                                 quantum_circuit_config_t* config, QuantumWorkspace* qws);
void quantum_verify_protection(NetworkPartition* partitions, size_t num_parts,
                              quantum_system_t* system, quantum_circuit_t* circuit,
                              quantum_circuit_config_t* config);

// =============================================================================
// Attention and monitoring operations
// =============================================================================

QuantumCircuit* init_quantum_attention_circuit(size_t dimension, size_t num_heads);
QuantumState* quantum_calculate_attention(quantum_topological_tensor_t* qgt, const AttentionConfig* config, QuantumCircuit* qc);
void quantum_apply_attention_chunk(quantum_topological_tensor_t* qgt, QuantumState* attention_state,
                                  size_t chunk, size_t chunk_size, QuantumCircuit* qc, QuantumWorkspace* qws);
void quantum_verify_tee(quantum_topological_tensor_t* qgt, size_t chunk, size_t chunk_size,
                       QuantumCircuit* qc, QuantumWorkspace* qws);
void update_global_state(quantum_topological_tensor_t* qgt);

QuantumCircuit* init_quantum_monitor_circuit(size_t dimension);
TopologicalMonitor* quantum_create_monitor(const MonitorConfig* config, QuantumCircuit* qc);
void free_topological_monitor(TopologicalMonitor* monitor);
double quantum_estimate_order(quantum_topological_tensor_t* qgt, QuantumCircuit* qc, QuantumWorkspace* qws);
double quantum_estimate_tee(quantum_topological_tensor_t* qgt, QuantumCircuit* qc, QuantumWorkspace* qws);
double quantum_verify_braiding_order(quantum_topological_tensor_t* qgt, QuantumCircuit* qc, QuantumWorkspace* qws);
void quantum_update_metrics(TopologicalMonitor* monitor, double order, double tee, double braiding,
                           QuantumCircuit* qc, QuantumWorkspace* qws);
void quantum_apply_correction(quantum_topological_tensor_t* qgt, TopologicalMonitor* monitor,
                             QuantumCircuit* qc, QuantumWorkspace* qws);
void quantum_verify_state(quantum_topological_tensor_t* qgt, QuantumCircuit* qc, QuantumWorkspace* qws);
void quantum_wait_interval(TopologicalMonitor* monitor, QuantumCircuit* qc);

// =============================================================================
// Error estimation operations
// =============================================================================

quantum_register_t* topo_register_create_state(quantum_topological_tensor_t* qgt, quantum_system_t* system);
double quantum_estimate_errors(quantum_register_t* reg, size_t chunk, size_t chunk_size,
                              quantum_system_t* system, quantum_circuit_t* circuit,
                              quantum_amplitude_config_t* config, QuantumWorkspace* qws);
bool quantum_check_threshold(double error_amplitude, double threshold, quantum_system_t* system,
                            quantum_circuit_t* circuit, quantum_amplitude_config_t* config,
                            QuantumWorkspace* qws);

// quantum_circuit_destroy for quantum_circuit_t is declared in quantum_circuit_operations.h

// Min helper macro
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_TOPOLOGICAL_OPERATIONS_H



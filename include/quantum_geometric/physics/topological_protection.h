/**
 * @file topological_protection.h
 * @brief Topological quantum error protection and correction
 *
 * Implements topological error correction codes including surface codes,
 * toric codes, color codes, and related syndrome decoding algorithms.
 */

#ifndef TOPOLOGICAL_PROTECTION_H
#define TOPOLOGICAL_PROTECTION_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct quantum_state;
struct quantum_circuit;

// =============================================================================
// Topological Code Types
// =============================================================================

/**
 * Types of topological quantum error correction codes
 */
typedef enum {
    TOPO_CODE_SURFACE,           // Surface code (planar)
    TOPO_CODE_TORIC,             // Toric code (periodic boundaries)
    TOPO_CODE_COLOR,             // Color code (2D triangular lattice)
    TOPO_CODE_STEANE,            // Steane [[7,1,3]] code
    TOPO_CODE_BACON_SHOR,        // Bacon-Shor subsystem code
    TOPO_CODE_FLOQUET,           // Floquet (dynamical) codes
    TOPO_CODE_HYPERBOLIC,        // Hyperbolic surface codes
    TOPO_CODE_CUSTOM             // User-defined topological code
} TopologicalCodeType;

/**
 * Boundary types for topological codes
 */
typedef enum {
    BOUNDARY_ROUGH,              // Rough (Z-type) boundary
    BOUNDARY_SMOOTH,             // Smooth (X-type) boundary
    BOUNDARY_PERIODIC,           // Periodic (toric) boundary
    BOUNDARY_TWIST,              // Twist defect boundary
    BOUNDARY_MIXED               // Mixed boundary conditions
} BoundaryType;

/**
 * Error types that can be corrected
 */
typedef enum {
    ERROR_TYPE_BIT_FLIP,         // X errors (bit-flip)
    ERROR_TYPE_PHASE_FLIP,       // Z errors (phase-flip)
    ERROR_TYPE_BOTH,             // Both X and Z errors
    ERROR_TYPE_ERASURE,          // Erasure errors (known location)
    ERROR_TYPE_LEAKAGE,          // Leakage to non-computational states
    ERROR_TYPE_CORRELATED        // Correlated multi-qubit errors
} TopologicalErrorType;

/**
 * Decoder types for syndrome processing
 */
typedef enum {
    DECODER_MWPM,                // Minimum Weight Perfect Matching
    DECODER_UNION_FIND,          // Union-Find decoder
    DECODER_NEURAL_NETWORK,      // Neural network decoder
    DECODER_TENSOR_NETWORK,      // Tensor network decoder
    DECODER_BELIEF_PROPAGATION,  // Belief propagation
    DECODER_CELLULAR_AUTOMATON,  // Cellular automaton decoder
    DECODER_RENORMALIZATION,     // Renormalization group decoder
    DECODER_CUSTOM               // User-defined decoder
} DecoderType;

// =============================================================================
// Lattice and Stabilizer Structures
// =============================================================================

/**
 * Position on the topological lattice
 */
typedef struct {
    int x;                       // X coordinate
    int y;                       // Y coordinate
    int layer;                   // Layer (for 3D codes)
} LatticePosition;

/**
 * Qubit role in the topological code
 */
typedef enum {
    QUBIT_ROLE_DATA,             // Data qubit
    QUBIT_ROLE_ANCILLA_X,        // X-stabilizer ancilla
    QUBIT_ROLE_ANCILLA_Z,        // Z-stabilizer ancilla
    QUBIT_ROLE_FLAG,             // Flag qubit for fault tolerance
    QUBIT_ROLE_LOGICAL           // Logical qubit representative
} QubitRole;

/**
 * Topological qubit on the lattice
 */
typedef struct {
    size_t qubit_id;             // Global qubit identifier
    LatticePosition position;    // Position on lattice
    QubitRole role;              // Role in the code
    size_t* connected_qubits;    // Neighboring qubits
    size_t num_connections;
    bool is_boundary;            // True if on boundary
    BoundaryType boundary_type;  // Type if on boundary
} TopologicalQubit;

/**
 * Stabilizer measurement
 */
typedef struct {
    size_t stabilizer_id;
    LatticePosition position;
    TopologicalErrorType type;   // X or Z stabilizer
    size_t* data_qubits;         // Data qubits in support
    size_t num_data_qubits;
    size_t ancilla_qubit;        // Ancilla for measurement
    int8_t measurement_result;   // +1 or -1
    uint64_t measurement_time;
} StabilizerMeasurement;

/**
 * Syndrome (pattern of stabilizer violations)
 */
typedef struct {
    int8_t* measurements;        // Array of stabilizer measurements
    size_t num_measurements;
    TopologicalErrorType error_type;
    LatticePosition* defect_positions;  // Positions of -1 measurements
    size_t num_defects;
    uint64_t round;              // Syndrome round number
} Syndrome;

/**
 * Error chain on the lattice
 */
typedef struct {
    LatticePosition* positions;  // Positions of errors
    size_t length;
    TopologicalErrorType type;
    double weight;               // Total weight (log-likelihood)
    bool is_logical;             // True if crosses logical operator
} ErrorChain;

/**
 * Logical operator definition
 */
typedef struct {
    char* name;                  // e.g., "X_L", "Z_L"
    TopologicalErrorType type;
    size_t* qubit_support;       // Qubits in the operator
    size_t support_size;
    LatticePosition start;       // Start position (for string operators)
    LatticePosition end;         // End position
} LogicalOperator;

// =============================================================================
// Code Configuration
// =============================================================================

/**
 * Configuration for topological code construction
 */
typedef struct {
    TopologicalCodeType code_type;
    size_t distance;             // Code distance d
    size_t width;                // Lattice width (may differ from distance)
    size_t height;               // Lattice height
    BoundaryType boundary_x;     // X-direction boundary
    BoundaryType boundary_y;     // Y-direction boundary
    bool use_flag_qubits;        // Use flag qubits for fault tolerance
    bool enable_leakage_reduction; // Enable leakage reduction
    double physical_error_rate;  // Target physical error rate
} TopologicalCodeConfig;

/**
 * Decoder configuration
 */
typedef struct {
    DecoderType decoder_type;
    double matching_weight_x;    // Weight for X errors
    double matching_weight_z;    // Weight for Z errors
    double matching_weight_y;    // Weight for Y errors
    double erasure_weight;       // Weight for erasure errors
    size_t max_iterations;       // For iterative decoders
    double convergence_threshold;
    bool use_correlated_matching; // Use correlated MWPM
    bool use_soft_information;   // Use soft syndrome information
    char* neural_model_path;     // Path to NN model (if applicable)
    void* custom_config;         // Decoder-specific configuration
} DecoderConfig;

// =============================================================================
// Main Context Structures
// =============================================================================

/**
 * Topological code instance
 */
typedef struct {
    TopologicalCodeConfig config;
    TopologicalQubit* qubits;    // All qubits in the code
    size_t num_qubits;
    size_t num_data_qubits;
    size_t num_ancilla_qubits;
    StabilizerMeasurement* x_stabilizers;
    size_t num_x_stabilizers;
    StabilizerMeasurement* z_stabilizers;
    size_t num_z_stabilizers;
    LogicalOperator* logical_x_ops;
    LogicalOperator* logical_z_ops;
    size_t num_logical_qubits;
    struct quantum_circuit* syndrome_circuit;
    struct quantum_circuit* initialization_circuit;
} TopologicalCode;

/**
 * Decoder context
 */
typedef struct {
    DecoderConfig config;
    TopologicalCode* code;       // Associated code
    void* decoder_state;         // Decoder-specific state
    double* edge_weights;        // Weights for matching graph
    size_t num_edges;
    Syndrome* syndrome_history;  // History of syndromes
    size_t history_size;
    size_t history_capacity;
    pthread_mutex_t lock;
} DecoderContext;

/**
 * Protection context (main interface)
 */
typedef struct {
    TopologicalCode* code;
    DecoderContext* decoder;
    struct quantum_state* logical_state;
    ErrorChain* detected_errors;
    size_t num_errors;
    size_t total_rounds;
    size_t successful_corrections;
    size_t failed_corrections;
    double logical_error_rate;
    uint64_t start_time;
} TopologicalProtectionContext;

// =============================================================================
// Protection Metrics
// =============================================================================

/**
 * Performance metrics for topological protection
 */
typedef struct {
    double physical_error_rate;
    double logical_error_rate;
    double threshold_estimate;
    size_t total_syndrome_rounds;
    size_t total_corrections;
    size_t logical_x_errors;
    size_t logical_z_errors;
    double average_decoding_time_us;
    double max_decoding_time_us;
    double syndrome_extraction_fidelity;
} TopologicalMetrics;

// =============================================================================
// Code Construction and Management
// =============================================================================

/**
 * Create a topological code
 */
int topological_code_create(TopologicalCode** code, TopologicalCodeConfig* config);

/**
 * Destroy a topological code
 */
void topological_code_destroy(TopologicalCode* code);

/**
 * Get code parameters
 */
int topological_code_get_params(TopologicalCode* code,
                                 size_t* n_out,      // Physical qubits
                                 size_t* k_out,      // Logical qubits
                                 size_t* d_out);     // Distance

/**
 * Build syndrome extraction circuit
 */
int topological_code_build_syndrome_circuit(TopologicalCode* code,
                                             struct quantum_circuit** circuit_out);

/**
 * Build logical state initialization circuit
 */
int topological_code_build_init_circuit(TopologicalCode* code,
                                         bool* initial_state,
                                         size_t num_logical,
                                         struct quantum_circuit** circuit_out);

/**
 * Build logical gate circuit
 */
int topological_code_build_logical_gate(TopologicalCode* code,
                                         const char* gate_name,
                                         size_t* logical_qubits,
                                         size_t num_qubits,
                                         struct quantum_circuit** circuit_out);

/**
 * Get physical qubit mapping for logical qubit
 */
int topological_code_get_qubit_mapping(TopologicalCode* code,
                                        size_t logical_qubit,
                                        size_t** physical_qubits,
                                        size_t* num_physical);

// =============================================================================
// Decoder Operations
// =============================================================================

/**
 * Create a decoder
 */
int topological_decoder_create(DecoderContext** decoder,
                                TopologicalCode* code,
                                DecoderConfig* config);

/**
 * Destroy a decoder
 */
void topological_decoder_destroy(DecoderContext* decoder);

/**
 * Decode a syndrome
 */
int topological_decoder_decode(DecoderContext* decoder,
                                Syndrome* syndrome,
                                ErrorChain** correction_out);

/**
 * Decode multiple syndrome rounds (space-time decoding)
 */
int topological_decoder_decode_spacetime(DecoderContext* decoder,
                                          Syndrome** syndromes,
                                          size_t num_rounds,
                                          ErrorChain** correction_out);

/**
 * Update decoder weights from calibration data
 */
int topological_decoder_update_weights(DecoderContext* decoder,
                                         double* error_rates,
                                         size_t num_qubits);

/**
 * Get decoder statistics
 */
int topological_decoder_get_stats(DecoderContext* decoder,
                                   double* avg_time_us,
                                   size_t* total_decodes,
                                   size_t* successful_decodes);

// =============================================================================
// Protection Context Operations
// =============================================================================

/**
 * Create topological protection context
 */
int topological_protection_create(TopologicalProtectionContext** ctx,
                                   TopologicalCodeConfig* code_config,
                                   DecoderConfig* decoder_config);

/**
 * Destroy protection context
 */
void topological_protection_destroy(TopologicalProtectionContext* ctx);

/**
 * Initialize logical state under protection
 */
int topological_protection_init_state(TopologicalProtectionContext* ctx,
                                       bool* initial_state,
                                       size_t num_logical);

/**
 * Run one round of error correction
 */
int topological_protection_round(TopologicalProtectionContext* ctx,
                                  struct quantum_state* physical_state,
                                  bool* error_detected,
                                  bool* correction_applied);

/**
 * Run multiple rounds of error correction
 */
int topological_protection_run(TopologicalProtectionContext* ctx,
                                struct quantum_state* physical_state,
                                size_t num_rounds,
                                TopologicalMetrics* metrics_out);

/**
 * Apply logical gate under protection
 */
int topological_protection_apply_gate(TopologicalProtectionContext* ctx,
                                       const char* gate_name,
                                       size_t* logical_qubits,
                                       size_t num_qubits);

/**
 * Extract logical measurement
 */
int topological_protection_measure(TopologicalProtectionContext* ctx,
                                    size_t logical_qubit,
                                    bool* result_out);

/**
 * Get protection metrics
 */
int topological_protection_get_metrics(TopologicalProtectionContext* ctx,
                                        TopologicalMetrics* metrics_out);

// =============================================================================
// Syndrome Processing
// =============================================================================

/**
 * Create syndrome from measurement results
 */
int syndrome_create(Syndrome** syndrome,
                    int8_t* measurements,
                    size_t num_measurements,
                    TopologicalErrorType type);

/**
 * Destroy syndrome
 */
void syndrome_destroy(Syndrome* syndrome);

/**
 * Compare two syndromes
 */
int syndrome_compare(Syndrome* s1, Syndrome* s2, Syndrome** diff_out);

/**
 * Check if syndrome is trivial (no defects)
 */
bool syndrome_is_trivial(Syndrome* syndrome);

/**
 * Get defect positions from syndrome
 */
int syndrome_get_defects(Syndrome* syndrome,
                          LatticePosition** positions_out,
                          size_t* num_defects_out);

// =============================================================================
// Error Chain Operations
// =============================================================================

/**
 * Create error chain
 */
int error_chain_create(ErrorChain** chain,
                        LatticePosition* positions,
                        size_t length,
                        TopologicalErrorType type);

/**
 * Destroy error chain
 */
void error_chain_destroy(ErrorChain* chain);

/**
 * Concatenate two error chains
 */
int error_chain_concat(ErrorChain* chain1, ErrorChain* chain2,
                        ErrorChain** result_out);

/**
 * Check if chain represents a logical error
 */
bool error_chain_is_logical(ErrorChain* chain, TopologicalCode* code);

/**
 * Convert error chain to correction circuit
 */
int error_chain_to_circuit(ErrorChain* chain,
                            TopologicalCode* code,
                            struct quantum_circuit** circuit_out);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Calculate code threshold from error rates
 */
double topological_calculate_threshold(TopologicalCodeType code_type,
                                        DecoderType decoder_type);

/**
 * Estimate logical error rate
 */
double topological_estimate_logical_error(TopologicalCode* code,
                                           double physical_error_rate);

/**
 * Get optimal code distance for target logical error rate
 */
size_t topological_optimal_distance(TopologicalCodeType code_type,
                                     double physical_error_rate,
                                     double target_logical_error);

/**
 * Print code information
 */
void topological_code_print(TopologicalCode* code);

/**
 * Print syndrome
 */
void syndrome_print(Syndrome* syndrome, TopologicalCode* code);

/**
 * Get code type name
 */
const char* topological_code_type_name(TopologicalCodeType type);

/**
 * Get decoder type name
 */
const char* topological_decoder_type_name(DecoderType type);

#ifdef __cplusplus
}
#endif

#endif // TOPOLOGICAL_PROTECTION_H

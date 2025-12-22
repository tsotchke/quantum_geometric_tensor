/**
 * @file anyon_detection.h
 * @brief Header file for anyon detection and tracking system
 */

#ifndef ANYON_DETECTION_H
#define ANYON_DETECTION_H

#include "quantum_geometric/physics/quantum_state_operations.h"
#include <stdbool.h>
#include <complex.h>

// Error syndrome types (for error correction)
typedef enum {
    SYNDROME_NONE,    // No syndrome present
    SYNDROME_X,       // X-type syndrome
    SYNDROME_Z,       // Z-type syndrome
    SYNDROME_Y        // Y-type syndrome (combined X and Z)
} syndrome_type_t;

// Anyon particle types (physical anyon classification)
typedef enum {
    ANYON_NONE,           // No anyon present
    ANYON_X,              // X-type error anyon
    ANYON_Z,              // Z-type error anyon
    ANYON_Y,              // Y-type error anyon
    ANYON_ABELIAN,        // Abelian anyon (commutative statistics)
    ANYON_NON_ABELIAN,    // Non-abelian anyon (non-commutative statistics)
    ANYON_MAJORANA,       // Majorana fermion (self-conjugate)
    ANYON_FIBONACCI,      // Fibonacci anyon (universal TQC)
    ANYON_ISING           // Ising anyon
} anyon_type_t;

// Forward declaration for quantum state
struct quantum_state_t;
typedef struct quantum_state_t quantum_state;

// Position in 3D lattice with stability metric
typedef struct {
    size_t x;              // X coordinate
    size_t y;              // Y coordinate
    size_t z;              // Z coordinate
    anyon_type_t type;     // Type of anyon at this position
    double stability;      // Position stability metric [0,1]
} AnyonPosition;

// Anyon charge structure (topological quantum numbers)
typedef struct {
    double electric;       // Electric charge component
    double magnetic;       // Magnetic charge component
    double topological;    // Topological charge (fractional statistics parameter)
} AnyonCharge;

// Full anyon structure
typedef struct Anyon {
    anyon_type_t type;     // Anyon type classification
    AnyonPosition position; // Current position
    AnyonCharge charge;    // Topological charges
    double lifetime;       // Anyon lifetime (coherence time)
    double energy;         // Anyon energy
    bool is_mobile;        // Whether anyon can move
} Anyon;

// Anyon pair for braiding/fusion operations
typedef struct AnyonPair {
    Anyon* anyon1;         // First anyon
    Anyon* anyon2;         // Second anyon
    double interaction_strength; // Pair interaction strength
    double braiding_phase; // Accumulated braiding phase
} AnyonPair;

// Configuration for braiding operations
typedef struct {
    double min_separation;        // Minimum anyon separation
    double max_interaction_strength; // Maximum allowed interaction
    size_t braiding_steps;        // Number of discrete steps for braiding path
    bool verify_topology;         // Verify topological protection during braiding
} BraidingConfig;

// Configuration for fusion operations
typedef struct {
    double energy_threshold;      // Maximum energy for fusion to occur
    double coherence_requirement; // Minimum coherence for valid fusion
    size_t fusion_attempts;       // Number of fusion attempts
    bool track_statistics;        // Track fusion statistics
} FusionConfig;

// Fusion outcome result
typedef struct {
    double probability;           // Fusion probability
    anyon_type_t result_type;     // Resulting anyon type
    AnyonCharge result_charge;    // Resulting charge
    double energy_delta;          // Energy change from fusion
} FusionOutcome;

// Detection configuration (extended)
typedef struct {
    double detection_threshold;   // Anyon detection threshold
    double noise_tolerance;       // Noise tolerance level
    size_t measurement_cycles;    // Number of measurement cycles
    bool use_error_correction;    // Enable error correction
} DetectionConfig;

// Cell in anyon grid
typedef struct {
    anyon_type_t type;     // Type of anyon in this cell
    double charge;         // Anyon charge value
    double velocity[3];    // Velocity components [vx, vy, vz]
    double confidence;     // Detection confidence [0,1]
    bool is_fused;        // Whether anyon has fused with another
} AnyonCell;

// Grid structure for anyon tracking
typedef struct {
    AnyonCell* cells;     // Array of cells
    size_t width;         // Grid width
    size_t height;        // Grid height
    size_t depth;         // Grid depth (layers)
} AnyonGrid;

// Configuration for anyon detection
typedef struct {
    size_t grid_width;           // Width of detection grid
    size_t grid_height;          // Height of detection grid
    size_t grid_depth;           // Depth of detection grid
    double detection_threshold;   // Threshold for anyon detection
    double max_movement_speed;    // Maximum allowed movement speed
    double charge_threshold;      // Threshold for charge measurement
} AnyonConfig;

// State for anyon detection system
typedef struct {
    AnyonGrid* grid;             // Detection grid
    AnyonPosition* last_positions; // Previous anyon positions
    size_t measurement_count;     // Number of measurements taken
    size_t total_anyons;         // Total anyons detected
} AnyonState;

/**
 * Initialize anyon detection system
 * @param state Pointer to state structure to initialize
 * @param config Configuration parameters
 * @return true if initialization successful, false otherwise
 */
bool init_anyon_detection(AnyonState* state, const AnyonConfig* config);

/**
 * Clean up anyon detection system
 * @param state State structure to clean up
 */
void cleanup_anyon_detection(AnyonState* state);

/**
 * Detect and track anyons in quantum state
 * @param state Detection system state
 * @param qstate Quantum state to analyze
 * @return true if detection successful, false otherwise
 */
bool detect_and_track_anyons(AnyonState* state, const quantum_state* qstate);

/**
 * Count total number of anyons in grid
 * @param grid Grid to count anyons in
 * @return Number of anyons detected
 */
size_t count_anyons(const AnyonGrid* grid);

/**
 * Get positions of all anyons
 * @param grid Grid containing anyons
 * @param positions Array to store positions (must be pre-allocated)
 * @return true if positions retrieved successfully, false otherwise
 */
bool get_anyon_positions(const AnyonGrid* grid, AnyonPosition* positions);

// ============================================================================
// Anyon Physics Helper Functions
// ============================================================================

/**
 * Calculate Euclidean distance between two anyon positions
 */
double calculate_distance(const AnyonPosition* pos1, const AnyonPosition* pos2);

/**
 * Calculate braiding phase increment for a single step
 * Uses Aharonov-Bohm phase: φ = q₁q₂/ℏ × solid_angle
 */
complex double calculate_braiding_phase_step(const quantum_state* state,
                                            const Anyon* moving_anyon,
                                            const Anyon* stationary_anyon,
                                            const AnyonPosition* prev_pos);

/**
 * Verify topological protection is maintained during operation
 * Checks that the topological order is preserved
 */
bool verify_topological_protection(const quantum_state* state,
                                  const Anyon* const* anyons,
                                  size_t num_anyons);

/**
 * Apply braiding unitary to quantum state
 * Implements R-matrix for anyon exchange
 */
bool apply_braiding_operation(quantum_state* state,
                             const Anyon* anyon1,
                             const Anyon* anyon2,
                             complex double phase);

/**
 * Calculate fusion energy from anyon pair
 * E = E₁ + E₂ - binding_energy(type1, type2)
 */
double calculate_fusion_energy(const Anyon* anyon1, const Anyon* anyon2);

/**
 * Calculate fusion probability based on anyon types and charges
 * Uses fusion rules for the anyon model
 */
double calculate_fusion_probability(const Anyon* anyon1, const Anyon* anyon2);

/**
 * Determine resulting anyon type from fusion
 * Implements fusion rules: a × b = Σ N^c_{ab} c
 */
anyon_type_t determine_fusion_type(const Anyon* anyon1, const Anyon* anyon2);

/**
 * Calculate resulting charge from fusion (charge addition rules)
 */
AnyonCharge calculate_fusion_charge(AnyonCharge charge1, AnyonCharge charge2);

/**
 * Calculate energy difference from fusion process
 */
double calculate_energy_difference(const quantum_state* state,
                                  const Anyon* anyon1,
                                  const Anyon* anyon2,
                                  const AnyonCharge* result_charge);

/**
 * Apply fusion operation to quantum state
 * Implements F-matrix transformation
 */
bool apply_fusion_operation(quantum_state* state,
                           const Anyon* anyon1,
                           const Anyon* anyon2,
                           FusionOutcome* outcome);

/**
 * Update fusion statistics tracking
 */
void update_fusion_statistics(anyon_type_t result_type,
                             double probability,
                             double energy_delta);

/**
 * Calculate charge-charge interaction strength
 */
double calculate_charge_interaction(AnyonCharge charge1, AnyonCharge charge2);

/**
 * Calculate type-dependent interaction factor
 */
double calculate_type_interaction(anyon_type_t type1, anyon_type_t type2);

/**
 * Calculate statistical angle θ for anyon exchange
 * θ = π for fermions, 0 for bosons, fractional for anyons
 */
double calculate_statistical_angle(anyon_type_t type1, anyon_type_t type2);

/**
 * Calculate phase factor from charge interaction
 */
double calculate_charge_phase_factor(AnyonCharge charge1, AnyonCharge charge2);

/**
 * Calculate topological correction to braiding phase
 */
double calculate_topological_correction(const Anyon* anyon1, const Anyon* anyon2);

/**
 * Check if two anyon types are compatible for fusion
 */
bool are_types_compatible(anyon_type_t type1, anyon_type_t type2);

/**
 * Verify charge conservation in fusion process
 */
bool verify_charge_conservation(AnyonCharge charge1, AnyonCharge charge2);

/**
 * Verify topological rules for fusion
 */
bool verify_topological_rules(const Anyon* anyon1, const Anyon* anyon2);

/**
 * Calculate all possible fusion channels
 */
void calculate_fusion_channels(anyon_type_t type1,
                              anyon_type_t type2,
                              FusionOutcome* outcomes,
                              size_t* num_outcomes,
                              size_t max_outcomes);

/**
 * Calculate probability for specific fusion channel
 */
double calculate_channel_probability(const AnyonPair* pair, const FusionOutcome* outcome);

/**
 * Calculate energy for specific fusion channel
 */
double calculate_channel_energy(const AnyonPair* pair, const FusionOutcome* outcome);

/**
 * Detect anyons in quantum state
 */
size_t detect_anyons(const quantum_state* state, Anyon* anyons, size_t max_anyons);

/**
 * Track anyon movement over time steps
 */
bool track_anyon_movement(quantum_state* state, Anyon* anyon, size_t num_steps);

/**
 * Inject test error at position (for testing)
 */
void inject_test_error(quantum_state* state, size_t position);

/**
 * Initialize topological state
 */
void initialize_topological_state(quantum_state* state);

/**
 * Create quantum state with given size
 */
quantum_state* create_quantum_state(size_t size);

/**
 * Destroy quantum state
 */
void destroy_quantum_state(quantum_state* state);

#endif // ANYON_DETECTION_H

/**
 * @file anyon_detection.h
 * @brief Header file for anyon detection and tracking system
 */

#ifndef ANYON_DETECTION_H
#define ANYON_DETECTION_H

#include "quantum_geometric/physics/quantum_state_operations.h"
#include <stdbool.h>

// Anyon types
typedef enum {
    ANYON_NONE,    // No anyon present
    ANYON_X,       // X-type anyon
    ANYON_Z,       // Z-type anyon
    ANYON_Y        // Y-type anyon (combined X and Z)
} anyon_type_t;

// Position in 3D lattice
typedef struct {
    size_t x;          // X coordinate
    size_t y;          // Y coordinate
    size_t z;          // Z coordinate
    anyon_type_t type; // Type of anyon at this position
} AnyonPosition;

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

#endif // ANYON_DETECTION_H

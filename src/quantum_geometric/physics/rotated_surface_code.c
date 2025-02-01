/**
 * @file rotated_surface_code.c
 * @brief Implementation of rotated surface code
 */

#include "quantum_geometric/physics/rotated_surface_code.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Internal rotated lattice state
typedef struct {
    RotatedLatticeConfig config;
    size_t num_stabilizers;
    size_t num_data_qubits;
    bool initialized;
    double* stabilizer_weights;
    StabilizerType* stabilizer_types;
    size_t** qubit_indices;
    size_t* qubit_counts;
    bool* is_boundary;
} RotatedLatticeState;

static RotatedLatticeState lattice_state = {0};

bool init_rotated_lattice(const RotatedLatticeConfig* config) {
    if (!config || !validate_rotated_config(config)) {
        return false;
    }

    // Initialize state
    memset(&lattice_state, 0, sizeof(RotatedLatticeState));
    memcpy(&lattice_state.config, config, sizeof(RotatedLatticeConfig));

    // Calculate dimensions
    calculate_rotated_dimensions(config->distance,
                               &lattice_state.config.width,
                               &lattice_state.config.height);

    // Allocate memory
    size_t max_stabilizers = config->width * config->height;
    lattice_state.stabilizer_weights = calloc(max_stabilizers, sizeof(double));
    lattice_state.stabilizer_types = calloc(max_stabilizers, sizeof(StabilizerType));
    lattice_state.qubit_indices = calloc(max_stabilizers, sizeof(size_t*));
    lattice_state.qubit_counts = calloc(max_stabilizers, sizeof(size_t));
    lattice_state.is_boundary = calloc(max_stabilizers, sizeof(bool));

    for (size_t i = 0; i < max_stabilizers; i++) {
        lattice_state.qubit_indices[i] = calloc(4, sizeof(size_t));
    }

    // Setup lattice
    setup_rotated_stabilizers();
    if (config->use_boundary_stabilizers) {
        setup_rotated_boundaries();
    }
    calculate_rotated_weights();

    // Verify setup
    if (!check_rotated_consistency()) {
        cleanup_rotated_lattice();
        return false;
    }

    lattice_state.initialized = true;
    return true;
}

void cleanup_rotated_lattice(void) {
    if (lattice_state.stabilizer_weights) {
        free(lattice_state.stabilizer_weights);
    }
    if (lattice_state.stabilizer_types) {
        free(lattice_state.stabilizer_types);
    }
    if (lattice_state.qubit_indices) {
        for (size_t i = 0; i < lattice_state.config.width * lattice_state.config.height; i++) {
            if (lattice_state.qubit_indices[i]) {
                free(lattice_state.qubit_indices[i]);
            }
        }
        free(lattice_state.qubit_indices);
    }
    if (lattice_state.qubit_counts) {
        free(lattice_state.qubit_counts);
    }
    if (lattice_state.is_boundary) {
        free(lattice_state.is_boundary);
    }
    memset(&lattice_state, 0, sizeof(RotatedLatticeState));
}

bool get_rotated_coordinates(size_t stabilizer_idx,
                           double* x,
                           double* y) {
    if (!lattice_state.initialized || !x || !y ||
        stabilizer_idx >= lattice_state.num_stabilizers) {
        return false;
    }

    // Calculate base coordinates
    size_t row = stabilizer_idx / lattice_state.config.width;
    size_t col = stabilizer_idx % lattice_state.config.width;

    // Apply rotation transformation
    double angle = lattice_state.config.angle * M_PI / 180.0;
    double base_x = (double)col;
    double base_y = (double)row;

    *x = base_x * cos(angle) - base_y * sin(angle);
    *y = base_x * sin(angle) + base_y * cos(angle);

    return true;
}

size_t get_rotated_qubits(size_t stabilizer_idx,
                         size_t* qubits,
                         size_t max_qubits) {
    if (!lattice_state.initialized || !qubits || max_qubits == 0 ||
        stabilizer_idx >= lattice_state.num_stabilizers) {
        return 0;
    }

    size_t count = lattice_state.qubit_counts[stabilizer_idx];
    if (count > max_qubits) {
        count = max_qubits;
    }

    memcpy(qubits, lattice_state.qubit_indices[stabilizer_idx],
           count * sizeof(size_t));
    return count;
}

bool is_boundary_stabilizer(size_t stabilizer_idx) {
    if (!lattice_state.initialized ||
        stabilizer_idx >= lattice_state.num_stabilizers) {
        return false;
    }
    return lattice_state.is_boundary[stabilizer_idx];
}

StabilizerType get_rotated_stabilizer_type(size_t stabilizer_idx) {
    if (!lattice_state.initialized ||
        stabilizer_idx >= lattice_state.num_stabilizers) {
        return STABILIZER_X; // Default return value
    }
    return lattice_state.stabilizer_types[stabilizer_idx];
}

size_t get_rotated_neighbors(size_t stabilizer_idx,
                           size_t* neighbors,
                           size_t max_neighbors) {
    if (!lattice_state.initialized || !neighbors || max_neighbors == 0 ||
        stabilizer_idx >= lattice_state.num_stabilizers) {
        return 0;
    }

    size_t count = 0;
    size_t row = stabilizer_idx / lattice_state.config.width;
    size_t col = stabilizer_idx % lattice_state.config.width;

    // Check potential neighbors
    size_t potential_neighbors[4][2] = {
        {row - 1, col}, // North
        {row, col + 1}, // East
        {row + 1, col}, // South
        {row, col - 1}  // West
    };

    for (size_t i = 0; i < 4 && count < max_neighbors; i++) {
        size_t r = potential_neighbors[i][0];
        size_t c = potential_neighbors[i][1];

        if (r < lattice_state.config.height && c < lattice_state.config.width) {
            size_t neighbor_idx = r * lattice_state.config.width + c;
            
            // Check if valid stabilizer
            if (neighbor_idx < lattice_state.num_stabilizers) {
                // Check if stabilizers share qubits
                bool shares_qubits = false;
                for (size_t j = 0; j < lattice_state.qubit_counts[stabilizer_idx]; j++) {
                    for (size_t k = 0; k < lattice_state.qubit_counts[neighbor_idx]; k++) {
                        if (lattice_state.qubit_indices[stabilizer_idx][j] ==
                            lattice_state.qubit_indices[neighbor_idx][k]) {
                            shares_qubits = true;
                            break;
                        }
                    }
                    if (shares_qubits) break;
                }

                if (shares_qubits) {
                    neighbors[count++] = neighbor_idx;
                }
            }
        }
    }

    return count;
}

void calculate_rotated_dimensions(size_t distance,
                                size_t* width,
                                size_t* height) {
    if (!width || !height) {
        return;
    }

    // For rotated surface code:
    // width = distance + 1
    // height = distance + 1
    *width = distance + 1;
    *height = distance + 1;
}

bool validate_rotated_config(const RotatedLatticeConfig* config) {
    if (!config) {
        return false;
    }

    // Check basic parameters
    if (config->distance < 3 || config->distance % 2 == 0) {
        return false;
    }

    // Check dimensions
    size_t required_width, required_height;
    calculate_rotated_dimensions(config->distance, &required_width, &required_height);

    if (config->width < required_width || config->height < required_height) {
        return false;
    }

    if (config->width > MAX_ROTATED_SIZE || config->height > MAX_ROTATED_SIZE) {
        return false;
    }

    // Check angle
    if (config->angle < 0.0 || config->angle >= 360.0) {
        return false;
    }

    return true;
}

void setup_rotated_stabilizers(void) {
    size_t width = lattice_state.config.width;
    size_t height = lattice_state.config.height;
    size_t stabilizer_idx = 0;

    // Setup alternating X and Z stabilizers
    for (size_t row = 0; row < height; row++) {
        for (size_t col = 0; col < width; col++) {
            // Skip positions that don't need stabilizers in rotated layout
            if ((row + col) % 2 == 0) {
                continue;
            }

            // Determine stabilizer type based on position
            StabilizerType type = ((row + col) % 4 == 1) ? STABILIZER_X : STABILIZER_Z;
            lattice_state.stabilizer_types[stabilizer_idx] = type;

            // Add surrounding qubits
            size_t qubit_count = 0;
            size_t qubit_positions[4][2] = {
                {row - 1, col}, // North
                {row, col + 1}, // East
                {row + 1, col}, // South
                {row, col - 1}  // West
            };

            for (size_t i = 0; i < 4; i++) {
                size_t r = qubit_positions[i][0];
                size_t c = qubit_positions[i][1];
                
                if (r < height && c < width) {
                    size_t qubit_idx = r * width + c;
                    lattice_state.qubit_indices[stabilizer_idx][qubit_count++] = qubit_idx;
                }
            }

            lattice_state.qubit_counts[stabilizer_idx] = qubit_count;
            stabilizer_idx++;
        }
    }

    lattice_state.num_stabilizers = stabilizer_idx;
}

void setup_rotated_boundaries(void) {
    size_t width = lattice_state.config.width;
    size_t height = lattice_state.config.height;

    // Mark boundary stabilizers
    for (size_t i = 0; i < lattice_state.num_stabilizers; i++) {
        size_t row = i / width;
        size_t col = i % width;

        // Check if on boundary
        if (row == 0 || row == height - 1 || col == 0 || col == width - 1) {
            lattice_state.is_boundary[i] = true;

            // Adjust qubit connections for boundary stabilizers
            size_t new_count = 0;
            for (size_t j = 0; j < lattice_state.qubit_counts[i]; j++) {
                size_t qubit_idx = lattice_state.qubit_indices[i][j];
                size_t qubit_row = qubit_idx / width;
                size_t qubit_col = qubit_idx % width;

                // Keep only qubits within boundaries
                if (qubit_row < height && qubit_col < width) {
                    lattice_state.qubit_indices[i][new_count++] = qubit_idx;
                }
            }
            lattice_state.qubit_counts[i] = new_count;
        }
    }
}

void calculate_rotated_weights(void) {
    // Calculate weights based on stabilizer properties
    for (size_t i = 0; i < lattice_state.num_stabilizers; i++) {
        double weight = 1.0;

        // Adjust weight based on number of qubits
        weight *= (double)lattice_state.qubit_counts[i] / 4.0;

        // Adjust weight for boundary stabilizers
        if (lattice_state.is_boundary[i]) {
            weight *= 0.8;
        }

        lattice_state.stabilizer_weights[i] = weight;
    }
}

bool check_rotated_consistency(void) {
    if (!lattice_state.initialized) {
        return false;
    }

    // Check stabilizer count
    if (lattice_state.num_stabilizers == 0) {
        return false;
    }

    // Verify each stabilizer
    for (size_t i = 0; i < lattice_state.num_stabilizers; i++) {
        // Check qubit count
        if (lattice_state.qubit_counts[i] == 0) {
            return false;
        }

        // Verify qubit indices
        for (size_t j = 0; j < lattice_state.qubit_counts[i]; j++) {
            size_t qubit_idx = lattice_state.qubit_indices[i][j];
            if (qubit_idx >= lattice_state.config.width * lattice_state.config.height) {
                return false;
            }
        }

        // Check weight
        if (lattice_state.stabilizer_weights[i] <= 0.0) {
            return false;
        }
    }

    return true;
}

/**
 * @file floquet_surface_code.c
 * @brief Implementation of Floquet surface code for time-dependent quantum error correction
 */

#include "quantum_geometric/physics/floquet_surface_code.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Internal Floquet lattice state
typedef struct {
    FloquetConfig config;
    size_t num_stabilizers;
    size_t num_data_qubits;
    bool initialized;
    double* stabilizer_weights;
    StabilizerType* stabilizer_types;
    size_t** qubit_indices;
    size_t* qubit_counts;
    bool* is_boundary;
    double** coupling_matrices;  // One coupling matrix per time step
    double* evolution_operators; // Time evolution operators between steps
} FloquetState;

static FloquetState floquet_state = {0};

bool init_floquet_lattice(const FloquetConfig* config) {
    if (!config || !validate_floquet_config(config)) {
        return false;
    }

    // Initialize state
    memset(&floquet_state, 0, sizeof(FloquetState));
    memcpy(&floquet_state.config, config, sizeof(FloquetConfig));

    // Calculate dimensions
    calculate_floquet_dimensions(config->distance,
                               &floquet_state.config.width,
                               &floquet_state.config.height);

    // Allocate memory
    size_t max_stabilizers = config->width * config->height;
    size_t max_qubits = max_stabilizers * 4; // Each stabilizer can have up to 4 qubits

    floquet_state.stabilizer_weights = calloc(max_stabilizers, sizeof(double));
    floquet_state.stabilizer_types = calloc(max_stabilizers, sizeof(StabilizerType));
    floquet_state.qubit_indices = calloc(max_stabilizers, sizeof(size_t*));
    floquet_state.qubit_counts = calloc(max_stabilizers, sizeof(size_t));
    floquet_state.is_boundary = calloc(max_stabilizers, sizeof(bool));
    
    // Allocate coupling matrices for each time step
    floquet_state.coupling_matrices = calloc(config->time_steps, sizeof(double*));
    for (size_t t = 0; t < config->time_steps; t++) {
        floquet_state.coupling_matrices[t] = calloc(max_qubits * max_qubits, sizeof(double));
    }

    // Allocate evolution operators
    size_t num_operators = config->time_steps * (config->time_steps - 1) / 2;
    floquet_state.evolution_operators = calloc(num_operators * max_qubits * max_qubits, sizeof(double));

    for (size_t i = 0; i < max_stabilizers; i++) {
        floquet_state.qubit_indices[i] = calloc(4, sizeof(size_t)); // Max 4 qubits per stabilizer
    }

    // Setup lattice
    setup_floquet_stabilizers();
    if (config->use_boundary_stabilizers) {
        setup_floquet_boundaries();
    }
    calculate_floquet_weights();

    // Initialize time-dependent couplings
    for (size_t t = 0; t < config->time_steps; t++) {
        update_floquet_couplings(t);
    }

    // Verify setup
    if (!check_floquet_consistency()) {
        cleanup_floquet_lattice();
        return false;
    }

    floquet_state.initialized = true;
    return true;
}

void cleanup_floquet_lattice(void) {
    if (floquet_state.stabilizer_weights) {
        free(floquet_state.stabilizer_weights);
    }
    if (floquet_state.stabilizer_types) {
        free(floquet_state.stabilizer_types);
    }
    if (floquet_state.qubit_indices) {
        for (size_t i = 0; i < floquet_state.config.width * floquet_state.config.height; i++) {
            if (floquet_state.qubit_indices[i]) {
                free(floquet_state.qubit_indices[i]);
            }
        }
        free(floquet_state.qubit_indices);
    }
    if (floquet_state.qubit_counts) {
        free(floquet_state.qubit_counts);
    }
    if (floquet_state.is_boundary) {
        free(floquet_state.is_boundary);
    }
    if (floquet_state.coupling_matrices) {
        for (size_t t = 0; t < floquet_state.config.time_steps; t++) {
            if (floquet_state.coupling_matrices[t]) {
                free(floquet_state.coupling_matrices[t]);
            }
        }
        free(floquet_state.coupling_matrices);
    }
    if (floquet_state.evolution_operators) {
        free(floquet_state.evolution_operators);
    }
    memset(&floquet_state, 0, sizeof(FloquetState));
}

bool get_floquet_coordinates(size_t stabilizer_idx,
                           size_t time_step,
                           double* x,
                           double* y) {
    if (!floquet_state.initialized || !x || !y ||
        stabilizer_idx >= floquet_state.num_stabilizers ||
        time_step >= floquet_state.config.time_steps) {
        return false;
    }

    // Calculate base coordinates
    size_t row = stabilizer_idx / floquet_state.config.width;
    size_t col = stabilizer_idx % floquet_state.config.width;

    // Apply time-dependent modulation
    double t = time_step * floquet_state.config.period / floquet_state.config.time_steps;
    double modulation = sin(2.0 * M_PI * t / floquet_state.config.period);
    
    *x = col + 0.1 * modulation;
    *y = row + 0.1 * modulation;

    return true;
}

size_t get_floquet_qubits(size_t stabilizer_idx,
                         size_t time_step,
                         size_t* qubits,
                         size_t max_qubits) {
    if (!floquet_state.initialized || !qubits || max_qubits == 0 ||
        stabilizer_idx >= floquet_state.num_stabilizers ||
        time_step >= floquet_state.config.time_steps) {
        return 0;
    }

    size_t count = floquet_state.qubit_counts[stabilizer_idx];
    if (count > max_qubits) {
        count = max_qubits;
    }

    memcpy(qubits, floquet_state.qubit_indices[stabilizer_idx],
           count * sizeof(size_t));
    return count;
}

bool is_floquet_boundary_stabilizer(size_t stabilizer_idx,
                                  size_t time_step) {
    if (!floquet_state.initialized ||
        stabilizer_idx >= floquet_state.num_stabilizers ||
        time_step >= floquet_state.config.time_steps) {
        return false;
    }
    return floquet_state.is_boundary[stabilizer_idx];
}

StabilizerType get_floquet_stabilizer_type(size_t stabilizer_idx,
                                         size_t time_step) {
    if (!floquet_state.initialized ||
        stabilizer_idx >= floquet_state.num_stabilizers ||
        time_step >= floquet_state.config.time_steps) {
        return STABILIZER_X; // Default return value
    }

    // Time-dependent stabilizer type
    double t = time_step * floquet_state.config.period / floquet_state.config.time_steps;
    bool flip = (sin(2.0 * M_PI * t / floquet_state.config.period) > 0);
    return flip ? 
        (floquet_state.stabilizer_types[stabilizer_idx] == STABILIZER_X ? STABILIZER_Z : STABILIZER_X) :
        floquet_state.stabilizer_types[stabilizer_idx];
}

size_t get_floquet_neighbors(size_t stabilizer_idx,
                           size_t time_step,
                           size_t* neighbors,
                           size_t max_neighbors) {
    if (!floquet_state.initialized || !neighbors || max_neighbors == 0 ||
        stabilizer_idx >= floquet_state.num_stabilizers ||
        time_step >= floquet_state.config.time_steps) {
        return 0;
    }

    size_t count = 0;
    size_t row = stabilizer_idx / floquet_state.config.width;
    size_t col = stabilizer_idx % floquet_state.config.width;

    // Check potential neighbors
    const int offsets[4][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}};
    for (size_t i = 0; i < 4 && count < max_neighbors; i++) {
        int new_row = row + offsets[i][0];
        int new_col = col + offsets[i][1];
        
        if (new_row >= 0 && new_row < (int)floquet_state.config.height &&
            new_col >= 0 && new_col < (int)floquet_state.config.width) {
            neighbors[count++] = new_row * floquet_state.config.width + new_col;
        }
    }

    return count;
}

void calculate_floquet_dimensions(size_t distance,
                                size_t* width,
                                size_t* height) {
    if (!width || !height) {
        return;
    }

    // For Floquet code:
    // width = distance
    // height = distance
    *width = distance;
    *height = distance;
}

bool validate_floquet_config(const FloquetConfig* config) {
    if (!config) {
        return false;
    }

    // Check basic parameters
    if (config->distance < 3 || config->distance % 2 == 0) {
        return false;
    }

    // Check dimensions
    size_t required_width, required_height;
    calculate_floquet_dimensions(config->distance, &required_width, &required_height);

    if (config->width < required_width || config->height < required_height) {
        return false;
    }

    if (config->width > MAX_FLOQUET_SIZE || config->height > MAX_FLOQUET_SIZE) {
        return false;
    }

    // Check time parameters
    if (config->time_steps == 0 || config->time_steps > MAX_TIME_STEPS) {
        return false;
    }

    if (config->period <= 0.0) {
        return false;
    }

    // Check coupling parameters
    if (config->coupling_strength <= 0.0 || config->coupling_strength > 1.0) {
        return false;
    }

    if (!config->time_dependent_couplings) {
        return false;
    }

    return true;
}

double get_floquet_coupling_strength(size_t qubit1_idx,
                                   size_t qubit2_idx,
                                   size_t time_step) {
    if (!floquet_state.initialized ||
        qubit1_idx >= floquet_state.num_data_qubits ||
        qubit2_idx >= floquet_state.num_data_qubits ||
        time_step >= floquet_state.config.time_steps) {
        return 0.0;
    }

    size_t idx = qubit1_idx * floquet_state.num_data_qubits + qubit2_idx;
    return floquet_state.coupling_matrices[time_step][idx];
}

bool get_floquet_evolution_operator(size_t time_step1,
                                  size_t time_step2,
                                  double* operator,
                                  size_t max_size) {
    if (!floquet_state.initialized || !operator ||
        time_step1 >= floquet_state.config.time_steps ||
        time_step2 >= floquet_state.config.time_steps ||
        max_size < floquet_state.num_data_qubits * floquet_state.num_data_qubits) {
        return false;
    }

    // Get index into evolution operators array
    size_t t1 = time_step1 < time_step2 ? time_step1 : time_step2;
    size_t t2 = time_step1 < time_step2 ? time_step2 : time_step1;
    size_t idx = t1 * floquet_state.config.time_steps + t2;
    size_t matrix_size = floquet_state.num_data_qubits * floquet_state.num_data_qubits;

    memcpy(operator, &floquet_state.evolution_operators[idx * matrix_size],
           matrix_size * sizeof(double));
    return true;
}

void setup_floquet_stabilizers(void) {
    size_t width = floquet_state.config.width;
    size_t height = floquet_state.config.height;
    size_t stabilizer_idx = 0;

    // Setup alternating X and Z stabilizers
    for (size_t row = 0; row < height; row++) {
        for (size_t col = 0; col < width; col++) {
            // Determine stabilizer type based on checkerboard pattern
            StabilizerType type = ((row + col) % 2 == 0) ? STABILIZER_X : STABILIZER_Z;
            floquet_state.stabilizer_types[stabilizer_idx] = type;

            // Add surrounding qubits
            size_t qubit_count = 0;
            const int offsets[4][2] = {{-1,0}, {1,0}, {0,-1}, {0,1}};

            for (size_t i = 0; i < 4; i++) {
                int r = row + offsets[i][0];
                int c = col + offsets[i][1];
                
                if (r >= 0 && r < (int)height && c >= 0 && c < (int)width) {
                    size_t qubit_idx = r * width + c;
                    floquet_state.qubit_indices[stabilizer_idx][qubit_count++] = qubit_idx;
                }
            }

            floquet_state.qubit_counts[stabilizer_idx] = qubit_count;
            stabilizer_idx++;
        }
    }

    floquet_state.num_stabilizers = stabilizer_idx;
}

void setup_floquet_boundaries(void) {
    size_t width = floquet_state.config.width;
    size_t height = floquet_state.config.height;

    // Mark boundary stabilizers
    for (size_t i = 0; i < floquet_state.num_stabilizers; i++) {
        size_t row = i / width;
        size_t col = i % width;

        if (row == 0 || row == height - 1 || col == 0 || col == width - 1) {
            floquet_state.is_boundary[i] = true;
        }
    }
}

void calculate_floquet_weights(void) {
    // Calculate weights based on stabilizer properties
    for (size_t i = 0; i < floquet_state.num_stabilizers; i++) {
        double weight = 1.0;

        // Adjust weight based on number of qubits
        weight *= (double)floquet_state.qubit_counts[i] / 4.0;

        // Adjust weight for boundary stabilizers
        if (floquet_state.is_boundary[i]) {
            weight *= 0.8;
        }

        floquet_state.stabilizer_weights[i] = weight;
    }
}

void update_floquet_couplings(size_t time_step) {
    if (time_step >= floquet_state.config.time_steps) {
        return;
    }

    double t = time_step * floquet_state.config.period / floquet_state.config.time_steps;
    double base_coupling = floquet_state.config.coupling_strength;
    double modulation = floquet_state.config.time_dependent_couplings[time_step];

    // Update coupling matrix for this time step
    for (size_t i = 0; i < floquet_state.num_stabilizers; i++) {
        for (size_t j = 0; j < floquet_state.qubit_counts[i]; j++) {
            size_t q1 = floquet_state.qubit_indices[i][j];
            for (size_t k = j + 1; k < floquet_state.qubit_counts[i]; k++) {
                size_t q2 = floquet_state.qubit_indices[i][k];
                double coupling = base_coupling * (1.0 + modulation * sin(2.0 * M_PI * t / floquet_state.config.period));
                
                size_t idx = q1 * floquet_state.num_data_qubits + q2;
                floquet_state.coupling_matrices[time_step][idx] = coupling;
                floquet_state.coupling_matrices[time_step][q2 * floquet_state.num_data_qubits + q1] = coupling;
            }
        }
    }
}

bool check_floquet_consistency(void) {
    if (!floquet_state.initialized) {
        return false;
    }

    // Check stabilizer count
    if (floquet_state.num_stabilizers == 0) {
        return false;
    }

    // Verify each stabilizer
    for (size_t i = 0; i < floquet_state.num_stabilizers; i++) {
        // Check qubit count
        if (floquet_state.qubit_counts[i] == 0) {
            return false;
        }

        // Verify qubit indices
        for (size_t j = 0; j < floquet_state.qubit_counts[i]; j++) {
            size_t qubit_idx = floquet_state.qubit_indices[i][j];
            if (qubit_idx >= floquet_state.config.width * floquet_state.config.height) {
                return false;
            }
        }

        // Check weight
        if (floquet_state.stabilizer_weights[i] <= 0.0) {
            return false;
        }
    }

    // Verify coupling matrices
    for (size_t t = 0; t < floquet_state.config.time_steps; t++) {
        if (!floquet_state.coupling_matrices[t]) {
            return false;
        }
    }

    return true;
}

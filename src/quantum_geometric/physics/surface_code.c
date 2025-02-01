/**
 * @file surface_code.c
 * @brief Implementation of surface code for quantum error correction
 */

#include "quantum_geometric/physics/surface_code.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper function declarations
static void setup_standard_lattice(SurfaceCode* state);
static void setup_rotated_lattice(SurfaceCode* state);
static void setup_heavy_hex_lattice(SurfaceCode* state);
static void setup_floquet_lattice(SurfaceCode* state);
static bool validate_lattice_configuration(SurfaceCode* state);
static void calculate_stabilizer_weights(SurfaceCode* state);
static void update_stabilizer_correlations(SurfaceCode* state);

SurfaceCode* init_surface_code(const SurfaceConfig* config) {
    if (!config || !validate_surface_config(config)) {
        return NULL;
    }

    // Allocate and initialize state
    SurfaceCode* state = (SurfaceCode*)calloc(1, sizeof(SurfaceCode));
    if (!state) {
        return NULL;
    }

    // Copy configuration
    memcpy(&state->config, config, sizeof(SurfaceConfig));

    // Setup lattice based on type
    switch (config->type) {
        case SURFACE_CODE_STANDARD:
            setup_standard_lattice(state);
            break;
        case SURFACE_CODE_ROTATED:
            setup_rotated_lattice(state);
            break;
        case SURFACE_CODE_HEAVY_HEX:
            setup_heavy_hex_lattice(state);
            break;
        case SURFACE_CODE_FLOQUET:
            setup_floquet_lattice(state);
            break;
        default:
            free(state);
            return NULL;
    }

    // Validate lattice setup
    if (!validate_lattice_configuration(state)) {
        free(state);
        return NULL;
    }

    // Initialize stabilizers and logical qubits
    initialize_stabilizers(state);
    initialize_logical_qubits(state);

    state->initialized = true;
    return state;
}

void cleanup_surface_code(SurfaceCode* state) {
    if (state) {
        free(state);
    }
}

size_t measure_stabilizers(SurfaceCode* state, StabilizerResult* results) {
    if (!state || !state->initialized || !results) {
        return 0;
    }

    // Try Metal acceleration first
    if (state->config.use_metal_acceleration) {
        QuantumGeometricMetal* metal = get_metal_context();
        if (metal) {
            // Prepare Metal measurement configuration
            ZStabilizerConfig metal_config = {
                .enable_optimization = true,
                .num_measurements = state->num_stabilizers,
                .error_threshold = state->config.error_threshold,
                .confidence_threshold = 0.9,
                .use_phase_tracking = true,
                .track_correlations = true,
                .history_capacity = 100,
                .use_metal_acceleration = true,
                .num_stabilizers = state->num_stabilizers,
                .parallel_group_size = 4,
                .phase_calibration = 1.0,
                .correlation_factor = 0.8
            };

            // Collect stabilizer indices
            size_t* stabilizer_indices = malloc(state->num_stabilizers * 4 * sizeof(size_t));
            for (size_t i = 0; i < state->num_stabilizers; i++) {
                Stabilizer* stabilizer = &state->stabilizers[i];
                memcpy(&stabilizer_indices[i * 4], stabilizer->qubits,
                       stabilizer->num_qubits * sizeof(size_t));
            }

            // Perform Metal-accelerated measurement
            ZStabilizerResults metal_results;
            bool success = [metal measureZStabilizers:state->quantum_state
                                        stabilizers:stabilizer_indices
                                            config:&metal_config
                                           results:&metal_results];
            free(stabilizer_indices);

            if (success) {
                // Copy results and update state
                for (size_t i = 0; i < state->num_stabilizers; i++) {
                    results[i].value = metal_results.measurements[i].value;
                    results[i].confidence = metal_results.measurements[i].confidence;
                    results[i].needs_correction = (metal_results.measurements[i].value < 0);

                    // Update stabilizer state
                    state->stabilizers[i].result = results[i];
                    state->stabilizers[i].error_rate = 1.0 - results[i].confidence;
                }

                // Update error rates and correlations
                update_error_rates(state, results, state->num_stabilizers);
                update_stabilizer_correlations(state);

                return state->num_stabilizers;
            }
        }
    }

    // Fallback to CPU implementation
    size_t measurements = 0;
    
    // Measure each stabilizer
    for (size_t i = 0; i < state->num_stabilizers; i++) {
        Stabilizer* stabilizer = &state->stabilizers[i];
        
        // Perform measurement
        double measurement_value = 0.0;
        double confidence = 1.0;
        
        // Accumulate contributions from each qubit
        for (size_t j = 0; j < stabilizer->num_qubits; j++) {
            size_t qubit_idx = stabilizer->qubits[j];
            
            // Apply appropriate Pauli operator based on stabilizer type
            switch (stabilizer->type) {
                case STABILIZER_X: {
                    // Apply X measurement with error tracking
                    double x_value = 0.0;
                    double x_conf = 0.0;
                    measure_pauli_x_with_confidence(state, qubit_idx, &x_value, &x_conf);
                    measurement_value *= x_value;
                    confidence *= x_conf * (1.0 - state->config.measurement_error_rate);
                    break;
                }
                    
                case STABILIZER_Z: {
                    // Apply Z measurement with error tracking
                    double z_value = 0.0;
                    double z_conf = 0.0;
                    measure_pauli_z_with_confidence(state, qubit_idx, &z_value, &z_conf);
                    measurement_value *= z_value;
                    confidence *= z_conf * (1.0 - state->config.measurement_error_rate);
                    break;
                }
                    
                case STABILIZER_Y: {
                    // Apply Y measurement (composite X and Z)
                    double x_value = 0.0, z_value = 0.0;
                    double x_conf = 0.0, z_conf = 0.0;
                    measure_pauli_x_with_confidence(state, qubit_idx, &x_value, &x_conf);
                    measure_pauli_z_with_confidence(state, qubit_idx, &z_value, &z_conf);
                    measurement_value *= x_value * z_value;
                    confidence *= x_conf * z_conf * (1.0 - state->config.measurement_error_rate);
                    break;
                }
            }
        }

        // Record measurement result
        results[measurements].value = (measurement_value > 0) ? 1 : -1;
        results[measurements].confidence = confidence;
        results[measurements].needs_correction = (measurement_value < 0);
        
        // Update stabilizer state
        stabilizer->result = results[measurements];
        stabilizer->error_rate = 1.0 - confidence;
        
        measurements++;
    }

    // Update error rates and correlations
    update_error_rates(state, results, measurements);
    update_stabilizer_correlations(state);

    return measurements;
}

size_t apply_corrections(SurfaceCode* state,
                        const SyndromeVertex* syndromes,
                        size_t num_syndromes) {
    if (!state || !state->initialized || !syndromes) {
        return 0;
    }

    size_t corrections = 0;

    // Process each syndrome
    for (size_t i = 0; i < num_syndromes; i++) {
        // Find affected stabilizers
        for (size_t j = 0; j < state->num_stabilizers; j++) {
            Stabilizer* stabilizer = &state->stabilizers[j];
            
            // Check if syndrome affects this stabilizer
            bool affects_stabilizer = false;
            for (size_t k = 0; k < stabilizer->num_qubits; k++) {
                // Check if syndrome vertex intersects with stabilizer qubits
                size_t qubit_idx = stabilizer->qubits[k];
                size_t qubit_x = qubit_idx % state->config.width;
                size_t qubit_y = qubit_idx / state->config.width;
                
                // Calculate syndrome-qubit distance based on lattice type
                double distance = 0.0;
                switch (state->config.type) {
                    case SURFACE_CODE_STANDARD:
                    case SURFACE_CODE_ROTATED:
                        // Manhattan distance for standard/rotated lattices
                        distance = abs((int)qubit_x - (int)syndromes[i].x) +
                                 abs((int)qubit_y - (int)syndromes[i].y);
                        affects_stabilizer = (distance <= 2);
                        break;
                        
                    case SURFACE_CODE_HEAVY_HEX:
                        // Hexagonal distance metric
                        double dx = abs((int)qubit_x - (int)syndromes[i].x);
                        double dy = abs((int)qubit_y - (int)syndromes[i].y);
                        distance = dx + max(0.0, (dy - dx/2));
                        affects_stabilizer = (distance <= 1.5);
                        break;
                        
                    case SURFACE_CODE_FLOQUET:
                        // Space-time distance including temporal component
                        distance = sqrt(pow(qubit_x - syndromes[i].x, 2) +
                                     pow(qubit_y - syndromes[i].y, 2) +
                                     pow(stabilizer->time_step - syndromes[i].t, 2));
                        affects_stabilizer = (distance <= sqrt(2));
                        break;
                }
                
                if (affects_stabilizer) {
                    // Update error metrics
                    update_error_metrics(qubit_idx, ERROR_SYNDROME,
                                      1.0 - syndromes[i].confidence);
                    
                    // Record syndrome-stabilizer correlation
                    record_syndrome_correlation(state, stabilizer,
                                             &syndromes[i], distance);
                    break;
                }
            }
            
            if (affects_stabilizer) {
                // Apply correction
                apply_stabilizer_corrections(stabilizer);
                corrections++;
            }
        }
    }

    // Update error rates after corrections
    if (corrections > 0) {
        calculate_stabilizer_weights(state);
        update_logical_error_rates(state);
    }

    return corrections;
}

int encode_logical_qubit(SurfaceCode* state,
                        const size_t* data_qubits,
                        size_t num_qubits) {
    if (!state || !state->initialized || !data_qubits ||
        num_qubits > MAX_SURFACE_SIZE ||
        state->num_logical_qubits >= MAX_LOGICAL_QUBITS) {
        return -1;
    }

    // Create new logical qubit
    LogicalQubit* logical = &state->logical_qubits[state->num_logical_qubits];
    
    // Copy data qubits
    memcpy(logical->data_qubits, data_qubits, num_qubits * sizeof(size_t));
    logical->num_data_qubits = num_qubits;
    
    // Find associated stabilizers
    logical->num_stabilizers = 0;
    for (size_t i = 0; i < state->num_stabilizers; i++) {
        // Check if stabilizer acts on any data qubit
        for (size_t j = 0; j < num_qubits; j++) {
            if (get_qubit_neighbors(state, data_qubits[j],
                                  logical->stabilizers + logical->num_stabilizers,
                                  MAX_STABILIZERS - logical->num_stabilizers) > 0) {
                logical->num_stabilizers++;
                break;
            }
        }
    }

    // Initialize error rate
    logical->logical_error_rate = 0.0;
    update_logical_error_rates(state);

    return state->num_logical_qubits++;
}

bool measure_logical_qubit(SurfaceCode* state,
                          size_t logical_idx,
                          StabilizerResult* result) {
    if (!state || !state->initialized || !result ||
        logical_idx >= state->num_logical_qubits) {
        return false;
    }

    LogicalQubit* logical = &state->logical_qubits[logical_idx];
    
    // Measure all stabilizers
    double total_value = 0.0;
    double total_confidence = 1.0;
    
    for (size_t i = 0; i < logical->num_stabilizers; i++) {
        Stabilizer* stabilizer = &state->stabilizers[logical->stabilizers[i]];
        
        // Combine stabilizer results
        total_value += stabilizer->result.value;
        total_confidence *= stabilizer->result.confidence;
    }

    // Compute logical measurement result
    result->value = (total_value > 0) ? 1 : -1;
    result->confidence = total_confidence;
    result->needs_correction = (total_value < 0);

    return true;
}

double update_error_rates(SurfaceCode* state,
                         const StabilizerResult* measurements,
                         size_t num_measurements) {
    if (!state || !state->initialized || !measurements) {
        return 0.0;
    }

    // Update individual stabilizer error rates
    for (size_t i = 0; i < num_measurements; i++) {
        state->stabilizers[i].error_rate =
            1.0 - measurements[i].confidence;
    }

    // Calculate total error rate
    double total_error = 0.0;
    for (size_t i = 0; i < state->num_stabilizers; i++) {
        total_error += state->stabilizers[i].error_rate;
    }
    
    state->total_error_rate = total_error / state->num_stabilizers;
    return state->total_error_rate;
}

const Stabilizer* get_stabilizer(const SurfaceCode* state, size_t stabilizer_idx) {
    if (!state || !state->initialized ||
        stabilizer_idx >= state->num_stabilizers) {
        return NULL;
    }
    return &state->stabilizers[stabilizer_idx];
}

const LogicalQubit* get_logical_qubit(const SurfaceCode* state, size_t logical_idx) {
    if (!state || !state->initialized ||
        logical_idx >= state->num_logical_qubits) {
        return NULL;
    }
    return &state->logical_qubits[logical_idx];
}

// Helper function implementations
bool validate_surface_config(const SurfaceConfig* config) {
    if (!config) return false;
    
    // Check basic parameters
    if (config->distance < 3 || config->distance % 2 == 0) return false;
    if (config->width < config->distance || config->height < config->distance) return false;
    if (config->width > MAX_SURFACE_SIZE || config->height > MAX_SURFACE_SIZE) return false;
    if (config->threshold <= 0.0 || config->threshold >= 1.0) return false;
    
    return true;
}

void initialize_stabilizers(SurfaceCode* state) {
    if (!state) return;

    // Clear existing stabilizers
    state->num_stabilizers = 0;
    
    // Initialize based on lattice type
    switch (state->config.type) {
        case SURFACE_CODE_STANDARD:
            // Add X stabilizers
            for (size_t i = 1; i < state->config.height; i += 2) {
                for (size_t j = 1; j < state->config.width; j += 2) {
                    Stabilizer* stabilizer = &state->stabilizers[state->num_stabilizers++];
                    stabilizer->type = STABILIZER_X;
                    // Add surrounding qubits
                    stabilizer->qubits[0] = (i-1) * state->config.width + j;
                    stabilizer->qubits[1] = i * state->config.width + (j-1);
                    stabilizer->qubits[2] = i * state->config.width + (j+1);
                    stabilizer->qubits[3] = (i+1) * state->config.width + j;
                    stabilizer->num_qubits = 4;
                }
            }
            
            // Add Z stabilizers
            for (size_t i = 2; i < state->config.height; i += 2) {
                for (size_t j = 2; j < state->config.width; j += 2) {
                    Stabilizer* stabilizer = &state->stabilizers[state->num_stabilizers++];
                    stabilizer->type = STABILIZER_Z;
                    // Add surrounding qubits
                    stabilizer->qubits[0] = (i-1) * state->config.width + j;
                    stabilizer->qubits[1] = i * state->config.width + (j-1);
                    stabilizer->qubits[2] = i * state->config.width + (j+1);
                    stabilizer->qubits[3] = (i+1) * state->config.width + j;
                    stabilizer->num_qubits = 4;
                }
            }
            break;
            
        case SURFACE_CODE_ROTATED: {
            // Initialize rotated lattice stabilizers
            for (size_t i = 0; i < state->config.height; i++) {
                for (size_t j = 0; j < state->config.width; j++) {
                    Stabilizer* stabilizer = &state->stabilizers[state->num_stabilizers++];
                    
                    // Alternate X and Z stabilizers in checkerboard pattern
                    stabilizer->type = ((i + j) % 2 == 0) ? STABILIZER_X : STABILIZER_Z;
                    
                    // Add surrounding qubits in rotated configuration
                    stabilizer->qubits[0] = i * state->config.width + j;  // Center
                    stabilizer->qubits[1] = ((i+1) % state->config.height) * state->config.width + j;  // North
                    stabilizer->qubits[2] = i * state->config.width + ((j+1) % state->config.width);   // East
                    stabilizer->qubits[3] = ((i > 0 ? i-1 : state->config.height-1)) * state->config.width + j; // South
                    stabilizer->qubits[4] = i * state->config.width + (j > 0 ? j-1 : state->config.width-1);    // West
                    stabilizer->num_qubits = 5;
                }
            }
            break;
        }
            
        case SURFACE_CODE_HEAVY_HEX: {
            // Initialize heavy hexagonal lattice stabilizers
            for (size_t i = 0; i < state->config.height; i += 2) {
                for (size_t j = 0; j < state->config.width; j += 3) {
                    // Add X stabilizer at center of hex
                    Stabilizer* x_stabilizer = &state->stabilizers[state->num_stabilizers++];
                    x_stabilizer->type = STABILIZER_X;
                    x_stabilizer->qubits[0] = i * state->config.width + j;           // Center
                    x_stabilizer->qubits[1] = i * state->config.width + (j+1);       // Right
                    x_stabilizer->qubits[2] = (i+1) * state->config.width + j;       // Bottom
                    x_stabilizer->qubits[3] = i * state->config.width + (j > 0 ? j-1 : state->config.width-1); // Left
                    x_stabilizer->num_qubits = 4;
                    
                    // Add Z stabilizers at vertices
                    if (j + 2 < state->config.width) {
                        Stabilizer* z_stabilizer = &state->stabilizers[state->num_stabilizers++];
                        z_stabilizer->type = STABILIZER_Z;
                        z_stabilizer->qubits[0] = i * state->config.width + (j+1);     // Left
                        z_stabilizer->qubits[1] = i * state->config.width + (j+2);     // Right
                        z_stabilizer->qubits[2] = (i+1) * state->config.width + (j+1); // Bottom
                        z_stabilizer->num_qubits = 3;
                    }
                }
            }
            break;
        }
            
        case SURFACE_CODE_FLOQUET: {
            // Initialize Floquet code stabilizers
            for (size_t t = 0; t < state->config.time_steps; t++) {
                for (size_t i = 1; i < state->config.height; i += 2) {
                    for (size_t j = 1; j < state->config.width; j += 2) {
                        Stabilizer* stabilizer = &state->stabilizers[state->num_stabilizers++];
                        
                        // Alternate between X and Z stabilizers in time
                        stabilizer->type = (t % 2 == 0) ? STABILIZER_X : STABILIZER_Z;
                        
                        // Add surrounding qubits
                        stabilizer->qubits[0] = (i-1) * state->config.width + j;     // Top
                        stabilizer->qubits[1] = i * state->config.width + (j-1);     // Left
                        stabilizer->qubits[2] = i * state->config.width + (j+1);     // Right
                        stabilizer->qubits[3] = (i+1) * state->config.width + j;     // Bottom
                        stabilizer->num_qubits = 4;
                        
                        // Store time step
                        stabilizer->time_step = t;
                    }
                }
            }
            break;
        }
    }
    
    // Initialize stabilizer states
    for (size_t i = 0; i < state->num_stabilizers; i++) {
        state->stabilizers[i].result.value = 1;
        state->stabilizers[i].result.confidence = 1.0;
        state->stabilizers[i].result.needs_correction = false;
        state->stabilizers[i].error_rate = 0.0;
    }
}

void initialize_logical_qubits(SurfaceCode* state) {
    if (!state) return;
    state->num_logical_qubits = 0;
}

bool check_error_threshold(const SurfaceCode* state) {
    if (!state) return false;
    return state->total_error_rate < state->config.threshold;
}

void update_logical_error_rates(SurfaceCode* state) {
    if (!state) return;
    
    for (size_t i = 0; i < state->num_logical_qubits; i++) {
        LogicalQubit* logical = &state->logical_qubits[i];
        
        // Calculate logical error rate from stabilizer error rates
        double error_rate = 0.0;
        for (size_t j = 0; j < logical->num_stabilizers; j++) {
            error_rate += state->stabilizers[logical->stabilizers[j]].error_rate;
        }
        logical->logical_error_rate = error_rate / logical->num_stabilizers;
    }
}

size_t get_qubit_neighbors(const SurfaceCode* state,
                          size_t qubit_idx,
                          size_t* neighbors,
                          size_t max_neighbors) {
    if (!state || !neighbors || max_neighbors == 0) {
        return 0;
    }

    size_t count = 0;
    
    // Find stabilizers that act on this qubit
    for (size_t i = 0; i < state->num_stabilizers && count < max_neighbors; i++) {
        Stabilizer* stabilizer = &state->stabilizers[i];
        
        for (size_t j = 0; j < stabilizer->num_qubits; j++) {
            if (stabilizer->qubits[j] == qubit_idx) {
                neighbors[count++] = i;
                break;
            }
        }
    }
    
    return count;
}

bool is_valid_stabilizer_configuration(const SurfaceCode* state,
                                     const Stabilizer* stabilizer) {
    if (!state || !stabilizer) return false;
    
    // Check number of qubits
    if (stabilizer->num_qubits < 2 || stabilizer->num_qubits > 4) {
        return false;
    }
    
    // Verify qubit indices
    for (size_t i = 0; i < stabilizer->num_qubits; i++) {
        if (stabilizer->qubits[i] >= state->config.width * state->config.height) {
            return false;
        }
    }
    
    return true;
}

void apply_stabilizer_corrections(SurfaceCode* state, const Stabilizer* stabilizer) {
    if (!state || !stabilizer || !stabilizer->result.needs_correction) {
        qgt_error_log(QGT_ERROR_INVALID_ARGUMENT, "Invalid arguments to apply_stabilizer_corrections");
        return;
    }
    
    qgt_error_t result = QGT_SUCCESS;
    
    // Apply correction operations based on stabilizer type
    switch (stabilizer->type) {
        case STABILIZER_X: {
            // Apply X corrections with error tracking
            for (size_t i = 0; i < stabilizer->num_qubits; i++) {
                size_t qubit_idx = stabilizer->qubits[i];
                result = apply_pauli_x(state, qubit_idx);
                if (result != QGT_SUCCESS) {
                    qgt_error_log(result, "Failed to apply X correction");
                    return;
                }
                update_error_metrics(state, qubit_idx, GATE_X, 1.0 - stabilizer->result.confidence);
            }
            break;
        }
            
        case STABILIZER_Z: {
            // Apply Z corrections with error tracking
            for (size_t i = 0; i < stabilizer->num_qubits; i++) {
                size_t qubit_idx = stabilizer->qubits[i];
                result = apply_pauli_z(state, qubit_idx);
                if (result != QGT_SUCCESS) {
                    qgt_error_log(result, "Failed to apply Z correction");
                    return;
                }
                update_error_metrics(state, qubit_idx, GATE_Z, 1.0 - stabilizer->result.confidence);
            }
            break;
        }
            
        case STABILIZER_Y: {
            // Apply Y corrections (composite X and Z)
            for (size_t i = 0; i < stabilizer->num_qubits; i++) {
                size_t qubit_idx = stabilizer->qubits[i];
                result = apply_pauli_x(state, qubit_idx);
                if (result != QGT_SUCCESS) {
                    qgt_error_log(result, "Failed to apply X part of Y correction");
                    return;
                }
                result = apply_pauli_z(state, qubit_idx);
                if (result != QGT_SUCCESS) {
                    qgt_error_log(result, "Failed to apply Z part of Y correction");
                    return;
                }
                update_error_metrics(state, qubit_idx, GATE_Y, 1.0 - stabilizer->result.confidence);
            }
            break;
        }
            
        default:
            qgt_error_log(QGT_ERROR_INVALID_ARGUMENT, "Unknown stabilizer type");
            return;
    }
    
    // Update error rates after corrections
    calculate_stabilizer_weights(state);
    update_logical_error_rates(state);
}

static void setup_standard_lattice(SurfaceCode* state) {
    if (!state) return;
    // Standard surface code lattice setup
    state->config.width = state->config.distance * 2 + 1;
    state->config.height = state->config.distance * 2 + 1;
}

static void setup_rotated_lattice(SurfaceCode* state) {
    if (!state) return;
    
    // Rotated surface code has distance d stabilizers arranged in a d x d grid
    // Each stabilizer alternates between X and Z type
    state->config.width = state->config.distance;
    state->config.height = state->config.distance;
    
    // Initialize stabilizers in rotated configuration
    for (size_t i = 0; i < state->config.height; i++) {
        for (size_t j = 0; j < state->config.width; j++) {
            Stabilizer* stabilizer = &state->stabilizers[state->num_stabilizers++];
            
            // Alternate X and Z stabilizers in checkerboard pattern
            stabilizer->type = ((i + j) % 2 == 0) ? STABILIZER_X : STABILIZER_Z;
            
            // Add surrounding qubits in rotated configuration
            stabilizer->qubits[0] = i * state->config.width + j;  // Center
            stabilizer->qubits[1] = ((i+1) % state->config.height) * state->config.width + j;  // North
            stabilizer->qubits[2] = i * state->config.width + ((j+1) % state->config.width);   // East
            stabilizer->qubits[3] = ((i > 0 ? i-1 : state->config.height-1)) * state->config.width + j; // South
            stabilizer->qubits[4] = i * state->config.width + (j > 0 ? j-1 : state->config.width-1);    // West
            stabilizer->num_qubits = 5;
        }
    }
}

static void setup_heavy_hex_lattice(SurfaceCode* state) {
    if (!state) return;
    
    // Heavy hexagonal lattice has additional qubits at vertices
    // This creates a hexagonal pattern with alternating X and Z stabilizers
    state->config.width = state->config.distance * 3;  // 3 qubits per hex cell
    state->config.height = state->config.distance * 2; // 2 rows per hex cell
    
    // Add stabilizers in hexagonal pattern
    for (size_t i = 0; i < state->config.height; i += 2) {
        for (size_t j = 0; j < state->config.width; j += 3) {
            // Add X stabilizer at center of hex
            Stabilizer* x_stabilizer = &state->stabilizers[state->num_stabilizers++];
            x_stabilizer->type = STABILIZER_X;
            x_stabilizer->qubits[0] = i * state->config.width + j;           // Center
            x_stabilizer->qubits[1] = i * state->config.width + (j+1);       // Right
            x_stabilizer->qubits[2] = (i+1) * state->config.width + j;       // Bottom
            x_stabilizer->qubits[3] = i * state->config.width + (j > 0 ? j-1 : state->config.width-1); // Left
            x_stabilizer->num_qubits = 4;
            
            // Add Z stabilizers at vertices
            if (j + 2 < state->config.width) {
                Stabilizer* z_stabilizer = &state->stabilizers[state->num_stabilizers++];
                z_stabilizer->type = STABILIZER_Z;
                z_stabilizer->qubits[0] = i * state->config.width + (j+1);     // Left
                z_stabilizer->qubits[1] = i * state->config.width + (j+2);     // Right
                z_stabilizer->qubits[2] = (i+1) * state->config.width + (j+1); // Bottom
                z_stabilizer->num_qubits = 3;
            }
        }
    }
}

static void setup_floquet_lattice(SurfaceCode* state) {
    if (!state) return;
    
    // Floquet surface code uses time-dependent stabilizer measurements
    // This creates a 3D space-time lattice
    state->config.width = state->config.distance * 2;
    state->config.height = state->config.distance * 2;
    state->config.time_steps = 4; // Number of measurement cycles
    
    // Initialize stabilizers for each time step
    for (size_t t = 0; t < state->config.time_steps; t++) {
        // Add stabilizers in alternating pattern
        for (size_t i = 1; i < state->config.height; i += 2) {
            for (size_t j = 1; j < state->config.width; j += 2) {
                Stabilizer* stabilizer = &state->stabilizers[state->num_stabilizers++];
                
                // Alternate between X and Z stabilizers in time
                stabilizer->type = (t % 2 == 0) ? STABILIZER_X : STABILIZER_Z;
                
                // Add surrounding qubits
                stabilizer->qubits[0] = (i-1) * state->config.width + j;     // Top
                stabilizer->qubits[1] = i * state->config.width + (j-1);     // Left
                stabilizer->qubits[2] = i * state->config.width + (j+1);     // Right
                stabilizer->qubits[3] = (i+1) * state->config.width + j;     // Bottom
                stabilizer->num_qubits = 4;
                
                // Store time step
                stabilizer->time_step = t;
            }
        }
    }
}

static bool validate_lattice_configuration(SurfaceCode* state) {
    if (!state) return false;
    
    // Check lattice dimensions
    if (state->config.width * state->config.height > MAX_SURFACE_SIZE * MAX_SURFACE_SIZE) {
        return false;
    }
    
    // Verify stabilizer placement
    for (size_t i = 0; i < state->num_stabilizers; i++) {
        if (!is_valid_stabilizer_configuration(state, &state->stabilizers[i])) {
            return false;
        }
    }
    
    return true;
}

static void calculate_stabilizer_weights(SurfaceCode* state) {
    if (!state) return;

    // Calculate weights for each stabilizer
    for (size_t i = 0; i < state->num_stabilizers; i++) {
        Stabilizer* stabilizer = &state->stabilizers[i];
        
        // Start with base confidence
        double weight = stabilizer->result.confidence;
        
        // Factor in error history
        size_t error_count = 0;
        for (size_t j = 0; j < stabilizer->num_qubits; j++) {
            size_t qubit_idx = stabilizer->qubits[j];
            error_count += get_error_count(state, qubit_idx);
        }
        
        // Adjust weight based on error history
        if (error_count > 0) {
            weight *= exp(-error_count * state->config.error_weight_factor);
        }
        
        // Factor in spatial correlations
        for (size_t j = 0; j < state->num_stabilizers; j++) {
            if (i == j) continue;
            
            Stabilizer* other = &state->stabilizers[j];
            if (are_neighbors(stabilizer, other)) {
                weight *= (1.0 - fabs(other->error_rate));
            }
        }
        
        // Update stabilizer weight
        stabilizer->weight = weight;
    }
}

static void update_stabilizer_correlations(SurfaceCode* state) {
    if (!state) return;

    // Update correlations between stabilizers
    for (size_t i = 0; i < state->num_stabilizers; i++) {
        Stabilizer* stabilizer1 = &state->stabilizers[i];
        
        for (size_t j = i + 1; j < state->num_stabilizers; j++) {
            Stabilizer* stabilizer2 = &state->stabilizers[j];
            
            // Skip if not neighbors
            if (!are_neighbors(stabilizer1, stabilizer2)) continue;
            
            // Calculate correlation based on measurement results
            double correlation = 0.0;
            size_t shared_errors = 0;
            
            for (size_t k = 0; k < stabilizer1->num_qubits; k++) {
                for (size_t l = 0; l < stabilizer2->num_qubits; l++) {
                    if (stabilizer1->qubits[k] == stabilizer2->qubits[l]) {
                        correlation += stabilizer1->result.value * 
                                     stabilizer2->result.value;
                        if (stabilizer1->result.needs_correction &&
                            stabilizer2->result.needs_correction) {
                            shared_errors++;
                        }
                    }
                }
            }
            
            // Update correlation based on shared errors
            if (shared_errors > 0) {
                correlation *= (1.0 + shared_errors * state->config.correlation_factor);
            }
            
            // Store correlation
            size_t idx = i * state->num_stabilizers + j;
            state->correlations[idx] = correlation;
            state->correlations[j * state->num_stabilizers + i] = correlation;
        }
    }
}

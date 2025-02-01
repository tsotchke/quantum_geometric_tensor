/**
 * @file anyon_detection.c
 * @brief Implementation of anyon detection and tracking system
 */

#include "quantum_geometric/physics/anyon_detection.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// Helper functions
static bool allocate_grid(AnyonGrid* grid, size_t width, size_t height, size_t depth) {
    if (!grid) return false;

    grid->width = width;
    grid->height = height;
    grid->depth = depth;
    size_t total_cells = width * height * depth;

    grid->cells = calloc(total_cells, sizeof(AnyonCell));
    if (!grid->cells) return false;

    // Initialize all cells
    for (size_t i = 0; i < total_cells; i++) {
        grid->cells[i].type = ANYON_NONE;
        grid->cells[i].charge = 0.0;
        grid->cells[i].velocity[0] = 0.0;
        grid->cells[i].velocity[1] = 0.0;
        grid->cells[i].velocity[2] = 0.0;
        grid->cells[i].confidence = 1.0;
        grid->cells[i].is_fused = false;
    }

    return true;
}

static void free_grid(AnyonGrid* grid) {
    if (grid && grid->cells) {
        free(grid->cells);
        grid->cells = NULL;
    }
}

static bool is_valid_config(const AnyonConfig* config) {
    if (!config) return false;

    return (config->grid_width > 0 &&
            config->grid_height > 0 &&
            config->grid_depth > 0 &&
            config->detection_threshold >= 0.0 &&
            config->detection_threshold <= 1.0 &&
            config->max_movement_speed >= 0.0 &&
            config->charge_threshold >= 0.0);
}

static anyon_type_t detect_anyon_type(const QuantumState* qstate, size_t x, size_t y, size_t z,
                                    double threshold) {
    if (!qstate) return ANYON_NONE;

    size_t idx = (z * qstate->width * qstate->width + y * qstate->width + x) * 2;
    if (idx >= qstate->dimension * 2) return ANYON_NONE;

    // Get amplitudes
    double complex zero_amp = qstate->amplitudes[idx];     // |0⟩ amplitude
    double complex one_amp = qstate->amplitudes[idx + 1];  // |1⟩ amplitude

    // Calculate probabilities
    double p0 = cabs(zero_amp) * cabs(zero_amp);
    double p1 = cabs(one_amp) * cabs(one_amp);
    double phase = carg(one_amp) - carg(zero_amp);

    // Detect anyon type based on state characteristics
    if (p1 > threshold && fabs(phase) < 0.1) {
        return ANYON_X;
    } else if (p0 > threshold && p1 > threshold) {
        return ANYON_Z;
    } else if (p1 > threshold && fabs(phase - M_PI/2) < 0.1) {
        return ANYON_Y;
    }

    return ANYON_NONE;
}

static double calculate_charge(const QuantumState* qstate, size_t x, size_t y, size_t z) {
    if (!qstate) return 0.0;

    size_t idx = (z * qstate->width * qstate->width + y * qstate->width + x) * 2;
    if (idx >= qstate->dimension * 2) return 0.0;

    // Calculate charge based on state amplitudes
    double complex zero_amp = qstate->amplitudes[idx];
    double complex one_amp = qstate->amplitudes[idx + 1];
    
    return cabs(one_amp) * cabs(one_amp);  // Use excitation probability as charge
}

static void update_velocities(AnyonState* state) {
    if (!state || !state->grid) return;

    // Calculate velocities based on position changes
    for (size_t i = 0; i < state->total_anyons; i++) {
        AnyonPosition* prev = &state->last_positions[i];
        
        // Find corresponding anyon in current grid
        for (size_t z = 0; z < state->grid->depth; z++) {
            for (size_t y = 0; y < state->grid->height; y++) {
                for (size_t x = 0; x < state->grid->width; x++) {
                    size_t idx = z * state->grid->height * state->grid->width +
                                y * state->grid->width + x;
                    AnyonCell* cell = &state->grid->cells[idx];
                    
                    if (cell->type == prev->type && !cell->is_fused) {
                        // Update velocity components
                        cell->velocity[0] = (double)((int)x - (int)prev->x);
                        cell->velocity[1] = (double)((int)y - (int)prev->y);
                        cell->velocity[2] = (double)((int)z - (int)prev->z);
                    }
                }
            }
        }
    }
}

static void apply_fusion_rules(AnyonGrid* grid) {
    if (!grid) return;

    // Check for adjacent anyons that can fuse
    for (size_t z = 0; z < grid->depth; z++) {
        for (size_t y = 0; y < grid->height; y++) {
            for (size_t x = 0; x < grid->width; x++) {
                size_t idx = z * grid->height * grid->width + y * grid->width + x;
                AnyonCell* cell = &grid->cells[idx];
                
                if (cell->type == ANYON_NONE || cell->is_fused) continue;

                // Check neighbors
                for (int dz = -1; dz <= 1; dz++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dx == 0 && dy == 0 && dz == 0) continue;
                            
                            int nx = x + dx;
                            int ny = y + dy;
                            int nz = z + dz;
                            
                            if (nx < 0 || nx >= (int)grid->width ||
                                ny < 0 || ny >= (int)grid->height ||
                                nz < 0 || nz >= (int)grid->depth)
                                continue;

                            size_t nidx = nz * grid->height * grid->width +
                                        ny * grid->width + nx;
                            AnyonCell* neighbor = &grid->cells[nidx];
                            
                            if (neighbor->type == ANYON_NONE || neighbor->is_fused)
                                continue;

                            // Apply fusion rules
                            if (cell->type == neighbor->type) {
                                // Same type anyons annihilate
                                cell->type = ANYON_NONE;
                                neighbor->type = ANYON_NONE;
                                cell->is_fused = true;
                                neighbor->is_fused = true;
                            } else if ((cell->type == ANYON_X && neighbor->type == ANYON_Z) ||
                                     (cell->type == ANYON_Z && neighbor->type == ANYON_X)) {
                                // X and Z fuse to Y
                                cell->type = ANYON_Y;
                                neighbor->type = ANYON_NONE;
                                cell->charge = (cell->charge + neighbor->charge) / 2.0;
                                cell->is_fused = false;
                                neighbor->is_fused = true;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Public functions
bool init_anyon_detection(AnyonState* state, const AnyonConfig* config) {
    if (!state || !config || !is_valid_config(config)) {
        return false;
    }

    // Allocate grid
    state->grid = malloc(sizeof(AnyonGrid));
    if (!state->grid) {
        return false;
    }

    if (!allocate_grid(state->grid, config->grid_width,
                      config->grid_height, config->grid_depth)) {
        free(state->grid);
        return false;
    }

    // Allocate position tracking array (maximum possible anyons)
    size_t max_anyons = config->grid_width * config->grid_height * config->grid_depth;
    state->last_positions = malloc(max_anyons * sizeof(AnyonPosition));
    if (!state->last_positions) {
        free_grid(state->grid);
        free(state->grid);
        return false;
    }

    // Initialize state
    state->measurement_count = 0;
    state->total_anyons = 0;

    return true;
}

void cleanup_anyon_detection(AnyonState* state) {
    if (!state) {
        return;
    }

    if (state->grid) {
        free_grid(state->grid);
        free(state->grid);
    }

    free(state->last_positions);
}

bool detect_and_track_anyons(AnyonState* state, const QuantumState* qstate) {
    if (!state || !state->grid || !qstate) {
        return false;
    }

    // Store previous positions for tracking
    AnyonPosition* prev_positions = NULL;
    if (state->total_anyons > 0) {
        prev_positions = malloc(state->total_anyons * sizeof(AnyonPosition));
        if (prev_positions) {
            memcpy(prev_positions, state->last_positions,
                   state->total_anyons * sizeof(AnyonPosition));
        }
    }

    // Reset anyon count
    state->total_anyons = 0;

    // Detect anyons in each cell
    for (size_t z = 0; z < state->grid->depth; z++) {
        for (size_t y = 0; y < state->grid->height; y++) {
            for (size_t x = 0; x < state->grid->width; x++) {
                size_t idx = z * state->grid->height * state->grid->width +
                            y * state->grid->width + x;
                
                // Detect anyon type
                anyon_type_t type = detect_anyon_type(qstate, x, y, z, 0.1);
                state->grid->cells[idx].type = type;

                if (type != ANYON_NONE) {
                    // Calculate charge
                    state->grid->cells[idx].charge = calculate_charge(qstate, x, y, z);

                    // Store position
                    if (state->total_anyons < state->grid->width *
                                            state->grid->height *
                                            state->grid->depth) {
                        state->last_positions[state->total_anyons].x = x;
                        state->last_positions[state->total_anyons].y = y;
                        state->last_positions[state->total_anyons].z = z;
                        state->last_positions[state->total_anyons].type = type;
                        state->total_anyons++;
                    }
                }
            }
        }
    }

    // Update tracking if we have previous positions
    if (prev_positions) {
        update_velocities(state);
        free(prev_positions);
    }

    // Apply fusion rules
    apply_fusion_rules(state->grid);

    state->measurement_count++;
    return true;
}

size_t count_anyons(const AnyonGrid* grid) {
    if (!grid) return 0;

    size_t count = 0;
    size_t total_cells = grid->width * grid->height * grid->depth;

    for (size_t i = 0; i < total_cells; i++) {
        if (grid->cells[i].type != ANYON_NONE && !grid->cells[i].is_fused) {
            count++;
        }
    }

    return count;
}

bool get_anyon_positions(const AnyonGrid* grid, AnyonPosition* positions) {
    if (!grid || !positions) return false;

    size_t pos = 0;
    for (size_t z = 0; z < grid->depth; z++) {
        for (size_t y = 0; y < grid->height; y++) {
            for (size_t x = 0; x < grid->width; x++) {
                size_t idx = z * grid->height * grid->width +
                            y * grid->width + x;
                
                if (grid->cells[idx].type != ANYON_NONE &&
                    !grid->cells[idx].is_fused) {
                    positions[pos].x = x;
                    positions[pos].y = y;
                    positions[pos].z = z;
                    positions[pos].type = grid->cells[idx].type;
                    pos++;
                }
            }
        }
    }

    return true;
}

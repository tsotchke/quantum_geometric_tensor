/**
 * @file anyon_detection.c
 * @brief Implementation of anyon detection and tracking system
 */

#include "quantum_geometric/physics/anyon_detection.h"
#include "quantum_geometric/core/quantum_complex.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

static anyon_type_t detect_anyon_type(const quantum_state_t* qstate, size_t x, size_t y, size_t z,
                                    size_t grid_width, size_t grid_height, double threshold) {
    if (!qstate || !qstate->coordinates) return ANYON_NONE;

    // Calculate linear index using grid dimensions
    size_t idx = (z * grid_height * grid_width + y * grid_width + x) * 2;
    if (idx + 1 >= qstate->dimension) return ANYON_NONE;

    // Get amplitudes as ComplexFloat
    ComplexFloat zero_amp = qstate->coordinates[idx];     // |0⟩ amplitude
    ComplexFloat one_amp = qstate->coordinates[idx + 1];  // |1⟩ amplitude

    // Calculate probabilities using ComplexFloat accessors
    float p0 = complex_float_abs_squared(zero_amp);
    float p1 = complex_float_abs_squared(one_amp);
    float phase = complex_float_arg(one_amp) - complex_float_arg(zero_amp);

    // Detect anyon type based on state characteristics
    if (p1 > threshold && fabsf(phase) < 0.1f) {
        return ANYON_X;
    } else if (p0 > threshold && p1 > threshold) {
        return ANYON_Z;
    } else if (p1 > threshold && fabsf(phase - (float)M_PI/2.0f) < 0.1f) {
        return ANYON_Y;
    }

    return ANYON_NONE;
}

static double calculate_charge(const quantum_state_t* qstate, size_t x, size_t y, size_t z,
                              size_t grid_width, size_t grid_height) {
    if (!qstate || !qstate->coordinates) return 0.0;

    // Calculate linear index using grid dimensions
    size_t idx = (z * grid_height * grid_width + y * grid_width + x) * 2;
    if (idx + 1 >= qstate->dimension) return 0.0;

    // Calculate charge based on state amplitudes using ComplexFloat
    ComplexFloat one_amp = qstate->coordinates[idx + 1];

    return (double)complex_float_abs_squared(one_amp);  // Use excitation probability as charge
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

// init_anyon_detection() - Canonical implementation in anyon_operations.c
// (removed: this version incorrectly set confidence=1.0, canonical uses confidence=0.0)

// cleanup_anyon_detection() - Canonical implementation in anyon_operations.c
// (removed due to potential double-free bug with free_grid followed by free)

bool detect_and_track_anyons(AnyonState* state, const quantum_state* qstate) {
    if (!state || !state->grid || !qstate) {
        return false;
    }

    // Cast to quantum_state_t* for internal use
    const quantum_state_t* qs = (const quantum_state_t*)qstate;

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

    // Get grid dimensions
    size_t grid_width = state->grid->width;
    size_t grid_height = state->grid->height;

    // Detect anyons in each cell
    for (size_t z = 0; z < state->grid->depth; z++) {
        for (size_t y = 0; y < grid_height; y++) {
            for (size_t x = 0; x < grid_width; x++) {
                size_t idx = z * grid_height * grid_width +
                            y * grid_width + x;

                // Detect anyon type with grid dimensions
                anyon_type_t type = detect_anyon_type(qs, x, y, z,
                                                      grid_width, grid_height, 0.1);
                state->grid->cells[idx].type = type;

                if (type != ANYON_NONE) {
                    // Calculate charge with grid dimensions
                    state->grid->cells[idx].charge = calculate_charge(qs, x, y, z,
                                                                      grid_width, grid_height);

                    // Store position
                    if (state->total_anyons < grid_width * grid_height * state->grid->depth) {
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

/**
 * @file anyon_operations.c
 * @brief Implementation of anyon braiding and fusion operations
 */

#include "quantum_geometric/physics/anyon_detection.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include <stdlib.h>
#include <math.h>

// Helper function to calculate optimal braiding path
static bool calculate_braiding_path(const AnyonPair* pair, 
                                  const BraidingConfig* config,
                                  AnyonPosition** path,
                                  size_t* path_length) {
    if (!pair || !config || !path || !path_length) {
        return false;
    }

    // Calculate distance between anyons
    double distance = calculate_distance(&pair->anyon1->position, 
                                      &pair->anyon2->position);
    
    if (distance < config->min_separation) {
        return false;
    }

    // Calculate number of steps needed
    *path_length = (size_t)(distance * config->braiding_steps);
    *path = calloc(*path_length, sizeof(AnyonPosition));
    
    if (!*path) {
        return false;
    }

    // Calculate intermediate positions for smooth braiding
    double dx = (double)(pair->anyon2->position.x - pair->anyon1->position.x);
    double dy = (double)(pair->anyon2->position.y - pair->anyon1->position.y);
    double dz = (double)(pair->anyon2->position.z - pair->anyon1->position.z);

    for (size_t i = 0; i < *path_length; i++) {
        double t = (double)i / (double)(*path_length - 1);
        double angle = 2.0 * M_PI * t;
        
        // Calculate position along braiding path
        (*path)[i].x = (size_t)(pair->anyon1->position.x + dx * t + 
                               config->min_separation * cos(angle));
        (*path)[i].y = (size_t)(pair->anyon1->position.y + dy * t + 
                               config->min_separation * sin(angle));
        (*path)[i].z = (size_t)(pair->anyon1->position.z + dz * t);
        
        // Set stability based on position
        (*path)[i].stability = 1.0;
    }

    return true;
}

bool braid_anyons(quantum_state* state, AnyonPair* pair, const BraidingConfig* config) {
    if (!state || !pair || !config || !pair->anyon1 || !pair->anyon2) {
        return false;
    }

    // Calculate braiding path
    AnyonPosition* path = NULL;
    size_t path_length = 0;
    
    if (!calculate_braiding_path(pair, config, &path, &path_length)) {
        return false;
    }

    // Initialize braiding phase
    complex double total_phase = 0.0;
    bool success = true;

    // Perform braiding operation
    for (size_t i = 0; i < path_length && success; i++) {
        // Move first anyon along path
        AnyonPosition prev_pos = pair->anyon1->position;
        pair->anyon1->position = path[i];

        // Calculate and accumulate braiding phase
        complex double step_phase = calculate_braiding_phase_step(state, 
                                                               pair->anyon1,
                                                               pair->anyon2,
                                                               &prev_pos);
        total_phase += step_phase;

        // Verify topological protection if required
        if (config->verify_topology) {
            success = verify_topological_protection(state, 
                                                 (const Anyon*[]){pair->anyon1, pair->anyon2},
                                                 2);
        }

        // Update quantum state
        if (success) {
            success = apply_braiding_operation(state, pair->anyon1, pair->anyon2, step_phase);
        }
    }

    // Store final braiding phase
    if (success) {
        pair->braiding_phase = carg(total_phase);
        
        // Update interaction strength based on braiding result
        pair->interaction_strength = cabs(total_phase) / (2.0 * M_PI);
    }

    free(path);
    return success;
}

FusionOutcome fuse_anyons(quantum_state* state, const AnyonPair* pair, const FusionConfig* config) {
    FusionOutcome outcome = {0};
    
    if (!state || !pair || !config || !pair->anyon1 || !pair->anyon2) {
        return outcome;
    }

    // Check if fusion is energetically allowed
    double fusion_energy = calculate_fusion_energy(pair->anyon1, pair->anyon2);
    if (fusion_energy > config->energy_threshold) {
        return outcome;
    }

    // Calculate fusion probability based on anyon types and charges
    outcome.probability = calculate_fusion_probability(pair->anyon1, pair->anyon2);
    
    // Only proceed if probability exceeds coherence requirement
    if (outcome.probability < config->coherence_requirement) {
        return outcome;
    }

    // Determine fusion outcome type
    outcome.result_type = determine_fusion_type(pair->anyon1, pair->anyon2);
    
    // Calculate resulting charge
    outcome.result_charge = calculate_fusion_charge(pair->anyon1->charge,
                                                  pair->anyon2->charge);
    
    // Calculate energy change
    outcome.energy_delta = calculate_energy_difference(state,
                                                     pair->anyon1,
                                                     pair->anyon2,
                                                     &outcome.result_charge);

    // Attempt fusion operation
    bool fusion_success = false;
    for (size_t attempt = 0; attempt < config->fusion_attempts && !fusion_success; attempt++) {
        fusion_success = apply_fusion_operation(state,
                                             pair->anyon1,
                                             pair->anyon2,
                                             &outcome);
        
        if (fusion_success && config->track_statistics) {
            update_fusion_statistics(outcome.result_type,
                                   outcome.probability,
                                   outcome.energy_delta);
        }
    }

    return outcome;
}

double calculate_interaction_energy(const Anyon* anyon1, const Anyon* anyon2) {
    if (!anyon1 || !anyon2) {
        return 0.0;
    }

    // Calculate base interaction energy
    double distance = calculate_distance(&anyon1->position, &anyon2->position);
    if (distance < 1e-6) {
        return INFINITY;
    }

    // Calculate charge-dependent interaction
    double charge_interaction = calculate_charge_interaction(anyon1->charge,
                                                          anyon2->charge);

    // Calculate type-dependent interaction
    double type_interaction = calculate_type_interaction(anyon1->type,
                                                       anyon2->type);

    // Combine all interaction terms
    return (charge_interaction * type_interaction) / distance;
}

double calculate_braiding_phase(const AnyonPair* pair) {
    if (!pair || !pair->anyon1 || !pair->anyon2) {
        return 0.0;
    }

    // Calculate statistical angle based on anyon types
    double statistical_angle = calculate_statistical_angle(pair->anyon1->type,
                                                         pair->anyon2->type);

    // Modify by charge interaction
    double charge_factor = calculate_charge_phase_factor(pair->anyon1->charge,
                                                       pair->anyon2->charge);

    // Include topological correction
    double topological_factor = calculate_topological_correction(pair->anyon1,
                                                               pair->anyon2);

    return statistical_angle * charge_factor * topological_factor;
}

bool check_fusion_rules(const Anyon* anyon1, const Anyon* anyon2) {
    if (!anyon1 || !anyon2) {
        return false;
    }

    // Check type compatibility
    if (!are_types_compatible(anyon1->type, anyon2->type)) {
        return false;
    }

    // Check charge conservation
    if (!verify_charge_conservation(anyon1->charge, anyon2->charge)) {
        return false;
    }

    // Check topological constraints
    if (!verify_topological_rules(anyon1, anyon2)) {
        return false;
    }

    return true;
}

void predict_fusion_outcomes(const AnyonPair* pair, 
                           FusionOutcome* outcomes,
                           size_t* num_outcomes) {
    if (!pair || !outcomes || !num_outcomes || !pair->anyon1 || !pair->anyon2) {
        if (num_outcomes) {
            *num_outcomes = 0;
        }
        return;
    }

    // Get possible fusion channels
    size_t max_channels = *num_outcomes;
    *num_outcomes = 0;

    // Calculate possible outcomes based on anyon types
    calculate_fusion_channels(pair->anyon1->type,
                            pair->anyon2->type,
                            outcomes,
                            num_outcomes,
                            max_channels);

    // Calculate probabilities and energies for each outcome
    for (size_t i = 0; i < *num_outcomes; i++) {
        outcomes[i].probability = calculate_channel_probability(pair, &outcomes[i]);
        outcomes[i].energy_delta = calculate_channel_energy(pair, &outcomes[i]);
    }
}

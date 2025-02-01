#include "quantum_geometric/core/quantum_geometric_simulation.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Create geometric simulation
qgt_error_t geometric_create_simulation(quantum_geometric_simulation_t** simulation,
                                      geometric_simulation_type_t type,
                                      size_t dimension) {
    QGT_CHECK_NULL(simulation);
    QGT_CHECK_ARGUMENT(dimension > 0 && dimension <= QGT_MAX_DIMENSIONS);
    
    *simulation = (quantum_geometric_simulation_t*)calloc(1, sizeof(quantum_geometric_simulation_t));
    if (!*simulation) {
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    
    // Allocate simulation state
    size_t size = dimension * sizeof(ComplexFloat);
    (*simulation)->state = (ComplexFloat*)malloc(size);
    if (!(*simulation)->state) {
        free(*simulation);
        return QGT_ERROR_ALLOCATION_FAILED;
    }
    
    (*simulation)->type = type;
    (*simulation)->dimension = dimension;
    (*simulation)->time = 0.0f;
    (*simulation)->dt = QGT_DEFAULT_TIME_STEP;
    
    // Initialize state to ground state
    memset((*simulation)->state, 0, size);
    (*simulation)->state[0] = complex_float_create(1.0f, 0.0f);
    
    return QGT_SUCCESS;
}

// Destroy geometric simulation
void geometric_destroy_simulation(quantum_geometric_simulation_t* simulation) {
    if (simulation) {
        free(simulation->state);
        free(simulation);
    }
}

// Clone geometric simulation
qgt_error_t geometric_clone_simulation(quantum_geometric_simulation_t** dest,
                                     const quantum_geometric_simulation_t* src) {
    QGT_CHECK_NULL(dest);
    QGT_CHECK_NULL(src);
    
    qgt_error_t err = geometric_create_simulation(dest, src->type, src->dimension);
    if (err != QGT_SUCCESS) {
        return err;
    }
    
    size_t size = src->dimension * sizeof(ComplexFloat);
    memcpy((*dest)->state, src->state, size);
    (*dest)->time = src->time;
    (*dest)->dt = src->dt;
    
    return QGT_SUCCESS;
}

// Step geometric simulation
qgt_error_t geometric_step_simulation(quantum_geometric_simulation_t* simulation,
                                    const quantum_geometric_metric_t* metric,
                                    const quantum_geometric_connection_t* connection) {
    QGT_CHECK_NULL(simulation);
    QGT_CHECK_NULL(metric);
    QGT_CHECK_NULL(connection);
    
    if (simulation->dimension != metric->dimension ||
        simulation->dimension != connection->dimension) {
        return QGT_ERROR_INVALID_PARAMETER;
    }
    
    size_t dim = simulation->dimension;
    
    // Perform simulation step based on type
    switch (simulation->type) {
        case GEOMETRIC_SIMULATION_SCHRODINGER:
            // Simple Euler step for Schrödinger equation
            ComplexFloat* new_state = (ComplexFloat*)malloc(dim * sizeof(ComplexFloat));
            if (!new_state) {
                return QGT_ERROR_ALLOCATION_FAILED;
            }
            
            // Compute evolution using metric and connection
            for (size_t i = 0; i < dim; i++) {
                ComplexFloat sum = COMPLEX_FLOAT_ZERO;
                
                for (size_t j = 0; j < dim; j++) {
                    ComplexFloat metric_term = metric->components[i * dim + j];
                    ComplexFloat connection_term = connection->coefficients[(i * dim + j) * dim];
                    
                    ComplexFloat term = complex_float_multiply(
                        complex_float_add(metric_term, connection_term),
                        simulation->state[j]
                    );
                    sum = complex_float_add(sum, term);
                }
                
                // Apply time evolution
                new_state[i] = complex_float_multiply(
                    complex_float_create(0.0f, -simulation->dt),
                    sum
                );
            }
            
            // Update state
            memcpy(simulation->state, new_state, dim * sizeof(ComplexFloat));
            free(new_state);
            break;
            
        default:
            return QGT_ERROR_NOT_IMPLEMENTED;
    }
    
    simulation->time += simulation->dt;
    
    return QGT_SUCCESS;
}

// Get simulation observables
qgt_error_t geometric_get_observables(const quantum_geometric_simulation_t* simulation,
                                    simulation_observables_t* observables) {
    QGT_CHECK_NULL(simulation);
    QGT_CHECK_NULL(observables);
    
    // Compute basic observables
    observables->time = simulation->time;
    
    // Compute norm
    observables->norm = 0.0f;
    for (size_t i = 0; i < simulation->dimension; i++) {
        observables->norm += complex_float_abs_squared(simulation->state[i]);
    }
    observables->norm = sqrtf(observables->norm);
    
    // Compute energy (simplified)
    observables->energy = 0.0f;
    for (size_t i = 0; i < simulation->dimension; i++) {
        observables->energy += complex_float_abs_squared(simulation->state[i]) * i;
    }
    
    return QGT_SUCCESS;
}

#ifndef QUANTUM_GEOMETRIC_SIMULATION_H
#define QUANTUM_GEOMETRIC_SIMULATION_H

#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_metric.h"
#include "quantum_geometric/core/quantum_geometric_connection.h"

// Simulation types
typedef enum {
    GEOMETRIC_SIMULATION_NONE = 0,
    GEOMETRIC_SIMULATION_SCHRODINGER = 1,
    GEOMETRIC_SIMULATION_HEISENBERG = 2,
    GEOMETRIC_SIMULATION_INTERACTION = 3
} geometric_simulation_type_t;

// Simulation observables structure
typedef struct {
    float time;
    float norm;
    float energy;
} simulation_observables_t;

// Create geometric simulation
qgt_error_t geometric_create_simulation(quantum_geometric_simulation_t** simulation,
                                      geometric_simulation_type_t type,
                                      size_t dimension);

// Destroy geometric simulation
void geometric_destroy_simulation(quantum_geometric_simulation_t* simulation);

// Clone geometric simulation
qgt_error_t geometric_clone_simulation(quantum_geometric_simulation_t** dest,
                                     const quantum_geometric_simulation_t* src);

// Step geometric simulation
qgt_error_t geometric_step_simulation(quantum_geometric_simulation_t* simulation,
                                    const quantum_geometric_metric_t* metric,
                                    const quantum_geometric_connection_t* connection);

// Get simulation observables
qgt_error_t geometric_get_observables(const quantum_geometric_simulation_t* simulation,
                                    simulation_observables_t* observables);

#endif // QUANTUM_GEOMETRIC_SIMULATION_H

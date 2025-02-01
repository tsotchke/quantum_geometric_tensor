#include "quantum_geometric/physics/quantum_field_calculations.h"
#include "quantum_geometric/core/tensor_operations.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

// Transform field components at a spacetime point
void transform_field_components(
    Tensor* field,
    const Tensor* transformation,
    size_t t,
    size_t x,
    size_t y,
    size_t z) {
    
    size_t num_components = field->dims[4];
    
    // Get field components at point
    complex double* components = malloc(
        num_components * sizeof(complex double)
    );
    
    for (size_t i = 0; i < num_components; i++) {
        size_t idx = ((t * field->dims[1] + x) * field->dims[2] + y)
                    * field->dims[3] + z;
        components[i] = field->data[idx * field->dims[4] + i];
    }
    
    // Apply transformation
    complex double* transformed = malloc(
        num_components * sizeof(complex double)
    );
    
    for (size_t i = 0; i < num_components; i++) {
        transformed[i] = 0;
        for (size_t j = 0; j < num_components; j++) {
            size_t trans_idx = i * num_components + j;
            transformed[i] += transformation->data[trans_idx] * components[j];
        }
    }
    
    // Update field
    for (size_t i = 0; i < num_components; i++) {
        size_t idx = ((t * field->dims[1] + x) * field->dims[2] + y)
                    * field->dims[3] + z;
        field->data[idx * field->dims[4] + i] = transformed[i];
    }
    
    free(components);
    free(transformed);
}

// Transform gauge field
void transform_gauge_field(
    Tensor* gauge_field,
    const Tensor* transformation) {
    
    size_t num_components = gauge_field->dims[5];
    size_t num_generators = gauge_field->dims[4];
    
    #pragma omp parallel for collapse(4)
    for (size_t t = 0; t < gauge_field->dims[0]; t++) {
        for (size_t x = 0; x < gauge_field->dims[1]; x++) {
            for (size_t y = 0; y < gauge_field->dims[2]; y++) {
                for (size_t z = 0; z < gauge_field->dims[3]; z++) {
                    // Transform each generator
                    for (size_t g = 0; g < num_generators; g++) {
                        transform_generator(
                            gauge_field,
                            transformation,
                            t, x, y, z, g
                        );
                    }
                }
            }
        }
    }
}

// Calculate kinetic terms
void calculate_kinetic_terms(
    const QuantumField* field,
    Tensor* equations) {
    
    size_t n = field->field_tensor->dims[4];
    
    #pragma omp parallel for collapse(4)
    for (size_t t = 0; t < field->field_tensor->dims[0]; t++) {
        for (size_t x = 0; x < field->field_tensor->dims[1]; x++) {
            for (size_t y = 0; y < field->field_tensor->dims[2]; y++) {
                for (size_t z = 0; z < field->field_tensor->dims[3]; z++) {
                    // Calculate derivatives
                    complex double* derivatives = calculate_derivatives(
                        field->field_tensor,
                        t, x, y, z
                    );
                    
                    // Contract with metric
                    for (size_t i = 0; i < n; i++) {
                        complex double sum = 0;
                        for (size_t mu = 0; mu < QG_SPACETIME_DIMS; mu++) {
                            for (size_t nu = 0; nu < QG_SPACETIME_DIMS; nu++) {
                                sum += field->metric[mu * QG_SPACETIME_DIMS + nu] *
                                      derivatives[mu * n + i] *
                                      conj(derivatives[nu * n + i]);
                            }
                        }
                        
                        size_t idx = ((t * field->field_tensor->dims[1] + x) *
                                    field->field_tensor->dims[2] + y) *
                                    field->field_tensor->dims[3] + z;
                        equations->data[idx * n + i] = sum;
                    }
                    
                    free(derivatives);
                }
            }
        }
    }
}

// Add mass terms
void add_mass_terms(
    const QuantumField* field,
    Tensor* equations) {
    
    size_t n = field->field_tensor->dims[4];
    double mass_squared = field->mass * field->mass;
    
    #pragma omp parallel for collapse(4)
    for (size_t t = 0; t < field->field_tensor->dims[0]; t++) {
        for (size_t x = 0; x < field->field_tensor->dims[1]; x++) {
            for (size_t y = 0; y < field->field_tensor->dims[2]; y++) {
                for (size_t z = 0; z < field->field_tensor->dims[3]; z++) {
                    size_t idx = ((t * field->field_tensor->dims[1] + x) *
                                field->field_tensor->dims[2] + y) *
                                field->field_tensor->dims[3] + z;
                    
                    for (size_t i = 0; i < n; i++) {
                        equations->data[idx * n + i] +=
                            mass_squared * field->field_tensor->data[idx * n + i];
                    }
                }
            }
        }
    }
}

// Add interaction terms
void add_interaction_terms(
    const QuantumField* field,
    Tensor* equations) {
    
    size_t n = field->field_tensor->dims[4];
    
    #pragma omp parallel for collapse(4)
    for (size_t t = 0; t < field->field_tensor->dims[0]; t++) {
        for (size_t x = 0; x < field->field_tensor->dims[1]; x++) {
            for (size_t y = 0; y < field->field_tensor->dims[2]; y++) {
                for (size_t z = 0; z < field->field_tensor->dims[3]; z++) {
                    size_t idx = ((t * field->field_tensor->dims[1] + x) *
                                field->field_tensor->dims[2] + y) *
                                field->field_tensor->dims[3] + z;
                    
                    // Calculate field magnitude squared
                    double phi_squared = 0;
                    for (size_t i = 0; i < n; i++) {
                        complex double phi = field->field_tensor->data[idx * n + i];
                        phi_squared += creal(phi * conj(phi));
                    }
                    
                    // Add phi^4 interaction
                    for (size_t i = 0; i < n; i++) {
                        equations->data[idx * n + i] +=
                            field->coupling * phi_squared *
                            field->field_tensor->data[idx * n + i];
                    }
                }
            }
        }
    }
}

// Add gauge coupling
void add_gauge_coupling(
    const QuantumField* field,
    Tensor* equations) {
    
    if (!field->gauge_field) return;
    
    size_t n = field->field_tensor->dims[4];
    
    #pragma omp parallel for collapse(4)
    for (size_t t = 0; t < field->field_tensor->dims[0]; t++) {
        for (size_t x = 0; x < field->field_tensor->dims[1]; x++) {
            for (size_t y = 0; y < field->field_tensor->dims[2]; y++) {
                for (size_t z = 0; z < field->field_tensor->dims[3]; z++) {
                    // Calculate covariant derivatives
                    complex double* cov_derivatives = calculate_covariant_derivatives(
                        field,
                        t, x, y, z
                    );
                    
                    // Add to equations
                    size_t idx = ((t * field->field_tensor->dims[1] + x) *
                                field->field_tensor->dims[2] + y) *
                                field->field_tensor->dims[3] + z;
                    
                    for (size_t i = 0; i < n; i++) {
                        for (size_t mu = 0; mu < QG_SPACETIME_DIMS; mu++) {
                            equations->data[idx * n + i] +=
                                field->field_strength * cov_derivatives[mu * n + i];
                        }
                    }
                    
                    free(cov_derivatives);
                }
            }
        }
    }
}

// Calculate field energies
double calculate_kinetic_energy(const QuantumField* field) {
    double energy = 0.0;
    size_t n = field->field_tensor->dims[4];
    
    #pragma omp parallel for collapse(4) reduction(+:energy)
    for (size_t t = 0; t < field->field_tensor->dims[0]; t++) {
        for (size_t x = 0; x < field->field_tensor->dims[1]; x++) {
            for (size_t y = 0; y < field->field_tensor->dims[2]; y++) {
                for (size_t z = 0; z < field->field_tensor->dims[3]; z++) {
                    size_t idx = ((t * field->field_tensor->dims[1] + x) *
                                field->field_tensor->dims[2] + y) *
                                field->field_tensor->dims[3] + z;
                    
                    for (size_t i = 0; i < n; i++) {
                        complex double pi = field->conjugate_momentum->data[idx * n + i];
                        energy += creal(pi * conj(pi));
                    }
                }
            }
        }
    }
    
    return QG_HALF * energy;
}

double calculate_potential_energy(const QuantumField* field) {
    double energy = 0.0;
    size_t n = field->field_tensor->dims[4];
    
    #pragma omp parallel for collapse(4) reduction(+:energy)
    for (size_t t = 0; t < field->field_tensor->dims[0]; t++) {
        for (size_t x = 0; x < field->field_tensor->dims[1]; x++) {
            for (size_t y = 0; y < field->field_tensor->dims[2]; y++) {
                for (size_t z = 0; z < field->field_tensor->dims[3]; z++) {
                    size_t idx = ((t * field->field_tensor->dims[1] + x) *
                                field->field_tensor->dims[2] + y) *
                                field->field_tensor->dims[3] + z;
                    
                    double phi_squared = 0;
                    for (size_t i = 0; i < n; i++) {
                        complex double phi = field->field_tensor->data[idx * n + i];
                        phi_squared += creal(phi * conj(phi));
                    }
                    
                    energy += QG_HALF * field->mass * field->mass * phi_squared +
                             QG_QUARTER * field->coupling * phi_squared * phi_squared;
                }
            }
        }
    }
    
    return energy;
}

double calculate_gauge_energy(const QuantumField* field) {
    if (!field->gauge_field) return 0.0;
    
    double energy = 0.0;
    size_t n = field->gauge_field->dims[5];
    
    #pragma omp parallel for collapse(4) reduction(+:energy)
    for (size_t t = 0; t < field->gauge_field->dims[0]; t++) {
        for (size_t x = 0; x < field->gauge_field->dims[1]; x++) {
            for (size_t y = 0; y < field->gauge_field->dims[2]; y++) {
                for (size_t z = 0; z < field->gauge_field->dims[3]; z++) {
                    // Calculate field strength tensor
                    complex double* F_munu = calculate_field_strength(
                        field,
                        t, x, y, z
                    );
                    
                    // Calculate energy density
                    for (size_t mu = 0; mu < QG_SPACETIME_DIMS; mu++) {
                        for (size_t nu = 0; nu < QG_SPACETIME_DIMS; nu++) {
                            energy += creal(F_munu[mu * QG_SPACETIME_DIMS + nu] *
                                         conj(F_munu[mu * QG_SPACETIME_DIMS + nu]));
                        }
                    }
                    
                    free(F_munu);
                }
            }
        }
    }
    
    return QG_QUARTER * field->field_strength * field->field_strength * energy;
}

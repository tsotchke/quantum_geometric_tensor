#include "quantum_geometric_tensor_network.h"
#include "quantum_geometric/core/tensor_network_operations.h"
#include "quantum_geometric/core/quantum_geometric_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper functions
static void copy_spin_states(quantum_geometric_tensor* dst,
                           const quantum_geometric_tensor* src) {
    memcpy(dst->spin_system.spin_states,
           src->spin_system.spin_states,
           src->num_spins * sizeof(double complex));
}

static void copy_metric_tensor(quantum_geometric_tensor* dst,
                             const quantum_geometric_tensor* src) {
    memcpy(dst->geometry.metric_tensor,
           src->geometry.metric_tensor,
           src->dimension * src->dimension * sizeof(double));
}

// Conversion functions
PhysicsMLTensor* qgt_to_physicsml_tensor(const quantum_geometric_tensor* qgt) {
    if (!qgt) return NULL;
    
    // Create PhysicsML tensor
    PhysicsMLTensor* pml = physicsml_tensor_create(qgt->dimension,
                                                  qgt->num_spins);
    if (!pml) return NULL;
    
    // Copy spin states
    memcpy(pml->data, qgt->spin_system.spin_states,
           qgt->num_spins * sizeof(double complex));
    
    // Copy geometric data
    pml->metadata.metric = malloc(qgt->dimension * qgt->dimension * sizeof(double));
    if (!pml->metadata.metric) {
        physicsml_tensor_destroy(pml);
        return NULL;
    }
    memcpy(pml->metadata.metric, qgt->geometry.metric_tensor,
           qgt->dimension * qgt->dimension * sizeof(double));
    
    return pml;
}

quantum_geometric_tensor* physicsml_to_qgt_tensor(const PhysicsMLTensor* pml) {
    if (!pml) return NULL;
    
    // Create quantum geometric tensor
    quantum_geometric_tensor* qgt = create_quantum_tensor(pml->dimension,
                                                        pml->num_spins,
                                                        QGT_MEM_HUGE_PAGES);
    if (!qgt) return NULL;
    
    // Copy spin states
    memcpy(qgt->spin_system.spin_states, pml->data,
           pml->num_spins * sizeof(double complex));
    
    // Copy geometric data if available
    if (pml->metadata.metric) {
        memcpy(qgt->geometry.metric_tensor, pml->metadata.metric,
               pml->dimension * pml->dimension * sizeof(double));
    }
    
    return qgt;
}

// Network creation and manipulation
TreeTensorNetwork* create_geometric_network(const quantum_geometric_tensor* qgt,
                                          size_t bond_dim) {
    if (!qgt || bond_dim == 0) return NULL;
    
    // Create tensor network
    TreeTensorNetwork* ttn = ttn_create(qgt->dimension,
                                      qgt->dimension,
                                      bond_dim,
                                      1e-6);
    if (!ttn) return NULL;
    
    // Initialize with quantum geometric data
    PhysicsMLTensor* pml = qgt_to_physicsml_tensor(qgt);
    if (!pml) {
        ttn_destroy(ttn);
        return NULL;
    }
    
    // Apply geometric constraints
    if (apply_geometric_constraints(ttn, qgt) != PHYSICSML_SUCCESS) {
        physicsml_tensor_destroy(pml);
        ttn_destroy(ttn);
        return NULL;
    }
    
    physicsml_tensor_destroy(pml);
    return ttn;
}

quantum_geometric_tensor* extract_geometric_properties(const TreeTensorNetwork* ttn) {
    if (!ttn) return NULL;
    
    // Contract network to get effective tensor
    PhysicsMLTensor* pml = physicsml_contract_network(ttn);
    if (!pml) return NULL;
    
    // Convert to quantum geometric tensor
    quantum_geometric_tensor* qgt = physicsml_to_qgt_tensor(pml);
    
    physicsml_tensor_destroy(pml);
    return qgt;
}

// Constraint application
int apply_physical_constraints(quantum_geometric_tensor* qgt,
                             const PhysicalConstraints* constraints) {
    if (!qgt || !constraints) return QGT_ERROR_INVALID_ARGUMENT;
    
    // Apply energy constraint
    double energy = 0.0;
    for (size_t i = 0; i < qgt->num_spins; i++) {
        energy += creal(conj(qgt->spin_system.spin_states[i]) *
                       qgt->spin_system.spin_states[i]);
    }
    
    double scale = sqrt(constraints->energy_threshold / energy);
    for (size_t i = 0; i < qgt->num_spins; i++) {
        qgt->spin_system.spin_states[i] *= scale;
    }
    
    // Apply symmetry constraints
    for (size_t i = 0; i < qgt->dimension; i++) {
        for (size_t j = i + 1; j < qgt->dimension; j++) {
            size_t ij = i * qgt->dimension + j;
            size_t ji = j * qgt->dimension + i;
            double avg = 0.5 * (qgt->geometry.metric_tensor[ij] +
                              qgt->geometry.metric_tensor[ji]);
            qgt->geometry.metric_tensor[ij] = avg;
            qgt->geometry.metric_tensor[ji] = avg;
        }
    }
    
    return QGT_SUCCESS;
}

int apply_geometric_constraints(TreeTensorNetwork* ttn,
                              const quantum_geometric_tensor* qgt) {
    if (!ttn || !qgt) return PHYSICSML_ERROR_INVALID_ARGUMENT;
    
    // Extract current geometric properties
    quantum_geometric_tensor* current = extract_geometric_properties(ttn);
    if (!current) return PHYSICSML_ERROR_INTERNAL;
    
    // Compute correction to match target geometry
    for (size_t i = 0; i < qgt->dimension * qgt->dimension; i++) {
        double diff = qgt->geometry.metric_tensor[i] -
                     current->geometry.metric_tensor[i];
        if (fabs(diff) > 1e-6) {
            // Apply correction through network optimization
            TTNConfig config = {
                .max_bond_dimension = ttn->config.max_bond_dimension,
                .compression_tolerance = 1e-6,
                .use_gpu = false,
                .use_metal = false,
                .num_threads = 1,
                .cache_size = 1024 * 1024
            };
            ttn_set_config(ttn, &config);
            ttn_optimize_structure(ttn, 1e-6);
        }
    }
    
    free_quantum_tensor(current);
    return PHYSICSML_SUCCESS;
}

// Verification utilities
bool verify_tensor_consistency(const quantum_geometric_tensor* qgt,
                             const PhysicsMLTensor* pml,
                             double tolerance) {
    if (!qgt || !pml) return false;
    
    // Check dimensions match
    if (qgt->dimension != pml->dimension ||
        qgt->num_spins != pml->num_spins) {
        return false;
    }
    
    // Check spin states match
    for (size_t i = 0; i < qgt->num_spins; i++) {
        double complex diff = qgt->spin_system.spin_states[i] -
                            ((double complex*)pml->data)[i];
        if (cabs(diff) > tolerance) return false;
    }
    
    // Check geometric data matches if available
    if (pml->metadata.metric) {
        for (size_t i = 0; i < qgt->dimension * qgt->dimension; i++) {
            if (fabs(qgt->geometry.metric_tensor[i] -
                    pml->metadata.metric[i]) > tolerance) {
                return false;
            }
        }
    }
    
    return true;
}

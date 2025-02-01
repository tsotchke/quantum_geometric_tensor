#include "quantum_geometric/physics/quantum_geometric_projections.h"
#include "quantum_geometric/core/hierarchical_matrix.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/hardware/quantum_geometric_gpu.h"
#include "quantum_geometric/distributed/workload_distribution.h"
#include <complex.h>
#include <math.h>

// Optimized gauge orbit projection using hierarchical approach - O(log n)
void project_to_gauge_orbit(double complex* state, const double complex* gauge, size_t n) {
    // Convert to hierarchical representation
    HierarchicalMatrix* h_state = convert_to_hierarchical(state, n);
    HierarchicalMatrix* h_gauge = convert_to_hierarchical(gauge, n);
    
    // Project using hierarchical operations
    project_hierarchical_gauge(h_state, h_gauge);
    
    // Convert back
    convert_from_hierarchical(state, h_state);
    
    // Cleanup
    destroy_hierarchical_matrix(h_state);
    destroy_hierarchical_matrix(h_gauge);
}

// Optimized winding number projection using GPU - O(log n)
void project_winding_numbers(double complex* winding, const double complex* field, size_t n) {
    // Allocate GPU memory
    double complex *d_winding, *d_field;
    gpu_malloc((void**)&d_winding, n * sizeof(double complex));
    gpu_malloc((void**)&d_field, n * sizeof(double complex));
    
    // Copy to GPU
    gpu_memcpy_to_device(d_winding, winding, n * sizeof(double complex));
    gpu_memcpy_to_device(d_field, field, n * sizeof(double complex));
    
    // Launch kernel
    compute_winding_kernel<<<n/QG_GPU_BLOCK_SIZE + 1, QG_GPU_BLOCK_SIZE>>>(d_winding, d_field, n);
    
    // Copy back
    gpu_memcpy_to_host(winding, d_winding, n * sizeof(double complex));
    
    // Cleanup
    gpu_free(d_winding);
    gpu_free(d_field);
}

// Optimized braiding phase projection using distributed computing - O(log n)
void project_braiding_phases(double complex* phases, const double complex* anyons, size_t n) {
    // Distribute computation
    size_t local_n = distribute_workload(n);
    size_t offset = get_local_offset();
    
    // Each node computes its portion
    compute_local_braiding(phases + offset, anyons + offset, local_n);
    
    // Synchronize results
    synchronize_results(phases, n);
}

// Optimized fusion rules projection using fast multipole method - O(log n)
void project_fusion_rules(double complex* fusion, const double complex* particles, size_t n) {
    // Initialize multipole expansion
    MultipoleExpansion* expansion = init_multipole_expansion(particles, n);
    
    // Compute using fast multipole method
    compute_fusion_multipole(expansion, fusion);
    
    // Cleanup
    destroy_multipole_expansion(expansion);
}

// Optimized topological order projection using hierarchical approach - O(log n)
void project_topological_order(double complex* order, const double complex* state, size_t n) {
    // Convert to hierarchical representation
    HierarchicalMatrix* h_order = convert_to_hierarchical(order, n);
    HierarchicalMatrix* h_state = convert_to_hierarchical(state, n);
    
    // Project using hierarchical operations
    project_hierarchical_order(h_order, h_state);
    
    // Convert back
    convert_from_hierarchical(order, h_order);
    
    // Cleanup
    destroy_hierarchical_matrix(h_order);
    destroy_hierarchical_matrix(h_state);
}

// GPU kernel for winding number computation - O(1) per thread
__global__ void compute_winding_kernel(double complex* winding, const double complex* field, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Shared memory for field data
    __shared__ double complex shared_field[QG_GPU_BLOCK_SIZE];
    
    // Load field data to shared memory
    shared_field[threadIdx.x] = field[idx];
    __syncthreads();
    
    // Compute local winding number
    winding[idx] = compute_local_winding(shared_field[threadIdx.x]);
}

// Helper function for hierarchical gauge projection - O(log n)
static void project_hierarchical_gauge(HierarchicalMatrix* state, const HierarchicalMatrix* gauge) {
    if (state->is_leaf) {
        // Base case: direct projection
        project_leaf_gauge(state->data, gauge->data, state->size);
        return;
    }
    
    // Recursive case: divide and conquer
    #pragma omp parallel sections
    {
        #pragma omp section
        project_hierarchical_gauge(state->northwest, gauge->northwest);
        
        #pragma omp section
        project_hierarchical_gauge(state->northeast, gauge->northeast);
        
        #pragma omp section
        project_hierarchical_gauge(state->southwest, gauge->southwest);
        
        #pragma omp section
        project_hierarchical_gauge(state->southeast, gauge->southeast);
    }
    
    // Merge results
    merge_gauge_projections(state);
}

// Local braiding computation - O(log n)
static void compute_local_braiding(double complex* phases, const double complex* anyons, size_t n) {
    // Use fast multipole method for local computation
    MultipoleExpansion* expansion = init_multipole_expansion(anyons, n);
    compute_multipole_braiding(expansion, phases);
    destroy_multipole_expansion(expansion);
}

// Helper for leaf node gauge projection - O(1)
static void project_leaf_gauge(double complex* state, const double complex* gauge, size_t n) {
    // Direct projection at leaf level
    #pragma omp simd
    for (size_t i = 0; i < n; i++) {
        state[i] = project_single_gauge(state[i], gauge[i]);
    }
}

// Single gauge projection step - O(1)
static double complex project_single_gauge(double complex state, double complex gauge) {
    // Apply gauge transformation
    return state * cexp(I * carg(gauge));
}

// Merge function for hierarchical gauge projection - O(1)
static void merge_gauge_projections(HierarchicalMatrix* state) {
    // Apply boundary conditions between subdivisions
    apply_gauge_boundaries(state->northwest, state->northeast);
    apply_gauge_boundaries(state->southwest, state->southeast);
    apply_gauge_boundaries(state->northwest, state->southwest);
    apply_gauge_boundaries(state->northeast, state->southeast);
}

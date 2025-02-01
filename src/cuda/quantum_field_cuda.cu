#include "quantum_geometric/physics/quantum_field_operations.h"
#include "quantum_geometric/hardware/quantum_geometric_cuda.h"
#include <cuda_runtime.h>

// CUDA parameters
#define BLOCK_SIZE 256
#define MAX_BLOCKS 65535

// CUDA kernels for field operations
__global__ void apply_rotation_kernel(
    QuantumAmplitude* field,
    size_t field_size,
    size_t num_components,
    size_t qubit,
    cuDoubleComplex* rotation) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= field_size) return;
    
    size_t mask = 1ULL << qubit;
    if (idx & mask) {
        // Get field components
        cuDoubleComplex psi_0 = to_cuda_complex(field[idx ^ mask].amplitude);
        cuDoubleComplex psi_1 = to_cuda_complex(field[idx].amplitude);
        
        // Apply rotation
        cuDoubleComplex new_psi_0 = cuCmul(rotation[0], psi_0);
        new_psi_0 = cuCadd(new_psi_0, cuCmul(rotation[1], psi_1));
        
        cuDoubleComplex new_psi_1 = cuCmul(rotation[2], psi_0);
        new_psi_1 = cuCadd(new_psi_1, cuCmul(rotation[3], psi_1));
        
        // Update field
        field[idx ^ mask].amplitude = from_cuda_complex(new_psi_0);
        field[idx].amplitude = from_cuda_complex(new_psi_1);
    }
}

__global__ void calculate_field_energy_kernel(
    QuantumAmplitude* field,
    QuantumAmplitude* momentum,
    size_t field_size,
    size_t num_components,
    double* energy) {
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= field_size) return;
    
    double local_energy = 0.0;
    
    // Kinetic energy
    for (size_t i = 0; i < num_components; i++) {
        cuDoubleComplex pi = to_cuda_complex(momentum[idx * num_components + i].amplitude);
        local_energy += cuCreal(cuCmul(pi, cuConj(pi)));
    }
    
    // Potential energy
    double phi_squared = 0.0;
    for (size_t i = 0; i < num_components; i++) {
        cuDoubleComplex phi = to_cuda_complex(field[idx * num_components + i].amplitude);
        phi_squared += cuCreal(cuCmul(phi, cuConj(phi)));
    }
    
    local_energy += 0.5 * phi_squared;
    
    // Atomic add to total energy
    atomicAdd(energy, local_energy);
}

// CUDA wrapper functions
extern "C" {

int apply_rotation_cuda(
    QuantumField* field,
    size_t qubit,
    double theta,
    double phi) {
    
    // Allocate device memory
    QuantumAmplitude* d_field;
    size_t field_size = field->field_tensor->size * sizeof(QuantumAmplitude);
    cudaMalloc(&d_field, field_size);
    cudaMemcpy(d_field, field->field_tensor->data, field_size, cudaMemcpyHostToDevice);
    
    // Create rotation matrix
    cuDoubleComplex rotation[4];
    rotation[0] = make_cuDoubleComplex(cos(theta/2), 0);
    rotation[1] = make_cuDoubleComplex(-sin(theta/2) * cos(phi), -sin(theta/2) * sin(phi));
    rotation[2] = make_cuDoubleComplex(sin(theta/2) * cos(phi), sin(theta/2) * sin(phi));
    rotation[3] = make_cuDoubleComplex(cos(theta/2), 0);
    
    cuDoubleComplex* d_rotation;
    cudaMalloc(&d_rotation, 4 * sizeof(cuDoubleComplex));
    cudaMemcpy(d_rotation, rotation, 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    
    // Launch kernel
    size_t num_blocks = (field->field_tensor->size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > MAX_BLOCKS) num_blocks = MAX_BLOCKS;
    
    apply_rotation_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_field,
        field->field_tensor->size,
        field->field_tensor->dims[4],
        qubit,
        d_rotation
    );
    
    // Copy result back
    cudaMemcpy(field->field_tensor->data, d_field, field_size, cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_field);
    cudaFree(d_rotation);
    
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

double calculate_field_energy_cuda(const QuantumField* field) {
    // Allocate device memory
    QuantumAmplitude* d_field;
    QuantumAmplitude* d_momentum;
    double* d_energy;
    
    size_t field_size = field->field_tensor->size * sizeof(QuantumAmplitude);
    cudaMalloc(&d_field, field_size);
    cudaMalloc(&d_momentum, field_size);
    cudaMalloc(&d_energy, sizeof(double));
    
    cudaMemcpy(d_field, field->field_tensor->data, field_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_momentum, field->conjugate_momentum->data, field_size, cudaMemcpyHostToDevice);
    cudaMemset(d_energy, 0, sizeof(double));
    
    // Launch kernel
    size_t num_blocks = (field->field_tensor->size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > MAX_BLOCKS) num_blocks = MAX_BLOCKS;
    
    calculate_field_energy_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_field,
        d_momentum,
        field->field_tensor->size,
        field->field_tensor->dims[4],
        d_energy
    );
    
    // Get result
    double energy;
    cudaMemcpy(&energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_field);
    cudaFree(d_momentum);
    cudaFree(d_energy);
    
    return energy;
}

} // extern "C"

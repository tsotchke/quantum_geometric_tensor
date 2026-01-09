/**
 * @file quantum_field_cuda.cu
 * @brief Full CUDA GPU backend for quantum field operations
 *
 * Production-quality GPU-accelerated quantum field operations
 * using NVIDIA CUDA. Implements Klein-Gordon field theory
 * calculations with mass and interaction terms.
 */

#include "quantum_geometric/physics/quantum_field_calculations.h"
#include "quantum_geometric/hardware/quantum_field_gpu.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>
#include <stdio.h>

// CUDA parameters
#define BLOCK_SIZE 256
#define MAX_BLOCKS 65535

// ============================================================================
// Helper: Convert complex double to/from cuDoubleComplex
// ============================================================================

__host__ __device__ inline cuDoubleComplex to_cuda_complex(double real, double imag) {
    return make_cuDoubleComplex(real, imag);
}

__device__ inline double cuCabs2(cuDoubleComplex z) {
    return cuCreal(z) * cuCreal(z) + cuCimag(z) * cuCimag(z);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void apply_rotation_kernel_cuda(
    cuDoubleComplex* field,
    size_t field_size,
    size_t num_components,
    size_t qubit,
    cuDoubleComplex* rotation)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= field_size) return;

    size_t mask = 1ULL << qubit;
    if (idx & mask) {
        // Get field components
        cuDoubleComplex psi_0 = field[idx ^ mask];
        cuDoubleComplex psi_1 = field[idx];

        // Apply rotation matrix
        cuDoubleComplex new_psi_0 = cuCadd(cuCmul(rotation[0], psi_0),
                                           cuCmul(rotation[1], psi_1));
        cuDoubleComplex new_psi_1 = cuCadd(cuCmul(rotation[2], psi_0),
                                           cuCmul(rotation[3], psi_1));

        // Update field
        field[idx ^ mask] = new_psi_0;
        field[idx] = new_psi_1;
    }
}

__global__ void calculate_field_energy_kernel_cuda(
    cuDoubleComplex* field,
    cuDoubleComplex* momentum,
    size_t field_size,
    size_t num_components,
    double* energy)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= field_size) return;

    double local_energy = 0.0;

    // Kinetic energy: |pi|^2
    for (size_t i = 0; i < num_components; i++) {
        size_t comp_idx = idx * num_components + i;
        if (comp_idx < field_size) {
            cuDoubleComplex pi = momentum[comp_idx];
            local_energy += cuCabs2(pi);
        }
    }

    // Potential energy: 0.5 * |phi|^2
    double phi_squared = 0.0;
    for (size_t i = 0; i < num_components; i++) {
        size_t comp_idx = idx * num_components + i;
        if (comp_idx < field_size) {
            cuDoubleComplex phi = field[comp_idx];
            phi_squared += cuCabs2(phi);
        }
    }

    local_energy += 0.5 * phi_squared;

    // Atomic add to total energy
    atomicAdd(energy, local_energy);
}

__global__ void calculate_field_equations_kernel_cuda(
    cuDoubleComplex* field,
    cuDoubleComplex* equations,
    size_t field_size,
    size_t num_components,
    double mass,
    double coupling)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= field_size) return;

    // Calculate |phi|^2 for this point
    double phi_squared = 0.0;
    for (size_t i = 0; i < num_components; i++) {
        size_t comp_idx = idx * num_components + i;
        if (comp_idx < field_size) {
            cuDoubleComplex phi = field[comp_idx];
            phi_squared += cuCabs2(phi);
        }
    }

    // Calculate field equation: (m^2 + lambda*|phi|^2) * phi
    for (size_t i = 0; i < num_components; i++) {
        size_t comp_idx = idx * num_components + i;
        if (comp_idx < field_size) {
            cuDoubleComplex phi = field[comp_idx];
            double coeff = mass * mass + coupling * phi_squared;
            equations[comp_idx] = make_cuDoubleComplex(
                coeff * cuCreal(phi),
                coeff * cuCimag(phi)
            );
        }
    }
}

// ============================================================================
// Helper: Convert host complex double array to device cuDoubleComplex
// ============================================================================

static cudaError_t copyFieldToDevice(const Tensor* tensor, cuDoubleComplex** d_ptr) {
    if (!tensor || !tensor->data || tensor->total_size == 0) {
        return cudaErrorInvalidValue;
    }

    size_t total = tensor->total_size;
    size_t byteSize = total * sizeof(cuDoubleComplex);

    // Allocate host buffer for conversion
    cuDoubleComplex* h_buffer = (cuDoubleComplex*)malloc(byteSize);
    if (!h_buffer) return cudaErrorMemoryAllocation;

    // Convert complex double to cuDoubleComplex
    for (size_t i = 0; i < total; i++) {
        h_buffer[i] = make_cuDoubleComplex(creal(tensor->data[i]), cimag(tensor->data[i]));
    }

    // Allocate and copy to device
    cudaError_t err = cudaMalloc(d_ptr, byteSize);
    if (err != cudaSuccess) {
        free(h_buffer);
        return err;
    }

    err = cudaMemcpy(*d_ptr, h_buffer, byteSize, cudaMemcpyHostToDevice);
    free(h_buffer);
    return err;
}

static cudaError_t copyDeviceToField(cuDoubleComplex* d_ptr, Tensor* tensor) {
    if (!d_ptr || !tensor || !tensor->data) {
        return cudaErrorInvalidValue;
    }

    size_t total = tensor->total_size;
    size_t byteSize = total * sizeof(cuDoubleComplex);

    // Allocate host buffer
    cuDoubleComplex* h_buffer = (cuDoubleComplex*)malloc(byteSize);
    if (!h_buffer) return cudaErrorMemoryAllocation;

    // Copy from device
    cudaError_t err = cudaMemcpy(h_buffer, d_ptr, byteSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        free(h_buffer);
        return err;
    }

    // Convert back to complex double
    for (size_t i = 0; i < total; i++) {
        tensor->data[i] = (double _Complex)(cuCreal(h_buffer[i]) + I * cuCimag(h_buffer[i]));
    }

    free(h_buffer);
    return cudaSuccess;
}

// ============================================================================
// CUDA Wrapper Functions
// ============================================================================

extern "C" {

int apply_rotation_cuda(
    QuantumField* field,
    size_t qubit,
    double theta,
    double phi)
{
    if (!field || !field->field_tensor || !field->field_tensor->data) {
        return -1;
    }

    size_t total_size = field->field_tensor->total_size;
    size_t num_components = field->num_components;

    // Copy field to device
    cuDoubleComplex* d_field;
    if (copyFieldToDevice(field->field_tensor, &d_field) != cudaSuccess) {
        return -1;
    }

    // Create rotation matrix (RZ(phi) * RY(theta))
    cuDoubleComplex rotation[4];
    double cos_t = cos(theta / 2.0);
    double sin_t = sin(theta / 2.0);
    double cos_p = cos(phi);
    double sin_p = sin(phi);

    rotation[0] = make_cuDoubleComplex(cos_t, 0.0);
    rotation[1] = make_cuDoubleComplex(-sin_t * cos_p, -sin_t * sin_p);
    rotation[2] = make_cuDoubleComplex(sin_t * cos_p, sin_t * sin_p);
    rotation[3] = make_cuDoubleComplex(cos_t, 0.0);

    cuDoubleComplex* d_rotation;
    cudaMalloc(&d_rotation, 4 * sizeof(cuDoubleComplex));
    cudaMemcpy(d_rotation, rotation, 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // Launch kernel
    size_t num_blocks = (total_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > MAX_BLOCKS) num_blocks = MAX_BLOCKS;

    apply_rotation_kernel_cuda<<<num_blocks, BLOCK_SIZE>>>(
        d_field,
        total_size,
        num_components,
        qubit,
        d_rotation
    );

    // Synchronize and check for errors
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: apply_rotation kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_field);
        cudaFree(d_rotation);
        return -1;
    }

    // Copy result back
    err = copyDeviceToField(d_field, field->field_tensor);

    // Clean up
    cudaFree(d_field);
    cudaFree(d_rotation);

    return (err == cudaSuccess) ? 0 : -1;
}

double calculate_field_energy_cuda(const QuantumField* field) {
    if (!field || !field->field_tensor || !field->field_tensor->data) {
        return 0.0;
    }

    size_t total_size = field->field_tensor->total_size;
    size_t num_components = field->num_components;

    // Copy field to device
    cuDoubleComplex* d_field;
    if (copyFieldToDevice(field->field_tensor, &d_field) != cudaSuccess) {
        return 0.0;
    }

    // Copy momentum to device (or use zeros)
    cuDoubleComplex* d_momentum;
    if (field->conjugate_momentum && field->conjugate_momentum->data) {
        if (copyFieldToDevice(field->conjugate_momentum, &d_momentum) != cudaSuccess) {
            cudaFree(d_field);
            return 0.0;
        }
    } else {
        cudaMalloc(&d_momentum, total_size * sizeof(cuDoubleComplex));
        cudaMemset(d_momentum, 0, total_size * sizeof(cuDoubleComplex));
    }

    // Allocate energy result
    double* d_energy;
    cudaMalloc(&d_energy, sizeof(double));
    cudaMemset(d_energy, 0, sizeof(double));

    // Launch kernel
    size_t num_blocks = (total_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > MAX_BLOCKS) num_blocks = MAX_BLOCKS;

    calculate_field_energy_kernel_cuda<<<num_blocks, BLOCK_SIZE>>>(
        d_field,
        d_momentum,
        total_size,
        num_components,
        d_energy
    );

    // Synchronize
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: calculate_energy kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_field);
        cudaFree(d_momentum);
        cudaFree(d_energy);
        return 0.0;
    }

    // Get result
    double energy;
    cudaMemcpy(&energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_field);
    cudaFree(d_momentum);
    cudaFree(d_energy);

    return energy;
}

int calculate_field_equations_cuda(const QuantumField* field, Tensor* equations) {
    if (!field || !field->field_tensor || !field->field_tensor->data ||
        !equations || !equations->data) {
        return -1;
    }

    size_t total_size = field->field_tensor->total_size;
    size_t num_components = field->num_components;
    double mass = field->mass;
    double coupling = field->coupling;

    // Copy field to device
    cuDoubleComplex* d_field;
    if (copyFieldToDevice(field->field_tensor, &d_field) != cudaSuccess) {
        return -1;
    }

    // Allocate equations buffer on device
    cuDoubleComplex* d_equations;
    cudaMalloc(&d_equations, equations->total_size * sizeof(cuDoubleComplex));

    // Launch kernel
    size_t num_blocks = (total_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > MAX_BLOCKS) num_blocks = MAX_BLOCKS;

    calculate_field_equations_kernel_cuda<<<num_blocks, BLOCK_SIZE>>>(
        d_field,
        d_equations,
        total_size,
        num_components,
        mass,
        coupling
    );

    // Synchronize
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: calculate_equations kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_field);
        cudaFree(d_equations);
        return -1;
    }

    // Copy results back
    err = copyDeviceToField(d_equations, equations);

    // Clean up
    cudaFree(d_field);
    cudaFree(d_equations);

    return (err == cudaSuccess) ? 0 : -1;
}

} // extern "C"

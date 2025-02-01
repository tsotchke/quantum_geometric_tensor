#include "quantum_geometric/hardware/quantum_geometric_tensor_error.h"
#include "quantum_geometric/hardware/quantum_geometric_tensor_gpu.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Test configuration
#define NUM_ITERATIONS 10

// Test utilities
static void generate_random_state(double complex *state, size_t size) {
  for (size_t i = 0; i < size; i++) {
    double real = (double)rand() / RAND_MAX;
    double imag = (double)rand() / RAND_MAX;
    state[i] = real + imag * I;
  }
}

static void print_test_header(const char *test_name) {
  printf("\n=== Running %s ===\n", test_name);
}

static void print_test_result(const char *test_name, QGTError error) {
  printf("%s: %s (%s)\n", test_name, error == QGT_SUCCESS ? "PASSED" : "FAILED",
         get_error_message(error));
}

static QGTError test_metric_tensor_performance(size_t matrix_size) {
  print_test_header("Quantum Metric Tensor Performance Test");

  // Allocate host memory
  size_t total_size = matrix_size * matrix_size;
  double complex *h_state =
      (double complex *)malloc(total_size * sizeof(double complex));
  double complex *h_metric =
      (double complex *)malloc(total_size * sizeof(double complex));

  if (!h_state || !h_metric) {
    printf("Error: Host memory allocation failed\n");
    return QGT_ERROR_ALLOCATION_FAILED;
  }

  // Generate random state vector
  generate_random_state(h_state, total_size);

  // Allocate device memory
  double complex *d_state, *d_metric;
  cudaError_t cuda_err;

  cuda_err = cudaMalloc(&d_state, total_size * sizeof(double complex));
  if (cuda_err != cudaSuccess) {
    printf("Error: Device memory allocation failed for state\n");
    return QGT_ERROR_ALLOCATION_FAILED;
  }

  cuda_err = cudaMalloc(&d_metric, total_size * sizeof(double complex));
  if (cuda_err != cudaSuccess) {
    printf("Error: Device memory allocation failed for metric\n");
    return QGT_ERROR_ALLOCATION_FAILED;
  }

  // Copy data to device
  cuda_err = cudaMemcpy(d_state, h_state, total_size * sizeof(double complex),
                        cudaMemcpyHostToDevice);
  if (cuda_err != cudaSuccess) {
    printf("Error: Host to device memory copy failed\n");
    return QGT_ERROR_MEMORY_COPY_FAILED;
  }

  // Create CUDA stream
  cudaStream_t stream;
  cuda_err = cudaStreamCreate(&stream);
  if (cuda_err != cudaSuccess) {
    printf("Error: Stream creation failed\n");
    return QGT_ERROR_SYSTEM_ERROR;
  }

  // Performance measurement
  float total_time = 0.0f;
  float min_time = INFINITY;
  float max_time = 0.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < NUM_ITERATIONS; i++) {
    cudaEventRecord(start, stream);

    QGTError err = launch_quantum_metric_kernel(d_state, d_metric, matrix_size,
                                                matrix_size, stream);
    if (err != QGT_SUCCESS) {
      printf("Error: Kernel launch failed on iteration %d\n", i);
      return err;
    }

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);

    total_time += kernel_time;
    min_time = fminf(min_time, kernel_time);
    max_time = fmaxf(max_time, kernel_time);
  }

  // Calculate statistics
  float avg_time = total_time / NUM_ITERATIONS;
  printf("Performance Results:\n");
  printf("  Average Time: %.2f ms\n", avg_time);
  printf("  Min Time: %.2f ms\n", min_time);
  printf("  Max Time: %.2f ms\n", max_time);

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cudaFree(d_state);
  cudaFree(d_metric);
  free(h_state);
  free(h_metric);

  return QGT_SUCCESS;
}

int main(int argc, char **argv) {
  // Initialize random seed
  srand(time(NULL));

  // Get matrix size from command line
  size_t matrix_size = 1024; // Default size
  if (argc > 1) {
    matrix_size = atoi(argv[1]);
    if (matrix_size == 0) {
      printf("Error: Invalid matrix size\n");
      return 1;
    }
  }

  // Initialize CUDA
  cudaError_t cuda_err = cudaSetDevice(0);
  if (cuda_err != cudaSuccess) {
    printf("Error: Failed to initialize CUDA device\n");
    return 1;
  }

  // Run performance test
  QGTError err = test_metric_tensor_performance(matrix_size);
  print_test_result("Performance Test", err);

  return err == QGT_SUCCESS ? 0 : 1;
}

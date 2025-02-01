#include "../src/quantum_geometric/core/amx_operations.h"
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <complex.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

// Test configuration
#define NUM_ITERATIONS 10 // Reduced for testing
#define NUM_WARMUP 2
#define TILE_SIZE 32          // AMX register size is 32x32 for FP64
#define VERIFY_BLOCK_SIZE 128 // Smaller blocks for verification

// Timing utilities
static double get_time_ms() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

// Initialize matrix with random values
static void init_matrix(double *matrix, size_t rows, size_t cols) {
  printf("Initializing %zux%zu matrix at %p\n", rows, cols, (void *)matrix);
  fflush(stdout);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      matrix[i * cols + j] = (double)rand() / RAND_MAX;
    }
  }
}

// Print matrix for debugging
static void print_matrix(const char *name, const double *matrix, size_t rows,
                         size_t cols) {
  printf("\nMatrix %s (%zux%zu):\n", name, rows, cols);
  fflush(stdout);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      printf("%8.4f ", matrix[i * cols + j]);
    }
    printf("\n");
    fflush(stdout);
  }
  printf("\n");
  fflush(stdout);
}

// Verify matrix multiplication result using BLAS and NEON
static bool verify_result(const double *a, const double *b, const double *c,
                          size_t m, size_t n, size_t k) {
  printf("Starting verification...\n");
  fflush(stdout);

  // Compute reference result using BLAS
  double *ref = aligned_alloc(64, m * n * sizeof(double));
  if (!ref) {
    printf("Failed to allocate reference matrix\n");
    fflush(stdout);
    return false;
  }

  // Copy C matrix to reference since BLAS will accumulate
  printf("Copying result matrix...\n");
  fflush(stdout);
  memcpy(ref, c, m * n * sizeof(double));

  // Use BLAS dgemm for reference computation
  // C = alpha * A * B + beta * C
  double alpha = 1.0;
  double beta = -1.0; // Subtract from our result

  printf("Computing reference result with BLAS...\n");
  fflush(stdout);

  // Process in blocks to reduce memory pressure
  for (size_t i = 0; i < m; i += VERIFY_BLOCK_SIZE) {
    size_t block_rows =
        (i + VERIFY_BLOCK_SIZE > m) ? (m - i) : VERIFY_BLOCK_SIZE;
    printf("Processing verification block %zu/%zu\n",
           (i / VERIFY_BLOCK_SIZE) + 1,
           (m + VERIFY_BLOCK_SIZE - 1) / VERIFY_BLOCK_SIZE);
    fflush(stdout);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, block_rows, n, k,
                alpha, &a[i * k], k,   // lda = k
                b, n,                  // ldb = n
                beta, &ref[i * n], n); // ldc = n
  }

  printf("Checking differences...\n");
  fflush(stdout);

  // Now ref contains the difference between our result and BLAS result
  // Check if the difference is within epsilon using NEON
  bool correct = true;
  const double epsilon = 1e-10;
  float64x2_t vepsilon = vdupq_n_f64(epsilon);
  float64x2_t vmax_diff = vdupq_n_f64(0.0);

  // Process 2 elements at a time using NEON
  for (size_t i = 0; i < m * n; i += 2) {
    if ((i % (m * n / 10)) == 0) {
      printf("Verification progress: %zu%%\n", (i * 100) / (m * n));
      fflush(stdout);
    }

    float64x2_t vdiff = vabsq_f64(vld1q_f64(&ref[i]));
    vmax_diff = vmaxq_f64(vmax_diff, vdiff);
    uint64x2_t vcmp = vcgtq_f64(vdiff, vepsilon);
    if (vgetq_lane_u64(vcmp, 0) || vgetq_lane_u64(vcmp, 1)) {
      correct = false;
      break;
    }
  }

  // Handle any remaining elements
  size_t remaining = (m * n) % 2;
  if (remaining && correct) {
    double diff = fabs(ref[m * n - 1]);
    if (diff > epsilon) {
      correct = false;
    }
  }

  // Get maximum difference for reporting
  double max_diff = vgetq_lane_f64(vmax_diff, 0);
  max_diff = fmax(max_diff, vgetq_lane_f64(vmax_diff, 1));

  printf("Maximum difference from BLAS: %g\n", max_diff);
  printf("Verification %s\n", correct ? "PASSED" : "FAILED");
  fflush(stdout);

  free(ref);
  return correct;
}

int main(int argc, char **argv) {
  // Get matrix size from command line
  size_t matrix_size = TILE_SIZE; // Default to one tile
  if (argc > 1) {
    matrix_size = atoi(argv[1]);
    if (matrix_size == 0 || matrix_size % TILE_SIZE != 0) {
      printf("Error: Matrix size must be a multiple of %d\n", TILE_SIZE);
      fflush(stdout);
      return 1;
    }
  }

  printf("Starting test with matrix size %zux%zu\n", matrix_size, matrix_size);
  fflush(stdout);

  // Allocate aligned memory for matrices
  double *a = aligned_alloc(64, matrix_size * matrix_size * sizeof(double));
  double *b = aligned_alloc(64, matrix_size * matrix_size * sizeof(double));
  double *c = aligned_alloc(64, matrix_size * matrix_size * sizeof(double));

  if (!a || !b || !c) {
    printf("Error: Memory allocation failed\n");
    fflush(stdout);
    free(a);
    free(b);
    free(c);
    return 1;
  }

  // Initialize matrices
  printf("Initializing matrices...\n");
  fflush(stdout);
  init_matrix(a, matrix_size, matrix_size);
  init_matrix(b, matrix_size, matrix_size);
  memset(c, 0, matrix_size * matrix_size * sizeof(double));

  if (matrix_size <= 8) {
    print_matrix("A", a, matrix_size, matrix_size);
    print_matrix("B", b, matrix_size, matrix_size);
  }

  // Initialize AMX
  printf("Initializing AMX...\n");
  fflush(stdout);
  bool using_amx = amx_init();
  if (using_amx) {
    printf("Using AMX acceleration\n");
  } else {
    printf("Using fallback implementation\n");
  }
  fflush(stdout);

  // Warmup runs
  printf("Performing %d warmup runs...\n", NUM_WARMUP);
  fflush(stdout);
  for (int i = 0; i < NUM_WARMUP; i++) {
    printf("Warmup run %d/%d\n", i + 1, NUM_WARMUP);
    fflush(stdout);
    if (!amx_matrix_multiply(a, b, c, matrix_size, matrix_size, matrix_size)) {
      printf("Error: Matrix multiplication failed during warmup\n");
      fflush(stdout);
      goto cleanup;
    }
  }

  // Performance measurement
  printf("Running performance test with %d iterations...\n", NUM_ITERATIONS);
  fflush(stdout);
  double total_time = 0.0;
  double min_time = INFINITY;
  double max_time = 0.0;

  for (int i = 0; i < NUM_ITERATIONS; i++) {
    printf("Performance run %d/%d\n", i + 1, NUM_ITERATIONS);
    fflush(stdout);
    double start_time = get_time_ms();

    if (!amx_matrix_multiply(a, b, c, matrix_size, matrix_size, matrix_size)) {
      printf("Error: Matrix multiplication failed during performance run\n");
      fflush(stdout);
      goto cleanup;
    }

    double end_time = get_time_ms();
    double elapsed_time = end_time - start_time;

    total_time += elapsed_time;
    min_time = fmin(min_time, elapsed_time);
    max_time = fmax(max_time, elapsed_time);
  }

  if (matrix_size <= 8) {
    print_matrix("C", c, matrix_size, matrix_size);
  }

  // Verify result
  printf("Starting result verification...\n");
  fflush(stdout);
  if (!verify_result(a, b, c, matrix_size, matrix_size, matrix_size)) {
    printf("Error: Result verification failed\n");
    fflush(stdout);
    goto cleanup;
  }

  // Calculate statistics
  double avg_time = total_time / NUM_ITERATIONS;
  double gflops =
      (2.0 * matrix_size * matrix_size * matrix_size) / (avg_time * 1e6);

  printf("\nPerformance Results:\n");
  printf("  Implementation: %s\n", using_amx ? "AMX" : "Fallback");
  printf("  Average Time: %.3f ms\n", avg_time);
  printf("  Min Time: %.3f ms\n", min_time);
  printf("  Max Time: %.3f ms\n", max_time);
  printf("  Matrix Size: %zux%zu\n", matrix_size, matrix_size);
  printf("  Performance: %.2f GFLOPS\n", gflops);
  printf("  Total Iterations: %d\n", NUM_ITERATIONS);
  fflush(stdout);

cleanup:
  // Cleanup
  printf("Cleaning up...\n");
  fflush(stdout);
  amx_shutdown();
  free(a);
  free(b);
  free(c);

  return 0;
}

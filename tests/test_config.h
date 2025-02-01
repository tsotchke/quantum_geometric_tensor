#ifndef TEST_CONFIG_H
#define TEST_CONFIG_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "quantum_geometric/core/quantum_complex.h"

// Maximum number of tests
#define MAX_TESTS 1000

// Test case structure
struct test_case {
    const char* name;
    void (*func)(void);
};

// Test registration macro
#define REGISTER_TEST(func) \
    do { \
        if (g_test_count < MAX_TESTS) { \
            g_tests[g_test_count].name = #func; \
            g_tests[g_test_count].func = func; \
            g_test_count++; \
        } \
    } while (0)

// Test setup/teardown macros
#define TEST_SETUP() \
    printf("Running test: %s\n", __func__)

#define TEST_TEARDOWN() \
    printf("Test passed: %s\n", __func__)

// Assertion macros
#define TEST_ASSERT(condition) \
    do { \
        if (!(condition)) { \
            printf("Assertion failed: %s\n", #condition); \
            printf("File: %s, Line: %d\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

#define TEST_ASSERT_FLOAT_EQ(actual, expected) \
    do { \
        float a = (actual); \
        float e = (expected); \
        if (fabsf(a - e) > 1e-6f) { \
            printf("Float assertion failed\n"); \
            printf("Expected: %f\n", e); \
            printf("Actual: %f\n", a); \
            printf("File: %s, Line: %d\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

#define TEST_ASSERT_COMPLEX_EQ(actual, expected) \
    do { \
        ComplexFloat a = (actual); \
        ComplexFloat e = (expected); \
        if (!complex_eq(a, e, 1e-6f)) { \
            printf("Complex assertion failed\n"); \
            printf("Expected: %f + %fi\n", crealf(e), cimagf(e)); \
            printf("Actual: %f + %fi\n", crealf(a), cimagf(a)); \
            printf("File: %s, Line: %d\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

// MPI initialization macros
#ifdef USE_MPI
#include <mpi.h>
#define TEST_MPI_INIT() \
    do { \
        int provided; \
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided); \
    } while (0)
#define TEST_MPI_FINALIZE() MPI_Finalize()
#else
#define TEST_MPI_INIT()
#define TEST_MPI_FINALIZE()
#endif

// Metal initialization macros
#ifdef USE_METAL
#include "quantum_geometric/hardware/metal/metal_common.h"
#define TEST_METAL_INIT() metal_initialize()
#define TEST_METAL_CLEANUP() metal_cleanup()
#else
#define TEST_METAL_INIT()
#define TEST_METAL_CLEANUP()
#endif

// CUDA initialization macros
#ifdef USE_CUDA
#include "quantum_geometric/hardware/cuda/cuda_common.h"
#define TEST_CUDA_INIT() cuda_initialize()
#define TEST_CUDA_CLEANUP() cuda_cleanup()
#else
#define TEST_CUDA_INIT()
#define TEST_CUDA_CLEANUP()
#endif

// Test runner (if not using CTest)
#ifdef STANDALONE_TEST
int main(int argc, char** argv) {
    TEST_MPI_INIT();
    TEST_METAL_INIT();
    TEST_CUDA_INIT();
    
    // Run all registered tests
    for (int i = 0; i < g_test_count; i++) {
        g_tests[i].func();
    }
    
    TEST_CUDA_CLEANUP();
    TEST_METAL_CLEANUP();
    TEST_MPI_FINALIZE();
    
    printf("All tests passed!\n");
    return 0;
}
#endif

#endif // TEST_CONFIG_H

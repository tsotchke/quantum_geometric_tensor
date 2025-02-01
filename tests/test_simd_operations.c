#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "quantum_geometric/core/simd_operations.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include "quantum_geometric/core/error_codes.h"

#define EPSILON 1e-6f
#define TEST_SIZE 1024
#define ALIGNMENT 64

// Test helper functions
static void init_test_data(ComplexFloat* a, ComplexFloat* b, size_t size) {
    for (size_t i = 0; i < size; i++) {
        a[i].real = (float)rand() / RAND_MAX;
        a[i].imag = (float)rand() / RAND_MAX;
        b[i].real = (float)rand() / RAND_MAX;
        b[i].imag = (float)rand() / RAND_MAX;
    }
}

static int compare_complex(ComplexFloat a, ComplexFloat b) {
    return fabsf(a.real - b.real) < EPSILON && 
           fabsf(a.imag - b.imag) < EPSILON;
}

static void scalar_complex_multiply_accumulate(ComplexFloat* result,
                                            const ComplexFloat* a,
                                            const ComplexFloat* b,
                                            size_t count) {
    ComplexFloat sum = {0.0f, 0.0f};
    for (size_t i = 0; i < count; i++) {
        ComplexFloat prod;
        prod.real = a[i].real * b[i].real - a[i].imag * b[i].imag;
        prod.imag = a[i].real * b[i].imag + a[i].imag * b[i].real;
        sum.real += prod.real;
        sum.imag += prod.imag;
    }
    *result = sum;
}

// Additional scalar implementations for testing
static void scalar_complex_add(ComplexFloat* result,
                             const ComplexFloat* a,
                             const ComplexFloat* b,
                             size_t count) {
    for (size_t i = 0; i < count; i++) {
        result[i].real = a[i].real + b[i].real;
        result[i].imag = a[i].imag + b[i].imag;
    }
}

static void scalar_complex_subtract(ComplexFloat* result,
                                 const ComplexFloat* a,
                                 const ComplexFloat* b,
                                 size_t count) {
    for (size_t i = 0; i < count; i++) {
        result[i].real = a[i].real - b[i].real;
        result[i].imag = a[i].imag - b[i].imag;
    }
}

static void scalar_complex_multiply(ComplexFloat* result,
                                 const ComplexFloat* a,
                                 const ComplexFloat* b,
                                 size_t count) {
    for (size_t i = 0; i < count; i++) {
        result[i].real = a[i].real * b[i].real - a[i].imag * b[i].imag;
        result[i].imag = a[i].real * b[i].imag + a[i].imag * b[i].real;
    }
}

static void scalar_complex_scale(ComplexFloat* result,
                               const ComplexFloat* input,
                               ComplexFloat scalar,
                               size_t count) {
    for (size_t i = 0; i < count; i++) {
        result[i].real = input[i].real * scalar.real - input[i].imag * scalar.imag;
        result[i].imag = input[i].real * scalar.imag + input[i].imag * scalar.real;
    }
}

static double scalar_complex_norm(const ComplexFloat* input, size_t count) {
    double norm = 0.0;
    for (size_t i = 0; i < count; i++) {
        norm += input[i].real * input[i].real + input[i].imag * input[i].imag;
    }
    return sqrt(norm);
}

static void scalar_complex_conjugate(ComplexFloat* result,
                                   const ComplexFloat* input,
                                   size_t count) {
    for (size_t i = 0; i < count; i++) {
        result[i].real = input[i].real;
        result[i].imag = -input[i].imag;
    }
}

// Test cases
static int test_complex_add() {
    printf("Testing complex addition...\n");
    
    ComplexFloat* a = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat* b = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat* simd_result = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat* scalar_result = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    
    if (!a || !b || !simd_result || !scalar_result) {
        printf("Failed to allocate memory\n");
        return 0;
    }
    
    init_test_data(a, b, TEST_SIZE);
    
    // Test SIMD implementation
    simd_complex_add(simd_result, a, b, TEST_SIZE);
    
    // Test scalar implementation
    scalar_complex_add(scalar_result, a, b, TEST_SIZE);
    
    // Compare results
    int success = 1;
    for (size_t i = 0; i < TEST_SIZE; i++) {
        if (!compare_complex(simd_result[i], scalar_result[i])) {
            success = 0;
            break;
        }
    }
    
    free(a);
    free(b);
    free(simd_result);
    free(scalar_result);
    
    return success;
}

static int test_complex_subtract() {
    printf("Testing complex subtraction...\n");
    
    ComplexFloat* a = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat* b = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat* simd_result = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat* scalar_result = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    
    if (!a || !b || !simd_result || !scalar_result) {
        printf("Failed to allocate memory\n");
        return 0;
    }
    
    init_test_data(a, b, TEST_SIZE);
    
    // Test SIMD implementation
    simd_complex_subtract(simd_result, a, b, TEST_SIZE);
    
    // Test scalar implementation
    scalar_complex_subtract(scalar_result, a, b, TEST_SIZE);
    
    // Compare results
    int success = 1;
    for (size_t i = 0; i < TEST_SIZE; i++) {
        if (!compare_complex(simd_result[i], scalar_result[i])) {
            success = 0;
            break;
        }
    }
    
    free(a);
    free(b);
    free(simd_result);
    free(scalar_result);
    
    return success;
}

static int test_complex_multiply() {
    printf("Testing complex multiplication...\n");
    
    ComplexFloat* a = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat* b = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat* simd_result = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat* scalar_result = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    
    if (!a || !b || !simd_result || !scalar_result) {
        printf("Failed to allocate memory\n");
        return 0;
    }
    
    init_test_data(a, b, TEST_SIZE);
    
    // Test SIMD implementation
    simd_complex_multiply(simd_result, a, b, TEST_SIZE);
    
    // Test scalar implementation
    scalar_complex_multiply(scalar_result, a, b, TEST_SIZE);
    
    // Compare results
    int success = 1;
    for (size_t i = 0; i < TEST_SIZE; i++) {
        if (!compare_complex(simd_result[i], scalar_result[i])) {
            success = 0;
            break;
        }
    }
    
    free(a);
    free(b);
    free(simd_result);
    free(scalar_result);
    
    return success;
}

static int test_complex_scale() {
    printf("Testing complex scaling...\n");
    
    ComplexFloat* input = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat* simd_result = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat* scalar_result = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat scalar = {(float)rand() / RAND_MAX, (float)rand() / RAND_MAX};
    
    if (!input || !simd_result || !scalar_result) {
        printf("Failed to allocate memory\n");
        return 0;
    }
    
    init_test_data(input, NULL, TEST_SIZE);
    
    // Test SIMD implementation
    simd_complex_scale(simd_result, input, scalar, TEST_SIZE);
    
    // Test scalar implementation
    scalar_complex_scale(scalar_result, input, scalar, TEST_SIZE);
    
    // Compare results
    int success = 1;
    for (size_t i = 0; i < TEST_SIZE; i++) {
        if (!compare_complex(simd_result[i], scalar_result[i])) {
            success = 0;
            break;
        }
    }
    
    free(input);
    free(simd_result);
    free(scalar_result);
    
    return success;
}

static int test_complex_norm() {
    printf("Testing complex norm...\n");
    
    ComplexFloat* input = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    
    if (!input) {
        printf("Failed to allocate memory\n");
        return 0;
    }
    
    init_test_data(input, NULL, TEST_SIZE);
    
    // Test SIMD implementation
    double simd_norm = simd_complex_norm(input, TEST_SIZE);
    
    // Test scalar implementation
    double scalar_norm = scalar_complex_norm(input, TEST_SIZE);
    
    // Compare results
    int success = fabs(simd_norm - scalar_norm) < EPSILON;
    
    free(input);
    
    return success;
}

static int test_complex_conjugate() {
    printf("Testing complex conjugate...\n");
    
    ComplexFloat* input = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat* simd_result = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat* scalar_result = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    
    if (!input || !simd_result || !scalar_result) {
        printf("Failed to allocate memory\n");
        return 0;
    }
    
    init_test_data(input, NULL, TEST_SIZE);
    
    // Test SIMD implementation
    simd_tensor_conjugate(simd_result, input, TEST_SIZE);
    
    // Test scalar implementation
    scalar_complex_conjugate(scalar_result, input, TEST_SIZE);
    
    // Compare results
    int success = 1;
    for (size_t i = 0; i < TEST_SIZE; i++) {
        if (!compare_complex(simd_result[i], scalar_result[i])) {
            success = 0;
            break;
        }
    }
    
    free(input);
    free(simd_result);
    free(scalar_result);
    
    return success;
}

static int test_multiply_accumulate() {
    printf("Testing complex multiply-accumulate...\n");
    
    // Allocate aligned memory
    ComplexFloat* a = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat* b = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat simd_result = {0.0f, 0.0f};
    ComplexFloat scalar_result = {0.0f, 0.0f};
    
    if (!a || !b) {
        printf("Failed to allocate memory\n");
        return 0;
    }
    
    // Initialize test data
    init_test_data(a, b, TEST_SIZE);
    
    // Test SIMD implementation
    simd_complex_multiply_accumulate(&simd_result, a, b, TEST_SIZE);
    
    // Test scalar implementation
    scalar_complex_multiply_accumulate(&scalar_result, a, b, TEST_SIZE);
    
    // Compare results
    int success = compare_complex(simd_result, scalar_result);
    
    // Performance comparison
    clock_t start, end;
    double simd_time, scalar_time;
    
    start = clock();
    for (int i = 0; i < 100; i++) {
        simd_complex_multiply_accumulate(&simd_result, a, b, TEST_SIZE);
    }
    end = clock();
    simd_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    start = clock();
    for (int i = 0; i < 100; i++) {
        scalar_complex_multiply_accumulate(&scalar_result, a, b, TEST_SIZE);
    }
    end = clock();
    scalar_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("SIMD time: %.3f s\n", simd_time);
    printf("Scalar time: %.3f s\n", scalar_time);
    printf("Speedup: %.2fx\n", scalar_time / simd_time);
    
    free(a);
    free(b);
    
    return success;
}

static int test_unaligned_data() {
    printf("Testing unaligned data handling...\n");
    
    // Allocate unaligned memory
    ComplexFloat* a = malloc((TEST_SIZE + 1) * sizeof(ComplexFloat));
    ComplexFloat* b = malloc((TEST_SIZE + 1) * sizeof(ComplexFloat));
    ComplexFloat simd_result = {0.0f, 0.0f};
    ComplexFloat scalar_result = {0.0f, 0.0f};
    
    if (!a || !b) {
        printf("Failed to allocate memory\n");
        return 0;
    }
    
    // Use unaligned pointers
    ComplexFloat* unaligned_a = (ComplexFloat*)((char*)a + 1);
    ComplexFloat* unaligned_b = (ComplexFloat*)((char*)b + 1);
    
    // Initialize test data
    init_test_data(unaligned_a, unaligned_b, TEST_SIZE);
    
    // Test SIMD implementation with unaligned data
    simd_complex_multiply_accumulate(&simd_result, unaligned_a, unaligned_b, TEST_SIZE);
    
    // Test scalar implementation
    scalar_complex_multiply_accumulate(&scalar_result, unaligned_a, unaligned_b, TEST_SIZE);
    
    // Compare results
    int success = compare_complex(simd_result, scalar_result);
    
    free(a);
    free(b);
    
    return success;
}

static int test_complex_copy() {
    printf("Testing complex copy...\n");
    
    ComplexFloat* input = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    ComplexFloat* simd_result = aligned_alloc(ALIGNMENT, TEST_SIZE * sizeof(ComplexFloat));
    
    if (!input || !simd_result) {
        printf("Failed to allocate memory\n");
        return 0;
    }
    
    init_test_data(input, NULL, TEST_SIZE);
    
    // Test SIMD implementation
    simd_complex_copy(simd_result, input, TEST_SIZE);
    
    // Compare results
    int success = 1;
    for (size_t i = 0; i < TEST_SIZE; i++) {
        if (!compare_complex(simd_result[i], input[i])) {
            success = 0;
            break;
        }
    }
    
    free(input);
    free(simd_result);
    
    return success;
}

static int test_division_by_zero() {
    printf("Testing division by zero handling...\n");
    
    ComplexFloat a = {1.0f, 1.0f};
    ComplexFloat b = {0.0f, 0.0f};
    ComplexFloat result;
    
    // Test division by zero
    simd_complex_divide(&result, &a, &b, 1);
    
    // Check if result is infinity or NaN
    return isinf(result.real) || isnan(result.real) ||
           isinf(result.imag) || isnan(result.imag);
}

// Additional scalar implementations for tensor operations
static void scalar_tensor_multiply(ComplexFloat* result,
                                const ComplexFloat* a,
                                const ComplexFloat* b,
                                const size_t* dimensions,
                                size_t rank) {
    size_t total_elements = 1;
    for (size_t i = 0; i < rank; i++) {
        total_elements *= dimensions[i];
    }
    
    for (size_t i = 0; i < total_elements; i++) {
        result[i].real = a[i].real * b[i].real - a[i].imag * b[i].imag;
        result[i].imag = a[i].real * b[i].imag + a[i].imag * b[i].real;
    }
}

static void scalar_tensor_contract(ComplexFloat* result,
                                const ComplexFloat* a,
                                const ComplexFloat* b,
                                const size_t* dimensions_a,
                                const size_t* dimensions_b,
                                const size_t* contract_indices,
                                size_t num_indices,
                                size_t rank_a,
                                size_t rank_b) {
    // Calculate contracted dimensions
    size_t contracted_dim = 1;
    for (size_t i = 0; i < num_indices; i++) {
        contracted_dim *= dimensions_a[contract_indices[i]];
    }
    
    // Calculate output dimensions
    size_t output_elements = 1;
    for (size_t i = 0; i < rank_a; i++) {
        bool is_contracted = false;
        for (size_t j = 0; j < num_indices; j++) {
            if (i == contract_indices[j]) {
                is_contracted = true;
                break;
            }
        }
        if (!is_contracted) {
            output_elements *= dimensions_a[i];
        }
    }
    for (size_t i = 0; i < rank_b; i++) {
        bool is_contracted = false;
        for (size_t j = 0; j < num_indices; j++) {
            if (i == contract_indices[j]) {
                is_contracted = true;
                break;
            }
        }
        if (!is_contracted) {
            output_elements *= dimensions_b[i];
        }
    }
    
    // Initialize result to zero
    memset(result, 0, output_elements * sizeof(ComplexFloat));
    
    // For each output element
    for (size_t i = 0; i < output_elements; i++) {
        // For each contracted dimension
        for (size_t j = 0; j < contracted_dim; j++) {
            ComplexFloat temp;
            temp.real = a[i*contracted_dim + j].real * b[j].real - 
                       a[i*contracted_dim + j].imag * b[j].imag;
            temp.imag = a[i*contracted_dim + j].real * b[j].imag + 
                       a[i*contracted_dim + j].imag * b[j].real;
            result[i].real += temp.real;
            result[i].imag += temp.imag;
        }
    }
}

// Test cases for tensor operations
static int test_tensor_multiply() {
    printf("Testing tensor multiplication...\n");
    
    // Create test dimensions
    size_t dimensions[] = {2, 2, 2}; // 2x2x2 tensor
    size_t rank = 3;
    size_t total_elements = 8; // 2*2*2
    
    ComplexFloat* a = aligned_alloc(ALIGNMENT, total_elements * sizeof(ComplexFloat));
    ComplexFloat* b = aligned_alloc(ALIGNMENT, total_elements * sizeof(ComplexFloat));
    ComplexFloat* simd_result = aligned_alloc(ALIGNMENT, total_elements * sizeof(ComplexFloat));
    ComplexFloat* scalar_result = aligned_alloc(ALIGNMENT, total_elements * sizeof(ComplexFloat));
    
    if (!a || !b || !simd_result || !scalar_result) {
        printf("Failed to allocate memory\n");
        return 0;
    }
    
    init_test_data(a, b, total_elements);
    
    // Test SIMD implementation
    simd_tensor_multiply(simd_result, a, b, dimensions, rank);
    
    // Test scalar implementation
    scalar_tensor_multiply(scalar_result, a, b, dimensions, rank);
    
    // Compare results
    int success = 1;
    for (size_t i = 0; i < total_elements; i++) {
        if (!compare_complex(simd_result[i], scalar_result[i])) {
            success = 0;
            break;
        }
    }
    
    free(a);
    free(b);
    free(simd_result);
    free(scalar_result);
    
    return success;
}

static int test_tensor_contract() {
    printf("Testing tensor contraction...\n");
    
    // Create test dimensions for a 2x2x2 tensor contracting with a 2x2 tensor
    size_t dimensions_a[] = {2, 2, 2};
    size_t dimensions_b[] = {2, 2};
    size_t contract_indices[] = {1, 2}; // Contract last two dimensions
    size_t rank_a = 3;
    size_t rank_b = 2;
    size_t num_indices = 2;
    
    size_t total_elements_a = 8; // 2*2*2
    size_t total_elements_b = 4; // 2*2
    size_t output_elements = 2; // Result is a 2x1 tensor
    
    ComplexFloat* a = aligned_alloc(ALIGNMENT, total_elements_a * sizeof(ComplexFloat));
    ComplexFloat* b = aligned_alloc(ALIGNMENT, total_elements_b * sizeof(ComplexFloat));
    ComplexFloat* simd_result = aligned_alloc(ALIGNMENT, output_elements * sizeof(ComplexFloat));
    ComplexFloat* scalar_result = aligned_alloc(ALIGNMENT, output_elements * sizeof(ComplexFloat));
    
    if (!a || !b || !simd_result || !scalar_result) {
        printf("Failed to allocate memory\n");
        return 0;
    }
    
    init_test_data(a, b, total_elements_a);
    
    // Test SIMD implementation
    simd_tensor_contract(simd_result, a, b, dimensions_a, dimensions_b,
                        contract_indices, num_indices, rank_a, rank_b);
    
    // Test scalar implementation
    scalar_tensor_contract(scalar_result, a, b, dimensions_a, dimensions_b,
                         contract_indices, num_indices, rank_a, rank_b);
    
    // Compare results
    int success = 1;
    for (size_t i = 0; i < output_elements; i++) {
        if (!compare_complex(simd_result[i], scalar_result[i])) {
            success = 0;
            break;
        }
    }
    
    free(a);
    free(b);
    free(simd_result);
    free(scalar_result);
    
    return success;
}

int main() {
    srand(time(NULL));
    int tests_passed = 0;
    int total_tests = 12;
    
    printf("Running SIMD operations tests...\n\n");
    
    // Run tests
    tests_passed += test_complex_add();
    tests_passed += test_complex_subtract();
    tests_passed += test_complex_multiply();
    tests_passed += test_complex_scale();
    tests_passed += test_complex_norm();
    tests_passed += test_complex_conjugate();
    tests_passed += test_multiply_accumulate();
    tests_passed += test_unaligned_data();
    tests_passed += test_division_by_zero();
    tests_passed += test_tensor_multiply();
    tests_passed += test_tensor_contract();
    tests_passed += test_complex_copy();
    
    // Print summary
    printf("\nTest Summary: %d/%d tests passed\n", tests_passed, total_tests);
    
    return tests_passed == total_tests ? 0 : 1;
}

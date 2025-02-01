#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "quantum_geometric/core/tensor_operations.h"
#include "quantum_geometric/core/quantum_geometric_logging.h"
#include "quantum_geometric/core/quantum_complex.h"

#define EPSILON 1e-6
#define TEST_SIZE 4

static int tests_run = 0;
static int tests_passed = 0;

static double get_time_ms(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 + (end->tv_nsec - start->tv_nsec) / 1000000.0;
}

static void print_tensor(const tensor_t* tensor) {
    if (!tensor || !tensor->data) {
        printf("Invalid tensor\n");
        return;
    }

    printf("Tensor dimensions: [");
    for (size_t i = 0; i < tensor->rank; i++) {
        printf("%zu%s", tensor->dimensions[i], i < tensor->rank - 1 ? ", " : "]\n");
    }

    printf("Data:\n");
    for (size_t i = 0; i < tensor->total_size; i++) {
        printf("%f + %fi ", tensor->data[i].real, tensor->data[i].imag);
        if ((i + 1) % tensor->dimensions[tensor->rank - 1] == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

static bool tensors_equal(const tensor_t* a, const tensor_t* b) {
    if (!a || !b || !a->data || !b->data) {
        return false;
    }

    if (a->rank != b->rank || a->total_size != b->total_size) {
        return false;
    }

    for (size_t i = 0; i < a->rank; i++) {
        if (a->dimensions[i] != b->dimensions[i]) {
            return false;
        }
    }

    for (size_t i = 0; i < a->total_size; i++) {
        if (fabs(a->data[i].real - b->data[i].real) > EPSILON ||
            fabs(a->data[i].imag - b->data[i].imag) > EPSILON) {
            return false;
        }
    }

    return true;
}

static bool test_tensor_init() {
    printf("Testing tensor initialization...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    tests_run++;

    size_t dims[] = {2, 3, 4};
    tensor_t tensor;
    struct timespec op_start, op_end;
    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool result = qg_tensor_init(&tensor, dims, 3);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (tensor init): %.3f ms\n", get_time_ms(&op_start, &op_end));

    if (!result || !tensor.data || !tensor.dimensions || 
        tensor.rank != 3 || tensor.total_size != 24) {
        printf("Tensor initialization failed\n");
        return false;
    }

    qg_tensor_cleanup(&tensor);
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time taken: %.3f ms\n", get_time_ms(&start, &end));
    tests_passed++;
    return true;
}

static bool test_tensor_add() {
    printf("Testing tensor addition...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    tests_run++;

    size_t dims[] = {2, 3};
    tensor_t a, b, result;

    struct timespec op_start, op_end;
    
    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool init_a = qg_tensor_init(&a, dims, 2);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (init tensor a): %.3f ms\n", get_time_ms(&op_start, &op_end));

    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool init_b = qg_tensor_init(&b, dims, 2);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (init tensor b): %.3f ms\n", get_time_ms(&op_start, &op_end));

    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool init_result = qg_tensor_init(&result, dims, 2);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (init result tensor): %.3f ms\n", get_time_ms(&op_start, &op_end));

    if (!init_a || !init_b || !init_result) {
        printf("Failed to initialize tensors\n");
        return false;
    }

    // Initialize test data with complex values
    for (size_t i = 0; i < a.total_size; i++) {
        a.data[i].real = (float)i;
        a.data[i].imag = (float)(i + 1);
        b.data[i].real = (float)(i * 2);
        b.data[i].imag = (float)(i * 3);
    }

    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool add_result = qg_tensor_add(&result, &a, &b);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (tensor addition): %.3f ms\n", get_time_ms(&op_start, &op_end));
    
    if (!add_result) {
        printf("Tensor addition failed\n");
        return false;
    }

    // Verify results
    bool success = true;
    for (size_t i = 0; i < result.total_size; i++) {
        float expected_real = (float)(i + i * 2);
        float expected_imag = (float)(i + 1 + i * 3);
        if (fabs(result.data[i].real - expected_real) > EPSILON ||
            fabs(result.data[i].imag - expected_imag) > EPSILON) {
            printf("Mismatch at index %zu: got %f + %fi, expected %f + %fi\n", 
                   i, result.data[i].real, result.data[i].imag, expected_real, expected_imag);
            success = false;
            break;
        }
    }

    qg_tensor_cleanup(&a);
    qg_tensor_cleanup(&b);
    qg_tensor_cleanup(&result);

    if (success) {
        tests_passed++;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time taken: %.3f ms\n", get_time_ms(&start, &end));
    return success;
}

static bool test_tensor_scale() {
    printf("Testing tensor scaling...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    tests_run++;

    size_t dims[] = {2, 3};
    tensor_t tensor;
    struct timespec op_start, op_end;
    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool init_result = qg_tensor_init(&tensor, dims, 2);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (tensor init): %.3f ms\n", get_time_ms(&op_start, &op_end));
    
    if (!init_result) {
        printf("Failed to initialize tensor\n");
        return false;
    }

    // Initialize test data with complex values
    for (size_t i = 0; i < tensor.total_size; i++) {
        tensor.data[i].real = (float)i;
        tensor.data[i].imag = (float)(i + 0.5);
    }

    float scalar = 2.0f;
    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool scale_result = qg_tensor_scale(&tensor, scalar);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (tensor scaling): %.3f ms\n", get_time_ms(&op_start, &op_end));
    
    if (!scale_result) {
        printf("Tensor scaling failed\n");
        return false;
    }

    // Verify results
    bool success = true;
    for (size_t i = 0; i < tensor.total_size; i++) {
        float expected_real = (float)i * scalar;
        float expected_imag = (float)(i + 0.5) * scalar;
        if (fabs(tensor.data[i].real - expected_real) > EPSILON ||
            fabs(tensor.data[i].imag - expected_imag) > EPSILON) {
            printf("Mismatch at index %zu: got %f + %fi, expected %f + %fi\n", 
                   i, tensor.data[i].real, tensor.data[i].imag, expected_real, expected_imag);
            success = false;
            break;
        }
    }

    qg_tensor_cleanup(&tensor);

    if (success) {
        tests_passed++;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time taken: %.3f ms\n", get_time_ms(&start, &end));
    return success;
}

static bool test_tensor_reshape() {
    printf("Testing tensor reshape...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    tests_run++;

    size_t dims[] = {2, 3};
    size_t new_dims[] = {3, 2};
    tensor_t tensor;
    struct timespec op_start, op_end;
    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool init_result = qg_tensor_init(&tensor, dims, 2);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (tensor init): %.3f ms\n", get_time_ms(&op_start, &op_end));
    
    if (!init_result) {
        printf("Failed to initialize tensor\n");
        return false;
    }

    // Initialize test data with complex values
    for (size_t i = 0; i < tensor.total_size; i++) {
        tensor.data[i].real = (float)i;
        tensor.data[i].imag = (float)(i * 1.5);
    }

    ComplexFloat* result = (ComplexFloat*)malloc(tensor.total_size * sizeof(ComplexFloat));
    if (!result) {
        printf("Failed to allocate result buffer\n");
        qg_tensor_cleanup(&tensor);
        return false;
    }

    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool reshape_result = qg_tensor_reshape(result, tensor.data, new_dims, 2);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (tensor reshape): %.3f ms\n", get_time_ms(&op_start, &op_end));
    
    if (!reshape_result) {
        printf("Tensor reshape failed\n");
        free(result);
        qg_tensor_cleanup(&tensor);
        return false;
    }

    // Verify total size remains the same
    bool success = true;
    size_t total_size = qg_tensor_get_size(new_dims, 2);
    if (total_size != tensor.total_size) {
        printf("Incorrect total size after reshape\n");
        success = false;
    }

    free(result);
    qg_tensor_cleanup(&tensor);

    if (success) {
        tests_passed++;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time taken: %.3f ms\n", get_time_ms(&start, &end));
    return success;
}

static bool test_tensor_transpose() {
    printf("Testing tensor transpose...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    tests_run++;

    // Test 2D transpose first
    printf("\nTesting 2D transpose (3x4):\n");
    size_t dims_2d[] = {3, 4};
    size_t perm_2d[] = {1, 0};
    tensor_t tensor_2d;
    ComplexFloat* result_2d = NULL;

    struct timespec op_start, op_end;
    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool init_result = qg_tensor_init(&tensor_2d, dims_2d, 2);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (2D tensor init): %.3f ms\n", get_time_ms(&op_start, &op_end));
    
    if (!init_result) {
        printf("Failed to initialize 2D tensor\n");
        return false;
    }

    // Initialize with recognizable pattern
    for (size_t i = 0; i < dims_2d[0]; i++) {
        for (size_t j = 0; j < dims_2d[1]; j++) {
            tensor_2d.data[i * dims_2d[1] + j] = complex_float_create(
                (float)(10 * (i + 1) + (j + 1)),
                (float)(5 * (i + 1) + (j + 1))
            );
        }
    }

    result_2d = (ComplexFloat*)malloc(tensor_2d.total_size * sizeof(ComplexFloat));
    if (!result_2d) {
        printf("Failed to allocate result buffer for 2D tensor\n");
        qg_tensor_cleanup(&tensor_2d);
        return false;
    }

    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool transpose_result = qg_tensor_transpose(result_2d, tensor_2d.data, dims_2d, 2, perm_2d);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (2D tensor transpose): %.3f ms\n", get_time_ms(&op_start, &op_end));
    
    if (!transpose_result) {
        printf("2D tensor transpose failed\n");
        free(result_2d);
        qg_tensor_cleanup(&tensor_2d);
        return false;
    }

    // Print and verify 2D transpose
    printf("Original 2D tensor (3x4):\n");
    for (size_t i = 0; i < dims_2d[0]; i++) {
        for (size_t j = 0; j < dims_2d[1]; j++) {
            printf("%6.1f + %6.1fi ", tensor_2d.data[i * dims_2d[1] + j].real, tensor_2d.data[i * dims_2d[1] + j].imag);
        }
        printf("\n");
    }

    printf("\nTransposed 2D tensor (4x3):\n");
    for (size_t i = 0; i < dims_2d[1]; i++) {
        for (size_t j = 0; j < dims_2d[0]; j++) {
            printf("%6.1f + %6.1fi ", result_2d[i * dims_2d[0] + j].real, result_2d[i * dims_2d[0] + j].imag);
        }
        printf("\n");
    }

    // Test 3D transpose
    printf("\nTesting 3D transpose (2x3x4):\n");
    size_t dims_3d[] = {2, 3, 4};  // 2x3x4 tensor
    size_t perm_3d[] = {2, 0, 1};  // Transpose to 4x2x3
    tensor_t tensor_3d;
    ComplexFloat* result_3d = NULL;

    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool init_3d_result = qg_tensor_init(&tensor_3d, dims_3d, 3);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (3D tensor init): %.3f ms\n", get_time_ms(&op_start, &op_end));
    
    if (!init_3d_result) {
        printf("Failed to initialize 3D tensor\n");
        free(result_2d);
        qg_tensor_cleanup(&tensor_2d);
        return false;
    }

    // Initialize with recognizable pattern
    for (size_t i = 0; i < dims_3d[0]; i++) {
        for (size_t j = 0; j < dims_3d[1]; j++) {
            for (size_t k = 0; k < dims_3d[2]; k++) {
                tensor_3d.data[i * dims_3d[1] * dims_3d[2] + j * dims_3d[2] + k] = 
                    complex_float_create(
                        (float)(100 * (i + 1) + 10 * (j + 1) + (k + 1)),
                        (float)(50 * (i + 1) + 5 * (j + 1) + (k + 1))
                    );
            }
        }
    }

    result_3d = (ComplexFloat*)malloc(tensor_3d.total_size * sizeof(ComplexFloat));
    if (!result_3d) {
        printf("Failed to allocate result buffer for 3D tensor\n");
        free(result_2d);
        qg_tensor_cleanup(&tensor_2d);
        qg_tensor_cleanup(&tensor_3d);
        return false;
    }

    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool transpose_3d_result = qg_tensor_transpose(result_3d, tensor_3d.data, dims_3d, 3, perm_3d);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (3D tensor transpose): %.3f ms\n", get_time_ms(&op_start, &op_end));
    
    if (!transpose_3d_result) {
        printf("3D tensor transpose failed\n");
        free(result_2d);
        free(result_3d);
        qg_tensor_cleanup(&tensor_2d);
        qg_tensor_cleanup(&tensor_3d);
        return false;
    }

    // Print and verify 3D transpose
    printf("Original 3D tensor (2x3x4):\n");
    for (size_t i = 0; i < dims_3d[0]; i++) {
        printf("Slice %zu:\n", i);
        for (size_t j = 0; j < dims_3d[1]; j++) {
            for (size_t k = 0; k < dims_3d[2]; k++) {
                printf("%7.1f + %7.1fi ", 
                       tensor_3d.data[i * dims_3d[1] * dims_3d[2] + j * dims_3d[2] + k].real,
                       tensor_3d.data[i * dims_3d[1] * dims_3d[2] + j * dims_3d[2] + k].imag);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("\nTransposed 3D tensor (4x2x3):\n");
    for (size_t i = 0; i < dims_3d[2]; i++) {
        printf("Slice %zu:\n", i);
        for (size_t j = 0; j < dims_3d[0]; j++) {
            for (size_t k = 0; k < dims_3d[1]; k++) {
                printf("%7.1f + %7.1fi ", 
                       result_3d[i * dims_3d[0] * dims_3d[1] + j * dims_3d[1] + k].real,
                       result_3d[i * dims_3d[0] * dims_3d[1] + j * dims_3d[1] + k].imag);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Verify 2D transpose
    bool success = true;
    for (size_t i = 0; i < dims_2d[0]; i++) {
        for (size_t j = 0; j < dims_2d[1]; j++) {
            ComplexFloat original = tensor_2d.data[i * dims_2d[1] + j];
            ComplexFloat transposed = result_2d[j * dims_2d[0] + i];
            if (fabs(original.real - transposed.real) > EPSILON ||
                fabs(original.imag - transposed.imag) > EPSILON) {
                printf("2D transpose mismatch: original[%zu,%zu]=%f+%fi should equal transposed[%zu,%zu]=%f+%fi\n",
                       i, j, original.real, original.imag, j, i, transposed.real, transposed.imag);
                success = false;
            }
        }
    }

    // Verify 3D transpose
    for (size_t i = 0; i < dims_3d[0]; i++) {
        for (size_t j = 0; j < dims_3d[1]; j++) {
            for (size_t k = 0; k < dims_3d[2]; k++) {
                ComplexFloat original = tensor_3d.data[i * dims_3d[1] * dims_3d[2] + j * dims_3d[2] + k];
                ComplexFloat transposed = result_3d[k * dims_3d[0] * dims_3d[1] + i * dims_3d[1] + j];
                if (fabs(original.real - transposed.real) > EPSILON ||
                    fabs(original.imag - transposed.imag) > EPSILON) {
                    printf("3D transpose mismatch: original[%zu,%zu,%zu]=%f+%fi should equal transposed[%zu,%zu,%zu]=%f+%fi\n",
                           i, j, k, original.real, original.imag, k, i, j, transposed.real, transposed.imag);
                    success = false;
                }
            }
        }
    }

    free(result_2d);
    free(result_3d);
    qg_tensor_cleanup(&tensor_2d);
    qg_tensor_cleanup(&tensor_3d);

    if (success) {
        tests_passed++;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time taken: %.3f ms\n", get_time_ms(&start, &end));
    return success;
}

static bool test_tensor_contract() {
    printf("Testing tensor contraction...\n");
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    tests_run++;

    // Test matrix multiplication (2D contraction)
    printf("\nTesting matrix multiplication (2x3 * 3x2):\n");
    size_t dims_a[] = {2, 3};
    size_t dims_b[] = {3, 2};
    size_t contract_a[] = {1};
    size_t contract_b[] = {0};
    tensor_t a, b;
    ComplexFloat* result = NULL;

    struct timespec op_start, op_end;
    
    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool init_a = qg_tensor_init(&a, dims_a, 2);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (init tensor a): %.3f ms\n", get_time_ms(&op_start, &op_end));

    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool init_b = qg_tensor_init(&b, dims_b, 2);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (init tensor b): %.3f ms\n", get_time_ms(&op_start, &op_end));

    if (!init_a || !init_b) {
        printf("Failed to initialize tensors\n");
        return false;
    }

    // Initialize with recognizable pattern
    for (size_t i = 0; i < dims_a[0]; i++) {
        for (size_t j = 0; j < dims_a[1]; j++) {
            a.data[i * dims_a[1] + j] = complex_float_create(
                (float)(i + j + 1),
                (float)(i - j + 0.5)
            );
        }
    }
    for (size_t i = 0; i < dims_b[0]; i++) {
        for (size_t j = 0; j < dims_b[1]; j++) {
            b.data[i * dims_b[1] + j] = complex_float_create(
                (float)(i + j + 1),
                (float)(i * j + 0.5)
            );
        }
    }

    size_t result_size = (dims_a[0] * dims_b[1]);
    result = (ComplexFloat*)malloc(result_size * sizeof(ComplexFloat));
    if (!result) {
        printf("Failed to allocate result buffer\n");
        qg_tensor_cleanup(&a);
        qg_tensor_cleanup(&b);
        return false;
    }

    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool contract_result = qg_tensor_contract(result, a.data, b.data, dims_a, dims_b, 2, 2,
                                            contract_a, contract_b, 1);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (2D tensor contraction): %.3f ms\n", get_time_ms(&op_start, &op_end));
    
    if (!contract_result) {
        printf("Matrix multiplication failed\n");
        free(result);
        qg_tensor_cleanup(&a);
        qg_tensor_cleanup(&b);
        return false;
    }

    // Print matrices and result
    printf("Matrix A (2x3):\n");
    for (size_t i = 0; i < dims_a[0]; i++) {
        for (size_t j = 0; j < dims_a[1]; j++) {
            printf("%6.1f + %6.1fi ", a.data[i * dims_a[1] + j].real, a.data[i * dims_a[1] + j].imag);
        }
        printf("\n");
    }

    printf("\nMatrix B (3x2):\n");
    for (size_t i = 0; i < dims_b[0]; i++) {
        for (size_t j = 0; j < dims_b[1]; j++) {
            printf("%6.1f + %6.1fi ", b.data[i * dims_b[1] + j].real, b.data[i * dims_b[1] + j].imag);
        }
        printf("\n");
    }

    printf("\nResult (2x2):\n");
    for (size_t i = 0; i < dims_a[0]; i++) {
        for (size_t j = 0; j < dims_b[1]; j++) {
            printf("%6.1f + %6.1fi ", result[i * dims_b[1] + j].real, result[i * dims_b[1] + j].imag);
        }
        printf("\n");
    }

    // Test 3D tensor contraction
    printf("\nTesting 3D tensor contraction (2x3x4 * 4x3x2):\n");
    size_t dims_3d_a[] = {2, 3, 4};
    size_t dims_3d_b[] = {4, 3, 2};
    size_t contract_3d_a[] = {1, 2};
    size_t contract_3d_b[] = {1, 0};
    tensor_t a_3d, b_3d;
    ComplexFloat* result_3d = NULL;

    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool init_a_3d = qg_tensor_init(&a_3d, dims_3d_a, 3);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (init 3D tensor a): %.3f ms\n", get_time_ms(&op_start, &op_end));

    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool init_b_3d = qg_tensor_init(&b_3d, dims_3d_b, 3);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (init 3D tensor b): %.3f ms\n", get_time_ms(&op_start, &op_end));

    if (!init_a_3d || !init_b_3d) {
        printf("Failed to initialize 3D tensors\n");
        free(result);
        qg_tensor_cleanup(&a);
        qg_tensor_cleanup(&b);
        return false;
    }

    // Initialize 3D tensors with recognizable pattern
    for (size_t i = 0; i < dims_3d_a[0]; i++) {
        for (size_t j = 0; j < dims_3d_a[1]; j++) {
            for (size_t k = 0; k < dims_3d_a[2]; k++) {
                a_3d.data[i * dims_3d_a[1] * dims_3d_a[2] + j * dims_3d_a[2] + k] = 
                    complex_float_create(
                        (float)(100 * (i + 1) + 10 * (j + 1) + (k + 1)),
                        (float)(50 * (i + 1) + 5 * (j + 1) + (k + 1))
                    );
            }
        }
    }
    for (size_t i = 0; i < dims_3d_b[0]; i++) {
        for (size_t j = 0; j < dims_3d_b[1]; j++) {
            for (size_t k = 0; k < dims_3d_b[2]; k++) {
                b_3d.data[i * dims_3d_b[1] * dims_3d_b[2] + j * dims_3d_b[2] + k] = 
                    complex_float_create(
                        (float)(i + j + k + 1),
                        (float)((i + 1) * (j + 1) + (k + 1))
                    );
            }
        }
    }

    size_t result_3d_size = dims_3d_a[0] * dims_3d_b[2];
    result_3d = (ComplexFloat*)malloc(result_3d_size * sizeof(ComplexFloat));
    if (!result_3d) {
        printf("Failed to allocate 3D result buffer\n");
        free(result);
        qg_tensor_cleanup(&a);
        qg_tensor_cleanup(&b);
        qg_tensor_cleanup(&a_3d);
        qg_tensor_cleanup(&b_3d);
        return false;
    }

    clock_gettime(CLOCK_MONOTONIC, &op_start);
    bool contract_3d_result = qg_tensor_contract(result_3d, a_3d.data, b_3d.data, dims_3d_a, dims_3d_b, 3, 3,
                                               contract_3d_a, contract_3d_b, 2);
    clock_gettime(CLOCK_MONOTONIC, &op_end);
    printf("  Operation time (3D tensor contraction): %.3f ms\n", get_time_ms(&op_start, &op_end));
    
    if (!contract_3d_result) {
        printf("3D tensor contraction failed\n");
        free(result);
        free(result_3d);
        qg_tensor_cleanup(&a);
        qg_tensor_cleanup(&b);
        qg_tensor_cleanup(&a_3d);
        qg_tensor_cleanup(&b_3d);
        return false;
    }

    // Print 3D tensors and result
    printf("Tensor A (2x3x4):\n");
    for (size_t i = 0; i < dims_3d_a[0]; i++) {
        printf("Slice %zu:\n", i);
        for (size_t j = 0; j < dims_3d_a[1]; j++) {
            for (size_t k = 0; k < dims_3d_a[2]; k++) {
                printf("%7.1f + %7.1fi ", 
                       a_3d.data[i * dims_3d_a[1] * dims_3d_a[2] + j * dims_3d_a[2] + k].real,
                       a_3d.data[i * dims_3d_a[1] * dims_3d_a[2] + j * dims_3d_a[2] + k].imag);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("\nTensor B (4x3x2):\n");
    for (size_t i = 0; i < dims_3d_b[0]; i++) {
        printf("Slice %zu:\n", i);
        for (size_t j = 0; j < dims_3d_b[1]; j++) {
            for (size_t k = 0; k < dims_3d_b[2]; k++) {
                printf("%7.1f + %7.1fi ", 
                       b_3d.data[i * dims_3d_b[1] * dims_3d_b[2] + j * dims_3d_b[2] + k].real,
                       b_3d.data[i * dims_3d_b[1] * dims_3d_b[2] + j * dims_3d_b[2] + k].imag);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("\nResult (2x2):\n");
    for (size_t i = 0; i < dims_3d_a[0]; i++) {
        for (size_t j = 0; j < dims_3d_b[2]; j++) {
            printf("%7.1f + %7.1fi ", result_3d[i * dims_3d_b[2] + j].real, result_3d[i * dims_3d_b[2] + j].imag);
        }
        printf("\n");
    }

    // Verify 2D contraction
    bool success = true;
    // Manual calculation for verification
    ComplexFloat expected[4] = {COMPLEX_FLOAT_ZERO}; // 2x2 result
    for (size_t i = 0; i < dims_a[0]; i++) {
        for (size_t j = 0; j < dims_b[1]; j++) {
            ComplexFloat sum = COMPLEX_FLOAT_ZERO;
            for (size_t k = 0; k < dims_a[1]; k++) {
                ComplexFloat prod = complex_float_multiply(a.data[i * dims_a[1] + k], b.data[k * dims_b[1] + j]);
                sum = complex_float_add(sum, prod);
            }
            expected[i * dims_b[1] + j] = sum;
            ComplexFloat res = result[i * dims_b[1] + j];
            if (fabs(res.real - sum.real) > EPSILON || fabs(res.imag - sum.imag) > EPSILON) {
                printf("Matrix multiplication mismatch at [%zu,%zu]: got %f+%fi, expected %f+%fi\n",
                       i, j, res.real, res.imag, sum.real, sum.imag);
                success = false;
            }
        }
    }

    free(result);
    free(result_3d);
    qg_tensor_cleanup(&a);
    qg_tensor_cleanup(&b);
    qg_tensor_cleanup(&a_3d);
    qg_tensor_cleanup(&b_3d);

    if (success) {
        tests_passed++;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time taken: %.3f ms\n", get_time_ms(&start, &end));
    return success;
}

int main() {
    printf("Running tensor operations tests...\n\n");
    struct timespec suite_start, suite_end;
    clock_gettime(CLOCK_MONOTONIC, &suite_start);

    test_tensor_init();
    test_tensor_add();
    test_tensor_scale();
    test_tensor_reshape();
    test_tensor_transpose();
    test_tensor_contract();

    clock_gettime(CLOCK_MONOTONIC, &suite_end);
    printf("\nTest summary: %d/%d tests passed\n", tests_passed, tests_run);
    printf("Total time: %.3f ms\n", get_time_ms(&suite_start, &suite_end));
    return tests_passed == tests_run ? 0 : 1;
}
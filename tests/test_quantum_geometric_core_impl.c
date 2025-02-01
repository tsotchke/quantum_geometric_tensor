#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_geometric_types.h"
#include "quantum_geometric/core/quantum_geometric_constants.h"
#include "quantum_geometric/core/error_codes.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

// Test initialization and cleanup
static void test_initialization(void) {
    qgt_error_t status;
    
    // Test initialization
    status = geometric_core_initialize();
    assert(status == QGT_SUCCESS);
    
    // Test double initialization
    status = geometric_core_initialize();
    assert(status == QGT_ERROR_ALREADY_INITIALIZED);
    
    // Test cleanup
    geometric_core_shutdown();
    
    // Test initialization after cleanup
    status = geometric_core_initialize();
    assert(status == QGT_SUCCESS);
    
    geometric_core_shutdown();
}

// Test memory management
static void test_memory_management(void) {
    qgt_error_t status;
    void* ptr = NULL;
    size_t size = 1024;
    
    // Initialize core
    status = geometric_core_initialize();
    assert(status == QGT_SUCCESS);
    
    // Test allocation
    status = geometric_core_allocate(&ptr, size);
    assert(status == QGT_SUCCESS);
    assert(ptr != NULL);
    
    // Test memory operations
    status = geometric_core_memset(ptr, 0, size);
    assert(status == QGT_SUCCESS);
    
    void* ptr2 = NULL;
    status = geometric_core_allocate(&ptr2, size);
    assert(status == QGT_SUCCESS);
    assert(ptr2 != NULL);
    
    status = geometric_core_memcpy(ptr2, ptr, size);
    assert(status == QGT_SUCCESS);
    
    // Test memory stats
    size_t total, peak, count;
    status = geometric_core_get_memory_stats(&total, &peak, &count);
    assert(status == QGT_SUCCESS);
    assert(total >= 2 * size);
    assert(peak >= total);
    assert(count >= 2);
    
    // Test deallocation
    geometric_core_free(ptr);
    geometric_core_free(ptr2);
    
    geometric_core_shutdown();
}

// Test core operations
static void test_core_operations(void) {
    qgt_error_t status;
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float result[4];
    
    // Initialize core
    status = geometric_core_initialize();
    assert(status == QGT_SUCCESS);
    
    // Test addition
    status = geometric_core_add(result, a, b, 4);
    assert(status == QGT_SUCCESS);
    for (int i = 0; i < 4; i++) {
        assert(result[i] == a[i] + b[i]);
    }
    
    // Test subtraction
    status = geometric_core_subtract(result, a, b, 4);
    assert(status == QGT_SUCCESS);
    for (int i = 0; i < 4; i++) {
        assert(result[i] == a[i] - b[i]);
    }
    
    // Test multiplication
    status = geometric_core_multiply(result, a, b, 4);
    assert(status == QGT_SUCCESS);
    for (int i = 0; i < 4; i++) {
        assert(result[i] == a[i] * b[i]);
    }
    
    // Test division
    status = geometric_core_divide(result, a, b, 4);
    assert(status == QGT_SUCCESS);
    for (int i = 0; i < 4; i++) {
        assert(result[i] == a[i] / b[i]);
    }
    
    geometric_core_shutdown();
}

// Test error handling
static void test_error_handling(void) {
    qgt_error_t status;
    void* ptr = NULL;
    
    // Test uninitialized state
    status = geometric_core_allocate(&ptr, 1024);
    assert(status == QGT_ERROR_NOT_INITIALIZED);
    
    // Test null pointer
    status = geometric_core_initialize();
    assert(status == QGT_SUCCESS);
    
    status = geometric_core_allocate(NULL, 1024);
    assert(status == QGT_ERROR_INVALID_ARGUMENT);
    
    // Test zero size
    status = geometric_core_allocate(&ptr, 0);
    assert(status == QGT_ERROR_INVALID_ARGUMENT);
    
    // Test error string
    const char* error_str = geometric_core_get_error_string(QGT_ERROR_INVALID_ARGUMENT);
    assert(error_str != NULL);
    assert(strlen(error_str) > 0);
    
    geometric_core_shutdown();
}

// Test device management
static void test_device_management(void) {
    qgt_error_t status;
    size_t count;
    
    status = geometric_core_initialize();
    assert(status == QGT_SUCCESS);
    
    // Test device count
    status = geometric_core_get_device_count(&count);
    assert(status == QGT_SUCCESS);
    assert(count > 0);
    
    // Test device selection
    status = geometric_core_set_device(0);
    assert(status == QGT_SUCCESS);
    
    // Test invalid device
    status = geometric_core_set_device(count);
    assert(status == QGT_ERROR_INVALID_ARGUMENT);
    
    geometric_core_shutdown();
}

int main(void) {
    printf("Running quantum geometric core tests...\n");
    
    test_initialization();
    test_memory_management();
    test_core_operations();
    test_error_handling();
    test_device_management();
    
    printf("All tests passed!\n");
    return 0;
}

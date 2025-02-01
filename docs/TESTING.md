# Testing Guide

This document describes the testing infrastructure and procedures for the Quantum Geometric Learning library.

## Overview

The testing infrastructure consists of several components:
- Unit tests for individual components
- Integration tests for system-wide functionality
- Performance tests for benchmarking
- Memory leak detection
- Code coverage analysis
- Static code analysis

## Prerequisites

The following tools are required:
- CMake (3.12 or higher)
- GCC or Clang compiler
- gcov and lcov for coverage analysis
- clang-format for code formatting
- cppcheck for static analysis
- Valgrind for memory checking

Install on Ubuntu/Debian:
```bash
sudo apt-get install cmake gcc g++ gcov lcov clang-format cppcheck valgrind
```

Install on macOS:
```bash
brew install cmake lcov clang-format cppcheck valgrind
```

## Running Tests

### Quick Start

To run all tests with default settings:
```bash
./tools/run_test_suite.sh
```

This will:
1. Build the project with test coverage enabled
2. Run all unit tests
3. Run integration tests
4. Perform memory checks
5. Generate coverage reports
6. Run performance tests
7. Generate a test summary

### Test Categories

#### Unit Tests
Individual component tests located in `tests/`:
```bash
cd build && ctest --output-on-failure
```

#### Integration Tests
System-wide tests in `tests/integration/`:
```bash
cd build && make run_integration_tests
```

#### Performance Tests
Benchmarks in `benchmarks/`:
```bash
cd build && make run_performance_tests
```

### Coverage Analysis

Coverage reports are generated automatically when running the test suite. View the HTML report:
```bash
open build/coverage_report/index.html
```

### Memory Checking

Memory checks are performed using Valgrind:
```bash
cd build && ctest -T memcheck
```

### Static Analysis

Run static analysis tools:
```bash
# Format code
clang-format -i src/**/*.{c,h} include/**/*.h tests/**/*.c

# Static analysis
cppcheck --enable=all src/ include/ tests/
```

## Writing Tests

### Unit Tests

Create new unit tests in `tests/` following this template:
```c
#include "quantum_geometric/core/component.h"
#include <assert.h>

void test_feature() {
    // Setup
    ...
    
    // Test
    assert(condition);
    
    // Cleanup
    ...
}

int main() {
    test_feature();
    return 0;
}
```

### Integration Tests

Create new integration tests in `tests/integration/` following this template:
```c
#include "quantum_geometric/core/quantum_geometric_core.h"
#include <assert.h>

void test_workflow() {
    // Initialize system
    ...
    
    // Run complex workflow
    ...
    
    // Verify results
    assert(condition);
    
    // Cleanup
    ...
}

int main() {
    test_workflow();
    return 0;
}
```

### Performance Tests

Create new benchmarks in `benchmarks/` following this template:
```c
#include "quantum_geometric/core/component.h"
#include <time.h>

void benchmark_operation() {
    clock_t start = clock();
    
    // Run operation multiple times
    ...
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Operation time: %f seconds\n", time_spent);
}

int main() {
    benchmark_operation();
    return 0;
}
```

## Test Configuration

### CMake Options

Configure test builds with these CMake options:
```bash
cmake -DENABLE_TESTING=ON          # Enable testing
      -DENABLE_COVERAGE=ON         # Enable coverage analysis
      -DENABLE_SANITIZERS=ON       # Enable sanitizers
      -DENABLE_VALGRIND=ON         # Enable Valgrind integration
      ..
```

### Environment Variables

Set these environment variables to modify test behavior:
```bash
export QG_TEST_ENV=development     # Test environment (development/staging/production)
export QG_TEST_LOG_LEVEL=debug     # Log level for tests
export QG_TEST_TIMEOUT=300         # Test timeout in seconds
```

## Continuous Integration

Tests are automatically run in CI on:
- Every push to main branch
- Every pull request
- Nightly builds

CI configuration is in `.github/workflows/ci.yml`

## Best Practices

1. **Test Organization**
   - Keep tests focused and independent
   - Use clear, descriptive test names
   - Group related tests together
   - Clean up resources after tests

2. **Coverage**
   - Aim for >90% code coverage
   - Cover edge cases and error conditions
   - Test both success and failure paths
   - Include performance-critical paths

3. **Performance Testing**
   - Use consistent test data
   - Run multiple iterations
   - Test with varying load sizes
   - Compare against baselines

4. **Integration Testing**
   - Test complete workflows
   - Verify system interactions
   - Test configuration changes
   - Include error recovery

5. **Memory Management**
   - Check for memory leaks
   - Verify resource cleanup
   - Test memory-intensive operations
   - Monitor memory usage patterns

## Troubleshooting

### Common Issues

1. **Tests Failing**
   - Check test logs in `build/Testing/Temporary/`
   - Verify test environment setup
   - Check for resource conflicts
   - Review recent code changes

2. **Memory Leaks**
   - Run Valgrind with detailed output
   - Check resource cleanup
   - Review object lifecycles
   - Monitor system resources

3. **Performance Issues**
   - Profile the code
   - Check system load
   - Verify test data
   - Compare with baselines

4. **Coverage Problems**
   - Check build configuration
   - Verify test execution
   - Review uncovered code
   - Add missing test cases

## Support

For testing-related issues:
1. Check the troubleshooting guide
2. Review test logs
3. Search issue tracker
4. Contact development team

## Further Reading

- [Error Handling Guide](ERROR_HANDLING.md)
- [Performance Optimization](PERFORMANCE_OPTIMIZATION.md)
- [Memory Management](MEMORY_MANAGEMENT.md)
- [Production Monitoring](PRODUCTION_MONITORING.md)

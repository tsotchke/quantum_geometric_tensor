# Numerical Backend Documentation

## Overview

The numerical backend provides high-performance linear algebra operations for quantum geometric computations. It supports multiple providers including:

- CPU (basic implementation)
- Apple Accelerate Framework
- OpenBLAS (planned)
- Intel MKL (planned)

## Architecture

The backend is organized into several layers:

### 1. Complex Arithmetic Layer (`complex_arithmetic.h`)
- Basic complex number operations
- Vector/matrix operations
- Type conversions for different backends
- Platform-independent interface

### 2. LAPACK Wrapper (`lapack_wrapper.h`)
- Unified interface to LAPACK operations
- Automatic provider selection
- Error handling and workspace management
- Support for different matrix layouts

### 3. Backend Implementation Layer
- `numerical_backend_cpu.c`: Basic CPU implementation
- `numerical_backend_accelerate.c`: Apple Accelerate implementation
- Future: OpenBLAS and MKL implementations

### 4. Backend Selection Layer (`numerical_backend_selector.c`)
- Runtime backend selection
- Capability detection
- Performance metrics
- Error handling

## Usage

```c
// Initialize backend
numerical_config_t config = {
    .type = NUMERICAL_BACKEND_CPU,
    .max_threads = 4,
    .use_fma = true
};
initialize_numerical_backend(&config);

// Perform operations
ComplexFloat a[4] = {{1,0}, {0,1}, {-1,0}, {0,-1}};
ComplexFloat b[4] = {{1,0}, {1,0}, {1,0}, {1,0}};
ComplexFloat c[4];
numerical_matrix_add(a, b, c, 2, 2);

// Clean up
shutdown_numerical_backend();
```

## Error Handling

All operations return a boolean success indicator and set an error code that can be retrieved:

```c
numerical_error_t error = get_last_numerical_error();
const char* error_str = get_numerical_error_string(error);
```

## LAPACK Integration

LAPACK operations are available through the wrapper interface:

```c
// Perform SVD
ComplexFloat a[] = {...}; // Input matrix
ComplexFloat u[] = {...}; // Left singular vectors
float s[] = {...};        // Singular values
ComplexFloat vt[] = {...}; // Right singular vectors
lapack_svd(a, m, n, u, s, vt, LAPACK_ROW_MAJOR);
```

## Future Development

1. Complete OpenBLAS Integration
   - Add build system detection
   - Implement type conversions
   - Add performance benchmarks

2. Additional LAPACK Operations
   - QR decomposition
   - Eigendecomposition
   - Cholesky decomposition
   - LU factorization

3. Performance Optimizations
   - Workspace reuse
   - Thread pool integration
   - Cache-aware algorithms

4. Testing
   - Unit tests for all operations
   - Performance regression tests
   - Numerical stability tests

## Contributing

When adding new features:

1. Update the appropriate interface header
2. Implement for CPU backend first
3. Add accelerated implementations
4. Add tests and benchmarks
5. Update documentation

## Platform Support

- macOS: Full support (CPU + Accelerate)
- Linux: CPU support, OpenBLAS planned
- Windows: CPU support, MKL planned

## Performance Considerations

- Matrix layout (row vs column major)
- Memory alignment
- Cache utilization
- Thread synchronization
- SIMD operations

## Error Codes

- `NUMERICAL_SUCCESS`: Operation completed successfully
- `NUMERICAL_ERROR_INVALID_ARGUMENT`: Invalid input parameters
- `NUMERICAL_ERROR_MEMORY`: Memory allocation failed
- `NUMERICAL_ERROR_BACKEND`: Backend-specific error
- `NUMERICAL_ERROR_COMPUTATION`: Computation failed to converge
- `NUMERICAL_ERROR_NOT_IMPLEMENTED`: Operation not supported

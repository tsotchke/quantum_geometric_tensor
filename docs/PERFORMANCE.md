# Performance Optimization Guide

This guide provides detailed information on optimizing performance when using the Quantum Geometric Learning library.

## Performance Profiling

### Built-in Profiling Tools

1. Enable performance monitoring:
```c
physicsml_profiler_start();

// Your code here

physicsml_profiler_report();
```

2. Monitor specific operations:
```c
physicsml_profile_section("Tensor Contractions");
// Operations
physicsml_end_profile_section();
```

3. Generate performance report:
```bash
./tools/analyze_performance_log.py performance.log
```

### External Profiling

1. CPU profiling with perf:
```bash
perf record ./your_program
perf report
```

2. GPU profiling:
```bash
# NVIDIA
nvprof ./your_program
nsight-compute ./your_program

# AMD
rocprof ./your_program
```

## Optimization Strategies

### 1. Tensor Network Optimization

#### Bond Dimension Selection

Choose appropriate bond dimensions:
```c
// Small systems (< 10 qubits)
TreeTensorNetwork* ttn = physicsml_ttn_create(
    num_levels,
    16,     // Bond dimension
    2       // Physical dimension
);

// Large systems (> 20 qubits)
TreeTensorNetwork* ttn = physicsml_ttn_create(
    num_levels,
    32,     // Larger bond dimension
    2
);
```

#### Compression

Apply tensor compression:
```c
// Basic compression
physicsml_ttn_compress(ttn, 1e-8);

// Adaptive compression
physicsml_ttn_compress_adaptive(
    ttn,
    1e-8,           // Base tolerance
    0.1,            // Growth factor
    max_bond_dim    // Maximum bond dimension
);
```

#### Contraction Order

Optimize contraction sequence:
```c
// Get optimal contraction order
size_t* order;
size_t num_contractions;
physicsml_optimize_contraction_order(
    ttn,
    &order,
    &num_contractions
);

// Contract using optimal order
PhysicsMLTensor* result = physicsml_contract_network_ordered(
    ttn,
    order,
    num_contractions
);
```

### 2. Hardware Acceleration

#### GPU Optimization

1. Enable GPU support:
```bash
cmake -DUSE_CUDA=ON -DCUDA_ARCH=sm_80 ..  # For NVIDIA
cmake -DUSE_OPENCL=ON -DPREFER_AMD=ON ..   # For AMD
```

2. Configure GPU memory:
```c
// Set memory limit
physicsml_set_gpu_memory_limit(4ULL * 1024 * 1024 * 1024);  // 4GB

// Enable memory pooling
physicsml_enable_gpu_memory_pool(true);

// Set cache size
physicsml_set_gpu_cache_size(1024 * 1024 * 1024);  // 1GB
```

3. Batch operations:
```c
// Batch tensor contractions
physicsml_tensor_batch_contract(
    tensors,
    num_tensors,
    "ij,jk->ik"
);

// Batch parameter updates
physicsml_update_parameters_batch(
    parameters,
    gradients,
    num_params,
    learning_rate
);
```

#### Multi-GPU Support

1. Enable multiple GPUs:
```c
// Initialize multi-GPU
size_t num_gpus = physicsml_get_device_count();
physicsml_enable_multi_gpu(true);

// Distribute work
for (size_t i = 0; i < num_gpus; i++) {
    physicsml_set_device(i);
    // Device-specific work
}
```

2. Synchronize devices:
```c
// Wait for all devices
physicsml_synchronize_all_devices();

// Transfer between devices
physicsml_transfer_tensor(
    tensor,
    source_device,
    target_device
);
```

### 3. Memory Management

#### Memory Pooling

Enable memory pooling:
```c
// Configure memory pool
physicsml_memory_pool_config config = {
    .initial_size = 1024 * 1024 * 1024,  // 1GB
    .growth_factor = 1.5,
    .max_size = 8ULL * 1024 * 1024 * 1024  // 8GB
};

physicsml_enable_memory_pool_with_config(&config);
```

#### Out-of-Core Computing

Handle large datasets:
```c
// Enable disk offload
physicsml_enable_disk_offload(true);
physicsml_set_scratch_directory("/path/to/scratch");

// Configure chunking
physicsml_set_chunk_size(1024 * 1024);  // 1MB chunks
```

#### Memory Layout

Optimize tensor layout:
```c
// Use optimal memory layout
PhysicsMLTensor* tensor = physicsml_tensor_create_optimized(
    shape,
    ndim,
    dtype
);

// Convert existing tensor
physicsml_tensor_optimize_layout(tensor);
```

### 4. Algorithm Optimization

#### Numerical Precision

Choose appropriate precision:
```c
// High precision for critical computations
physicsml_set_computation_dtype(PHYSICSML_DTYPE_COMPLEX128);

// Lower precision for bulk operations
physicsml_set_computation_dtype(PHYSICSML_DTYPE_COMPLEX64);
```

#### Parallel Processing

Enable parallel execution:
```c
// Set number of threads
physicsml_set_num_threads(num_threads);

// Enable parallel regions
#pragma omp parallel
{
    // Parallel code
}
```

#### Caching

Use computation caching:
```c
// Enable operation caching
physicsml_enable_operation_cache(true);

// Set cache parameters
physicsml_set_cache_size(1024 * 1024);  // 1MB
physicsml_set_cache_policy(PHYSICSML_CACHE_LRU);
```

## Performance Monitoring

### 1. Memory Usage

Monitor memory patterns:
```c
// Start memory tracking
physicsml_memory_tracker_start();

// Get current usage
size_t current = physicsml_get_current_memory();
size_t peak = physicsml_get_peak_memory();

// Print report
physicsml_memory_tracker_report();
```

### 2. Operation Timing

Profile specific operations:
```c
// Time single operation
double start = physicsml_get_time();
// Operation
double end = physicsml_get_time();

// Profile section
physicsml_profile_section("Critical Path");
// Operations
physicsml_end_profile_section();
```

### 3. Hardware Utilization

Monitor hardware usage:
```c
// GPU utilization
float gpu_util = physicsml_get_gpu_utilization();
size_t gpu_memory = physicsml_get_gpu_memory_used();

// CPU utilization
float cpu_util = physicsml_get_cpu_utilization();
size_t cpu_memory = physicsml_get_system_memory_used();
```

## Best Practices

### 1. General Guidelines

- Profile before optimizing
- Focus on bottlenecks
- Monitor resource usage
- Test with realistic data
- Benchmark regularly

### 2. Tensor Operations

- Minimize tensor copies
- Use in-place operations
- Optimize contraction order
- Apply compression when possible
- Batch similar operations

### 3. Memory Management

- Reuse tensors when possible
- Clean up unused resources
- Monitor memory patterns
- Use appropriate data types
- Enable memory pooling

### 4. Hardware Utilization

- Match algorithm to hardware
- Balance CPU/GPU workload
- Monitor device utilization
- Use appropriate batch sizes
- Enable hardware-specific optimizations

## Performance Benchmarks

Run benchmarks:
```bash
cd benchmarks
./prepare_benchmark_env.sh
./run_benchmarks.sh
python3 analyze_benchmark_results.py
```

View results:
```bash
cat benchmark_report.txt
```

## Additional Resources

- [API Documentation](API.md)
- [Examples](EXAMPLES.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Hardware Guide](docs/advanced/HARDWARE.md)

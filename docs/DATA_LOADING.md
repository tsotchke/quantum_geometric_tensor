# Quantum Geometric Learning Data Loading

This guide explains how to use the enhanced data loading capabilities in the Quantum Geometric Learning library.

## Overview

The data loading system provides:
- Streaming support for large datasets
- GPU acceleration and caching
- Performance monitoring
- Memory optimization
- Parallel processing

## Basic Usage

Here's a simple example of loading a dataset:

```c
// Configure memory management
memory_config_t memory_config = {
    .streaming = false,
    .max_memory = 1024 * 1024 * 1024,  // 1GB limit
    .gpu_cache = false,
    .compress = false
};

// Configure dataset loading
dataset_config_t config = {
    .format = DATA_FORMAT_CSV,
    .csv_config = {
        .delimiter = ",",
        .has_header = true
    },
    .memory = memory_config,
    .normalize = true,
    .normalization_method = NORMALIZATION_ZSCORE
};

// Load dataset
dataset_t* dataset = quantum_load_dataset("data/features.csv", config);
```

## Advanced Features

### Streaming Mode

For large datasets that don't fit in memory:

```c
memory_config_t config = {
    .streaming = true,
    .chunk_size = 1024 * 1024,  // 1MB chunks
    .max_memory = 4ULL * 1024 * 1024 * 1024  // 4GB limit
};
```

### GPU Acceleration

To utilize GPU for data loading and preprocessing:

```c
memory_config_t config = {
    .gpu_cache = true,
    .max_memory = 4ULL * 1024 * 1024 * 1024
};

performance_config_t perf_config = {
    .num_workers = 4,
    .prefetch_size = 2,
    .pin_memory = true
};

quantum_configure_memory(config);
quantum_configure_performance(perf_config);
```

### Performance Monitoring

Track data loading performance:

```c
performance_metrics_t metrics;
quantum_get_performance_metrics(&metrics);

printf("Load time: %.2f seconds\n", metrics.load_time);
printf("Memory usage: %.2f MB\n", metrics.memory_usage / (1024.0 * 1024.0));
printf("Throughput: %.2f MB/s\n", metrics.throughput / (1024.0 * 1024.0));
```

## Configuration Options

### Memory Configuration

- `streaming`: Enable streaming mode for large datasets
- `chunk_size`: Size of data chunks when streaming
- `max_memory`: Maximum memory usage limit
- `gpu_cache`: Enable GPU caching
- `compress`: Enable data compression

### Performance Configuration

- `num_workers`: Number of worker threads
- `prefetch_size`: Number of chunks to prefetch
- `cache_size`: Size of memory cache
- `pin_memory`: Pin memory for GPU transfers
- `profile`: Enable performance profiling

### Dataset Configuration

- `format`: Data format (CSV, NUMPY, HDF5, IMAGE)
- `csv_config`: CSV-specific settings
- `normalize`: Enable data normalization
- `normalization_method`: Normalization method to use

## Best Practices

1. **Memory Management**
   - Use streaming mode for datasets larger than available RAM
   - Enable compression for large datasets with repetitive patterns
   - Configure max_memory based on system resources

2. **Performance Optimization**
   - Enable GPU cache for GPU-accelerated workloads
   - Adjust num_workers based on CPU cores
   - Use prefetching to reduce I/O latency

3. **Error Handling**
   - Always check return values for error conditions
   - Monitor performance metrics for bottlenecks
   - Clean up resources properly

## Example Workflow

1. Configure memory and performance settings
2. Load dataset with appropriate configuration
3. Split dataset into training/validation/test sets
4. Monitor performance metrics
5. Clean up resources when done

See `examples/advanced/ai/quantum_data_pipeline_example.c` for a complete example.

## Performance Tips

1. **Streaming Mode**
   - Ideal for datasets > 75% of available RAM
   - Adjust chunk_size based on memory constraints
   - Enable compression for network storage

2. **GPU Acceleration**
   - Enable for datasets < GPU memory size
   - Use pin_memory for faster transfers
   - Configure prefetch_size based on GPU memory

3. **Memory Usage**
   - Monitor memory_usage metric
   - Adjust max_memory based on system
   - Enable compression for large datasets

## Error Handling

Always check return values:

```c
if (!quantum_configure_memory(memory_config)) {
    fprintf(stderr, "Failed to configure memory\n");
    return 1;
}

dataset_t* dataset = quantum_load_dataset(path, config);
if (!dataset) {
    fprintf(stderr, "Failed to load dataset\n");
    return 1;
}
```

## Resource Cleanup

Always clean up resources:

```c
// Clean up dataset
quantum_dataset_destroy(dataset);

// Reset performance metrics if needed
quantum_reset_performance_metrics();
```

## See Also

- [Performance Optimization Guide](PERFORMANCE_OPTIMIZATION.md)
- [Memory Management Guide](MEMORY_MANAGEMENT.md)
- [GPU Acceleration Guide](HARDWARE_ACCELERATION.md)

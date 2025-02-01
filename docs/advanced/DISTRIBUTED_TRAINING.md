# Distributed Training Guide

This guide explains how to use the distributed training capabilities of the Quantum Geometric Learning library for training large-scale models across multiple nodes.

## Overview

The distributed training system provides:
- Data parallel training across multiple nodes
- Automatic workload distribution
- Fault tolerance with automatic recovery
- Gradient synchronization
- Checkpoint management
- Performance monitoring

## Requirements

- MPI implementation (OpenMPI recommended)
- Multiple compute nodes or processes
- Shared filesystem for checkpoints

## Basic Usage

Here's a minimal example of setting up distributed training:

```c
#include "quantum_geometric/distributed/distributed_training_manager.h"

// Initialize configuration
distributed_config_t config = {
    .world_size = size,              // Total number of processes
    .local_rank = rank,              // This process's rank
    .num_gpus_per_node = 1,          // GPUs per node
    .batch_size = 32,                // Global batch size
    .micro_batch_size = 8,           // Per-process batch size
    .use_data_parallel = true,       // Enable data parallelism
    .use_model_parallel = false,     // Disable model parallelism
    .learning_rate = 0.001f,         // Learning rate
    .warmup_steps = 100,             // LR warmup steps
    .max_steps = 1000,               // Total training steps
    .save_interval = 50,             // Checkpoint interval
    .checkpoint_dir = "/path/to/checkpoints"
};

// Create manager
distributed_manager_t* manager = distributed_manager_create(&config);

// Initialize environment
distributed_manager_init_environment(manager);

// Create and prepare pipeline
quantum_pipeline_t* pipeline = quantum_pipeline_create(...);
distributed_manager_prepare_training(manager, pipeline, total_samples);

// Training loop
for (size_t step = 0; step < max_steps; step++) {
    // Train step with automatic fault recovery
    if (distributed_manager_train_step(manager, pipeline, batch_data,
                                     batch_size, step, &metrics) != 0) {
        if (metrics.error_code == ERROR_PROCESS_FAILURE) {
            // Handle process failure
            distributed_manager_handle_failure(manager,
                                            metrics.failed_process_rank);
            // Retry step
            distributed_manager_train_step(manager, pipeline, batch_data,
                                         batch_size, step, &metrics);
        }
    }
}
```

## Fault Tolerance

The system automatically handles process failures through:

1. Failure detection via heartbeat monitoring
2. Communicator reconstruction
3. Checkpoint recovery
4. Workload redistribution

When a process fails:

```c
// In training loop
if (metrics.error_code == ERROR_PROCESS_FAILURE) {
    // Get failed rank from metrics
    size_t failed_rank = metrics.failed_process_rank;
    
    // Handle failure and reconstruct environment
    if (distributed_manager_handle_failure(manager, failed_rank) == 0) {
        // Retry failed operation
        distributed_manager_train_step(...);
    }
}
```

## Data Sharding

The manager automatically handles data distribution:

```c
// Get this process's data shard
size_t start_idx, end_idx;
distributed_manager_get_local_batch(manager, total_samples,
                                  &start_idx, &end_idx);

// Load shard
size_t shard_size = end_idx - start_idx;
load_data_range(start_idx, end_idx, ...);
```

## Checkpointing

Automatic checkpoint management:

```c
// Configuration
config.save_interval = 100;  // Save every 100 steps
config.checkpoint_dir = "/path/to/checkpoints";

// Saving (automatic in train_step)
distributed_manager_save_checkpoint(manager, pipeline, step);

// Loading
distributed_manager_load_checkpoint(manager, pipeline, step);
```

## Performance Monitoring

The training metrics provide detailed performance information:

```c
training_metrics_t metrics;
distributed_manager_train_step(manager, pipeline, data, size, step, &metrics);

printf("Step %zu:\n", step);
printf("- Loss: %.4f\n", metrics.loss);
printf("- Accuracy: %.2f%%\n", metrics.accuracy * 100);
printf("- Training time: %.2f s\n", metrics.training_time);
printf("- Communication time: %.2f s\n", metrics.comm_time);
printf("- Compute efficiency: %.2f%%\n", metrics.compute_efficiency * 100);
```

## Advanced Features

### Model Parallelism

Enable model parallelism for large models:

```c
config.use_model_parallel = true;
config.model_parallel_size = 2;  // Split model across 2 devices
```

### Pipeline Parallelism

Enable pipeline parallelism for deep models:

```c
config.use_pipeline_parallel = true;
config.num_pipeline_stages = 4;  // Split into 4 stages
```

### Gradient Checkpointing

Enable memory optimization:

```c
config.use_gradient_checkpointing = true;
config.checkpoint_granularity = CHECKPOINT_GRANULARITY_LAYER;
```

## Example

See `examples/advanced/ai/distributed_mnist_example.c` for a complete example of distributed training with MNIST.

To run the example:

```bash
# Compile with MPI support
cmake -DQGT_USE_MPI=ON ..
make

# Run with 4 processes
mpirun -np 4 ./bin/examples/advanced/ai/distributed_mnist_example \
    /path/to/mnist/train.bin \
    /path/to/mnist/test.bin
```

## Best Practices

1. **Batch Size**: Set micro_batch_size = batch_size / world_size for optimal scaling

2. **Checkpointing**: Save checkpoints frequently enough to minimize lost work on failure

3. **GPU Assignment**: Use num_gpus_per_node to control GPU allocation

4. **Learning Rate**: Scale learning rate with global batch size

5. **Monitoring**: Track metrics.compute_efficiency to identify bottlenecks

## Troubleshooting

Common issues and solutions:

1. **Process Failures**
   - Check system logs for OOM or hardware errors
   - Verify checkpoint directory permissions
   - Monitor GPU memory usage

2. **Poor Scaling**
   - Increase batch size
   - Check network bandwidth
   - Monitor communication/compute ratio

3. **Memory Issues**
   - Enable gradient checkpointing
   - Reduce model size or batch size
   - Use pipeline parallelism

4. **Slow Training**
   - Verify GPU utilization
   - Check process placement
   - Monitor I/O patterns

## Getting Started

The library now provides comprehensive distributed training capabilities with the following components:

1. Core Infrastructure:
   - Distributed training manager with fault tolerance
   - MPI-based communication layer
   - Automatic data sharding and workload distribution
   - Checkpoint management and recovery
   - Performance monitoring and optimization

2. User Tools:
   - Environment setup script (tools/setup_distributed_env.sh)
   - Configuration template (etc/quantum_geometric/distributed_config.json)
   - System verification script (tools/test_distributed_setup.sh)

3. Documentation & Examples:
   - This comprehensive guide
   - MNIST example for basic distributed training
   - CIFAR example for advanced distributed training

### Quick Start

1. Setup:
   ```bash
   # Install dependencies and configure environment
   sudo ./tools/setup_distributed_env.sh
   
   # Verify setup
   ./tools/test_distributed_setup.sh
   ```

2. Build:
   ```bash
   mkdir build && cd build
   cmake -DQGT_USE_MPI=ON ..
   make -j$(nproc)
   ```

3. Run Examples:
   ```bash
   # Start with MNIST
   mpirun -np 4 ./bin/examples/advanced/ai/distributed_mnist_example
   
   # Move to CIFAR for more complex tasks
   mpirun -np 4 ./bin/examples/advanced/ai/distributed_cifar_example
   ```

4. Monitor:
   ```bash
   # Watch training progress
   tail -f /tmp/quantum_geometric/logs/distributed_training.log
   
   # Check performance metrics
   quantum_geometric-monitor-distributed
   ```

## API Reference

See the following header files for detailed API documentation:

- `include/quantum_geometric/distributed/distributed_training_manager.h`
- `include/quantum_geometric/distributed/workload_distribution.h`
- `include/quantum_geometric/distributed/gradient_optimizer.h`
- `include/quantum_geometric/distributed/communication_optimizer.h`

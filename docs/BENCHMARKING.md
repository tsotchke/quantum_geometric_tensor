# Benchmarking Guide

This guide explains how to run performance benchmarks comparing Quantum Geometric Learning (QGL) against TensorFlow.

## Prerequisites

- macOS with M2 Ultra (or equivalent hardware)
- Python 3.8 or higher
- CMake 3.15 or higher
- OpenMPI (for distributed benchmarks)

## Setup

1. Set up the benchmark environment:
```bash
./tools/setup_benchmark_env.sh
source activate_benchmark_env.sh
```

2. Configure IBM Quantum (optional but recommended):
```bash
./tools/setup_ibm_quantum.sh
```

3. Build the library:
```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
cd ..
```

## Running Benchmarks

The benchmark suite compares QGL against TensorFlow across several dimensions:

1. **Quantum State Evolution**
   - System sizes from 1K to 1M qubits
   - Measures execution time and memory usage
   - Tests O(log n) quantum attention scaling

2. **Error Correction**
   - Error rates from 0.1% to 10%
   - Measures error reduction and correction time
   - Tests topological protection effectiveness

3. **Memory Efficiency**
   - State preparation
   - Evolution
   - Measurement
   - Error correction

4. **Distributed Performance**
   - Scaling from 2 to 16 processes
   - Tests MPI communication efficiency
   - Measures speedup and resource utilization

5. **GPU Utilization**
   - Quantum attention operations
   - Tensor contractions
   - FFT operations
   - State transformations

### Running All Benchmarks

To run the complete benchmark suite:
```bash
./benchmarks/run_comparison.sh
```

This will:
1. Warm up both frameworks
2. Run all benchmark categories
3. Generate visualizations
4. Produce a detailed report

### Running Individual Benchmarks

For quantum state evolution:
```bash
./build/benchmarks/benchmark_quantum_geometric --test=evolution --size=10000
python3 benchmarks/benchmark_tensorflow.py --test=evolution --size=10000
```

For error correction:
```bash
./build/benchmarks/benchmark_quantum_geometric --test=error --rate=0.01
python3 benchmarks/benchmark_tensorflow.py --test=error --rate=0.01
```

For memory usage:
```bash
./build/benchmarks/benchmark_quantum_geometric --test=memory --op="State prep"
python3 benchmarks/benchmark_tensorflow.py --test=memory --op="State prep"
```

For distributed performance:
```bash
mpirun -np 4 ./build/benchmarks/benchmark_quantum_geometric --test=distributed
python3 benchmarks/benchmark_tensorflow.py --test=distributed --procs=4
```

For GPU utilization:
```bash
./build/benchmarks/benchmark_quantum_geometric --test=gpu --op="Attention"
python3 benchmarks/benchmark_tensorflow.py --test=gpu --op="Attention"
```

## Understanding Results

The benchmark results are presented in several formats:

1. **Terminal Output**
   - Real-time benchmark progress
   - Raw performance numbers
   - System utilization metrics

2. **Visualizations**
   - `evolution_performance.png`: Execution time and speedup plots
   - `error_correction.png`: Error rates and improvement ratios
   - `memory_usage.png`: Memory consumption comparison
   - `distributed_scaling.png`: Scaling efficiency plots
   - `gpu_utilization.png`: GPU utilization charts

3. **PDF Report**
   - Detailed analysis of all benchmarks
   - Statistical significance tests
   - Hardware utilization insights
   - Scaling characteristics

## Interpreting Metrics

1. **Execution Time**
   - Reported in milliseconds
   - Lower is better
   - Includes warmup iterations
   - Excludes data transfer time

2. **Error Rates**
   - Reported as absolute values
   - Lower is better
   - Measured against ideal state
   - Includes quantum noise effects

3. **Memory Usage**
   - Reported in gigabytes
   - Lower is better
   - Peak memory consumption
   - Includes GPU memory

4. **Speedup**
   - Ratio of TensorFlow time to QGL time
   - Higher is better
   - Linear scaling = 1:1 ratio
   - Super-linear possible with quantum attention

5. **GPU Utilization**
   - Percentage of theoretical peak
   - Higher is better
   - Measured over operation duration
   - Accounts for memory bandwidth

## Common Issues

1. **Out of Memory**
   - Reduce system size
   - Adjust `QGL_GPU_MEMORY_FRACTION`
   - Enable state compression
   - Use distributed mode

2. **Poor Scaling**
   - Check process pinning
   - Verify MPI configuration
   - Monitor network bandwidth
   - Adjust workload distribution

3. **Low GPU Utilization**
   - Check thermal throttling
   - Monitor power limits
   - Verify backend selection
   - Adjust batch sizes

4. **High Error Rates**
   - Verify quantum state fidelity
   - Check error correction settings
   - Monitor syndrome measurements
   - Adjust protection parameters

## Customizing Benchmarks

The benchmark framework can be extended through:

1. **Configuration Files**
   - `etc/quantum_geometric/benchmark_config.json`
   - `etc/quantum_geometric/gpu_config.json`
   - `etc/quantum_geometric/mpi_config.json`

2. **Environment Variables**
   - `QGL_BENCHMARK_MODE`: Enable detailed profiling
   - `QGL_GPU_MEMORY_FRACTION`: Control memory usage
   - `QGL_ERROR_CORRECTION`: Set correction level
   - `QGL_CIRCUIT_OPTIMIZATION`: Enable optimizations

3. **Command Line Options**
   - `--size`: System size
   - `--rate`: Error rate
   - `--op`: Operation type
   - `--procs`: Process count

## Contributing Results

When submitting benchmark results:

1. **Hardware Details**
   - CPU model and configuration
   - GPU type and memory
   - Memory size and speed
   - Storage specifications

2. **Software Versions**
   - Operating system
   - Compiler toolchain
   - Library dependencies
   - Driver versions

3. **Methodology**
   - Number of iterations
   - Warmup procedures
   - System conditions
   - Error margins

4. **Raw Data**
   - Timing measurements
   - Memory statistics
   - GPU metrics
   - Log files

Submit results through:
- GitHub Issues
- Pull Requests
- Discussion Forums
- Email to maintainers

For more information, see:
- [Performance Comparison](PERFORMANCE_COMPARISON.md)
- [Hardware Requirements](INSTALLATION.md#hardware-requirements)
- [Optimization Guide](PERFORMANCE_OPTIMIZATION.md)

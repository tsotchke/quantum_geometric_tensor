# Installation Guide

**Version:** 0.777 Beta
**Last Updated:** January 2026

---

## Overview

This guide provides comprehensive instructions for installing and configuring the Quantum Geometric Tensor Library (QGTL) across supported platforms.

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | x86_64 or ARM64 with SIMD | AVX2/AVX-512 or Apple Silicon |
| Memory | 8 GB RAM | 32 GB RAM |
| Storage | 2 GB | 10 GB (with tests and examples) |
| GPU (optional) | CUDA 7.0+ or Metal-capable | NVIDIA Ampere or Apple M-series |

### Supported Platforms

| Platform | Version | GPU Support |
|----------|---------|-------------|
| macOS | 12.0+ (Monterey or later) | Metal |
| Linux | Kernel 4.19+ | CUDA |
| Windows | 10/11 (experimental) | CUDA |

### Quantum Hardware Access (Beta)

QGTL supports integration with the following quantum computing platforms:

**IBM Quantum**
- Account: [quantum.ibm.com](https://quantum.ibm.com)
- Systems: IBM Eagle (127 qubits), IBM Osprey (433 qubits)
- Integration: Via Qiskit Runtime

**Rigetti**
- Account: [qcs.rigetti.com](https://qcs.rigetti.com)
- Systems: Aspen-M, Ankaa processors
- Integration: Via PyQuil

**D-Wave**
- Account: [cloud.dwavesys.com](https://cloud.dwavesys.com)
- Systems: Advantage (5000+ qubits)
- Integration: Via Ocean SDK

## Software Prerequisites

### Required Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| CMake | 3.16+ | Build system |
| GCC/Clang | 9+/11+ | C compiler |
| BLAS/LAPACK | Any | Linear algebra |
| Threads | POSIX | Parallelization |

### Optional Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| MPI | OpenMPI 4.0+ or MPICH 3.3+ | Distributed training |
| CUDA | 11.0+ | NVIDIA GPU acceleration |
| hwloc | 2.0+ | Topology detection |
| json-c | 0.15+ | Backend configuration |
| libcurl | 7.0+ | Cloud backend communication |
| zlib | 1.2+ | Data compression |

## Installation Procedures

### macOS (Homebrew)

```bash
# Install build dependencies
brew install cmake

# Install MPI (optional, for distributed training)
brew install open-mpi

# Install optional backend dependencies
brew install hwloc json-c curl

# Clone repository
git clone https://github.com/tsotchke/quantum_geometric_tensor.git
cd quantum_geometric_tensor

# Configure build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DQGT_ENABLE_METAL=ON

# Build
make -j$(sysctl -n hw.ncpu)

# Run tests
ctest --output-on-failure

# Install (optional)
sudo make install
```

### Linux (Ubuntu/Debian)

```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install build-essential cmake

# Install MPI and numerical libraries
sudo apt-get install libopenmpi-dev libopenblas-dev liblapack-dev

# Install optional dependencies
sudo apt-get install libhwloc-dev libjson-c-dev libcurl4-openssl-dev zlib1g-dev

# Clone repository
git clone https://github.com/tsotchke/quantum_geometric_tensor.git
cd quantum_geometric_tensor

# Configure build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Install (optional)
sudo make install
```

### Linux with CUDA Support

```bash
# Install CUDA toolkit (Ubuntu 20.04/22.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-0

# Configure with CUDA
cmake .. -DCMAKE_BUILD_TYPE=Release -DQGT_ENABLE_CUDA=ON

# Build
make -j$(nproc)
```

## Build Configuration Options

| CMake Option | Default | Description |
|--------------|---------|-------------|
| `QGT_BUILD_TESTS` | ON | Build test executables |
| `QGT_BUILD_EXAMPLES` | ON | Build example programs |
| `QGT_BUILD_SHARED` | ON | Build shared library |
| `QGT_BUILD_STATIC` | ON | Build static library |
| `QGT_ENABLE_MPI` | Auto | Enable MPI support |
| `QGT_ENABLE_CUDA` | OFF | Enable CUDA GPU support |
| `QGT_ENABLE_METAL` | Auto | Enable Metal GPU support (macOS) |
| `QGT_MODULAR_BUILD` | OFF | Build separate module libraries |
| `QGT_PORTABLE_BUILD` | OFF | Disable -march=native for portable binaries |

Example configuration for maximum performance:

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DQGT_ENABLE_MPI=ON \
    -DQGT_ENABLE_METAL=ON \
    -DQGT_BUILD_TESTS=ON
```

## Quantum Hardware Configuration

### IBM Quantum Setup

```bash
# Install Qiskit (for credential management)
pip install qiskit

# Create credentials directory
mkdir -p ~/.quantum_geometric/ibm

# Configure credentials
cat > ~/.quantum_geometric/ibm/credentials.json << 'EOF'
{
    "api_token": "YOUR_IBM_API_TOKEN",
    "hub": "ibm-q",
    "group": "open",
    "project": "main"
}
EOF

chmod 600 ~/.quantum_geometric/ibm/credentials.json
```

### Rigetti Setup

```bash
# Install PyQuil
pip install pyquil

# Create credentials directory
mkdir -p ~/.quantum_geometric/rigetti

# Configure credentials
cat > ~/.quantum_geometric/rigetti/credentials.json << 'EOF'
{
    "api_key": "YOUR_RIGETTI_API_KEY",
    "qvm_url": "tcp://127.0.0.1:5000",
    "quilc_url": "tcp://127.0.0.1:5555"
}
EOF

chmod 600 ~/.quantum_geometric/rigetti/credentials.json
```

### D-Wave Setup

```bash
# Install Ocean SDK
pip install dwave-ocean-sdk

# Create credentials directory
mkdir -p ~/.quantum_geometric/dwave

# Configure credentials
cat > ~/.quantum_geometric/dwave/credentials.json << 'EOF'
{
    "token": "YOUR_DWAVE_TOKEN",
    "solver": "Advantage_system4.1",
    "region": "na-west-1"
}
EOF

chmod 600 ~/.quantum_geometric/dwave/credentials.json
```

## Environment Configuration

Add to shell configuration (`~/.bashrc` or `~/.zshrc`):

```bash
# Library paths (if not using system install)
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Quantum credentials location
export QGT_QUANTUM_CREDENTIALS=~/.quantum_geometric

# MPI configuration (if using distributed training)
export OMPI_MCA_btl_vader_single_copy_mechanism=none

# GPU memory allocation (fraction of total)
export QGT_GPU_MEMORY_FRACTION=0.75

# Checkpoint directory for distributed training
export QGT_CHECKPOINT_DIR=/tmp/quantum_geometric/checkpoints
```

## Verification

### Basic Verification

```bash
# Run all tests
cd build
ctest --output-on-failure

# Run specific test categories
ctest -R quantum_geometric  # Core tests
ctest -R surface_code       # Error correction tests
ctest -R distributed        # Distributed training tests
```

### GPU Verification

```bash
# Metal (macOS)
ctest -R metal

# CUDA (Linux)
ctest -R cuda
```

### Performance Verification

```bash
# Run performance benchmarks
./tests/test_quantum_performance
./tests/test_quantum_geometric_tensor_perf
```

## System Optimization

### Linux Performance Tuning

```bash
# Enable huge pages for large tensor operations
HUGE_PAGES=$((4 * 1024 * 1024 * 1024 / 2048 / 1024))
sudo sysctl -w vm.nr_hugepages=$HUGE_PAGES

# Enable NUMA balancing
sudo sysctl -w kernel.numa_balancing=1

# Increase file descriptor limits
ulimit -n 65536
```

### macOS Performance Tuning

```bash
# Increase shared memory limits
sudo sysctl -w kern.sysv.shmmax=4294967296
sudo sysctl -w kern.sysv.shmall=1048576
```

## Troubleshooting

### Common Issues

**CMake cannot find BLAS/LAPACK**
```bash
# macOS: Uses Accelerate framework automatically
# Linux: Install OpenBLAS
sudo apt-get install libopenblas-dev liblapack-dev
```

**MPI not found**
```bash
# Verify MPI installation
mpicc --version
mpirun --version

# Install if missing
sudo apt-get install libopenmpi-dev  # Linux
brew install open-mpi                 # macOS
```

**Metal compilation errors (macOS)**
```bash
# Ensure Xcode command-line tools are installed
xcode-select --install

# Verify Metal framework availability
xcrun --sdk macosx --find metal
```

**CUDA compilation errors**
```bash
# Verify CUDA installation
nvcc --version

# Ensure CUDA paths are set
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Build Diagnostics

```bash
# Verbose CMake output
cmake .. -DCMAKE_BUILD_TYPE=Release --trace-expand

# Verbose build output
make VERBOSE=1

# Check library dependencies
ldd lib/libquantum_geometric.so  # Linux
otool -L lib/libquantum_geometric.dylib  # macOS
```

## Additional Resources

- [Quickstart Guide](QUICKSTART.md): Rapid introduction to QGTL
- [API Reference](API_REFERENCE.md): Complete API documentation
- [Performance Tuning](PERFORMANCE_TUNING.md): Optimization strategies
- [Distributed Training](DISTRIBUTED_COMPUTING.md): Multi-node configuration
- [Hardware Integration](QUANTUM_HARDWARE.md): Backend setup details

## External Documentation

- [IBM Quantum Documentation](https://quantum-computing.ibm.com/docs/)
- [Rigetti QCS Documentation](https://docs.rigetti.com/qcs/)
- [D-Wave Documentation](https://docs.dwavesys.com/docs/latest/)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)
- [Metal Documentation](https://developer.apple.com/metal/)

---

*For installation issues, please open an issue on [GitHub](https://github.com/tsotchke/quantum_geometric_tensor/issues).*

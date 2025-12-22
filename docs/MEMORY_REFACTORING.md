# Memory System Refactoring Plan

## Current State (March 17, 2025)

### Memory Systems
1. **Memory Pool System** (`memory_pool.c`)
   - Size class based allocation
   - Thread caching
   - SIMD optimizations
   - Currently: Working but not integrated with other systems

2. **Advanced Memory System** (`advanced_memory_system.c`)
   - Geometric layouts
   - Hardware optimizations (NUMA, NEON)
   - Currently: Working but conflicts with memory pool

3. **Memory Singleton** (`memory_singleton.c`)
   - Global state management
   - Fixed-size block tracking (1024 blocks)
   - Currently: Limited and causing conflicts

### Critical Issues
1. Tree tensor network creates its own memory pool instead of using global system
2. Memory systems are not properly integrated
3. Fixed-size limitations in memory singleton
4. Memory leaks in tensor contractions

## Immediate Fixes Needed

### Phase 1: Fix Tree Tensor Network Memory Management
1. Modify `create_tree_tensor_network` to use global memory system
2. Implement proper memory cleanup in error paths
3. Fix streaming tensor contractions to use unified memory system

### Phase 2: Integrate Memory Systems
1. Make memory pool use advanced memory system's geometric layouts
2. Update memory singleton to use dynamic block tracking
3. Implement proper reference counting

### Phase 3: Optimize for Large Tensors
1. Implement streaming memory operations
2. Add hierarchical compression
3. Enable out-of-core processing

## Implementation Plan

### Memory System Integration
```c
// New unified memory interface
typedef struct {
    // Core allocation functions
    void* (*allocate)(size_t size, size_t alignment);
    void (*free)(void* ptr);
    void* (*realloc)(void* ptr, size_t size);
    
    // Advanced features
    void* (*allocate_geometric)(size_t* dimensions, size_t num_dims);
    void* (*allocate_hierarchical)(size_t size, double tolerance);
    void* (*allocate_streaming)(size_t total_size, size_t chunk_size);
    
    // Memory tracking
    bool (*track_allocation)(void* ptr, size_t size);
    bool (*untrack_allocation)(void* ptr);
    
    // Statistics
    void (*get_stats)(memory_stats_t* stats);
} unified_memory_interface_t;
```

### Memory Pool Integration
```c
// Updated memory pool configuration
typedef struct {
    size_t initial_size;
    size_t max_size;
    bool enable_growth;
    bool use_geometric_layout;
    bool enable_streaming;
    size_t chunk_size;
    double compression_tolerance;
} unified_pool_config_t;
```

### Tree Tensor Network Changes
```c
// Updated tree tensor network structure
struct tree_tensor_network {
    tree_tensor_node_t* root;
    size_t num_nodes;
    size_t max_rank;
    size_t num_qubits;
    double tolerance;
    unified_memory_interface_t* memory;  // Use unified interface
    tensor_network_metrics_t metrics;
};
```

## Migration Steps

1. Create unified memory interface
2. Update memory pool to implement interface
3. Update advanced memory system to implement interface
4. Replace memory singleton with dynamic tracking
5. Update tree tensor network to use unified interface
6. Fix all memory leaks and cleanup paths
7. Implement streaming operations
8. Add compression support

## Testing Plan

1. Unit tests for unified memory interface
2. Integration tests for memory systems
3. Stress tests for large tensor operations
4. Memory leak detection
5. Performance benchmarks

## Status Tracking

- [ ] Create unified memory interface
- [ ] Update memory pool implementation
- [ ] Update advanced memory system
- [ ] Replace memory singleton
- [ ] Fix tree tensor network memory usage
- [ ] Implement streaming operations
- [ ] Add compression support
- [ ] Complete testing suite

## Next Steps

1. Implement unified memory interface
2. Update tree tensor network to use interface
3. Fix memory leaks in tensor contractions
4. Add streaming support for large tensors

## Notes

- Critical for supporting 500M+ parameter models
- Must handle large tensor operations efficiently
- Need to prevent memory fragmentation
- Must support streaming operations

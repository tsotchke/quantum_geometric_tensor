#include <quantum_geometric/core/resource_validation.h>
#include <quantum_geometric/ai/quantum_llm_core.h>
#include <stdio.h>
#include <assert.h>

static void test_basic_validation(void) {
    printf("Testing basic resource validation...\n");
    
    quantum_llm_config_t config = {
        .model_config = {
            .total_parameters = 1ULL * 1024 * 1024 * 1024, // 1B parameters
            .model_layers = 96,
            .embedding_dimension = 8192,
            .attention_dimension = 128
        },
        .encoding_config = {
            .geometric_dimension = 256,
            .compression_ratio = 1000.0f,
            .encoding_qubits = 1000,
            .use_topological_protection = true,
            .code_distance = 7,
            .error_threshold = 1e-6f,
            .use_holographic_encoding = true,
            .holographic_dimension = 256,
            .holographic_fidelity = 0.995f,
            .use_error_mitigation = true,
            .error_mitigation_rounds = 5,
            .use_compression = true,
            .target_compression_ratio = 2000.0f
        },
        .distributed_config = {
            .quantum_nodes = 2048,
            .qubits_per_node = 2000,
            .coherence_time = 100.0f,
            .topology = TOPOLOGY_HYPERCUBE,
            .use_error_correction = true,
            .syndrome_qubits = 32
        }
    };
    
    quantum_status_t status = validate_resource_requirements(&config);
    assert(status == QUANTUM_STATUS_SUCCESS);
    printf("Basic validation passed\n");
}

static void test_insufficient_qubits(void) {
    printf("Testing insufficient qubit detection...\n");
    
    quantum_llm_config_t config = {
        .model_config = {
            .total_parameters = 32768ULL * 1024 * 1024 * 1024, // 32T parameters
            .model_layers = 96,
            .embedding_dimension = 8192,
            .attention_dimension = 128
        },
        .encoding_config = {
            .geometric_dimension = 256,
            .compression_ratio = 100.0f, // Low compression ratio
            .encoding_qubits = 1000,
            .use_topological_protection = true,
            .code_distance = 7,
            .error_threshold = 1e-6f,
            .use_holographic_encoding = true,
            .holographic_dimension = 256,
            .holographic_fidelity = 0.995f,
            .use_error_mitigation = true,
            .error_mitigation_rounds = 5,
            .use_compression = true,
            .target_compression_ratio = 200.0f // Low target compression
        },
        .distributed_config = {
            .quantum_nodes = 100, // Few nodes
            .qubits_per_node = 100, // Few qubits per node
            .coherence_time = 100.0f,
            .topology = TOPOLOGY_HYPERCUBE,
            .use_error_correction = true,
            .syndrome_qubits = 32
        }
    };
    
    quantum_status_t status = validate_resource_requirements(&config);
    assert(status == QUANTUM_STATUS_INSUFFICIENT_RESOURCES);
    printf("Insufficient qubit detection passed\n");
}

static void test_compression_adjustment(void) {
    printf("Testing compression ratio adjustment...\n");
    
    quantum_llm_config_t config = {
        .model_config = {
            .total_parameters = 4096ULL * 1024 * 1024 * 1024, // 4T parameters
            .model_layers = 96,
            .embedding_dimension = 8192,
            .attention_dimension = 128
        },
        .encoding_config = {
            .geometric_dimension = 256,
            .compression_ratio = 500.0f, // Initial compression
            .encoding_qubits = 1000,
            .use_topological_protection = true,
            .code_distance = 7,
            .error_threshold = 1e-6f,
            .use_holographic_encoding = true,
            .holographic_dimension = 256,
            .holographic_fidelity = 0.995f,
            .use_error_mitigation = true,
            .error_mitigation_rounds = 5,
            .use_compression = true,
            .target_compression_ratio = 2000.0f // Higher target allowed
        },
        .distributed_config = {
            .quantum_nodes = 2048,
            .qubits_per_node = 2000,
            .coherence_time = 100.0f,
            .topology = TOPOLOGY_HYPERCUBE,
            .use_error_correction = true,
            .syndrome_qubits = 32
        }
    };
    
    float initial_compression = config.encoding_config.compression_ratio;
    quantum_status_t status = validate_resource_requirements(&config);
    assert(status == QUANTUM_STATUS_SUCCESS);
    assert(config.encoding_config.compression_ratio > initial_compression);
    printf("Compression ratio adjustment passed\n");
}

static void test_invalid_holographic_params(void) {
    printf("Testing invalid holographic parameter detection...\n");
    
    quantum_llm_config_t config = {
        .model_config = {
            .total_parameters = 1ULL * 1024 * 1024 * 1024,
            .model_layers = 96,
            .embedding_dimension = 8192,
            .attention_dimension = 128
        },
        .encoding_config = {
            .geometric_dimension = 256,
            .compression_ratio = 1000.0f,
            .encoding_qubits = 1000,
            .use_topological_protection = true,
            .code_distance = 7,
            .error_threshold = 1e-6f,
            .use_holographic_encoding = true,
            .holographic_dimension = 16, // Too small
            .holographic_fidelity = 0.995f,
            .use_error_mitigation = true,
            .error_mitigation_rounds = 5,
            .use_compression = true,
            .target_compression_ratio = 2000.0f
        },
        .distributed_config = {
            .quantum_nodes = 2048,
            .qubits_per_node = 2000,
            .coherence_time = 100.0f,
            .topology = TOPOLOGY_HYPERCUBE,
            .use_error_correction = true,
            .syndrome_qubits = 32
        }
    };
    
    quantum_status_t status = validate_resource_requirements(&config);
    assert(status == QUANTUM_STATUS_INVALID_CONFIGURATION);
    printf("Invalid holographic parameter detection passed\n");
}

static void test_memory_limits(void) {
    printf("Testing memory limit validation...\n");
    
    quantum_llm_config_t config = {
        .model_config = {
            .total_parameters = (1ULL << 48) / sizeof(float), // Just at memory limit
            .model_layers = 96,
            .embedding_dimension = 8192,
            .attention_dimension = 128
        },
        .encoding_config = {
            .geometric_dimension = 256,
            .compression_ratio = 1000.0f,
            .encoding_qubits = 1000,
            .use_topological_protection = true,
            .code_distance = 7,
            .error_threshold = 1e-6f,
            .use_holographic_encoding = true,
            .holographic_dimension = 256,
            .holographic_fidelity = 0.995f,
            .use_error_mitigation = true,
            .error_mitigation_rounds = 5,
            .use_compression = true,
            .target_compression_ratio = 2000.0f
        },
        .distributed_config = {
            .quantum_nodes = 2048,
            .qubits_per_node = 2000,
            .coherence_time = 100.0f,
            .topology = TOPOLOGY_HYPERCUBE,
            .use_error_correction = true,
            .syndrome_qubits = 32
        }
    };
    
    // Test at limit
    quantum_status_t status = validate_resource_requirements(&config);
    assert(status == QUANTUM_STATUS_SUCCESS);
    
    // Test beyond limit
    config.model_config.total_parameters *= 2;
    config.tensor_config.use_quantum_memory = true; // Double memory usage
    status = validate_resource_requirements(&config);
    assert(status == QUANTUM_STATUS_INSUFFICIENT_MEMORY);
    
    printf("Memory limit validation passed\n");
}

int main(void) {
    printf("Running resource validation tests...\n\n");
    
    test_basic_validation();
    test_insufficient_qubits();
    test_compression_adjustment();
    test_invalid_holographic_params();
    test_memory_limits();
    
    printf("\nAll resource validation tests passed!\n");
    return 0;
}

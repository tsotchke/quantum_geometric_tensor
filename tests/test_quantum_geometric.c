#include "test_config.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_geometric_tensor.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/core/quantum_geometric_types.h"

// Test data
static const int NUM_QUBITS = 2;
static const int DIM = 4;  // 2^NUM_QUBITS

// Configuration structure
typedef struct {
    float error_rate;
    unsigned int flags;
} GeometricConfig;

// Global test registration
struct test_case g_tests[MAX_TESTS];
int g_test_count = 0;

// Test quantum geometric tensor computation
void test_quantum_geometric_tensor_basic() {
    TEST_SETUP();
    
    // Create quantum state (|00⟩ + |11⟩)/√2
    GeometricConfig config = {
        .error_rate = 0.01f,
        .flags = QG_FLAG_OPTIMIZE | QG_FLAG_ERROR_CORRECT
    };
    
    quantum_state_t* state;
    qgt_error_t err = quantum_state_create(&state, QUANTUM_STATE_PURE, 1 << NUM_QUBITS);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    // Initialize to Bell state
    ComplexFloat bell_state[4] = {
        CMPLX(1.0f/sqrt(2.0f), 0.0f),  // |00⟩
        CMPLX(0.0f, 0.0f),             // |01⟩
        CMPLX(0.0f, 0.0f),             // |10⟩
        CMPLX(1.0f/sqrt(2.0f), 0.0f)   // |11⟩
    };
    err = quantum_state_initialize(state, bell_state);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    // Create parameter generators
    quantum_operator_t* generators[2];
    for (int i = 0; i < 2; i++) {
        qgt_error_t err = quantum_operator_create(&generators[i], QUANTUM_OPERATOR_HERMITIAN, DIM);
        TEST_ASSERT(err == QGT_SUCCESS);
    }
    
    // Set X and Z operators as generators
    err = quantum_operator_pauli_x(generators[0], 0);  // σx on first qubit
    TEST_ASSERT(err == QGT_SUCCESS);
    err = quantum_operator_pauli_z(generators[1], 1);  // σz on second qubit
    TEST_ASSERT(err == QGT_SUCCESS);
    
    // Compute quantum geometric tensor
    double metric[4];  // 2x2 tensor for 2 parameters
    quantum_geometric_tensor_t* metric_tensor;
    size_t dims[] = {2, 2}; // 2x2 tensor for 2 parameters
    err = geometric_tensor_create(&metric_tensor, GEOMETRIC_TENSOR_SCALAR, dims, 2);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    quantum_geometric_metric_t* metric;
    err = geometric_create_metric(&metric, GEOMETRIC_METRIC_EUCLIDEAN, DIM);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    err = geometric_compute_metric(metric, state);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    err = geometric_tensor_initialize(metric_tensor, metric->components);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    // Expected values for this state
    // Expected values for this state
    ComplexFloat expected[4] = {
        CMPLX(0.5f, 0.0f), CMPLX(0.0f, 0.0f),  // G_00 = 1/2
        CMPLX(0.0f, 0.0f), CMPLX(0.5f, 0.0f)   // G_11 = 1/2
    };
    
    // Verify results
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_COMPLEX_EQ(metric_tensor->components[i], expected[i]);
    }
    
    // Cleanup
    geometric_tensor_destroy(metric_tensor);
    geometric_destroy_metric(metric);
    quantum_state_destroy(state);
    for (int i = 0; i < 2; i++) {
        quantum_operator_destroy(generators[i]);
    }
    
    TEST_TEARDOWN();
}

// Test Berry curvature computation
void test_berry_curvature() {
    TEST_SETUP();
    
    // Create quantum state
    GeometricConfig config = {
        .error_rate = 0.01f,
        .flags = QG_FLAG_OPTIMIZE | QG_FLAG_ERROR_CORRECT
    };
    
    quantum_state_t* state;
    qgt_error_t err = quantum_state_create(&state, QUANTUM_STATE_PURE, 1 << NUM_QUBITS);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    // Initialize to |+⟩⊗|0⟩ state
    ComplexFloat plus_state[4] = {
        CMPLX(1.0f/sqrt(2.0f), 0.0f),  // |00⟩
        CMPLX(0.0f, 0.0f),             // |01⟩
        CMPLX(1.0f/sqrt(2.0f), 0.0f),  // |10⟩
        CMPLX(0.0f, 0.0f)              // |11⟩
    };
    err = quantum_state_initialize(state, plus_state);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    // Create parameter generators
    quantum_operator_t* generators[2];
    for (int i = 0; i < 2; i++) {
        qgt_error_t err = quantum_operator_create(&generators[i], QUANTUM_OPERATOR_HERMITIAN, DIM);
        TEST_ASSERT(err == QGT_SUCCESS);
    }
    
    // Set generators for rotation angles
    err = quantum_operator_pauli_x(generators[0], 0);  // θ generator
    TEST_ASSERT(err == QGT_SUCCESS);
    err = quantum_operator_pauli_y(generators[1], 0);  // φ generator
    TEST_ASSERT(err == QGT_SUCCESS);
    
    // Create and compute curvature tensor
    quantum_geometric_curvature_t* curvature_tensor;
    size_t dims[] = {2, 2}; // 2x2 curvature tensor
    err = geometric_tensor_create(&curvature_tensor, GEOMETRIC_TENSOR_SCALAR, dims, 2);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    quantum_geometric_metric_t* metric;
    err = geometric_create_metric(&metric, GEOMETRIC_METRIC_EUCLIDEAN, DIM);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    err = geometric_compute_metric(metric, state);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    quantum_geometric_connection_t* connection;
    err = geometric_create_connection(&connection, GEOMETRIC_CONNECTION_LEVI_CIVITA, DIM);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    err = geometric_compute_connection(connection, metric);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    quantum_geometric_curvature_t* curvature;
    err = geometric_create_curvature(&curvature, GEOMETRIC_CURVATURE_RIEMANN, DIM);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    err = geometric_compute_curvature(curvature, connection);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    err = geometric_tensor_initialize(curvature_tensor, curvature->components);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    // For this state and these generators, expect curvature = 1/2
    TEST_ASSERT_COMPLEX_EQ(curvature_tensor->components[0], CMPLX(0.5f, 0.0f));
    
    // Cleanup
    geometric_tensor_destroy(curvature_tensor);
    geometric_destroy_curvature(curvature);
    geometric_destroy_connection(connection);
    geometric_destroy_metric(metric);
    quantum_state_destroy(state);
    for (int i = 0; i < 2; i++) {
        quantum_operator_destroy(generators[i]);
    }
    
    TEST_TEARDOWN();
}

// Test geometric phase computation
void test_geometric_phase() {
    TEST_SETUP();
    
    // Create initial state |0⟩
    GeometricConfig config = {
        .error_rate = 0.01f,
        .flags = QG_FLAG_OPTIMIZE | QG_FLAG_ERROR_CORRECT
    };
    
    quantum_state_t* state;
    qgt_error_t err = quantum_state_create(&state, QUANTUM_STATE_PURE, 2);  // 2 = 2^1 for 1 qubit
    TEST_ASSERT(err == QGT_SUCCESS);
    
    // Initialize to |0⟩ state
    ComplexFloat zero_state[2] = {
        CMPLX(1.0f, 0.0f),
        CMPLX(0.0f, 0.0f)
    };
    err = quantum_state_initialize(state, zero_state);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    // Create path parameters (closed loop on Bloch sphere)
    const int num_steps = 100;
    float theta[num_steps], phi[num_steps];
    for (int i = 0; i < num_steps; i++) {
        float t = 2.0f * M_PI * i / (num_steps - 1);
        theta[i] = M_PI / 3.0f;  // Fixed polar angle
        phi[i] = t;              // Azimuthal angle
    }
    
    // Create geometric phase tensor
    quantum_geometric_tensor_t* phase_tensor;
    size_t dims[] = {1}; // Scalar phase
    err = geometric_tensor_create(&phase_tensor, GEOMETRIC_TENSOR_SCALAR, dims, 1);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    quantum_geometric_metric_t* metric;
    err = geometric_create_metric(&metric, GEOMETRIC_METRIC_EUCLIDEAN, DIM);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    err = geometric_compute_metric(metric, state);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    quantum_geometric_connection_t* connection;
    err = geometric_create_connection(&connection, GEOMETRIC_CONNECTION_LEVI_CIVITA, DIM);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    err = geometric_compute_connection(connection, metric);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    ComplexFloat phase;
    err = geometric_compute_phase(&phase, state, connection);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    err = geometric_tensor_initialize(phase_tensor, &phase);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    // Expected phase for this loop = 2π(1 - cos(θ))
    float expected_phase = 2.0f * M_PI * (1.0f - cos(M_PI / 3.0f));
    TEST_ASSERT_COMPLEX_EQ(phase_tensor->components[0], CMPLX(expected_phase, 0.0f));
    
    // Cleanup
    geometric_tensor_destroy(phase_tensor);
    geometric_destroy_connection(connection);
    geometric_destroy_metric(metric);
    quantum_state_destroy(state);
    
    TEST_TEARDOWN();
}

// Test parallel transport
void test_parallel_transport() {
    TEST_SETUP();
    
    // Create initial state |0⟩
    GeometricConfig config = {
        .error_rate = 0.01f,
        .flags = QG_FLAG_OPTIMIZE | QG_FLAG_ERROR_CORRECT
    };
    
    quantum_state_t* state;
    qgt_error_t err = quantum_state_create(&state, QUANTUM_STATE_PURE, 2);  // 2 = 2^1 for 1 qubit
    TEST_ASSERT(err == QGT_SUCCESS);
    TEST_ASSERT(state != NULL);
    
    // Initialize to |0⟩ state
    state->amplitudes[0] = CMPLX(1.0f, 0.0f);
    state->amplitudes[1] = CMPLX(0.0f, 0.0f);
    
    // Create connection tensor
    quantum_geometric_connection_t* connection;
    size_t dims[] = {2, 2}; // 2x2 connection coefficients
    err = geometric_tensor_create(&connection, GEOMETRIC_TENSOR_SCALAR, dims, 2);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    quantum_geometric_metric_t* metric;
    err = geometric_create_metric(&metric, GEOMETRIC_METRIC_EUCLIDEAN, DIM);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    err = geometric_compute_metric(metric, state);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    quantum_geometric_connection_t* conn;
    err = geometric_create_connection(&conn, GEOMETRIC_CONNECTION_LEVI_CIVITA, DIM);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    err = geometric_compute_connection(conn, metric);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    err = geometric_tensor_initialize(connection, conn->coefficients);
    TEST_ASSERT(err == QGT_SUCCESS);
    
    // Verify connection coefficients are real (up to numerical error)
    for (int i = 0; i < 4; i++) {
        TEST_ASSERT_FLOAT_EQ(cimagf(connection->coefficients[i]), 0.0f);
    }
    
    // Cleanup
    geometric_tensor_destroy(connection);
    geometric_destroy_connection(conn);
    geometric_destroy_metric(metric);
    quantum_state_destroy(state);
    
    TEST_TEARDOWN();
}

// Register tests
REGISTER_TEST(test_quantum_geometric_tensor_basic);
REGISTER_TEST(test_berry_curvature);
REGISTER_TEST(test_geometric_phase);
REGISTER_TEST(test_parallel_transport);

// Test runner (if not using CTest)
#ifdef STANDALONE_TEST
int main(int argc, char** argv) {
    // Initialize core systems
    qgt_error_t err = geometric_core_initialize();
    if (err != QGT_SUCCESS) {
        printf("Failed to initialize geometric core\n");
        return 1;
    }
    
    TEST_MPI_INIT();
    TEST_METAL_INIT();
    TEST_CUDA_INIT();
    
    // Run all registered tests
    for (int i = 0; i < g_test_count; i++) {
        g_tests[i].func();
    }
    
    TEST_CUDA_CLEANUP();
    TEST_METAL_CLEANUP();
    TEST_MPI_FINALIZE();
    
    // Cleanup core systems
    geometric_core_shutdown();
    
    printf("All tests passed!\n");
    return 0;
}
#endif

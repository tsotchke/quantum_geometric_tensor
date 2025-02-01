#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <complex.h>

// Export for Python ctypes
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

static size_t get_memory_usage(void);
static void generate_random_unitary(double complex gate[2][2]);

EXPORT void quantum_geometric_simulate(int size, int num_gates, double* results) {
    clock_t start_time = clock();
    size_t start_mem = get_memory_usage();
    
    // Initialize quantum state
    double complex* state = (double complex*)malloc(size * sizeof(double complex));
    double norm = sqrt(size);
    for (int i = 0; i < size; i++) {
        state[i] = 1.0 / norm;
    }
    
    // Apply gates
    for (int g = 0; g < num_gates; g++) {
        double complex gate[2][2];
        generate_random_unitary(gate);
        
        // Apply 2-qubit gate
        for (int i = 0; i < size; i += 2) {
            double complex temp0 = state[i];
            double complex temp1 = state[i+1];
            state[i] = gate[0][0] * temp0 + gate[0][1] * temp1;
            state[i+1] = gate[1][0] * temp0 + gate[1][1] * temp1;
        }
    }
    
    // Calculate fidelity
    double complex fidelity = 0;
    for (int i = 0; i < size; i++) {
        fidelity += conj(state[i]) * state[i];
    }
    
    clock_t end_time = clock();
    size_t end_mem = get_memory_usage();
    
    // Store results
    results[0] = (double)(end_time - start_time) / CLOCKS_PER_SEC;  // Time in seconds
    results[1] = (double)(end_mem - start_mem) / 1024.0;  // Memory in MB
    results[2] = cabs(fidelity);  // Fidelity
    
    free(state);
}

static size_t get_memory_usage(void) {
    FILE* file = fopen("/proc/self/statm", "r");
    if (file == NULL) return 0;
    
    unsigned long size;
    fscanf(file, "%lu", &size);
    fclose(file);
    
    return size * sysconf(_SC_PAGESIZE);
}

static void generate_random_unitary(double complex gate[2][2]) {
    // Generate random complex numbers
    double complex a = (double)rand() / RAND_MAX + ((double)rand() / RAND_MAX) * I;
    double complex b = (double)rand() / RAND_MAX + ((double)rand() / RAND_MAX) * I;
    
    // Make unitary using QR decomposition
    double norm = sqrt(cabs(a)*cabs(a) + cabs(b)*cabs(b));
    a /= norm;
    b /= norm;
    
    gate[0][0] = a;
    gate[0][1] = b;
    gate[1][0] = -conj(b);
    gate[1][1] = conj(a);
}

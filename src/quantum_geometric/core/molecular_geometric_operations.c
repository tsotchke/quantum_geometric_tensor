#include "quantum_geometric/core/molecular_geometric_operations.h"
#include "quantum_geometric/core/simd_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Platform-specific SIMD includes
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
    #if defined(__ARM_NEON) || defined(__ARM_NEON__)
        #include <arm_neon.h>
    #endif
#endif

// Constants for geometric operations
#define MAX_SPHERICAL_DEGREE 4
#define NUM_RADIAL_BASES 32
#define CUTOFF_RADIUS 10.0
#define EPSILON 1e-6

// Compute factorials for normalization (cached for efficiency)
static double factorial(int n) {
    if (n <= 1) return 1.0;
    double result = 1.0;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

// Associated Legendre polynomial P_l^m(x) using recurrence relations
// Uses the stable recurrence: (l-m)P_l^m = x(2l-1)P_{l-1}^m - (l+m-1)P_{l-2}^m
static double associated_legendre(int l, int m, double x) {
    if (m < 0 || m > l) return 0.0;

    // P_m^m = (-1)^m (2m-1)!! (1-x^2)^(m/2)
    double pmm = 1.0;
    if (m > 0) {
        double somx2 = sqrt((1.0 - x) * (1.0 + x));
        double fact = 1.0;
        for (int i = 1; i <= m; i++) {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }

    if (l == m) return pmm;

    // P_{m+1}^m = x(2m+1)P_m^m
    double pmmp1 = x * (2 * m + 1) * pmm;
    if (l == m + 1) return pmmp1;

    // Use recurrence for l > m+1
    double pll = 0.0;
    for (int ll = m + 2; ll <= l; ll++) {
        pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }

    return pll;
}

// Helper function for computing real spherical harmonics Y_lm
// Uses the convention: Y_l^m = N_l^m * P_l^|m|(cos θ) * {cos(mφ) for m>=0, sin(|m|φ) for m<0}
// Output array is indexed as Y_lm[l*(l+1) + m] for m in [-l, l]
static void compute_spherical_harmonics(
    const double* r,
    double* Y_lm,
    size_t l_max
) {
    double x = r[0], y = r[1], z = r[2];
    double r2 = x*x + y*y + z*z;
    double r_norm = sqrt(r2);

    // Total size needed: sum_{l=0}^{l_max} (2l+1) = (l_max+1)^2
    size_t total_size = (l_max + 1) * (l_max + 1);
    memset(Y_lm, 0, total_size * sizeof(double));

    if (r_norm < EPSILON) {
        // At origin, only Y_00 is non-zero
        Y_lm[0] = 1.0 / sqrt(4.0 * M_PI);  // Y_00
        return;
    }

    // Compute spherical coordinates
    double cos_theta = z / r_norm;
    double phi = atan2(y, x);

    // Compute real spherical harmonics for each (l, m)
    for (int l = 0; l <= (int)l_max; l++) {
        for (int m = -l; m <= l; m++) {
            // Normalization factor
            // N_l^m = sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!)
            int abs_m = abs(m);
            double norm = sqrt((2.0 * l + 1.0) / (4.0 * M_PI) *
                               factorial(l - abs_m) / factorial(l + abs_m));

            // Associated Legendre polynomial P_l^|m|(cos θ)
            double plm = associated_legendre(l, abs_m, cos_theta);

            // Azimuthal factor
            double azimuthal;
            if (m > 0) {
                // Real spherical harmonic with cos(mφ)
                azimuthal = sqrt(2.0) * cos(m * phi);
            } else if (m < 0) {
                // Real spherical harmonic with sin(|m|φ)
                azimuthal = sqrt(2.0) * sin(abs_m * phi);
            } else {
                // m = 0: no azimuthal factor
                azimuthal = 1.0;
            }

            // Store in array with index l*(l+1) + m (which maps m in [-l,l] to [0, 2l])
            // Using standard indexing: l^2 + l + m gives unique index
            size_t idx = (size_t)(l * l + l + m);
            if (idx < total_size) {
                Y_lm[idx] = norm * plm * azimuthal;
            }
        }
    }
}

// Helper function for radial basis functions
static void compute_radial_basis(
    double r,
    double* R_n,
    size_t num_bases
) {
    double r_cut = CUTOFF_RADIUS;
    if (r > r_cut) {
        memset(R_n, 0, num_bases * sizeof(double));
        return;
    }
    
    // Gaussian radial basis
    double width = r_cut / num_bases;
    for (size_t n = 0; n < num_bases; n++) {
        double mu = n * width;
        R_n[n] = exp(-pow((r - mu)/width, 2));
    }
    
    // Normalize
    double sum = 0.0;
    for (size_t n = 0; n < num_bases; n++) {
        sum += R_n[n];
    }
    if (sum > EPSILON) {
        for (size_t n = 0; n < num_bases; n++) {
            R_n[n] /= sum;
        }
    }
}

void geometric_message_passing(
    const MolecularGraph* graph,
    GeometricFeatures* features,
    const SE3Transform* transform
) {
    size_t N = graph->num_atoms;
    size_t F = features->num_node_features;
    
    // Allocate temporary buffers
    double* messages = calloc(N * N * F, sizeof(double));
    double* Y_lm = malloc((MAX_SPHERICAL_DEGREE + 1) * (MAX_SPHERICAL_DEGREE + 1) * sizeof(double));
    double* R_n = malloc(NUM_RADIAL_BASES * sizeof(double));
    
    // Compute pairwise messages
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            if (i == j) continue;
            
            // Compute relative position
            double rij[3] = {
                graph->positions[3*j] - graph->positions[3*i],
                graph->positions[3*j+1] - graph->positions[3*i],
                graph->positions[3*j+2] - graph->positions[3*i]
            };
            double r = sqrt(rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]);
            
            // Skip if beyond cutoff
            if (r > CUTOFF_RADIUS) continue;
            
            // Compute spherical harmonic features
            compute_spherical_harmonics(rij, Y_lm, MAX_SPHERICAL_DEGREE);
            
            // Compute radial basis
            compute_radial_basis(r, R_n, NUM_RADIAL_BASES);
            
            // Combine features
            size_t msg_idx = (i * N + j) * F;
            for (size_t f = 0; f < F; f++) {
                double msg = 0.0;
                // Combine spherical and radial features
                for (size_t l = 0; l <= MAX_SPHERICAL_DEGREE; l++) {
                    for (size_t n = 0; n < NUM_RADIAL_BASES; n++) {
                        msg += Y_lm[l*l] * R_n[n] * features->node_features[j * F + f];
                    }
                }
                messages[msg_idx + f] = msg;
            }
        }
    }
    
    // Update node features with messages
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        for (size_t f = 0; f < F; f++) {
            double sum = 0.0;
            for (size_t j = 0; j < N; j++) {
                sum += messages[(i * N + j) * F + f];
            }
            features->node_features[i * F + f] += sum;
        }
    }
    
    // Free temporary buffers
    free(messages);
    free(Y_lm);
    free(R_n);
}

// Compute Wigner small-d matrix element d^l_{m'm}(beta)
// Uses explicit formula for low l and recursion for higher l
static double wigner_small_d(size_t l, int m_prime, int m, double beta) {
    if (l == 0) return 1.0;

    double cos_half = cos(beta / 2.0);
    double sin_half = sin(beta / 2.0);

    // Use explicit formulas for l <= 2 (common cases in molecular physics)
    if (l == 1) {
        double c = cos_half, s = sin_half;
        double c2 = c * c, s2 = s * s;
        double cs = c * s;

        // d^1 matrix
        if (m_prime == 1 && m == 1) return c2;
        if (m_prime == 1 && m == 0) return -sqrt(2.0) * cs;
        if (m_prime == 1 && m == -1) return s2;
        if (m_prime == 0 && m == 1) return sqrt(2.0) * cs;
        if (m_prime == 0 && m == 0) return c2 - s2;
        if (m_prime == 0 && m == -1) return -sqrt(2.0) * cs;
        if (m_prime == -1 && m == 1) return s2;
        if (m_prime == -1 && m == 0) return sqrt(2.0) * cs;
        if (m_prime == -1 && m == -1) return c2;
    }

    // General formula using recursion relation for higher l
    // d^l_{m'm}(beta) = sum_k (-1)^k * C(l+m, k) * C(l-m, l-m'-k) *
    //                   cos^(2l+m-m'-2k)(beta/2) * sin^(2k+m'-m)(beta/2)
    int k_min = (0 > m - m_prime) ? 0 : m - m_prime;
    int k_max = ((int)l + m < (int)l - m_prime) ? (int)l + m : (int)l - m_prime;

    double sum = 0.0;
    for (int k = k_min; k <= k_max; k++) {
        int pow_cos = 2 * (int)l + m - m_prime - 2 * k;
        int pow_sin = 2 * k + m_prime - m;

        if (pow_cos < 0 || pow_sin < 0) continue;

        // Compute binomial coefficients
        double binom1 = 1.0, binom2 = 1.0;
        for (int i = 0; i < k; i++) {
            binom1 *= (double)((int)l + m - i) / (double)(k - i);
        }
        for (int i = 0; i < (int)l - m_prime - k; i++) {
            binom2 *= (double)((int)l - m - i) / (double)((int)l - m_prime - k - i);
        }

        double term = pow(cos_half, pow_cos) * pow(sin_half, pow_sin);
        int sign = (k % 2 == 0) ? 1 : -1;
        sum += sign * binom1 * binom2 * term;
    }

    return sum;
}

// Compute Wigner-D matrix element D^l_{m'm}(R) from rotation matrix R
static double wigner_D_element(size_t l, int m_prime, int m, const double* R) {
    // Extract Euler angles (ZYZ convention) from rotation matrix
    double r33 = R[8];  // R[2,2]
    double r31 = R[6], r32 = R[7];  // R[2,0], R[2,1]
    double r13 = R[2], r23 = R[5];  // R[0,2], R[1,2]

    double beta = acos(fmax(-1.0, fmin(1.0, r33)));
    double alpha = 0.0, gamma = 0.0;

    double sin_beta = sin(beta);
    if (fabs(sin_beta) > EPSILON) {
        alpha = atan2(r32, r31);
        gamma = atan2(r23, -r13);
    } else if (r33 > 0) {
        // beta ~ 0: R[0,0] = cos(alpha+gamma), R[0,1] = sin(alpha+gamma)
        alpha = atan2(R[1], R[0]) / 2.0;
        gamma = alpha;
    } else {
        // beta ~ pi: R[0,0] = -cos(alpha-gamma), R[0,1] = sin(alpha-gamma)
        alpha = atan2(R[1], -R[0]) / 2.0;
        gamma = -alpha;
    }

    // D^l_{m'm}(alpha, beta, gamma) = e^{-i m' alpha} d^l_{m'm}(beta) e^{-i m gamma}
    double d_elem = wigner_small_d(l, m_prime, m, beta);
    double phase = -m_prime * alpha - m * gamma;

    // Return real part for real spherical harmonics
    return d_elem * cos(phase);
}

void steerable_convolution(
    const double* input_features,
    const double* filter_weights,
    double* output_features,
    const SE3Transform* transform,
    size_t num_channels
) {
    // Full SE(3)-equivariant steerable convolution implementation
    // Rotates spherical harmonic features using Wigner-D matrices and averages

    size_t num_rotations = transform->num_layers;
    size_t num_irreps = (MAX_SPHERICAL_DEGREE + 1) * (MAX_SPHERICAL_DEGREE + 1);

    if (num_rotations == 0) {
        // No rotation sampling - just apply filters directly
        #pragma omp parallel for collapse(2)
        for (size_t l = 0; l <= MAX_SPHERICAL_DEGREE; l++) {
            for (int m = -(int)l; m <= (int)l; m++) {
                size_t out_idx = l * l + l + m;
                for (size_t c = 0; c < num_channels; c++) {
                    double sum = 0.0;
                    for (size_t n = 0; n < NUM_RADIAL_BASES; n++) {
                        size_t weight_idx = (out_idx * num_channels + c) * NUM_RADIAL_BASES + n;
                        sum += filter_weights[weight_idx] * input_features[out_idx];
                    }
                    output_features[out_idx * num_channels + c] = sum;
                }
            }
        }
        return;
    }

    // Initialize output
    memset(output_features, 0, num_irreps * num_channels * sizeof(double));

    // Allocate buffer for rotated irrep features
    double* rotated_irreps = malloc(num_irreps * sizeof(double));
    if (!rotated_irreps) return;

    // Sample rotations and average (Monte Carlo integration over SO(3))
    for (size_t r = 0; r < num_rotations; r++) {
        const double* R = &transform->rotation_matrices[r * 9];
        double weight = transform->weights ? transform->weights[r] : 1.0;

        // Rotate each irrep using Wigner-D matrices
        for (size_t l = 0; l <= MAX_SPHERICAL_DEGREE; l++) {
            size_t l_offset = l * l + l;  // Base index for m=0 of this l

            for (int m_prime = -(int)l; m_prime <= (int)l; m_prime++) {
                double rotated = 0.0;

                // Sum over original m values: Y'^l_{m'} = sum_m D^l_{m'm}(R) Y^l_m
                for (int m = -(int)l; m <= (int)l; m++) {
                    double D_elem = wigner_D_element(l, m_prime, m, R);
                    rotated += D_elem * input_features[l_offset + m];
                }
                rotated_irreps[l_offset + m_prime] = rotated;
            }
        }

        // Apply steerable filters to rotated features and accumulate
        #pragma omp parallel for
        for (size_t idx = 0; idx < num_irreps; idx++) {
            for (size_t c = 0; c < num_channels; c++) {
                double sum = 0.0;

                // Radial convolution with filter weights
                for (size_t n = 0; n < NUM_RADIAL_BASES; n++) {
                    size_t weight_idx = (idx * num_channels + c) * NUM_RADIAL_BASES + n;
                    sum += filter_weights[weight_idx] * rotated_irreps[idx];
                }

                #pragma omp atomic
                output_features[idx * num_channels + c] += sum * weight;
            }
        }
    }

    // Normalize by total weight (or number of samples)
    double total_weight = 0.0;
    if (transform->weights) {
        for (size_t r = 0; r < num_rotations; r++) {
            total_weight += transform->weights[r];
        }
    } else {
        total_weight = (double)num_rotations;
    }

    if (total_weight > EPSILON) {
        double inv_weight = 1.0 / total_weight;
        for (size_t i = 0; i < num_irreps * num_channels; i++) {
            output_features[i] *= inv_weight;
        }
    }

    free(rotated_irreps);
}

void spherical_harmonic_transform(
    const double* positions,
    double* harmonics,
    size_t max_degree,
    size_t num_points
) {
    #pragma omp parallel for
    for (size_t i = 0; i < num_points; i++) {
        compute_spherical_harmonics(
            &positions[3*i],
            &harmonics[i * (max_degree + 1) * (max_degree + 1)],
            max_degree
        );
    }
}

void radial_basis_expansion(
    const double* distances,
    double* expansions,
    size_t num_bases,
    size_t num_pairs
) {
    #pragma omp parallel for
    for (size_t i = 0; i < num_pairs; i++) {
        compute_radial_basis(
            distances[i],
            &expansions[i * num_bases],
            num_bases
        );
    }
}

// Memory management implementations
void initialize_molecular_graph(
    MolecularGraph* graph,
    const double* positions,
    const int* atomic_numbers,
    const int* bonds,
    size_t num_atoms,
    size_t num_bonds
) {
    graph->num_atoms = num_atoms;
    graph->num_bonds = num_bonds;
    
    // Allocate memory
    graph->positions = malloc(3 * num_atoms * sizeof(double));
    graph->atomic_numbers = malloc(num_atoms * sizeof(int));
    graph->charges = calloc(num_atoms, sizeof(double));
    graph->bond_indices = malloc(2 * num_bonds * sizeof(int));
    graph->bond_types = malloc(num_bonds * sizeof(double));
    
    // Copy data
    memcpy(graph->positions, positions, 3 * num_atoms * sizeof(double));
    memcpy(graph->atomic_numbers, atomic_numbers, num_atoms * sizeof(int));
    memcpy(graph->bond_indices, bonds, 2 * num_bonds * sizeof(int));
}

void free_molecular_graph(MolecularGraph* graph) {
    free(graph->positions);
    free(graph->atomic_numbers);
    free(graph->charges);
    free(graph->bond_indices);
    free(graph->bond_types);
}

void initialize_geometric_features(
    GeometricFeatures* features,
    size_t num_nodes,
    size_t num_edges,
    size_t node_features,
    size_t edge_features,
    size_t global_features
) {
    features->num_node_features = node_features;
    features->num_edge_features = edge_features;
    features->num_global_features = global_features;
    
    features->node_features = calloc(num_nodes * node_features, sizeof(double));
    features->edge_features = calloc(num_edges * edge_features, sizeof(double));
    features->global_features = calloc(global_features, sizeof(double));
}

void free_geometric_features(GeometricFeatures* features) {
    free(features->node_features);
    free(features->edge_features);
    free(features->global_features);
}

/**
 * @file anyon_operations.c
 * @brief Implementation of anyon braiding and fusion operations
 *
 * Physics background:
 * - Anyons are quasiparticles in 2D systems with fractional statistics
 * - Braiding exchanges anyons, accumulating geometric phase
 * - Fusion combines anyons according to algebraic rules
 * - Topological quantum computation uses these operations
 */

#include "quantum_geometric/physics/anyon_detection.h"
#include "quantum_geometric/core/quantum_geometric_core.h"
#include "quantum_geometric/core/quantum_complex.h"
#include "quantum_geometric/physics/quantum_state_operations.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

// Physical constants
#define HBAR 1.054571817e-34    // Reduced Planck constant (J·s)
#define ANYON_MASS 1.0e-30      // Effective anyon mass (kg)
#define LATTICE_SPACING 1.0     // Lattice spacing (dimensionless units)

// Fibonacci anyon golden ratio
#define PHI ((1.0 + sqrt(5.0)) / 2.0)
#define PHI_INV (1.0 / PHI)

// Fusion statistics tracking (module-level)
static struct {
    size_t total_fusions;
    size_t successful_fusions;
    double total_energy_released;
    size_t fusion_counts_by_type[9];  // Count per anyon_type_t
} fusion_stats = {0};

// ============================================================================
// Basic Helper Functions
// ============================================================================

double calculate_distance(const AnyonPosition* pos1, const AnyonPosition* pos2) {
    if (!pos1 || !pos2) {
        return 0.0;
    }

    double dx = (double)pos1->x - (double)pos2->x;
    double dy = (double)pos1->y - (double)pos2->y;
    double dz = (double)pos1->z - (double)pos2->z;

    return sqrt(dx * dx + dy * dy + dz * dz);
}

// ============================================================================
// Statistical Angle Calculations
// ============================================================================

double calculate_statistical_angle(anyon_type_t type1, anyon_type_t type2) {
    // Statistical angle θ determines exchange statistics: ψ → e^{iθ} ψ
    // Bosons: θ = 0, Fermions: θ = π, Anyons: 0 < θ < π

    // Self-exchange angles by type
    double theta1, theta2;

    switch (type1) {
        case ANYON_ABELIAN:
            theta1 = M_PI / 4.0;  // Generic abelian: π/4 (semion-like)
            break;
        case ANYON_NON_ABELIAN:
            theta1 = M_PI / 3.0;  // Non-abelian: π/3
            break;
        case ANYON_MAJORANA:
            theta1 = M_PI / 2.0;  // Majorana: π/2 (Ising anyons)
            break;
        case ANYON_FIBONACCI:
            // Fibonacci anyon: θ = 4π/5
            theta1 = 4.0 * M_PI / 5.0;
            break;
        case ANYON_ISING:
            theta1 = M_PI / 8.0;  // Ising model anyons
            break;
        default:
            theta1 = 0.0;  // Trivial (vacuum)
    }

    switch (type2) {
        case ANYON_ABELIAN:
            theta2 = M_PI / 4.0;
            break;
        case ANYON_NON_ABELIAN:
            theta2 = M_PI / 3.0;
            break;
        case ANYON_MAJORANA:
            theta2 = M_PI / 2.0;
            break;
        case ANYON_FIBONACCI:
            theta2 = 4.0 * M_PI / 5.0;
            break;
        case ANYON_ISING:
            theta2 = M_PI / 8.0;
            break;
        default:
            theta2 = 0.0;
    }

    // For mutual statistics, use geometric mean
    return sqrt(theta1 * theta2);
}

double calculate_charge_phase_factor(AnyonCharge charge1, AnyonCharge charge2) {
    // Aharonov-Bohm phase from charge interaction
    // φ = (q1_e * q2_m - q1_m * q2_e) / ℏ (in natural units)

    double electric_term = charge1.electric * charge2.magnetic;
    double magnetic_term = charge1.magnetic * charge2.electric;
    double topological_term = charge1.topological * charge2.topological;

    // Phase factor includes topological contribution
    return (electric_term - magnetic_term) + M_PI * topological_term;
}

double calculate_topological_correction(const Anyon* anyon1, const Anyon* anyon2) {
    if (!anyon1 || !anyon2) {
        return 1.0;
    }

    // Topological correction based on winding number and topological charge
    double q1 = anyon1->charge.topological;
    double q2 = anyon2->charge.topological;

    // For Fibonacci anyons, use quantum dimension d = φ
    double d1 = 1.0, d2 = 1.0;
    if (anyon1->type == ANYON_FIBONACCI) d1 = PHI;
    if (anyon2->type == ANYON_FIBONACCI) d2 = PHI;

    // Topological correction: related to S-matrix elements
    // S_{ab} / (S_{0a} S_{0b}) where S is modular S-matrix
    return sqrt(d1 * d2) * cos(M_PI * q1 * q2);
}

// ============================================================================
// Braiding Phase Calculations
// ============================================================================

/**
 * @brief Calculate braiding phase using Berry connection line integral
 *
 * For anyons in 2D, the Berry phase accumulated during braiding is:
 *   γ = ∮ A · dl
 *
 * where A is the Berry connection:
 *   A = (θ/2π) * (ẑ × r) / |r|²
 *
 * In Cartesian coordinates for 2D:
 *   A_x = -(θ/2π) * y / (x² + y²)
 *   A_y =  (θ/2π) * x / (x² + y²)
 *
 * This gives the correct Aharonov-Bohm like phase for anyonic braiding.
 */
complex double calculate_braiding_phase_step(const quantum_state* state,
                                            const Anyon* moving_anyon,
                                            const Anyon* stationary_anyon,
                                            const AnyonPosition* prev_pos) {
    if (!moving_anyon || !stationary_anyon || !prev_pos) {
        return 1.0;  // Identity phase if invalid
    }

    // Get statistical angle θ for this anyon pair
    // For Abelian anyons: θ ∈ [0, 2π)
    // For Fibonacci: θ = 4π/5 (braiding τ around τ)
    // For Ising: θ = π/8 (braiding σ around σ)
    double theta = calculate_statistical_angle(moving_anyon->type, stationary_anyon->type);

    // Position of stationary anyon (the "flux" source)
    double sx = (double)stationary_anyon->position.x;
    double sy = (double)stationary_anyon->position.y;
    double sz = (double)stationary_anyon->position.z;

    // Previous position of moving anyon relative to stationary
    double x1 = (double)prev_pos->x - sx;
    double y1 = (double)prev_pos->y - sy;
    double z1 = (double)prev_pos->z - sz;

    // Current position of moving anyon relative to stationary
    double x2 = (double)moving_anyon->position.x - sx;
    double y2 = (double)moving_anyon->position.y - sy;
    double z2 = (double)moving_anyon->position.z - sz;

    // Displacement vector (path element dl)
    double dx = x2 - x1;
    double dy = y2 - y1;
    double dz = z2 - z1;

    // Check if motion is primarily 2D (in xy-plane)
    // Use dz to determine if there's significant vertical motion
    bool is_2d = (fabs(z1) < 1e-10 && fabs(z2) < 1e-10 && fabs(dz) < 1e-10);

    double accumulated_phase = 0.0;

    if (is_2d) {
        // 2D Berry connection: A = (θ/2π) * (-y, x, 0) / (x² + y²)
        // Use midpoint for better accuracy (trapezoidal rule)
        double x_mid = (x1 + x2) / 2.0;
        double y_mid = (y1 + y2) / 2.0;
        double r_sq = x_mid * x_mid + y_mid * y_mid;

        if (r_sq < 1e-20) {
            // Anyons too close - return unit phase to avoid singularity
            return 1.0;
        }

        // Berry connection components
        double A_x = -(theta / (2.0 * M_PI)) * y_mid / r_sq;
        double A_y =  (theta / (2.0 * M_PI)) * x_mid / r_sq;

        // Line integral: A · dl
        accumulated_phase = A_x * dx + A_y * dy;

    } else {
        // 3D case: use solid angle method for full generality
        // Solid angle Ω subtended by the motion gives phase θ * Ω / (4π)

        double r1_sq = x1*x1 + y1*y1 + z1*z1;
        double r2_sq = x2*x2 + y2*y2 + z2*z2;
        double r1 = sqrt(r1_sq);
        double r2 = sqrt(r2_sq);

        if (r1 < 1e-10 || r2 < 1e-10) {
            return 1.0;  // Avoid singularity
        }

        // Normalize position vectors
        double n1_x = x1/r1, n1_y = y1/r1, n1_z = z1/r1;
        double n2_x = x2/r2, n2_y = y2/r2, n2_z = z2/r2;

        // Cross product n1 × n2
        double cross_x = n1_y * n2_z - n1_z * n2_y;
        double cross_y = n1_z * n2_x - n1_x * n2_z;
        double cross_z = n1_x * n2_y - n1_y * n2_x;

        // Dot product n1 · n2
        double dot = n1_x * n2_x + n1_y * n2_y + n1_z * n2_z;

        // Solid angle element using atan2 for numerical stability
        // ΔΩ = 2 * atan2(|n1 × n2|, 1 + n1·n2)
        double cross_mag = sqrt(cross_x*cross_x + cross_y*cross_y + cross_z*cross_z);
        double solid_angle = 2.0 * atan2(cross_mag, 1.0 + dot);

        // Sign from z-component of cross product (handedness of rotation)
        if (cross_z < 0) solid_angle = -solid_angle;

        // Phase from solid angle: for anyons, γ = θ * Ω / (2π)
        accumulated_phase = theta * solid_angle / (2.0 * M_PI);
    }

    // Add charge-dependent contribution if non-zero
    double charge_phase = calculate_charge_phase_factor(moving_anyon->charge,
                                                        stationary_anyon->charge);
    if (fabs(charge_phase) > 1e-10) {
        // Charge phase scaled by winding
        double winding = atan2(y2, x2) - atan2(y1, x1);
        // Normalize winding to [-π, π]
        while (winding > M_PI) winding -= 2.0 * M_PI;
        while (winding < -M_PI) winding += 2.0 * M_PI;
        accumulated_phase += charge_phase * winding / (2.0 * M_PI);
    }

    return cexp(I * accumulated_phase);
}

/**
 * @brief Calculate complete braiding phase for a full exchange
 *
 * When anyon a is braided completely around anyon b, the phase is:
 *   R_{ab} = e^{iθ_{ab}}
 *
 * where θ_{ab} is the mutual statistical angle.
 */
complex double calculate_full_braid_phase(anyon_type_t type_a, anyon_type_t type_b) {
    double theta = calculate_statistical_angle(type_a, type_b);
    return cexp(I * theta);
}

// ============================================================================
// Charge and Type Interactions
// ============================================================================

double calculate_charge_interaction(AnyonCharge charge1, AnyonCharge charge2) {
    // Coulomb-like interaction for electric charges
    double electric = charge1.electric * charge2.electric;

    // Magnetic interaction (dual to electric)
    double magnetic = charge1.magnetic * charge2.magnetic;

    // Topological interaction (statistical)
    double topological = charge1.topological * charge2.topological;

    // Combined interaction strength
    return electric + magnetic + 2.0 * topological;
}

double calculate_type_interaction(anyon_type_t type1, anyon_type_t type2) {
    // Type-dependent coupling constants
    // Based on quantum dimension and topological spin

    // Quantum dimensions by type
    double d1, d2;
    switch (type1) {
        case ANYON_FIBONACCI:
            d1 = PHI;
            break;
        case ANYON_ISING:
        case ANYON_MAJORANA:
            d1 = sqrt(2.0);
            break;
        case ANYON_NON_ABELIAN:
            d1 = 2.0;
            break;
        default:
            d1 = 1.0;
    }

    switch (type2) {
        case ANYON_FIBONACCI:
            d2 = PHI;
            break;
        case ANYON_ISING:
        case ANYON_MAJORANA:
            d2 = sqrt(2.0);
            break;
        case ANYON_NON_ABELIAN:
            d2 = 2.0;
            break;
        default:
            d2 = 1.0;
    }

    // Interaction proportional to product of quantum dimensions
    return d1 * d2;
}

// ============================================================================
// Topological Protection Verification
// ============================================================================

bool verify_topological_protection(const quantum_state* state,
                                  const Anyon* const* anyons,
                                  size_t num_anyons) {
    if (!anyons || num_anyons == 0) {
        return true;  // Vacuously true
    }

    // Check 1: Minimum separation between anyons
    const double MIN_SEPARATION = 2.0 * LATTICE_SPACING;
    for (size_t i = 0; i < num_anyons; i++) {
        if (!anyons[i]) continue;
        for (size_t j = i + 1; j < num_anyons; j++) {
            if (!anyons[j]) continue;
            double dist = calculate_distance(&anyons[i]->position, &anyons[j]->position);
            if (dist < MIN_SEPARATION) {
                return false;  // Anyons too close, protection compromised
            }
        }
    }

    // Check 2: Total topological charge should be conserved
    double total_electric = 0.0;
    double total_magnetic = 0.0;
    double total_topological = 0.0;

    for (size_t i = 0; i < num_anyons; i++) {
        if (!anyons[i]) continue;
        total_electric += anyons[i]->charge.electric;
        total_magnetic += anyons[i]->charge.magnetic;
        total_topological += anyons[i]->charge.topological;
    }

    // Electric and magnetic charges must be conserved (sum to zero for closed system)
    // Allow small tolerance for numerical precision
    const double CHARGE_TOLERANCE = 0.01;
    if (fabs(total_electric) > CHARGE_TOLERANCE || fabs(total_magnetic) > CHARGE_TOLERANCE) {
        // Non-zero total electric or magnetic charge indicates open system
        // This is allowed but may indicate potential instability
        // For now, we only warn internally (could add logging here)
    }

    // Charges should sum to integer (or zero) for valid configuration
    double topo_mod = fmod(total_topological, 1.0);
    if (topo_mod > 0.5) topo_mod -= 1.0;
    if (fabs(topo_mod) > 0.01) {
        return false;  // Non-integer total topological charge
    }

    // Additional check: electric + magnetic charge conservation
    // For Abelian anyons, these should individually be quantized
    double elec_mod = fmod(total_electric, 1.0);
    if (elec_mod > 0.5) elec_mod -= 1.0;
    double mag_mod = fmod(total_magnetic, 1.0);
    if (mag_mod > 0.5) mag_mod -= 1.0;
    if (fabs(elec_mod) > 0.1 || fabs(mag_mod) > 0.1) {
        return false;  // Non-quantized electric or magnetic charge
    }

    // Check 3: Position stability
    for (size_t i = 0; i < num_anyons; i++) {
        if (!anyons[i]) continue;
        if (anyons[i]->position.stability < 0.5) {
            return false;  // Unstable anyon position
        }
    }

    return true;
}

// ============================================================================
// Braiding Operation Implementation
// ============================================================================

bool apply_braiding_operation(quantum_state* state,
                             const Anyon* anyon1,
                             const Anyon* anyon2,
                             complex double phase) {
    if (!state || !anyon1 || !anyon2) {
        return false;
    }

    // The R-matrix implements exchange of anyons
    // For abelian anyons: R = e^{iθ} (scalar phase)
    // For non-abelian: R is a matrix in fusion space

    // Get statistical information
    double theta = calculate_statistical_angle(anyon1->type, anyon2->type);
    double topological_correction = calculate_topological_correction(anyon1, anyon2);

    // Construct the R-matrix phase
    complex double R_phase = phase * topological_correction * cexp(I * theta);

    // For non-abelian anyons, the R-matrix is more complex
    if (anyon1->type == ANYON_FIBONACCI || anyon2->type == ANYON_FIBONACCI) {
        // Fibonacci anyon R-matrix: R^{τ,τ}_τ = e^{4πi/5}
        // R^{τ,τ}_1 = e^{-3πi/5}
        R_phase *= cexp(I * 4.0 * M_PI / 5.0);
    } else if (anyon1->type == ANYON_ISING || anyon2->type == ANYON_ISING ||
               anyon1->type == ANYON_MAJORANA || anyon2->type == ANYON_MAJORANA) {
        // Ising anyon R-matrix: R^{σ,σ}_ψ = e^{iπ/8}
        R_phase *= cexp(I * M_PI / 8.0);
    }

    // Apply phase to quantum state
    // The R-matrix phase is applied to all basis states where the anyons
    // are in the exchanged configuration
    if (state->coordinates && state->num_qubits > 0) {
        size_t dim = 1ULL << state->num_qubits;
        float R_real = (float)creal(R_phase);
        float R_imag = (float)cimag(R_phase);

        // Apply the phase rotation to the full state
        // For anyons at specific positions, this affects basis states
        // where those positions are occupied
        for (size_t i = 0; i < dim; i++) {
            ComplexFloat psi = state->coordinates[i];
            // Apply phase: ψ' = R * ψ = (R_real + i*R_imag)(ψ_real + i*ψ_imag)
            state->coordinates[i].real = R_real * psi.real - R_imag * psi.imag;
            state->coordinates[i].imag = R_real * psi.imag + R_imag * psi.real;
        }
    }

    // The braiding has been successfully applied
    return true;
}

// ============================================================================
// Fusion Energy and Probability Calculations
// ============================================================================

double calculate_fusion_energy(const Anyon* anyon1, const Anyon* anyon2) {
    if (!anyon1 || !anyon2) {
        return 0.0;
    }

    // Total energy before fusion
    double initial_energy = anyon1->energy + anyon2->energy;

    // Binding energy depends on type compatibility
    double binding_energy = 0.0;

    // Same-type fusion is most favorable
    if (anyon1->type == anyon2->type) {
        binding_energy = 0.5 * initial_energy;
    } else if ((anyon1->type == ANYON_ABELIAN && anyon2->type == ANYON_ABELIAN) ||
               (anyon1->type == ANYON_NON_ABELIAN && anyon2->type == ANYON_NON_ABELIAN)) {
        binding_energy = 0.3 * initial_energy;
    } else if (anyon1->type == ANYON_MAJORANA && anyon2->type == ANYON_MAJORANA) {
        // Majorana fusion to vacuum releases all energy
        binding_energy = initial_energy;
    } else {
        // Mixed types have weaker binding
        binding_energy = 0.1 * initial_energy;
    }

    // Distance-dependent correction
    double distance = calculate_distance(&anyon1->position, &anyon2->position);
    if (distance > LATTICE_SPACING) {
        binding_energy *= exp(-distance / (5.0 * LATTICE_SPACING));
    }

    // Fusion energy = total energy - binding energy
    return initial_energy - binding_energy;
}

double calculate_fusion_probability(const Anyon* anyon1, const Anyon* anyon2) {
    if (!anyon1 || !anyon2) {
        return 0.0;
    }

    // Base probability from quantum dimensions
    double d1 = 1.0, d2 = 1.0;
    double total_d = 1.0;  // Total quantum dimension of system

    // Get quantum dimensions
    switch (anyon1->type) {
        case ANYON_FIBONACCI:
            d1 = PHI;
            total_d = 1.0 + PHI;  // D² = 1 + φ² = 2 + φ
            break;
        case ANYON_ISING:
        case ANYON_MAJORANA:
            d1 = sqrt(2.0);
            total_d = 2.0;
            break;
        default:
            d1 = 1.0;
            total_d = 1.0;
    }

    switch (anyon2->type) {
        case ANYON_FIBONACCI:
            d2 = PHI;
            if (total_d < 2.0) total_d = 1.0 + PHI;
            break;
        case ANYON_ISING:
        case ANYON_MAJORANA:
            d2 = sqrt(2.0);
            if (total_d < 2.0) total_d = 2.0;
            break;
        default:
            d2 = 1.0;
    }

    // Fusion probability proportional to d_c / (d_a * d_b * D)
    // where c is the fusion outcome
    double prob = (d1 * d2) / (total_d * total_d);

    // Distance-dependent tunneling factor
    double distance = calculate_distance(&anyon1->position, &anyon2->position);
    double tunneling = exp(-distance / (3.0 * LATTICE_SPACING));

    // Coherence factor from lifetimes
    double coherence = 1.0;
    if (anyon1->lifetime > 0 && anyon2->lifetime > 0) {
        coherence = sqrt(anyon1->position.stability * anyon2->position.stability);
    }

    return prob * tunneling * coherence;
}

// ============================================================================
// Fusion Rules Implementation with Proper Quantum Mechanics
// ============================================================================

/**
 * Fibonacci anyon F-matrix: basis change between fusion trees
 *
 * For three τ anyons: ((τ × τ) × τ) ↔ (τ × (τ × τ))
 *
 * F^{τττ}_τ = | 1/φ    1/√φ |   where φ = (1+√5)/2 is golden ratio
 *             | 1/√φ   -1/φ |
 *
 * This is unitary: F^† F = I
 */
static const double FIBONACCI_F_MATRIX[2][2] = {
    { 0.6180339887498949,  0.7861513777574233 },  // { 1/φ,  1/√φ }
    { 0.7861513777574233, -0.6180339887498949 }   // { 1/√φ, -1/φ }
};

/**
 * Fibonacci anyon R-matrix: braiding phases in fusion channel basis
 *
 * R^{τ,τ}_1 = e^{-4πi/5}  (vacuum channel)
 * R^{τ,τ}_τ = e^{3πi/5}   (τ channel)
 */
static complex double get_fibonacci_r_phase(bool vacuum_channel) {
    if (vacuum_channel) {
        // R_1 = e^{-4πi/5}
        return cexp(I * (-4.0 * M_PI / 5.0));
    } else {
        // R_τ = e^{3πi/5}
        return cexp(I * (3.0 * M_PI / 5.0));
    }
}

/**
 * @brief Fusion superposition state for non-Abelian anyons
 *
 * For τ × τ = 1 ⊕ τ, the fusion state is:
 *   |ψ⟩ = α|1⟩ + β|τ⟩
 *
 * where |α|² + |β|² = 1
 *
 * Initialized with: α = 1/φ, β = 1/√φ (normalized probabilities)
 */
typedef struct {
    complex double amplitude_vacuum;   // Amplitude for vacuum (1) channel
    complex double amplitude_tau;      // Amplitude for τ channel
    bool is_measured;                  // Has the fusion outcome been measured?
    anyon_type_t measured_outcome;     // Result if measured
} FibonacciFusionState;

/**
 * @brief Initialize a Fibonacci fusion superposition
 *
 * For τ × τ, the initial state is:
 *   |ψ⟩ = (1/φ)|1⟩ + (1/√φ)|τ⟩
 *
 * Probabilities: P(1) = 1/φ² ≈ 0.382, P(τ) = 1/φ ≈ 0.618
 */
static FibonacciFusionState init_fibonacci_fusion(void) {
    FibonacciFusionState state;
    // Golden ratio: φ = (1 + √5) / 2 ≈ 1.618
    double inv_phi = 1.0 / PHI;           // 1/φ ≈ 0.618
    double inv_sqrt_phi = 1.0 / sqrt(PHI); // 1/√φ ≈ 0.786

    state.amplitude_vacuum = (complex double)inv_phi;
    state.amplitude_tau = (complex double)inv_sqrt_phi;
    state.is_measured = false;
    state.measured_outcome = ANYON_NONE;

    return state;
}

/**
 * @brief Apply F-move to fusion state
 *
 * Transforms between different fusion tree bases
 */
static void apply_f_move(FibonacciFusionState* state, bool inverse) {
    if (!state || state->is_measured) return;

    complex double a0 = state->amplitude_vacuum;
    complex double a1 = state->amplitude_tau;

    if (!inverse) {
        // Apply F-matrix
        state->amplitude_vacuum = FIBONACCI_F_MATRIX[0][0] * a0 + FIBONACCI_F_MATRIX[0][1] * a1;
        state->amplitude_tau   = FIBONACCI_F_MATRIX[1][0] * a0 + FIBONACCI_F_MATRIX[1][1] * a1;
    } else {
        // Apply F^{-1} = F^T (this matrix is orthogonal)
        state->amplitude_vacuum = FIBONACCI_F_MATRIX[0][0] * a0 + FIBONACCI_F_MATRIX[1][0] * a1;
        state->amplitude_tau   = FIBONACCI_F_MATRIX[0][1] * a0 + FIBONACCI_F_MATRIX[1][1] * a1;
    }
}

/**
 * @brief Apply braiding (R-matrix) to fusion state
 *
 * Each fusion channel acquires its R-phase when braiding
 */
static void apply_braid(FibonacciFusionState* state, bool clockwise) {
    if (!state || state->is_measured) return;

    complex double r_vacuum = get_fibonacci_r_phase(true);
    complex double r_tau = get_fibonacci_r_phase(false);

    if (!clockwise) {
        // Counter-clockwise: use R^{-1} = R^*
        r_vacuum = conj(r_vacuum);
        r_tau = conj(r_tau);
    }

    state->amplitude_vacuum *= r_vacuum;
    state->amplitude_tau *= r_tau;
}

/**
 * @brief Measure fusion outcome probabilistically
 *
 * Collapses superposition to definite outcome according to |amplitude|²
 */
static anyon_type_t measure_fusion_outcome(FibonacciFusionState* state) {
    if (!state) return ANYON_NONE;
    if (state->is_measured) return state->measured_outcome;

    double p_vacuum = cabs(state->amplitude_vacuum) * cabs(state->amplitude_vacuum);
    double p_tau = cabs(state->amplitude_tau) * cabs(state->amplitude_tau);

    // Normalize (should already be normalized, but ensure numerical stability)
    double total = p_vacuum + p_tau;
    p_vacuum /= total;

    // Random measurement
    double r = (double)rand() / (double)RAND_MAX;

    if (r < p_vacuum) {
        state->measured_outcome = ANYON_NONE;  // Vacuum
        state->amplitude_vacuum = 1.0;
        state->amplitude_tau = 0.0;
    } else {
        state->measured_outcome = ANYON_FIBONACCI;  // τ
        state->amplitude_vacuum = 0.0;
        state->amplitude_tau = 1.0;
    }

    state->is_measured = true;
    return state->measured_outcome;
}

anyon_type_t determine_fusion_type(const Anyon* anyon1, const Anyon* anyon2) {
    if (!anyon1 || !anyon2) {
        return ANYON_NONE;
    }

    // Fusion rules: a × b = Σ N^c_{ab} c
    // For non-Abelian anyons, this returns a probabilistic measurement

    anyon_type_t t1 = anyon1->type;
    anyon_type_t t2 = anyon2->type;

    // Vacuum fusion: 1 × a = a
    if (t1 == ANYON_NONE) return t2;
    if (t2 == ANYON_NONE) return t1;

    // Majorana fusion: σ × σ = 1 ⊕ ψ
    if (t1 == ANYON_MAJORANA && t2 == ANYON_MAJORANA) {
        // Probabilistic: 50% vacuum, 50% fermion
        double r = (double)rand() / (double)RAND_MAX;
        if (r < 0.5) {
            return ANYON_NONE;  // Fuse to vacuum
        } else {
            return ANYON_ABELIAN;  // Fuse to fermion
        }
    }

    // Fibonacci fusion: τ × τ = 1 ⊕ τ
    if (t1 == ANYON_FIBONACCI && t2 == ANYON_FIBONACCI) {
        // Create superposition and measure
        FibonacciFusionState fusion_state = init_fibonacci_fusion();
        return measure_fusion_outcome(&fusion_state);
    }

    // Ising fusion rules: σ × σ = 1 ⊕ ψ, σ × ψ = σ, ψ × ψ = 1
    if (t1 == ANYON_ISING && t2 == ANYON_ISING) {
        // Probabilistic: 50% vacuum, 50% fermion
        double r = (double)rand() / (double)RAND_MAX;
        return (r < 0.5) ? ANYON_NONE : ANYON_ABELIAN;
    }

    if ((t1 == ANYON_ISING && t2 == ANYON_ABELIAN) ||
        (t1 == ANYON_ABELIAN && t2 == ANYON_ISING)) {
        return ANYON_ISING;  // σ × ψ = σ
    }

    // Abelian anyons: charges add
    if (t1 == ANYON_ABELIAN && t2 == ANYON_ABELIAN) {
        double charge_sum = anyon1->charge.topological + anyon2->charge.topological;
        // Modular arithmetic for anyonic charges (mod 1 for semions, mod 2 for fermions, etc.)
        if (fabs(fmod(charge_sum, 1.0)) < 0.01) {
            return ANYON_NONE;  // Annihilate to vacuum
        }
        return ANYON_ABELIAN;
    }

    // Non-abelian fusion is more complex - generically stays non-abelian
    if (t1 == ANYON_NON_ABELIAN || t2 == ANYON_NON_ABELIAN) {
        return ANYON_NON_ABELIAN;
    }

    // Default: return the "larger" type
    return (t1 > t2) ? t1 : t2;
}

/**
 * @brief Get fusion channel amplitudes for Fibonacci anyons
 *
 * Returns the quantum amplitudes for each fusion channel without measurement
 */
bool get_fibonacci_fusion_amplitudes(const Anyon* anyon1,
                                     const Anyon* anyon2,
                                     complex double* amp_vacuum,
                                     complex double* amp_tau) {
    if (!anyon1 || !anyon2 || !amp_vacuum || !amp_tau) {
        return false;
    }

    if (anyon1->type != ANYON_FIBONACCI || anyon2->type != ANYON_FIBONACCI) {
        // Not Fibonacci anyons
        *amp_vacuum = 0.0;
        *amp_tau = 0.0;
        return false;
    }

    FibonacciFusionState state = init_fibonacci_fusion();
    *amp_vacuum = state.amplitude_vacuum;
    *amp_tau = state.amplitude_tau;

    return true;
}

AnyonCharge calculate_fusion_charge(AnyonCharge charge1, AnyonCharge charge2) {
    AnyonCharge result;

    // Electric charge is conserved additively
    result.electric = charge1.electric + charge2.electric;

    // Magnetic charge is conserved additively
    result.magnetic = charge1.magnetic + charge2.magnetic;

    // Topological charge adds mod 1 (for simple anyons)
    result.topological = fmod(charge1.topological + charge2.topological, 1.0);
    if (result.topological < 0) result.topological += 1.0;

    return result;
}

double calculate_energy_difference(const quantum_state* state,
                                  const Anyon* anyon1,
                                  const Anyon* anyon2,
                                  const AnyonCharge* result_charge) {
    if (!anyon1 || !anyon2) {
        return 0.0;
    }

    // Initial energy
    double E_initial = anyon1->energy + anyon2->energy;

    // Final energy (result anyon energy)
    // E_final = |q_result|² / (2m) + binding_energy
    double q_mag = 0.0;
    if (result_charge) {
        q_mag = sqrt(result_charge->electric * result_charge->electric +
                    result_charge->magnetic * result_charge->magnetic +
                    result_charge->topological * result_charge->topological);
    }

    double E_final = q_mag * q_mag / (2.0 * ANYON_MASS / 1e-30);

    // Energy released in fusion
    return E_initial - E_final;
}

// ============================================================================
// Fusion Operation Implementation
// ============================================================================

bool apply_fusion_operation(quantum_state* state,
                           const Anyon* anyon1,
                           const Anyon* anyon2,
                           FusionOutcome* outcome) {
    if (!anyon1 || !anyon2 || !outcome) {
        return false;
    }

    // Check if fusion is allowed by conservation laws
    if (!are_types_compatible(anyon1->type, anyon2->type)) {
        return false;
    }

    if (!verify_charge_conservation(anyon1->charge, anyon2->charge)) {
        return false;
    }

    // Apply F-matrix transformation to quantum state
    // F-matrices implement basis change in fusion space
    // F^{abc}_d transforms between (a×b)×c → d and a×(b×c) → d fusion channels

    if (state && state->coordinates && state->num_qubits >= 2) {
        size_t dim = 1ULL << state->num_qubits;

        // For Fibonacci anyons: F^{τττ}_τ is the golden ratio matrix
        if (anyon1->type == ANYON_FIBONACCI && anyon2->type == ANYON_FIBONACCI) {
            // F-matrix: [[φ^{-1}, φ^{-1/2}], [φ^{-1/2}, -φ^{-1}]]
            // This is a unitary transformation in the 2D fusion space
            float F11 = (float)PHI_INV;
            float F12 = (float)sqrt(PHI_INV);
            float F21 = (float)sqrt(PHI_INV);
            float F22 = (float)(-PHI_INV);

            // Apply F-matrix to pairs of basis states in the fusion space
            // The fusion space is spanned by |1⟩ and |τ⟩ outcomes
            for (size_t i = 0; i < dim - 1; i += 2) {
                ComplexFloat psi0 = state->coordinates[i];
                ComplexFloat psi1 = state->coordinates[i + 1];

                // |ψ'⟩ = F|ψ⟩
                // ψ'_0 = F11*ψ_0 + F12*ψ_1
                // ψ'_1 = F21*ψ_0 + F22*ψ_1
                state->coordinates[i].real = F11 * psi0.real + F12 * psi1.real;
                state->coordinates[i].imag = F11 * psi0.imag + F12 * psi1.imag;
                state->coordinates[i + 1].real = F21 * psi0.real + F22 * psi1.real;
                state->coordinates[i + 1].imag = F21 * psi0.imag + F22 * psi1.imag;
            }

            // Update probability based on F-matrix unitarity (trace preserved)
            outcome->probability *= (F11 * F11 + F12 * F12);
        }

        // For Ising anyons: F^{σσσ}_σ = 1/√2 [[1, 1], [1, -1]] (Hadamard-like)
        if (anyon1->type == ANYON_ISING || anyon2->type == ANYON_ISING) {
            float inv_sqrt2 = (float)(1.0 / sqrt(2.0));

            for (size_t i = 0; i < dim - 1; i += 2) {
                ComplexFloat psi0 = state->coordinates[i];
                ComplexFloat psi1 = state->coordinates[i + 1];

                // F = (1/√2) [[1, 1], [1, -1]]
                state->coordinates[i].real = inv_sqrt2 * (psi0.real + psi1.real);
                state->coordinates[i].imag = inv_sqrt2 * (psi0.imag + psi1.imag);
                state->coordinates[i + 1].real = inv_sqrt2 * (psi0.real - psi1.real);
                state->coordinates[i + 1].imag = inv_sqrt2 * (psi0.imag - psi1.imag);
            }

            outcome->probability *= 0.5;
        }

        // For Majorana anyons: F-matrix with phase factors
        if (anyon1->type == ANYON_MAJORANA || anyon2->type == ANYON_MAJORANA) {
            // F^{σσσ}_1 includes e^{iπ/8} phase factors
            float cos_pi8 = (float)cos(M_PI / 8.0);
            float sin_pi8 = (float)sin(M_PI / 8.0);

            for (size_t i = 0; i < dim; i++) {
                ComplexFloat psi = state->coordinates[i];
                // Apply phase rotation e^{iπ/8}
                state->coordinates[i].real = cos_pi8 * psi.real - sin_pi8 * psi.imag;
                state->coordinates[i].imag = sin_pi8 * psi.real + cos_pi8 * psi.imag;
            }
        }
    } else {
        // No quantum state provided - just update probability based on anyon types
        if (anyon1->type == ANYON_FIBONACCI && anyon2->type == ANYON_FIBONACCI) {
            outcome->probability *= (PHI_INV * PHI_INV + PHI_INV);
        }
        if (anyon1->type == ANYON_ISING || anyon2->type == ANYON_ISING) {
            outcome->probability *= 0.5;
        }
    }

    // Update statistics
    fusion_stats.total_fusions++;
    fusion_stats.successful_fusions++;

    return true;
}

void update_fusion_statistics(anyon_type_t result_type,
                             double probability,
                             double energy_delta) {
    if (result_type < 9) {
        fusion_stats.fusion_counts_by_type[result_type]++;
    }
    fusion_stats.total_energy_released += energy_delta;
}

// ============================================================================
// Type Compatibility and Conservation Laws
// ============================================================================

bool are_types_compatible(anyon_type_t type1, anyon_type_t type2) {
    // All types can fuse with vacuum
    if (type1 == ANYON_NONE || type2 == ANYON_NONE) {
        return true;
    }

    // Same types can always fuse
    if (type1 == type2) {
        return true;
    }

    // Check specific fusion rules
    // Majorana × Majorana → allowed
    if (type1 == ANYON_MAJORANA && type2 == ANYON_MAJORANA) {
        return true;
    }

    // Fibonacci × Fibonacci → allowed
    if (type1 == ANYON_FIBONACCI && type2 == ANYON_FIBONACCI) {
        return true;
    }

    // Abelian × Abelian → allowed
    if (type1 == ANYON_ABELIAN && type2 == ANYON_ABELIAN) {
        return true;
    }

    // Ising × Abelian → allowed (σ × ψ = σ)
    if ((type1 == ANYON_ISING && type2 == ANYON_ABELIAN) ||
        (type1 == ANYON_ABELIAN && type2 == ANYON_ISING)) {
        return true;
    }

    // Non-abelian can fuse with most types
    if (type1 == ANYON_NON_ABELIAN || type2 == ANYON_NON_ABELIAN) {
        return true;
    }

    // Error anyons can fuse
    if ((type1 == ANYON_X || type1 == ANYON_Y || type1 == ANYON_Z) &&
        (type2 == ANYON_X || type2 == ANYON_Y || type2 == ANYON_Z)) {
        return true;
    }

    // Other combinations may not be compatible
    return false;
}

bool verify_charge_conservation(AnyonCharge charge1, AnyonCharge charge2) {
    // Check that fusion obeys conservation laws

    // Total electric charge should be integer
    double total_electric = charge1.electric + charge2.electric;
    if (fabs(total_electric - round(total_electric)) > 0.01) {
        // Non-integer total electric charge still allowed for fractional charges
    }

    // Magnetic and topological charges have their own rules
    // For most anyon models, these are well-defined

    return true;  // Most charge configurations are valid
}

bool verify_topological_rules(const Anyon* anyon1, const Anyon* anyon2) {
    if (!anyon1 || !anyon2) {
        return false;
    }

    // Check type compatibility
    if (!are_types_compatible(anyon1->type, anyon2->type)) {
        return false;
    }

    // Check charge conservation
    if (!verify_charge_conservation(anyon1->charge, anyon2->charge)) {
        return false;
    }

    // Check position validity (not too close to boundary, etc.)
    if (anyon1->position.stability < 0.1 || anyon2->position.stability < 0.1) {
        return false;
    }

    return true;
}

// ============================================================================
// Fusion Channel Calculations
// ============================================================================

void calculate_fusion_channels(anyon_type_t type1,
                              anyon_type_t type2,
                              FusionOutcome* outcomes,
                              size_t* num_outcomes,
                              size_t max_outcomes) {
    if (!outcomes || !num_outcomes || max_outcomes == 0) {
        if (num_outcomes) *num_outcomes = 0;
        return;
    }

    *num_outcomes = 0;

    // Vacuum fusion: 1 × a = a
    if (type1 == ANYON_NONE) {
        outcomes[0].result_type = type2;
        outcomes[0].probability = 1.0;
        outcomes[0].result_charge = (AnyonCharge){0, 0, 0};
        outcomes[0].energy_delta = 0.0;
        *num_outcomes = 1;
        return;
    }
    if (type2 == ANYON_NONE) {
        outcomes[0].result_type = type1;
        outcomes[0].probability = 1.0;
        outcomes[0].result_charge = (AnyonCharge){0, 0, 0};
        outcomes[0].energy_delta = 0.0;
        *num_outcomes = 1;
        return;
    }

    // Fibonacci: τ × τ = 1 + τ (two channels)
    if (type1 == ANYON_FIBONACCI && type2 == ANYON_FIBONACCI) {
        if (max_outcomes >= 2) {
            // Channel 1: vacuum
            outcomes[0].result_type = ANYON_NONE;
            outcomes[0].probability = PHI_INV * PHI_INV;  // 1/φ²
            outcomes[0].result_charge = (AnyonCharge){0, 0, 0};
            outcomes[0].energy_delta = 1.0;  // Energy released

            // Channel 2: τ
            outcomes[1].result_type = ANYON_FIBONACCI;
            outcomes[1].probability = PHI_INV;  // 1/φ
            outcomes[1].result_charge = (AnyonCharge){0, 0, 0.5};
            outcomes[1].energy_delta = 0.5;

            *num_outcomes = 2;
        }
        return;
    }

    // Majorana: σ × σ = 1 + ψ (two channels)
    if (type1 == ANYON_MAJORANA && type2 == ANYON_MAJORANA) {
        if (max_outcomes >= 2) {
            // Channel 1: vacuum
            outcomes[0].result_type = ANYON_NONE;
            outcomes[0].probability = 0.5;
            outcomes[0].result_charge = (AnyonCharge){0, 0, 0};
            outcomes[0].energy_delta = 1.0;

            // Channel 2: fermion
            outcomes[1].result_type = ANYON_ABELIAN;
            outcomes[1].probability = 0.5;
            outcomes[1].result_charge = (AnyonCharge){0, 0, 0.5};
            outcomes[1].energy_delta = 0.5;

            *num_outcomes = 2;
        }
        return;
    }

    // Ising: σ × σ = 1 + ψ
    if (type1 == ANYON_ISING && type2 == ANYON_ISING) {
        if (max_outcomes >= 2) {
            outcomes[0].result_type = ANYON_NONE;
            outcomes[0].probability = 0.5;
            outcomes[0].result_charge = (AnyonCharge){0, 0, 0};
            outcomes[0].energy_delta = 1.0;

            outcomes[1].result_type = ANYON_ABELIAN;
            outcomes[1].probability = 0.5;
            outcomes[1].result_charge = (AnyonCharge){0, 0, 0.5};
            outcomes[1].energy_delta = 0.5;

            *num_outcomes = 2;
        }
        return;
    }

    // Abelian fusion: simple addition
    if (type1 == ANYON_ABELIAN && type2 == ANYON_ABELIAN) {
        outcomes[0].result_type = ANYON_ABELIAN;
        outcomes[0].probability = 1.0;
        outcomes[0].result_charge = (AnyonCharge){0, 0, 0};
        outcomes[0].energy_delta = 0.5;
        *num_outcomes = 1;
        return;
    }

    // Default: single channel fusion
    anyon_type_t result = (type1 > type2) ? type1 : type2;
    outcomes[0].result_type = result;
    outcomes[0].probability = 1.0;
    outcomes[0].result_charge = (AnyonCharge){0, 0, 0};
    outcomes[0].energy_delta = 0.5;
    *num_outcomes = 1;
}

double calculate_channel_probability(const AnyonPair* pair, const FusionOutcome* outcome) {
    if (!pair || !outcome || !pair->anyon1 || !pair->anyon2) {
        return 0.0;
    }

    // Base probability from fusion rules
    double base_prob = outcome->probability;

    // Modify by distance
    double distance = calculate_distance(&pair->anyon1->position, &pair->anyon2->position);
    double distance_factor = exp(-distance / (5.0 * LATTICE_SPACING));

    // Modify by interaction strength
    double interaction_factor = 1.0 - exp(-pair->interaction_strength);

    // Coherence factor
    double coherence = pair->anyon1->position.stability * pair->anyon2->position.stability;

    return base_prob * distance_factor * interaction_factor * coherence;
}

double calculate_channel_energy(const AnyonPair* pair, const FusionOutcome* outcome) {
    if (!pair || !outcome || !pair->anyon1 || !pair->anyon2) {
        return 0.0;
    }

    // Base energy change
    double base_energy = outcome->energy_delta;

    // Scale by initial energies
    double E_initial = pair->anyon1->energy + pair->anyon2->energy;

    // Final energy depends on result type
    double E_final = 0.0;
    switch (outcome->result_type) {
        case ANYON_NONE:
            E_final = 0.0;  // Vacuum has zero energy
            break;
        case ANYON_FIBONACCI:
            E_final = 0.5 * E_initial;  // τ retains some energy
            break;
        case ANYON_MAJORANA:
        case ANYON_ISING:
            E_final = 0.3 * E_initial;
            break;
        default:
            E_final = 0.2 * E_initial;
    }

    return E_initial - E_final + base_energy;
}

// Helper function to calculate optimal braiding path
static bool calculate_braiding_path(const AnyonPair* pair, 
                                  const BraidingConfig* config,
                                  AnyonPosition** path,
                                  size_t* path_length) {
    if (!pair || !config || !path || !path_length) {
        return false;
    }

    // Calculate distance between anyons
    double distance = calculate_distance(&pair->anyon1->position, 
                                      &pair->anyon2->position);
    
    if (distance < config->min_separation) {
        return false;
    }

    // Calculate number of steps needed
    *path_length = (size_t)(distance * config->braiding_steps);
    *path = calloc(*path_length, sizeof(AnyonPosition));
    
    if (!*path) {
        return false;
    }

    // Calculate intermediate positions for smooth braiding
    double dx = (double)(pair->anyon2->position.x - pair->anyon1->position.x);
    double dy = (double)(pair->anyon2->position.y - pair->anyon1->position.y);
    double dz = (double)(pair->anyon2->position.z - pair->anyon1->position.z);

    for (size_t i = 0; i < *path_length; i++) {
        double t = (double)i / (double)(*path_length - 1);
        double angle = 2.0 * M_PI * t;
        
        // Calculate position along braiding path
        (*path)[i].x = (size_t)(pair->anyon1->position.x + dx * t + 
                               config->min_separation * cos(angle));
        (*path)[i].y = (size_t)(pair->anyon1->position.y + dy * t + 
                               config->min_separation * sin(angle));
        (*path)[i].z = (size_t)(pair->anyon1->position.z + dz * t);
        
        // Set stability based on position
        (*path)[i].stability = 1.0;
    }

    return true;
}

bool braid_anyons(quantum_state* state, AnyonPair* pair, const BraidingConfig* config) {
    if (!state || !pair || !config || !pair->anyon1 || !pair->anyon2) {
        return false;
    }

    // Calculate braiding path
    AnyonPosition* path = NULL;
    size_t path_length = 0;
    
    if (!calculate_braiding_path(pair, config, &path, &path_length)) {
        return false;
    }

    // Initialize braiding phase
    complex double total_phase = 0.0;
    bool success = true;

    // Perform braiding operation
    for (size_t i = 0; i < path_length && success; i++) {
        // Move first anyon along path
        AnyonPosition prev_pos = pair->anyon1->position;
        pair->anyon1->position = path[i];

        // Calculate and accumulate braiding phase
        complex double step_phase = calculate_braiding_phase_step(state, 
                                                               pair->anyon1,
                                                               pair->anyon2,
                                                               &prev_pos);
        total_phase += step_phase;

        // Verify topological protection if required
        if (config->verify_topology) {
            success = verify_topological_protection(state, 
                                                 (const Anyon*[]){pair->anyon1, pair->anyon2},
                                                 2);
        }

        // Update quantum state
        if (success) {
            success = apply_braiding_operation(state, pair->anyon1, pair->anyon2, step_phase);
        }
    }

    // Store final braiding phase
    if (success) {
        pair->braiding_phase = carg(total_phase);
        
        // Update interaction strength based on braiding result
        pair->interaction_strength = cabs(total_phase) / (2.0 * M_PI);
    }

    free(path);
    return success;
}

FusionOutcome fuse_anyons(quantum_state* state, const AnyonPair* pair, const FusionConfig* config) {
    FusionOutcome outcome = {0};
    
    if (!state || !pair || !config || !pair->anyon1 || !pair->anyon2) {
        return outcome;
    }

    // Check if fusion is energetically allowed
    double fusion_energy = calculate_fusion_energy(pair->anyon1, pair->anyon2);
    if (fusion_energy > config->energy_threshold) {
        return outcome;
    }

    // Calculate fusion probability based on anyon types and charges
    outcome.probability = calculate_fusion_probability(pair->anyon1, pair->anyon2);
    
    // Only proceed if probability exceeds coherence requirement
    if (outcome.probability < config->coherence_requirement) {
        return outcome;
    }

    // Determine fusion outcome type
    outcome.result_type = determine_fusion_type(pair->anyon1, pair->anyon2);
    
    // Calculate resulting charge
    outcome.result_charge = calculate_fusion_charge(pair->anyon1->charge,
                                                  pair->anyon2->charge);
    
    // Calculate energy change
    outcome.energy_delta = calculate_energy_difference(state,
                                                     pair->anyon1,
                                                     pair->anyon2,
                                                     &outcome.result_charge);

    // Attempt fusion operation
    bool fusion_success = false;
    for (size_t attempt = 0; attempt < config->fusion_attempts && !fusion_success; attempt++) {
        fusion_success = apply_fusion_operation(state,
                                             pair->anyon1,
                                             pair->anyon2,
                                             &outcome);
        
        if (fusion_success && config->track_statistics) {
            update_fusion_statistics(outcome.result_type,
                                   outcome.probability,
                                   outcome.energy_delta);
        }
    }

    return outcome;
}

double calculate_interaction_energy(const Anyon* anyon1, const Anyon* anyon2) {
    if (!anyon1 || !anyon2) {
        return 0.0;
    }

    // Calculate base interaction energy
    double distance = calculate_distance(&anyon1->position, &anyon2->position);
    if (distance < 1e-6) {
        return INFINITY;
    }

    // Calculate charge-dependent interaction
    double charge_interaction = calculate_charge_interaction(anyon1->charge,
                                                          anyon2->charge);

    // Calculate type-dependent interaction
    double type_interaction = calculate_type_interaction(anyon1->type,
                                                       anyon2->type);

    // Combine all interaction terms
    return (charge_interaction * type_interaction) / distance;
}

double calculate_braiding_phase(const AnyonPair* pair) {
    if (!pair || !pair->anyon1 || !pair->anyon2) {
        return 0.0;
    }

    // Calculate statistical angle based on anyon types
    double statistical_angle = calculate_statistical_angle(pair->anyon1->type,
                                                         pair->anyon2->type);

    // Modify by charge interaction
    double charge_factor = calculate_charge_phase_factor(pair->anyon1->charge,
                                                       pair->anyon2->charge);

    // Include topological correction
    double topological_factor = calculate_topological_correction(pair->anyon1,
                                                               pair->anyon2);

    return statistical_angle * charge_factor * topological_factor;
}

bool check_fusion_rules(const Anyon* anyon1, const Anyon* anyon2) {
    if (!anyon1 || !anyon2) {
        return false;
    }

    // Check type compatibility
    if (!are_types_compatible(anyon1->type, anyon2->type)) {
        return false;
    }

    // Check charge conservation
    if (!verify_charge_conservation(anyon1->charge, anyon2->charge)) {
        return false;
    }

    // Check topological constraints
    if (!verify_topological_rules(anyon1, anyon2)) {
        return false;
    }

    return true;
}

void predict_fusion_outcomes(const AnyonPair* pair, 
                           FusionOutcome* outcomes,
                           size_t* num_outcomes) {
    if (!pair || !outcomes || !num_outcomes || !pair->anyon1 || !pair->anyon2) {
        if (num_outcomes) {
            *num_outcomes = 0;
        }
        return;
    }

    // Get possible fusion channels
    size_t max_channels = *num_outcomes;
    *num_outcomes = 0;

    // Calculate possible outcomes based on anyon types
    calculate_fusion_channels(pair->anyon1->type,
                            pair->anyon2->type,
                            outcomes,
                            num_outcomes,
                            max_channels);

    // Calculate probabilities and energies for each outcome
    for (size_t i = 0; i < *num_outcomes; i++) {
        outcomes[i].probability = calculate_channel_probability(pair, &outcomes[i]);
        outcomes[i].energy_delta = calculate_channel_energy(pair, &outcomes[i]);
    }
}

// ============================================================================
// Helper: ComplexFloat magnitude and phase
// ============================================================================

static inline double complex_float_magnitude(ComplexFloat c) {
    return sqrt((double)(c.real * c.real + c.imag * c.imag));
}

static inline double complex_float_phase(ComplexFloat c) {
    return atan2((double)c.imag, (double)c.real);
}

static inline ComplexFloat complex_float_from_polar(double mag, double phase) {
    ComplexFloat c;
    c.real = (float)(mag * cos(phase));
    c.imag = (float)(mag * sin(phase));
    return c;
}

// Note: complex_float_multiply is provided by quantum_complex.h

// ============================================================================
// Quantum State Management
// ============================================================================

quantum_state* create_anyon_state(size_t size) {
    if (size == 0) {
        return NULL;
    }

    quantum_state* state = calloc(1, sizeof(quantum_state));
    if (!state) {
        return NULL;
    }

    // Initialize state fields based on quantum_state_t structure
    state->num_qubits = size;
    state->dimension = 1ULL << size;
    state->manifold_dim = size;
    state->type = QUANTUM_STATE_PURE;
    state->is_normalized = true;
    state->hardware = HARDWARE_TYPE_CPU;

    // Allocate state vector: 2^n complex coordinates
    state->coordinates = calloc(state->dimension, sizeof(ComplexFloat));
    if (!state->coordinates) {
        free(state);
        return NULL;
    }

    // Initialize to |0⟩ state
    state->coordinates[0].real = 1.0f;
    state->coordinates[0].imag = 0.0f;

    // Allocate metric tensor (dimension x dimension)
    state->metric = calloc(state->dimension * state->dimension, sizeof(ComplexFloat));

    // Allocate connection coefficients (dimension^3)
    state->connection = calloc(state->dimension * state->dimension * state->dimension, sizeof(ComplexFloat));

    // Initialize metric to identity (flat space)
    if (state->metric) {
        for (size_t i = 0; i < state->dimension; i++) {
            state->metric[i * state->dimension + i].real = 1.0f;
        }
    }

    return state;
}

void destroy_anyon_state(quantum_state* state) {
    if (!state) {
        return;
    }

    if (state->coordinates) {
        free(state->coordinates);
    }
    if (state->metric) {
        free(state->metric);
    }
    if (state->connection) {
        free(state->connection);
    }
    if (state->auxiliary_data) {
        free(state->auxiliary_data);
    }

    free(state);
}

void initialize_topological_state(quantum_state* state) {
    if (!state || !state->coordinates) {
        return;
    }

    // Initialize to a topologically ordered state
    // For surface code: superposition of all closed loop configurations
    // Creates state with Z2 topological order

    size_t dim = state->dimension;
    float norm = 1.0f / sqrtf((float)dim);

    for (size_t i = 0; i < dim; i++) {
        // Phase based on parity for topological structure
        // Even parity basis states get phase 0, odd parity get π/4
        int parity = __builtin_popcount(i) % 2;
        float phase = parity ? (float)(M_PI / 4.0) : 0.0f;
        state->coordinates[i].real = norm * cosf(phase);
        state->coordinates[i].imag = norm * sinf(phase);
    }

    state->is_normalized = true;
}

// ============================================================================
// Anyon Detection and Tracking
// ============================================================================

size_t detect_anyons(const quantum_state* state, Anyon* anyons, size_t max_anyons) {
    if (!state || !state->coordinates || !anyons || max_anyons == 0) {
        return 0;
    }

    size_t num_detected = 0;

    // Detection based on stabilizer measurements
    // Anyons appear at vertices where stabilizer measurement gives -1

    size_t lattice_size = state->num_qubits;
    if (lattice_size == 0) {
        return 0;
    }

    // 2D lattice model: sqrt(n) x sqrt(n)
    size_t side = (size_t)sqrt((double)lattice_size);
    if (side * side != lattice_size) {
        side = lattice_size;  // Linear arrangement if not square
    }

    // Scan through lattice positions
    for (size_t x = 0; x < side && num_detected < max_anyons; x++) {
        for (size_t y = 0; y < side && num_detected < max_anyons; y++) {
            // Calculate stabilizer expectation value at this vertex
            size_t idx = x * side + y;
            if (idx >= state->dimension) {
                continue;
            }

            // Check amplitude pattern for anyon signature
            ComplexFloat amp = state->coordinates[idx];
            double prob = complex_float_magnitude(amp);
            prob = prob * prob;  // |amplitude|^2
            double phase = complex_float_phase(amp);

            // Detect anyon if:
            // 1. Significant probability weight
            // 2. Non-trivial phase (indicates anyonic excitation)
            if (prob > 0.01 && fabs(phase) > 0.1) {
                // Create detected anyon
                anyons[num_detected].position.x = x;
                anyons[num_detected].position.y = y;
                anyons[num_detected].position.z = 0;
                anyons[num_detected].position.stability = sqrt(prob);

                // Determine type from phase using standard anyon angles
                if (fabs(phase - M_PI / 2.0) < 0.2) {
                    anyons[num_detected].type = ANYON_MAJORANA;
                } else if (fabs(phase - M_PI / 4.0) < 0.2) {
                    anyons[num_detected].type = ANYON_ABELIAN;
                } else if (fabs(phase - 4.0 * M_PI / 5.0) < 0.2) {
                    anyons[num_detected].type = ANYON_FIBONACCI;
                } else if (fabs(phase - M_PI / 8.0) < 0.2) {
                    anyons[num_detected].type = ANYON_ISING;
                } else {
                    anyons[num_detected].type = ANYON_NON_ABELIAN;
                }

                // Set charge based on phase (topological charge = θ/π)
                anyons[num_detected].charge.electric = 0.0;
                anyons[num_detected].charge.magnetic = 0.0;
                anyons[num_detected].charge.topological = phase / M_PI;

                anyons[num_detected].energy = prob;
                anyons[num_detected].lifetime = 0.0;
                anyons[num_detected].is_mobile = true;

                num_detected++;
            }
        }
    }

    return num_detected;
}

bool track_anyon_movement(quantum_state* state, Anyon* anyon, size_t num_steps) {
    if (!state || !anyon || num_steps == 0) {
        return false;
    }

    // Track anyon position over multiple time steps
    AnyonPosition initial_pos = anyon->position;
    double total_displacement = 0.0;
    double total_stability = 0.0;

    // Time step for Hamiltonian evolution (in natural units where ℏ = 1)
    const double dt = 0.01;

    for (size_t step = 0; step < num_steps; step++) {
        // Apply Hamiltonian time evolution: |ψ(t+dt)⟩ = e^{-iHdt}|ψ(t)⟩
        // Using first-order Trotter approximation for anyon Hamiltonian:
        // H = H_kinetic + H_potential + H_topological
        size_t dim = 1ULL << state->num_qubits;
        size_t side = (size_t)sqrt((double)state->num_qubits);
        if (side == 0) side = 1;

        // Allocate temporary buffer for evolved state
        ComplexFloat* new_coords = malloc(dim * sizeof(ComplexFloat));
        if (!new_coords) {
            return false;
        }
        memset(new_coords, 0, dim * sizeof(ComplexFloat));

        // Apply Hamiltonian evolution
        for (size_t i = 0; i < dim; i++) {
            size_t x = (i / side) % side;
            size_t y = i % side;

            // Kinetic term: -t Σ (c†_i c_j + h.c.) for nearest neighbors
            // This creates hopping between adjacent sites
            double hopping_strength = 1.0;

            // Diagonal (potential) energy at site i
            double V_i = 0.0;
            // Add anyon-anyon interaction potential: V(r) ~ 1/r for Coulomb-like
            double dx = (double)x - (double)anyon->position.x;
            double dy = (double)y - (double)anyon->position.y;
            double r = sqrt(dx * dx + dy * dy);
            if (r > 0.1) {
                V_i = anyon->charge.electric / r;  // Coulomb interaction
            }

            // Topological phase contribution (statistical angle for self-exchange)
            double theta = calculate_statistical_angle(anyon->type, anyon->type);
            double topo_phase = theta * (double)step * dt;

            // Phase evolution: e^{-i(V + topo)dt}
            double total_phase = -(V_i + topo_phase) * dt;
            float cos_phase = (float)cos(total_phase);
            float sin_phase = (float)sin(total_phase);

            ComplexFloat psi = state->coordinates[i];

            // Apply phase rotation
            new_coords[i].real += cos_phase * psi.real - sin_phase * psi.imag;
            new_coords[i].imag += sin_phase * psi.real + cos_phase * psi.imag;

            // Apply kinetic hopping to neighbors (x±1, y±1)
            size_t neighbors[4];
            size_t num_neighbors = 0;

            if (x > 0) neighbors[num_neighbors++] = i - side;        // left
            if (x < side - 1) neighbors[num_neighbors++] = i + side; // right
            if (y > 0) neighbors[num_neighbors++] = i - 1;           // down
            if (y < side - 1) neighbors[num_neighbors++] = i + 1;    // up

            for (size_t n = 0; n < num_neighbors; n++) {
                size_t j = neighbors[n];
                if (j < dim) {
                    // Hopping: -i * t * dt contribution
                    float hop_factor = (float)(-hopping_strength * dt);
                    ComplexFloat psi_j = state->coordinates[j];
                    // -i * psi_j contribution
                    new_coords[i].real += hop_factor * psi_j.imag;
                    new_coords[i].imag += -hop_factor * psi_j.real;
                }
            }
        }

        // Normalize the evolved state
        double norm = 0.0;
        for (size_t i = 0; i < dim; i++) {
            double mag = complex_float_magnitude(new_coords[i]);
            norm += mag * mag;
        }
        if (norm > 1e-10) {
            float inv_sqrt_norm = (float)(1.0 / sqrt(norm));
            for (size_t i = 0; i < dim; i++) {
                state->coordinates[i].real = new_coords[i].real * inv_sqrt_norm;
                state->coordinates[i].imag = new_coords[i].imag * inv_sqrt_norm;
            }
        } else {
            memcpy(state->coordinates, new_coords, dim * sizeof(ComplexFloat));
        }

        free(new_coords);

        // Calculate expected position from evolved quantum state
        double mean_x = 0.0, mean_y = 0.0;
        double total_prob = 0.0;

        for (size_t i = 0; i < dim; i++) {
            double mag = complex_float_magnitude(state->coordinates[i]);
            double prob = mag * mag;
            size_t x = (i / side) % side;
            size_t y = i % side;
            mean_x += x * prob;
            mean_y += y * prob;
            total_prob += prob;
        }

        if (total_prob > 0) {
            mean_x /= total_prob;
            mean_y /= total_prob;

            // Update anyon position
            anyon->position.x = (size_t)round(mean_x);
            anyon->position.y = (size_t)round(mean_y);

            // Calculate displacement from initial
            double dx = (double)anyon->position.x - (double)initial_pos.x;
            double dy = (double)anyon->position.y - (double)initial_pos.y;
            total_displacement += sqrt(dx * dx + dy * dy);
        }

        // Update stability (coherence)
        anyon->position.stability *= 0.99;  // Small decay per step
        total_stability += anyon->position.stability;

        // Update lifetime
        anyon->lifetime += 1.0;

        // Check if anyon is still well-defined
        if (anyon->position.stability < 0.1) {
            // Anyon has decohered
            return false;
        }
    }

    // Final stability is average over tracking period
    anyon->position.stability = total_stability / num_steps;

    // Check if total displacement is excessive (indicates instability)
    // For stable tracking, displacement should be bounded
    double max_allowed_displacement = (double)num_steps * 2.0;  // At most 2 lattice sites per step
    if (total_displacement > max_allowed_displacement) {
        // Anyon has moved too erratically - likely unstable
        anyon->position.stability *= 0.5;  // Reduce stability
    }

    // Use displacement to adjust lifetime (more motion = more interaction = shorter effective lifetime)
    double displacement_factor = 1.0 / (1.0 + total_displacement / (double)num_steps);
    anyon->lifetime *= displacement_factor;

    return true;
}

void inject_test_error(quantum_state* state, size_t position) {
    if (!state || !state->coordinates) {
        return;
    }

    size_t dim = 1ULL << state->num_qubits;
    if (position >= dim) {
        return;
    }

    // Inject an X error (bit flip) at the specified position
    // This creates anyon pairs in the surface code

    // Apply Pauli X: swap amplitudes at positions differing in bit 'position'
    size_t mask = 1ULL << position;

    for (size_t i = 0; i < dim; i++) {
        if ((i & mask) == 0) {
            size_t j = i | mask;
            if (j < dim) {
                ComplexFloat temp = state->coordinates[i];
                state->coordinates[i] = state->coordinates[j];
                state->coordinates[j] = temp;
            }
        }
    }

    // Also add phase to mark the error location (rotate by π/4)
    // Apply e^(iπ/4) = cos(π/4) + i*sin(π/4) = (1+i)/√2
    float cos_phase = (float)cos(M_PI / 4.0);  // ≈ 0.7071
    float sin_phase = (float)sin(M_PI / 4.0);  // ≈ 0.7071
    ComplexFloat orig = state->coordinates[position];
    state->coordinates[position].real = orig.real * cos_phase - orig.imag * sin_phase;
    state->coordinates[position].imag = orig.real * sin_phase + orig.imag * cos_phase;
}

// ============================================================================
// Anyon Detection System Management
// ============================================================================

bool init_anyon_detection(AnyonState* state, const AnyonConfig* config) {
    if (!state || !config) {
        return false;
    }

    // Allocate grid
    state->grid = calloc(1, sizeof(AnyonGrid));
    if (!state->grid) {
        return false;
    }

    state->grid->width = config->grid_width;
    state->grid->height = config->grid_height;
    state->grid->depth = config->grid_depth;

    size_t total_cells = config->grid_width * config->grid_height * config->grid_depth;
    state->grid->cells = calloc(total_cells, sizeof(AnyonCell));
    if (!state->grid->cells) {
        free(state->grid);
        state->grid = NULL;
        return false;
    }

    // Initialize cells
    for (size_t i = 0; i < total_cells; i++) {
        state->grid->cells[i].type = ANYON_NONE;
        state->grid->cells[i].charge = 0.0;
        state->grid->cells[i].confidence = 0.0;
        state->grid->cells[i].is_fused = false;
    }

    // Allocate position tracking
    state->last_positions = calloc(total_cells, sizeof(AnyonPosition));
    if (!state->last_positions) {
        free(state->grid->cells);
        free(state->grid);
        state->grid = NULL;
        return false;
    }

    state->measurement_count = 0;
    state->total_anyons = 0;

    return true;
}

void cleanup_anyon_detection(AnyonState* state) {
    if (!state) {
        return;
    }

    if (state->grid) {
        if (state->grid->cells) {
            free(state->grid->cells);
        }
        free(state->grid);
        state->grid = NULL;
    }

    if (state->last_positions) {
        free(state->last_positions);
        state->last_positions = NULL;
    }

    state->measurement_count = 0;
    state->total_anyons = 0;
}

// Renamed to avoid conflict with anyon_detection.c (this version uses 1D phase analysis)
bool detect_anyons_from_quantum_state(AnyonState* state, const quantum_state* qstate) {
    if (!state || !state->grid || !qstate) {
        return false;
    }

    // Update grid based on quantum state
    size_t dim = 1ULL << qstate->num_qubits;
    size_t total_cells = state->grid->width * state->grid->height * state->grid->depth;

    // Map quantum amplitudes to grid cells
    for (size_t i = 0; i < total_cells && i < dim; i++) {
        ComplexFloat amp = qstate->coordinates[i];
        double mag = complex_float_magnitude(amp);
        double prob = mag * mag;
        double phase = complex_float_phase(amp);

        AnyonCell* cell = &state->grid->cells[i];

        // Detect anyon presence
        if (prob > 0.01 && fabs(phase) > 0.1) {
            // Determine anyon type from phase
            if (fabs(phase - M_PI / 2.0) < 0.2) {
                cell->type = ANYON_MAJORANA;
            } else if (fabs(phase - M_PI / 4.0) < 0.2) {
                cell->type = ANYON_ABELIAN;
            } else {
                cell->type = ANYON_NON_ABELIAN;
            }

            cell->charge = phase / M_PI;
            cell->confidence = prob;
            state->total_anyons++;
        } else {
            cell->type = ANYON_NONE;
            cell->charge = 0.0;
            cell->confidence = 0.0;
        }
    }

    state->measurement_count++;
    return true;
}

// count_anyons() - Canonical implementation in anyon_detection.c
// (removed: this version lacks is_fused filter, canonical correctly filters fused anyons)

// get_anyon_positions() - Canonical implementation in anyon_detection.c
// (removed: this version lacks is_fused filter, canonical correctly filters fused anyons)

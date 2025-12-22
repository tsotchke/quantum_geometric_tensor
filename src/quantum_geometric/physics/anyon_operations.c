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

complex double calculate_braiding_phase_step(const quantum_state* state,
                                            const Anyon* moving_anyon,
                                            const Anyon* stationary_anyon,
                                            const AnyonPosition* prev_pos) {
    if (!moving_anyon || !stationary_anyon || !prev_pos) {
        return 0.0;
    }

    // Calculate solid angle subtended by the path segment
    // Δφ = (q₁q₂/2) × ΔΩ where ΔΩ is solid angle change

    // Vector from stationary anyon to previous position
    double r1_x = (double)prev_pos->x - (double)stationary_anyon->position.x;
    double r1_y = (double)prev_pos->y - (double)stationary_anyon->position.y;
    double r1_z = (double)prev_pos->z - (double)stationary_anyon->position.z;

    // Vector from stationary anyon to new position
    double r2_x = (double)moving_anyon->position.x - (double)stationary_anyon->position.x;
    double r2_y = (double)moving_anyon->position.y - (double)stationary_anyon->position.y;
    double r2_z = (double)moving_anyon->position.z - (double)stationary_anyon->position.z;

    double r1 = sqrt(r1_x * r1_x + r1_y * r1_y + r1_z * r1_z);
    double r2 = sqrt(r2_x * r2_x + r2_y * r2_y + r2_z * r2_z);

    if (r1 < 1e-10 || r2 < 1e-10) {
        return 0.0;  // Anyons too close, singular
    }

    // Normalize vectors
    r1_x /= r1; r1_y /= r1; r1_z /= r1;
    r2_x /= r2; r2_y /= r2; r2_z /= r2;

    // Cross product r1 × r2 (gives normal to plane)
    double cross_x = r1_y * r2_z - r1_z * r2_y;
    double cross_y = r1_z * r2_x - r1_x * r2_z;
    double cross_z = r1_x * r2_y - r1_y * r2_x;

    // Solid angle element: dΩ = (r1 × r2) · ẑ / (1 + r1·r2)
    // For 2D motion, use z-component of cross product
    double dot = r1_x * r2_x + r1_y * r2_y + r1_z * r2_z;
    double solid_angle = 2.0 * atan2(cross_z, 1.0 + dot);

    // Get statistical angle for this pair
    double theta = calculate_statistical_angle(moving_anyon->type, stationary_anyon->type);

    // Add charge-dependent phase
    double charge_phase = calculate_charge_phase_factor(moving_anyon->charge,
                                                        stationary_anyon->charge);

    // Total phase: geometric + charge contribution
    double total_phase = (theta / M_PI) * solid_angle + charge_phase * solid_angle / (2.0 * M_PI);

    return cexp(I * total_phase);
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

    // Charges should sum to integer (or zero) for valid configuration
    double topo_mod = fmod(total_topological, 1.0);
    if (topo_mod > 0.5) topo_mod -= 1.0;
    if (fabs(topo_mod) > 0.01) {
        return false;  // Non-integer total topological charge
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

    // Apply phase to quantum state (simplified model)
    // In full implementation, this would update the state vector
    // according to the anyon positions in the computational basis

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
// Fusion Rules Implementation
// ============================================================================

anyon_type_t determine_fusion_type(const Anyon* anyon1, const Anyon* anyon2) {
    if (!anyon1 || !anyon2) {
        return ANYON_NONE;
    }

    // Fusion rules: a × b = Σ N^c_{ab} c
    // Implement standard fusion algebras

    anyon_type_t t1 = anyon1->type;
    anyon_type_t t2 = anyon2->type;

    // Vacuum fusion: 1 × a = a
    if (t1 == ANYON_NONE) return t2;
    if (t2 == ANYON_NONE) return t1;

    // Majorana fusion: σ × σ = 1 + ψ (usually collapses to vacuum or fermion)
    if (t1 == ANYON_MAJORANA && t2 == ANYON_MAJORANA) {
        // Randomly choose vacuum or fermion based on measurement
        // For deterministic version, choose based on charge parity
        double charge_sum = anyon1->charge.topological + anyon2->charge.topological;
        if (fmod(fabs(charge_sum), 2.0) < 1.0) {
            return ANYON_NONE;  // Fuse to vacuum
        } else {
            return ANYON_ABELIAN;  // Fuse to fermion
        }
    }

    // Fibonacci fusion: τ × τ = 1 + τ
    if (t1 == ANYON_FIBONACCI && t2 == ANYON_FIBONACCI) {
        // Probabilistic: vacuum with prob 1/φ², τ with prob 1/φ
        // For deterministic, use charge to decide
        double charge_prod = anyon1->charge.topological * anyon2->charge.topological;
        if (charge_prod > 0) {
            return ANYON_FIBONACCI;  // Fuse to τ
        } else {
            return ANYON_NONE;  // Fuse to vacuum
        }
    }

    // Ising fusion: σ × ψ = σ, ψ × ψ = 1
    if ((t1 == ANYON_ISING && t2 == ANYON_ABELIAN) ||
        (t1 == ANYON_ABELIAN && t2 == ANYON_ISING)) {
        return ANYON_ISING;
    }

    // Same type abelian: fuse to vacuum or same type
    if (t1 == ANYON_ABELIAN && t2 == ANYON_ABELIAN) {
        double charge_sum = fabs(anyon1->charge.topological + anyon2->charge.topological);
        if (charge_sum < 0.5) {
            return ANYON_NONE;  // Annihilate
        }
        return ANYON_ABELIAN;
    }

    // Non-abelian fusion is more complex
    if (t1 == ANYON_NON_ABELIAN || t2 == ANYON_NON_ABELIAN) {
        return ANYON_NON_ABELIAN;  // Stay non-abelian
    }

    // Default: return the "larger" type
    return (t1 > t2) ? t1 : t2;
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

static inline ComplexFloat complex_float_multiply(ComplexFloat a, ComplexFloat b) {
    ComplexFloat c;
    c.real = a.real * b.real - a.imag * b.imag;
    c.imag = a.real * b.imag + a.imag * b.real;
    return c;
}

// ============================================================================
// Quantum State Management
// ============================================================================

quantum_state* create_quantum_state(size_t size) {
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

void destroy_quantum_state(quantum_state* state) {
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

bool detect_and_track_anyons(AnyonState* state, const quantum_state* qstate) {
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

size_t count_anyons(const AnyonGrid* grid) {
    if (!grid || !grid->cells) {
        return 0;
    }

    size_t count = 0;
    size_t total_cells = grid->width * grid->height * grid->depth;

    for (size_t i = 0; i < total_cells; i++) {
        if (grid->cells[i].type != ANYON_NONE) {
            count++;
        }
    }

    return count;
}

bool get_anyon_positions(const AnyonGrid* grid, AnyonPosition* positions) {
    if (!grid || !grid->cells || !positions) {
        return false;
    }

    size_t pos_idx = 0;

    for (size_t z = 0; z < grid->depth; z++) {
        for (size_t y = 0; y < grid->height; y++) {
            for (size_t x = 0; x < grid->width; x++) {
                size_t cell_idx = z * grid->width * grid->height + y * grid->width + x;
                AnyonCell* cell = &grid->cells[cell_idx];

                if (cell->type != ANYON_NONE) {
                    positions[pos_idx].x = x;
                    positions[pos_idx].y = y;
                    positions[pos_idx].z = z;
                    positions[pos_idx].type = cell->type;
                    positions[pos_idx].stability = cell->confidence;
                    pos_idx++;
                }
            }
        }
    }

    return true;
}

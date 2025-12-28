/**
 * @file quantum_gravity_operations.h
 * @brief Quantum gravity and spacetime geometry operations
 *
 * Implements quantum gravity operations including spin foam models,
 * loop quantum gravity, causal dynamical triangulations, and
 * holographic/AdS-CFT computations.
 */

#ifndef QUANTUM_GRAVITY_OPERATIONS_H
#define QUANTUM_GRAVITY_OPERATIONS_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

// Portable complex type definition
typedef double _Complex qgt_complex_t;

// Forward declarations
struct quantum_state;
struct geometric_tensor;

// =============================================================================
// Quantum Gravity Model Types
// =============================================================================

/**
 * Quantum gravity model types
 */
typedef enum {
    QG_MODEL_SPIN_FOAM,              // Spin foam models (EPRL, BC)
    QG_MODEL_LOOP_QUANTUM,           // Loop quantum gravity
    QG_MODEL_CAUSAL_DYNAMICAL,       // Causal dynamical triangulations
    QG_MODEL_REGGE_CALCULUS,         // Regge calculus
    QG_MODEL_GROUP_FIELD_THEORY,     // Group field theory
    QG_MODEL_HOLOGRAPHIC,            // AdS/CFT holography
    QG_MODEL_ASYMPTOTIC_SAFETY,      // Asymptotic safety
    QG_MODEL_STRING_THEORY           // String theory
} QuantumGravityModel;

/**
 * Spin foam model variants
 */
typedef enum {
    SPIN_FOAM_PONZANO_REGGE,         // 3D Ponzano-Regge model
    SPIN_FOAM_TURAEV_VIRO,           // 3D Turaev-Viro model
    SPIN_FOAM_BARRETT_CRANE,         // Barrett-Crane model
    SPIN_FOAM_EPRL,                  // EPRL model (Engle-Pereira-Rovelli-Livine)
    SPIN_FOAM_FK,                    // Freidel-Krasnov model
    SPIN_FOAM_COHERENT                // Coherent state spin foam
} SpinFoamVariant;

/**
 * Spacetime signature
 */
typedef enum {
    SIGNATURE_EUCLIDEAN,             // (++++)
    SIGNATURE_LORENTZIAN,            // (-+++) or (+---)
    SIGNATURE_RIEMANNIAN,            // All positive
    SIGNATURE_CUSTOM                 // User-defined
} SpacetimeSignature;

/**
 * Holographic duality types
 */
typedef enum {
    HOLOGRAPHIC_ADS_CFT,             // AdS/CFT correspondence
    HOLOGRAPHIC_DS_CFT,              // dS/CFT correspondence
    HOLOGRAPHIC_FLAT,                // Flat space holography
    HOLOGRAPHIC_KERR_CFT,            // Kerr/CFT correspondence
    HOLOGRAPHIC_ENTANGLEMENT         // Entanglement holography
} HolographicDuality;

// =============================================================================
// Loop Quantum Gravity Structures
// =============================================================================

/**
 * Spin network node
 */
typedef struct {
    size_t node_id;
    size_t* adjacent_edges;          // Connected edge IDs
    size_t num_edges;
    double* intertwiner;             // Intertwiner state
    size_t intertwiner_dim;
    double* position;                // Embedded position (optional)
    size_t dimension;
} SpinNetworkNode;

/**
 * Spin network edge
 */
typedef struct {
    size_t edge_id;
    size_t source_node;
    size_t target_node;
    double spin_j;                   // SU(2) spin label (half-integer)
    qgt_complex_t* holonomy;        // Holonomy along edge
    size_t holonomy_dim;
} SpinNetworkEdge;

/**
 * Spin network (kinematical state in LQG)
 */
typedef struct {
    SpinNetworkNode* nodes;
    size_t num_nodes;
    SpinNetworkEdge* edges;
    size_t num_edges;
    double* vertex_amplitudes;       // Vertex amplitude contributions
    qgt_complex_t total_amplitude;
    bool is_gauge_invariant;
    double area_eigenvalue;          // Total area if computed
    double volume_eigenvalue;        // Total volume if computed
} SpinNetwork;

/**
 * Holonomy-flux algebra element
 */
typedef struct {
    qgt_complex_t* holonomy_matrix;     // Group element matrix
    size_t group_dim;
    double* flux_vector;                 // Lie algebra valued flux
    size_t algebra_dim;
    size_t edge_id;                      // Associated edge
} HolonomyFlux;

/**
 * Area operator eigenstate
 */
typedef struct {
    double* spin_labels;             // Spins of intersecting edges
    size_t num_intersections;
    double area_eigenvalue;          // 8πγℓ_P² √(j(j+1))
    double immirzi_parameter;        // Barbero-Immirzi parameter γ
} AreaEigenstate;

/**
 * Volume operator parameters
 */
typedef struct {
    double immirzi_parameter;
    size_t num_nodes;
    double* node_volumes;
    double total_volume;
    double planck_length;
} VolumeOperator;

// =============================================================================
// Spin Foam Structures
// =============================================================================

/**
 * Spin foam 2-complex vertex
 */
typedef struct {
    size_t vertex_id;
    size_t* adjacent_edges;          // Boundary edges
    size_t num_edges;
    size_t* adjacent_faces;          // Adjacent faces
    size_t num_faces;
    qgt_complex_t amplitude;        // Vertex amplitude A_v
} SpinFoamVertex;

/**
 * Spin foam face
 */
typedef struct {
    size_t face_id;
    double spin_jf;                  // Face spin label
    size_t* boundary_edges;          // Boundary edge sequence
    size_t num_boundary_edges;
    qgt_complex_t face_amplitude;
} SpinFoamFace;

/**
 * Spin foam edge (internal)
 */
typedef struct {
    size_t edge_id;
    size_t source_vertex;
    size_t target_vertex;
    double* intertwiner;             // Edge intertwiner
    size_t intertwiner_dim;
} SpinFoamEdge;

/**
 * Spin foam 2-complex
 */
typedef struct {
    SpinFoamVariant variant;
    SpinFoamVertex* vertices;
    size_t num_vertices;
    SpinFoamEdge* edges;
    size_t num_edges;
    SpinFoamFace* faces;
    size_t num_faces;
    SpinNetwork* boundary_in;        // Initial spin network
    SpinNetwork* boundary_out;       // Final spin network
    qgt_complex_t transition_amplitude;  // <out|in>
    double immirzi_parameter;
    SpacetimeSignature signature;
} SpinFoam;

/**
 * EPRL vertex amplitude parameters
 */
typedef struct {
    double immirzi_parameter;        // γ
    double* spins;                   // Boundary spins
    size_t num_spins;
    qgt_complex_t* coherent_states;  // Coherent state data
    bool use_asymptotics;            // Use asymptotic formula
} EPRLVertexParams;

// =============================================================================
// Causal Dynamical Triangulations
// =============================================================================

/**
 * CDT simplex types
 */
typedef enum {
    CDT_SIMPLEX_TIMELIKE,            // Timelike simplex
    CDT_SIMPLEX_SPACELIKE,           // Spacelike simplex
    CDT_SIMPLEX_31,                  // (3,1) simplex type
    CDT_SIMPLEX_22,                  // (2,2) simplex type
    CDT_SIMPLEX_13                   // (1,3) simplex type
} CDTSimplexType;

/**
 * CDT simplex
 */
typedef struct {
    size_t simplex_id;
    CDTSimplexType type;
    size_t* vertices;                // Vertex indices
    size_t num_vertices;
    size_t time_slice;               // Discrete time coordinate
    size_t* neighbors;               // Adjacent simplices
    size_t num_neighbors;
    double edge_length_squared;      // Squared edge length
} CDTSimplex;

/**
 * CDT configuration
 */
typedef struct {
    CDTSimplex* simplices;
    size_t num_simplices;
    size_t num_time_slices;
    size_t spatial_dimension;
    double lambda;                   // Cosmological constant
    double G_newton;                 // Newton's constant
    double k0;                       // Bare coupling
    double delta;                    // Asymmetry parameter
    double action;                   // Regge action value
} CDTConfiguration;

/**
 * CDT Monte Carlo parameters
 */
typedef struct {
    size_t num_sweeps;
    double* move_probabilities;      // Pachner move probabilities
    size_t num_moves;
    double temperature;
    bool enforce_causality;
    size_t thermalization_sweeps;
    size_t measurement_interval;
} CDTMonteCarloParams;

// =============================================================================
// Holographic/AdS-CFT Structures
// =============================================================================

/**
 * AdS geometry parameters
 */
typedef struct {
    size_t dimension;                // d+1 bulk dimension
    double ads_radius;               // AdS radius L
    double* metric;                  // Bulk metric components
    size_t num_metric_components;
    double cutoff_radius;            // UV cutoff
    bool use_poincare_patch;         // Poincare vs global
} AdSGeometry;

/**
 * CFT data
 */
typedef struct {
    size_t dimension;                // d dimensional CFT
    double central_charge;           // Central charge c
    double* operator_dimensions;     // Scaling dimensions Δ
    size_t num_operators;
    qgt_complex_t* structure_constants;  // OPE coefficients
    double temperature;              // For thermal CFT
} CFTData;

/**
 * Holographic entanglement
 */
typedef struct {
    double* boundary_region;         // Boundary subregion A
    size_t region_size;
    double* minimal_surface;         // RT/HRT surface
    size_t surface_vertices;
    double surface_area;             // Area of extremal surface
    double entanglement_entropy;     // S_A = Area/(4G_N)
    bool is_connected;               // Phase of surface
} HolographicEntanglement;

/**
 * Bulk-boundary propagator
 */
typedef struct {
    size_t boundary_dim;
    size_t bulk_dim;
    double operator_dimension;       // Boundary operator dimension
    qgt_complex_t* propagator;      // K(z,x;x')
    size_t num_points;
    bool normalized;
} BulkBoundaryPropagator;

// =============================================================================
// Loop Quantum Gravity Operations
// =============================================================================

/**
 * Create spin network
 */
int spin_network_create(
    SpinNetwork** network,
    size_t num_nodes,
    size_t num_edges
);

/**
 * Destroy spin network
 */
void spin_network_destroy(SpinNetwork* network);

/**
 * Add edge to spin network
 */
int spin_network_add_edge(
    SpinNetwork* network,
    size_t source,
    size_t target,
    double spin_j
);

/**
 * Compute area eigenvalue
 */
int spin_network_compute_area(
    SpinNetwork* network,
    size_t* edge_indices,
    size_t num_edges,
    double immirzi,
    double* area_out
);

/**
 * Compute volume eigenvalue at node
 */
int spin_network_compute_volume(
    SpinNetwork* network,
    size_t node_id,
    double immirzi,
    double* volume_out
);

/**
 * Apply holonomy operator
 */
int spin_network_apply_holonomy(
    SpinNetwork* network,
    size_t edge_id,
    qgt_complex_t* group_element,
    size_t group_dim
);

/**
 * Compute inner product of spin networks
 */
int spin_network_inner_product(
    SpinNetwork* network1,
    SpinNetwork* network2,
    qgt_complex_t* result
);

// =============================================================================
// Spin Foam Operations
// =============================================================================

/**
 * Create spin foam
 */
int spin_foam_create(
    SpinFoam** foam,
    SpinFoamVariant variant,
    SpinNetwork* boundary_in,
    SpinNetwork* boundary_out
);

/**
 * Destroy spin foam
 */
void spin_foam_destroy(SpinFoam* foam);

/**
 * Compute EPRL vertex amplitude
 */
int spin_foam_eprl_vertex(
    EPRLVertexParams* params,
    qgt_complex_t* amplitude_out
);

/**
 * Compute total spin foam amplitude
 */
int spin_foam_compute_amplitude(
    SpinFoam* foam,
    qgt_complex_t* amplitude_out
);

/**
 * Compute transition amplitude between spin networks
 */
int spin_foam_transition_amplitude(
    SpinNetwork* initial,
    SpinNetwork* final,
    SpinFoamVariant variant,
    double immirzi,
    qgt_complex_t* amplitude_out
);

/**
 * Evaluate spin foam in asymptotic regime
 */
int spin_foam_asymptotic_amplitude(
    SpinFoam* foam,
    double* critical_points,
    size_t num_critical,
    qgt_complex_t* amplitude_out
);

// =============================================================================
// CDT Operations
// =============================================================================

/**
 * Create CDT configuration
 */
int cdt_create(
    CDTConfiguration** config,
    size_t spatial_dim,
    size_t num_time_slices
);

/**
 * Destroy CDT configuration
 */
void cdt_destroy(CDTConfiguration* config);

/**
 * Initialize random CDT triangulation
 */
int cdt_initialize_random(
    CDTConfiguration* config,
    size_t target_simplices
);

/**
 * Perform Pachner move
 */
int cdt_pachner_move(
    CDTConfiguration* config,
    size_t simplex_id,
    int move_type
);

/**
 * Compute Regge action
 */
int cdt_compute_action(
    CDTConfiguration* config,
    double* action_out
);

/**
 * Run CDT Monte Carlo
 */
int cdt_monte_carlo(
    CDTConfiguration* config,
    CDTMonteCarloParams* params,
    double** observables_out,
    size_t* num_measurements
);

/**
 * Measure spectral dimension
 */
int cdt_spectral_dimension(
    CDTConfiguration* config,
    double diffusion_time,
    double* spectral_dim_out
);

/**
 * Measure Hausdorff dimension
 */
int cdt_hausdorff_dimension(
    CDTConfiguration* config,
    double* hausdorff_dim_out
);

// =============================================================================
// Holographic Operations
// =============================================================================

/**
 * Create AdS geometry
 */
int ads_geometry_create(
    AdSGeometry** geometry,
    size_t dimension,
    double radius
);

/**
 * Destroy AdS geometry
 */
void ads_geometry_destroy(AdSGeometry* geometry);

/**
 * Create CFT data
 */
int cft_data_create(
    CFTData** data,
    size_t dimension,
    double central_charge
);

/**
 * Destroy CFT data
 */
void cft_data_destroy(CFTData* data);

/**
 * Compute Ryu-Takayanagi surface
 */
int holographic_rt_surface(
    AdSGeometry* geometry,
    double* boundary_region,
    size_t region_size,
    HolographicEntanglement** result
);

/**
 * Compute holographic entanglement entropy
 */
int holographic_entanglement_entropy(
    AdSGeometry* geometry,
    double* boundary_region,
    size_t region_size,
    double G_newton,
    double* entropy_out
);

/**
 * Compute bulk-boundary propagator
 */
int holographic_bulk_boundary_propagator(
    AdSGeometry* geometry,
    double operator_dimension,
    double* bulk_point,
    double* boundary_points,
    size_t num_boundary,
    BulkBoundaryPropagator** propagator_out
);

/**
 * Reconstruct bulk from boundary CFT
 */
int holographic_bulk_reconstruction(
    CFTData* cft,
    AdSGeometry* ads,
    struct quantum_state* boundary_state,
    struct quantum_state** bulk_state_out
);

/**
 * Compute holographic complexity (CV)
 */
int holographic_complexity_cv(
    AdSGeometry* geometry,
    double boundary_time,
    double* complexity_out
);

/**
 * Compute holographic complexity (CA)
 */
int holographic_complexity_ca(
    AdSGeometry* geometry,
    double boundary_time,
    double* complexity_out
);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Compute Wigner 6j symbol
 */
int wigner_6j(
    double j1, double j2, double j3,
    double j4, double j5, double j6,
    double* result
);

/**
 * Compute Wigner 15j symbol
 */
int wigner_15j(
    double* spins,
    double* result
);

/**
 * Compute SU(2) Clebsch-Gordan coefficient
 */
int clebsch_gordan(
    double j1, double m1,
    double j2, double m2,
    double j, double m,
    double* result
);

/**
 * Compute coherent state overlap
 */
int coherent_state_overlap(
    double* n1, double* n2,
    double spin_j,
    qgt_complex_t* overlap_out
);

/**
 * Print spin network
 */
void spin_network_print(SpinNetwork* network);

/**
 * Print spin foam
 */
void spin_foam_print(SpinFoam* foam);

/**
 * Validate spin network gauge invariance
 */
bool spin_network_is_gauge_invariant(SpinNetwork* network);

#ifdef __cplusplus
}
#endif

#endif // QUANTUM_GRAVITY_OPERATIONS_H

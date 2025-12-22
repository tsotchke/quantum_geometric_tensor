#ifndef MOLECULAR_GEOMETRIC_OPERATIONS_H
#define MOLECULAR_GEOMETRIC_OPERATIONS_H

#include <stdbool.h>
#include <stddef.h>
#include <complex.h>

// Geometry types
typedef enum {
    GEOM_EUCLIDEAN,       // Euclidean geometry
    GEOM_RIEMANNIAN,      // Riemannian geometry
    GEOM_SYMPLECTIC,      // Symplectic geometry
    GEOM_KAHLER,          // Kähler geometry
    GEOM_MOLECULAR       // Molecular geometry
} geometry_type_t;

// Operation types
typedef enum {
    OP_ROTATION,          // Rotation operation
    OP_TRANSLATION,       // Translation operation
    OP_DEFORMATION,       // Deformation operation
    OP_TRANSFORMATION,    // General transformation
    OP_QUANTUM           // Quantum operation
} operation_type_t;

// Coordinate systems
typedef enum {
    COORD_CARTESIAN,      // Cartesian coordinates
    COORD_INTERNAL,       // Internal coordinates
    COORD_NORMAL,         // Normal coordinates
    COORD_SYMMETRY,       // Symmetry coordinates
    COORD_QUANTUM        // Quantum coordinates
} coordinate_system_t;

// Symmetry types
typedef enum {
    SYM_POINT,           // Point symmetry
    SYM_SPACE,           // Space symmetry
    SYM_TIME,            // Time symmetry
    SYM_GAUGE,           // Gauge symmetry
    SYM_QUANTUM         // Quantum symmetry
} symmetry_type_t;

// Molecular structure
typedef struct {
    size_t num_atoms;             // Number of atoms
    double* coordinates;          // Atomic coordinates
    int* atomic_numbers;          // Atomic numbers
    double* masses;               // Atomic masses
    double* charges;              // Atomic charges
    void* structure_data;        // Additional data
} molecular_structure_t;

// Geometric parameters
typedef struct {
    geometry_type_t type;         // Geometry type
    double* metric_tensor;        // Metric tensor
    double* connection_coeffs;    // Connection coefficients
    double* curvature_tensor;     // Curvature tensor
    size_t dimensions;           // Space dimensions
    void* geometry_data;         // Additional data
} geometric_params_t;

// Operation parameters
typedef struct {
    operation_type_t type;        // Operation type
    double* parameters;           // Operation parameters
    size_t num_params;            // Number of parameters
    bool is_reversible;           // Reversibility flag
    double tolerance;             // Operation tolerance
    void* operation_data;        // Additional data
} operation_params_t;

// Transformation result
typedef struct {
    molecular_structure_t* initial;  // Initial structure
    molecular_structure_t* final;    // Final structure
    double energy_change;            // Energy change
    double* gradients;               // Energy gradients
    double* forces;                  // Atomic forces
    void* result_data;             // Additional data
} transformation_result_t;

// Opaque operations handle
typedef struct molecular_operations_t molecular_operations_t;

// Core functions
molecular_operations_t* create_molecular_operations(geometry_type_t type);
void destroy_molecular_operations(molecular_operations_t* operations);

// Structure functions
bool initialize_structure(molecular_operations_t* operations,
                        molecular_structure_t* structure);
bool validate_structure(molecular_operations_t* operations,
                       const molecular_structure_t* structure);
bool optimize_structure(molecular_operations_t* operations,
                       molecular_structure_t* structure);

// Geometric operations
bool compute_geometric_parameters(molecular_operations_t* operations,
                                const molecular_structure_t* structure,
                                geometric_params_t* params);
bool transform_coordinates(molecular_operations_t* operations,
                         coordinate_system_t from,
                         coordinate_system_t to,
                         double* coordinates);
bool apply_symmetry(molecular_operations_t* operations,
                   symmetry_type_t symmetry,
                   molecular_structure_t* structure);

// Transformation operations
bool apply_transformation(molecular_operations_t* operations,
                         const operation_params_t* params,
                         molecular_structure_t* structure);
bool compute_transformation(molecular_operations_t* operations,
                          const molecular_structure_t* initial,
                          const molecular_structure_t* final,
                          operation_params_t* params);
bool validate_transformation(molecular_operations_t* operations,
                           const transformation_result_t* result);

// Energy calculations
bool compute_energy(molecular_operations_t* operations,
                   const molecular_structure_t* structure,
                   double* energy);
bool compute_gradients(molecular_operations_t* operations,
                      const molecular_structure_t* structure,
                      double* gradients);
bool compute_forces(molecular_operations_t* operations,
                   const molecular_structure_t* structure,
                   double* forces);

// Quantum operations
bool apply_quantum_operation(molecular_operations_t* operations,
                           const operation_params_t* params,
                           molecular_structure_t* structure);
bool compute_quantum_properties(molecular_operations_t* operations,
                              const molecular_structure_t* structure,
                              void* properties);
bool validate_quantum_state(molecular_operations_t* operations,
                          const molecular_structure_t* structure);

// Utility functions
bool export_structure(const molecular_operations_t* operations,
                     const molecular_structure_t* structure,
                     const char* filename);
bool import_structure(molecular_operations_t* operations,
                     molecular_structure_t* structure,
                     const char* filename);
void free_molecular_structure(molecular_structure_t* structure);

// =============================================================================
// Graph Neural Network Types for Geometric Deep Learning
// =============================================================================

// Molecular graph representation for GNN
typedef struct {
    size_t num_atoms;              // Number of atoms (nodes)
    size_t num_bonds;              // Number of bonds (edges)
    double* positions;             // 3D positions (num_atoms x 3)
    int* edge_index;               // Edge indices (2 x num_bonds)
    double* edge_attr;             // Edge attributes
    int* atomic_numbers;           // Atomic numbers
    double* charges;               // Atomic charges
    int* bond_indices;             // Bond connectivity (2 x num_bonds)
    int* bond_types;               // Bond types
    void* graph_data;              // Additional graph data
} MolecularGraph;

// Geometric feature representation
typedef struct {
    size_t num_node_features;      // Number of node features
    size_t num_edge_features;      // Number of edge features
    size_t num_global_features;    // Number of global features
    double* node_features;         // Node feature matrix
    double* edge_features;         // Edge feature matrix
    double* global_features;       // Global feature vector
    double* invariants;            // Geometric invariants
    void* feature_data;            // Additional feature data
} GeometricFeatures;

// SE(3) equivariant transformation
typedef struct {
    size_t num_layers;             // Number of transform layers
    double* rotation_matrices;     // Rotation matrices (3x3 per layer)
    double* translations;          // Translation vectors (3 per layer)
    double* weights;               // Layer weights
    bool use_fiber_bundle;         // Use fiber bundle formulation
    void* transform_data;          // Additional transform data
} SE3Transform;

// Graph neural network message passing
void geometric_message_passing(
    const MolecularGraph* graph,
    GeometricFeatures* features,
    const SE3Transform* transform
);

// SE(3)-equivariant convolution
void equivariant_convolution(
    const double* input_features,
    double* output_features,
    const SE3Transform* transform,
    size_t num_features,
    size_t num_points
);

// Graph initialization and cleanup
void init_molecular_graph(
    MolecularGraph* graph,
    size_t num_atoms,
    size_t num_bonds
);

void free_molecular_graph(MolecularGraph* graph);

// Feature initialization and cleanup
void init_geometric_features(
    GeometricFeatures* features,
    size_t num_nodes,
    size_t num_node_features
);

void free_geometric_features(GeometricFeatures* features);

#endif // MOLECULAR_GEOMETRIC_OPERATIONS_H

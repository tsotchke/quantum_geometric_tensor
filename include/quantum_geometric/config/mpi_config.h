#ifndef MPI_CONFIG_H
#define MPI_CONFIG_H

#ifdef USE_MPI

#ifdef __APPLE__
#include <mpi/mpi.h>
#else
#include <mpi.h>
#endif

// MPI Error codes
#define QG_MPI_SUCCESS MPI_SUCCESS
#define QG_MPI_ERROR -1

// MPI Constants
#define QG_MPI_MAX_PROCESSOR_NAME MPI_MAX_PROCESSOR_NAME
#define QG_MPI_ANY_SOURCE MPI_ANY_SOURCE
#define QG_MPI_ANY_TAG MPI_ANY_TAG

// MPI Types
typedef MPI_Comm qg_mpi_comm_t;
typedef MPI_Status qg_mpi_status_t;
typedef MPI_Request qg_mpi_request_t;
typedef MPI_Datatype qg_mpi_datatype_t;
typedef MPI_Op qg_mpi_op_t;

// MPI Functions
#define qg_mpi_init MPI_Init
#define qg_mpi_finalize MPI_Finalize
#define qg_mpi_comm_rank MPI_Comm_rank
#define qg_mpi_comm_size MPI_Comm_size
#define qg_mpi_send MPI_Send
#define qg_mpi_recv MPI_Recv
#define qg_mpi_isend MPI_Isend
#define qg_mpi_irecv MPI_Irecv
#define qg_mpi_wait MPI_Wait
#define qg_mpi_waitall MPI_Waitall
#define qg_mpi_barrier MPI_Barrier
#define qg_mpi_bcast MPI_Bcast
#define qg_mpi_reduce MPI_Reduce
#define qg_mpi_allreduce MPI_Allreduce
#define qg_mpi_gather MPI_Gather
#define qg_mpi_allgather MPI_Allgather
#define qg_mpi_scatter MPI_Scatter
#define qg_mpi_alltoall MPI_Alltoall

// MPI Data types
#define QG_MPI_CHAR MPI_CHAR
#define QG_MPI_INT MPI_INT
#define QG_MPI_FLOAT MPI_FLOAT
#define QG_MPI_DOUBLE MPI_DOUBLE
#define QG_MPI_BYTE MPI_BYTE

// MPI Operations
#define QG_MPI_SUM MPI_SUM
#define QG_MPI_MAX MPI_MAX
#define QG_MPI_MIN MPI_MIN
#define QG_MPI_PROD MPI_PROD

// MPI Communicators
#define QG_MPI_COMM_WORLD MPI_COMM_WORLD
#define QG_MPI_COMM_SELF MPI_COMM_SELF

#else // !USE_MPI

// Stub types for non-MPI builds
typedef int qg_mpi_comm_t;
typedef int qg_mpi_status_t;
typedef int qg_mpi_request_t;
typedef int qg_mpi_datatype_t;
typedef int qg_mpi_op_t;

// Error codes
#define QG_MPI_SUCCESS 0
#define QG_MPI_ERROR -1

// Constants
#define QG_MPI_MAX_PROCESSOR_NAME 256
#define QG_MPI_ANY_SOURCE -1
#define QG_MPI_ANY_TAG -1

// Stub data types
#define QG_MPI_CHAR 1
#define QG_MPI_INT 2
#define QG_MPI_FLOAT 3
#define QG_MPI_DOUBLE 4
#define QG_MPI_BYTE 5

// Stub operations
#define QG_MPI_SUM 1
#define QG_MPI_MAX 2
#define QG_MPI_MIN 3
#define QG_MPI_PROD 4

// Stub communicators
#define QG_MPI_COMM_WORLD 0
#define QG_MPI_COMM_SELF 1

// Stub function declarations
static inline int qg_mpi_init(int* argc, char*** argv) { return QG_MPI_SUCCESS; }
static inline int qg_mpi_finalize(void) { return QG_MPI_SUCCESS; }
static inline int qg_mpi_comm_rank(qg_mpi_comm_t comm, int* rank) { *rank = 0; return QG_MPI_SUCCESS; }
static inline int qg_mpi_comm_size(qg_mpi_comm_t comm, int* size) { *size = 1; return QG_MPI_SUCCESS; }
static inline int qg_mpi_barrier(qg_mpi_comm_t comm) { return QG_MPI_SUCCESS; }

#endif // USE_MPI

#endif // MPI_CONFIG_H

#ifndef ACCELERATE_WRAPPER_H
#define ACCELERATE_WRAPPER_H

/**
 * @file accelerate_wrapper.h
 * @brief Cross-platform wrapper for BLAS/LAPACK operations
 *
 * On Apple platforms: Uses the native Accelerate framework
 * On other platforms: Provides fallback definitions for compatibility
 */

#ifdef __APPLE__

// On Apple, use the real Accelerate framework
#include <Accelerate/Accelerate.h>

// Helper macros for error checking
#define CHECK_LAPACK_ERROR(info) \
    if (info != 0) { \
        return false; \
    }

#define CHECK_NULL(ptr) \
    if (ptr == NULL) { \
        return false; \
    }

// Define MIN macro if not defined
#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

#else // !__APPLE__

// Non-Apple platforms - provide compatible definitions

#include <stddef.h>
#include <stdbool.h>

// CBLAS enum definitions
enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102
};

enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
};

enum CBLAS_UPLO {
    CblasUpper = 121,
    CblasLower = 122
};

enum CBLAS_DIAG {
    CblasNonUnit = 131,
    CblasUnit = 132
};

enum CBLAS_SIDE {
    CblasLeft = 141,
    CblasRight = 142
};

// CLAPACK types
typedef int __CLPK_integer;
typedef float __CLPK_real;
typedef double __CLPK_doublereal;

typedef struct {
    float r, i;
} __CLPK_complex;

typedef struct {
    double r, i;
} __CLPK_doublecomplex;

// DSP types for vDSP compatibility
typedef struct DSPComplex {
    float real;
    float imag;
} DSPComplex;

typedef struct DSPDoubleComplex {
    double real;
    double imag;
} DSPDoubleComplex;

typedef struct DSPSplitComplex {
    float* realp;
    float* imagp;
} DSPSplitComplex;

typedef struct DSPDoubleSplitComplex {
    double* realp;
    double* imagp;
} DSPDoubleSplitComplex;

typedef unsigned long vDSP_Length;
typedef long vDSP_Stride;

#ifdef __cplusplus
extern "C" {
#endif

// CBLAS function declarations (typically provided by OpenBLAS, MKL, etc.)
void cblas_sgemm(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB,
                 const int M, const int N, const int K,
                 const float alpha, const float* A, const int lda,
                 const float* B, const int ldb, const float beta,
                 float* C, const int ldc);

void cblas_dgemm(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB,
                 const int M, const int N, const int K,
                 const double alpha, const double* A, const int lda,
                 const double* B, const int ldb, const double beta,
                 double* C, const int ldc);

void cblas_cgemm(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB,
                 const int M, const int N, const int K,
                 const void* alpha, const void* A, const int lda,
                 const void* B, const int ldb, const void* beta,
                 void* C, const int ldc);

void cblas_zgemm(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB,
                 const int M, const int N, const int K,
                 const void* alpha, const void* A, const int lda,
                 const void* B, const int ldb, const void* beta,
                 void* C, const int ldc);

#ifdef __cplusplus
}
#endif

// Helper macros for error checking
#define CHECK_LAPACK_ERROR(info) \
    if (info != 0) { \
        return false; \
    }

#define CHECK_NULL(ptr) \
    if (ptr == NULL) { \
        return false; \
    }

// Define MIN/MAX macros if not defined
#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

#endif // __APPLE__

#endif // ACCELERATE_WRAPPER_H

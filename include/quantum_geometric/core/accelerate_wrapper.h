#ifndef ACCELERATE_WRAPPER_H
#define ACCELERATE_WRAPPER_H

#ifdef __APPLE__

// Include system headers
#include <TargetConditionals.h>

// CBLAS enum definitions - moved to global scope
enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102
};

enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
};

// Forward declarations for Accelerate types we need
typedef struct {
    float real;
    float imag;
} __CLPK_complex;

typedef struct {
    double real;
    double imag;
} __CLPK_doublecomplex;

typedef int __CLPK_integer;
typedef float __CLPK_real;
typedef double __CLPK_doublereal;

// Define MIN macro if not defined
#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Single precision SVD
int cgesvd_(char* jobu, char* jobvt,
            __CLPK_integer* m, __CLPK_integer* n,
            __CLPK_complex* a, __CLPK_integer* lda,
            __CLPK_real* s,
            __CLPK_complex* u, __CLPK_integer* ldu,
            __CLPK_complex* vt, __CLPK_integer* ldvt,
            __CLPK_complex* work, __CLPK_integer* lwork,
            __CLPK_real* rwork, __CLPK_integer* info);

// Double precision SVD
int zgesvd_(char* jobu, char* jobvt,
            __CLPK_integer* m, __CLPK_integer* n,
            __CLPK_doublecomplex* a, __CLPK_integer* lda,
            __CLPK_doublereal* s,
            __CLPK_doublecomplex* u, __CLPK_integer* ldu,
            __CLPK_doublecomplex* vt, __CLPK_integer* ldvt,
            __CLPK_doublecomplex* work, __CLPK_integer* lwork,
            __CLPK_doublereal* rwork,
            __CLPK_integer* info);

// Single precision matrix multiply
void cblas_cgemm(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB,
                 const int M, const int N, const int K,
                 const void* alpha, const void* A, const int lda,
                 const void* B, const int ldb, const void* beta,
                 void* C, const int ldc);

// Double precision matrix multiply
void cblas_zgemm(enum CBLAS_ORDER Order,
                 enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB,
                 const int M, const int N, const int K,
                 const void* alpha, const void* A, const int lda,
                 const void* B, const int ldb, const void* beta,
                 void* C, const int ldc);

// vDSP operations for complex arithmetic
typedef struct DSPComplex {
    float real;
    float imag;
} DSPComplex;

typedef struct DSPDoubleComplex {
    double real;
    double imag;
} DSPDoubleComplex;

// vDSP complex operations
void vDSP_zvmul(const DSPComplex* __A, __CLPK_integer __IA,
                const DSPComplex* __B, __CLPK_integer __IB,
                DSPComplex* __C, __CLPK_integer __IC,
                __CLPK_integer __N, __CLPK_integer __F);

void vDSP_zvadd(const DSPComplex* __A, __CLPK_integer __IA,
                const DSPComplex* __B, __CLPK_integer __IB,
                DSPComplex* __C, __CLPK_integer __IC,
                __CLPK_integer __N);

void vDSP_zvabs(const DSPComplex* __A, __CLPK_integer __IA,
                float* __C, __CLPK_integer __IC,
                __CLPK_integer __N);

void vDSP_zrvadd(const float* __A, __CLPK_integer __IA,
                 const DSPComplex* __B, __CLPK_integer __IB,
                 DSPComplex* __C, __CLPK_integer __IC,
                 __CLPK_integer __N);

void vDSP_zrvmul(const float* __A, __CLPK_integer __IA,
                 const DSPComplex* __B, __CLPK_integer __IB,
                 DSPComplex* __C, __CLPK_integer __IC,
                 __CLPK_integer __N);

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

#endif // __APPLE__

#endif // ACCELERATE_WRAPPER_H

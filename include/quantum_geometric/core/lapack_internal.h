#ifndef LAPACK_INTERNAL_H
#define LAPACK_INTERNAL_H

#include "quantum_geometric/core/quantum_complex.h"

// LAPACK function declarations
extern void cgemm_(const char* transa, const char* transb,
                  const int* m, const int* n, const int* k,
                  const ComplexFloat* alpha,
                  const ComplexFloat* a, const int* lda,
                  const ComplexFloat* b, const int* ldb,
                  const ComplexFloat* beta,
                  ComplexFloat* c, const int* ldc);

extern void cgesvd_(const char* jobu, const char* jobvt,
                   const int* m, const int* n,
                   ComplexFloat* a, const int* lda,
                   float* s,
                   ComplexFloat* u, const int* ldu,
                   ComplexFloat* vt, const int* ldvt,
                   ComplexFloat* work, const int* lwork,
                   float* rwork, int* info);

extern void cgeqrf_(const int* m, const int* n,
                    ComplexFloat* a, const int* lda,
                    ComplexFloat* tau,
                    ComplexFloat* work, const int* lwork,
                    int* info);

extern void cungqr_(const int* m, const int* n, const int* k,
                    ComplexFloat* a, const int* lda,
                    ComplexFloat* tau,
                    ComplexFloat* work, const int* lwork,
                    int* info);

extern void cheev_(const char* jobz, const char* uplo,
                   const int* n, ComplexFloat* a, const int* lda,
                   float* w,
                   ComplexFloat* work, const int* lwork,
                   float* rwork, int* info);

extern void cpotrf_(const char* uplo, const int* n,
                    ComplexFloat* a, const int* lda,
                    int* info);

extern void cgetrf_(const int* m, const int* n,
                    ComplexFloat* a, const int* lda,
                    int* ipiv, int* info);

extern void ctrtrs_(const char* uplo, const char* trans, const char* diag,
                    const int* n, const int* nrhs,
                    const ComplexFloat* a, const int* lda,
                    ComplexFloat* b, const int* ldb,
                    int* info);

extern void cposv_(const char* uplo, const int* n, const int* nrhs,
                   ComplexFloat* a, const int* lda,
                   ComplexFloat* b, const int* ldb,
                   int* info);

extern void cgesv_(const int* n, const int* nrhs,
                   ComplexFloat* a, const int* lda,
                   int* ipiv,
                   ComplexFloat* b, const int* ldb,
                   int* info);

#endif // LAPACK_INTERNAL_H

#ifndef VALIDATION_H
#define VALIDATION_H

#include "datalayout.h"

/**
 * @brief Prints || A x - lambda x||_F.
 *
 * Validates that the eigenvectors computed point in the right direction.
 *
 * @param[in]  T_blocks      The upper quasi-triangular matrix partitioned into
 *                           blocks.
 * @param[in]  ldT           Leading dimension of T.
 * @param[in]  Q_blocks      The orthogonal matrix of the Schur decomposition
 *                           partitioned into blocks.
 * @param[in]  ldQ           Leading dimension of Q.
 * @param[out] X_blocks      The eigenvectors partitioned into blocks.
 * @param[in]  ldX           Leading dimension of X.
 * @param[in]  lambda        Array of length n. Holds the actual eigenvalues.
 *                           If T(i,i) is 1-by-1, then lambda[i] = T(i,i). If
 *                           T(i:i+1,i:i+1) is 2-by-2, the complex eigenvalue is
 *                           stored as lambda[i] = T(i,i) = T(i+1,i+1) and 
 *                           lambda[i+1] = T(i+1,i) = -T(i,i+1).
 * @param[in]  lambda_type   Array of length n. Describes if T(i,i) is a real
 *                           or complex eigenvalue.
 * @param[in]  selected      Array of length n. Marks selected eigenvalues.
 * @param[in]  p             Partitioning (OpenMP grid) of T, Q.
 * @param[in]  p_rhs         Partitioning (OpenMP grid) of X.
 * @param[in]  layout        COLUMN_MAJOR or TILE_LAYOUT
 */
void validate(int n,
    double ***restrict const T_blocks, int ldT,
    double ***restrict const Q_blocks, int ldQ,
    double ***restrict const X_blocks, int ldX,
    double *restrict const lambda, const int *restrict const lambda_type,
    const int *restrict const selected,
    const partitioning_t *restrict const p, const partitioning_t *restrict const p_rhs,
    memory_layout_t layout);

#endif

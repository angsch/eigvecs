#ifndef EIGENVECTORS_H
#define EIGENVECTORS_H

#include "datalayout.h"

/**
 * @brief Compute selected eigenvectors
 * A = Q T Q^T
 * 
 * @param[in]  n             Dimension of T, Q, X.
 * @param[in]  T_blocks      The upper quasi-triangular matrix partitioned into
 *                           blocks.
 * @param[in]  ldT           Leading dimension of T
 * @param[in]  desc_T        Matrix description structure of T.
 * @param[in]  lambda        Array of length n. Holds the actual eigenvalues.
 *                           If T(i,i) is 1-by-1, then lambda[i] = T(i,i). If
 *                           T(i:i+1,i:i+1) is 2-by-2, the complex eigenvalue is
 *                           stored as lambda[i] = T(i,i) = T(i+1,i+1) and 
 *                           lambda[i+1] = T(i+1,i) = -T(i,i+1).
 * @param[in]  lambda_type   Array of length n. Describes if T(i,i) is a real
 *                           or complex eigenvalue
 * @param[in]  selected      Array of length n. Marks selected eigenvalues.
 * @param[in]  p             Partitioning (OpenMP grid) of T, Q.
 * @param[in]  p_rhs         Partitioning (OpenMP grid) of X.
 * @param[in]  Q_blocks      The orthogonal matrix of the Schur decomposition partitioned into blocks.
 * @param[out] X_blocks      On exit, the eigenvectors partitioned into blocks.
 * @param[in]  ldX           Leading dimension of X
 * @param[in]  scales        An num_selected-by-num_blk array for the scaling factors.
 * @param[in]  layout        COLUMN_MAJOR or TILE_LAYOUT
 */

void compute_eigenvectors_separate_phases_openmp(
    int n,
    double ***restrict const T_blocks, int ldT,
    double *restrict const lambda, int *restrict const lambda_type,
    const int *restrict const selected,
    const partitioning_t *restrict const p, const partitioning_t *restrict const p_rhs,
    double ***restrict const Q_blocks, int ldQ,
    double ***restrict const X_blocks, int ldX,
    scaling_t *restrict const scales,
    const double *restrict const Tnorms,
    double *restrict const Xnorms,
    memory_layout_t layout);



#endif

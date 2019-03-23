#ifndef BACKTRANSFORM_TASK_H
#define BACKTRANSFORM_TASK_H

#include "datalayout.h"

#include <omp.h>

/** 
 * @defgroup backtransform Backtransform Y := Q * X.
 * 
 * Collection of routines that implement the backtransform
 *
 *      Y := Q * X
 *
 * where Q is orthogonal matrix and X, the eigenvector matrix, has generalized
 * upper triangular form.
 */


/** @ingroup backtransform
 *
 * @brief Fine-grained backtransform.
 *
 * Implements one accumulation step of the matrix-matrix multiplication
 *
 *    Y    := first_write * Y +    Q   *   X       where first_write is {0, 1}
 * (n x m)                      (n x k) (k x m)
 *
 * The value of first_write is determined by cnt_adds.
 *
 * @param[in]     n             Number of rows of Q, Y.
 * @param[in]     m             Number of columns of X, Y.
 * @param[in]     k             Number of columns of Q. Number of rows of X.
 * @param[in]     Q             Orthogonal n-by-k matrix.
 * @param[in]     ldQ           Leading dimension of Q.
 * @param[in]     lock          Lock associated with Y.
 * @param[in,out] cnt_adds      Atomic variable that counts accumulations.
 * @param[in,out] Y             n-by-m accumulation matrix.
 * @param[in]     ldY           Leading dimension of Y.
 * @param[in]     X             Generalized upper triangular k-by-m matrix.
 * @param[in]     ldX           Leading dimension of X.
 */
void basetransform(
    int n, int m, int k, const double *restrict Q, int ldQ,
    omp_lock_t *lock, int *restrict const cnt_adds,
    double *restrict const Y, int ldY,
    double *restrict const X, int ldX);



/** @ingroup backtransform
 *
 * @brief Coarse-grained backtransform.
 *
 * In column major, compute the backtransform as one call to DGEMM, i.e.,
 * Yij := Q * X. In tile layout, the backtransform is computed as tiled DGEMM,
 * i.e., Yij := sum_k (Qik * Xkj).
 *
 * @param[in]     p_rhs         Partitioning of X, Y.
 * @param[in]     i             Block row index of Y.
 * @param[in]     j             Block column index of Y.
 * @param[in]     Q_blocks      The orthogonal matrix of the Schur
 *                              decomposition partitioned into blocks.
 * @param[in]     ldQ           Leading dimension of Q.
 * @param[in]     Y_blocks      On exit, the block (i,j) of Y is computed.
 * @param[in]     ldY           Leading dimension of Y.
 * @param[in]     X_blocks      The eigenvectors partitioned into blocks.
 * @param[in]     ldX           Leading dimension of X.
 * @param[in]     layout        COLUMN_MAJOR or TILE_LAYOUT
 */
void backtransform(
    const partitioning_t *restrict p_rhs,
    int i, int j,
    double ***restrict Q_blocks, int ldQ,
    double ***restrict Y_blocks, int ldY,
    double ***restrict X_blocks, int ldX,
    memory_layout_t layout);

#endif

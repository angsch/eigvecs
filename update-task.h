#ifndef UPDATE_TASK_H
#define UPDATE_TASK_H

#include <omp.h>


/**
 * Computes the linear update Y := Y - T * X.
 *
 * Executes the linear update
 *
 *       Y           := Y -   T   *   X
 *    (n x num_rhs)        (n x m) (m x num_rhs)
 *
 * robustly. For this purpose, each column in Y is associated with a scaling
 * factor. If the growth in Y triggered overflow, the columns are downscaled.
 *
 * @param[in]  n             The number of rows in Y, T. n >= 0.
 * @param[in]  m             The number of columns in T. m >= 0.
 * @param[in]  T             Dense matrix T with dimensions (ldT, m).
 * @param[in]  ldT           The leading dimension of T. ldT >= max(1,n).
 * @param[in]  tnorm         An upper bound for the matrix of T. The value can
 *                           be computed by compute_majorants().
 * @param[in]  num_rhs       The number of columns in Y, X. num_rhs >= 0.
 * @param[in]  lock          Lock associated with Y.
 * @param[in, out] Y         Matrix with dimensions (ldY, num_rhs).
 * @param[in]  ldY           The leading dimension of Y. ldY >= max(1,n).
 * @param[in, out] Yscales   Array of length num_rhs. Yscales[i] is the scaling
 *                           factor associated with the i-th column vector in Y.
 * @param[in,out] Ynorms     Array of length num_rhs. Ynorms[i] is the infinity
 *                           norm associated with the i-th column vector in Y.
 * @param[in]  Xin           Matrix with dimensions (ldY, num_rhs).
 * @param[in]  ldX           The leading dimension of X. ldX >= max(1,m).
 * @param[in]  Xscales       Array of length num_rhs. Xscales[i] is the scaling
 *                           factor associated with the i-th column vector in X.
 * @param[in]  Xinnorms      Array of length num_rhs. Xinnorms[i] is an upper
 *                           bound associated with the i-th column vector in X.
 * @param[in]  lambda_type   Array of length num_rhs. If lambda_type[i] == 0,
 *                           then the i-th column in X respectively Y is real-
 *                           valued. If lambda_type[i] == 1 and
 *                           lambda_type[i+1] == 1, the i-th and (i+1)-th column
 *                           in X respectively Y are the real and imaginary part
 *                           part of a complex-valued vector.
 * */
void update(
    int n, int m,
    const double *restrict const T, int ldT, const double tnorm,
    int num_rhs, omp_lock_t *lock,
    double *restrict const Y, int ldY,
    scaling_t *restrict const Yscales, double *restrict const Ynorms,
    double *restrict const Xin, int ldX,
    const scaling_t *restrict const Xscales, const double *restrict const Xinnorms,
    const int *restrict const lambda_type);

#endif

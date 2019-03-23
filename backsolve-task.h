#ifndef BACKSOLVE_TASK_H
#define BACKSOLVE_TASK_H

#include "typedefs.h"

/**
 * Compute selected eigenvectors of a real upper quasi-triangular matrix T.
 *
 * Computes selected right eigenvectors of a real upper quasi-triangular matrix
 * T. The right eigenvector x of T corresponding to an eigenvalue lambda is
 * defined by
 *
 *   T * x = lambda * x.
 *
 * The eigenvalues are read off from the diagonal of T. The output is the right
 * eigenvectors of T.
 *
 * For a 2-by-2 block on the diagonal of T, corresponding to a complex
 * eigenvalue, only one eigenvector is computed. The second eigenvector can
 * be trivially obtained by the user through complex conjugation.
 *
 * @param[in]  n             The order of the matrix T. n >= 0.
 * @param[in]  T             The upper quasi-triangular matrix T in Schur
 *                           canonical form with dimensions (ldT, n).
 * @param[in]  ldT           The leading dimension of T. ldT >= max(1,n).
 * @param[in]  tnorm         An upper bound for the matrix of T. The value can
 *                           be computed by compute_majorants().
 * @param[out] X             On exit, the right eigenvectors of T. A complex
 *                           eigenvector corresponding to a complex eigenvalue
 *                           is stored in two consecutive columns; the first
 *                           holding the real part and the second the imaginary
 *                           part. The dimensions are (ldX, num_sel), where
 *                           num_sel corresponds to the number of 1's in the
 *                           array selected.
 * @param[in]  ldX           The leading dimension of X. ldX >= max(1,n).
 * @param[out] scales        Array of length num_sel. On exit, the scaling
 *                           factors associated with the eigenvectors columns.
 * @param[out] Xnorms        Array of length num_sel. On exit, the infinity norm
 *                           of all eigenvectors. A complex eigenvalue spanning
 *                           two entries has a duplicate norm entry.
 * @param[in] lambda_type    Array of length n. Describes if T(i,i) is a real
 *                           (lambda_type[i] == 0) or a complex eigenvalue
 *                           (lambda_type[i] == 1).
 * @param[in] selected       Array of length n. Specifies the eigenvectors to be
 *                           computed. If lambda_type[i] is a real eigenvalue,
 *                           the corresponding real eigenvector is computed is
 *                           selected[i] == 1. If lambda_type[i] and
 *                           lambda_type[i + 1] are the real and imaginary parts
 *                           of a complex eigenvalue, the corresponding complex
 *                           eigenvector is computed if selected[i] == 1 and
 *                           selected[i] == 1. For a complex eigenvalue,
 *                           selected[i] == 1 and selected[i + 1] == 0 or
 *                           selected[i] == 0 and selected[i + 1] == 1 is
 *                           an invalid input.
 * */
void backsolve(
    int n,
    const double *restrict const T, int ldT, const double tnorm,
    double *restrict const X, int ldX,
    scaling_t *restrict const scales, double *restrict const Xnorms,
    const int *restrict const lambda_type,
    const int *restrict const selected);

#endif

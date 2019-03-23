#ifndef SOLVE_TASK_H
#define SOLVE_TASK_H

/**
 * Computes the solution for a multi-shift backward substitution.
 *
 * Solves a shifted generalized quasi upper triangular system through
 * backward substitution. The solution x of (T - lambda) is
 * 
 *   (T - lambda) * x = b.
 *
 * The vector b and the shift lambda is an input. The output is the solution x.
 * 
 * @param[in]  n             The order of the matrix T. n >= 0.
 * @param[in]  T             The upper quasi-triangular matrix T in Schur
 *                           canonical form with dimensions (ldT, n).
 * @param[in]  ldT           The leading dimension of T. ldT >= max(1,n).
 * @param[in]  tnorm         An upper bound for the matrix of T. The value can
 *                           be computed by compute_majorants().
 * @param[in]  diag_type     Array of length n. If diag_type[i] == 0, then
 *                           T(i,i) is interpreted as a 1-by-1 block.
 *                           If diag_type[i] == 1 and diag_type[i + 1] == 1,
 *                           then T(i : i + 1, i : i + 1) is interpreted as a
 *                           2-by-2 block.
 * @param[in,out] X          A matrix with dimensions (ldX, num_sel), where
 *                           num_sel corresponds to the number of 1's in
 *                           the array selected. On entry, the vectors b.
 * @param[in]  ldX           The leading dimension of X. ldX >= max(1,n).
 * @param[in]  num_shifts    Length of the arrays lambda, lamba_type, selected.
 * @param[out] scales        Array of length num_sel. On exit, the scaling
 *                           factors associated with the solution vectors.
 * @param[out] Xnorms        Array of length num_sel. On exit, the infinity norm
 *                           of the solution vectors.
 * @param[in]  lambda        Array of length num_shifts. Specifies the shifts.
 *                           If lambda_type[i] == 0, lambda[i] gives the real-
 *                           valued shift. If lambda_type[i] == 1 and
 *                           lambda_type[i + 1] == 1, then lambda[i] gives the
 *                           real part and lambda[i + 1] the imaginary part of
 *                           a complex eigenvalue.
 * @param[in] lambda_type    Array of length num_shifts. If lambda_type[i] == 0,
 *                           lambda[i] encodes a real-valued shift. If
 *                           lambda_type[i] == 1, lambda[i] encodes a part
 *                           (real or imaginary) of a complex eigenvalue.
 * @param[in] selected       Array of length num_shifts. Specifies what shifts
 *                           are used for computing b. If selected[i] == 1, then
 *                           the shift is used.
 * */
void solve(
    int n,
    const double *restrict const T, int ldT, const double tnorm,
    const int *restrict const diag_type,
    double *restrict const X, int ldX,
    int num_shifts,
    scaling_t *restrict const scales, double *restrict const Xnorms,
    const double *restrict const lambda, const int *restrict const lambda_type,
    const int *restrict const selected);

#endif

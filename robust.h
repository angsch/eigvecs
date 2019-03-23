#ifndef ROBUST_H
#define ROBUST_H

#include "typedefs.h"



/**
 * @brief Compute a column-wise scaling such that the matrix difference 
 * Y - X cannot overflow.
 *
 * If scales is of type double, this routine returns a scaling scales(i) such
 * that scales(i) * Y(:,i) - scales(i) * X(:,i) cannot overflow.
 *
 * If the return type is int, this routine returns a scaling scales(i) such
 * that 2^scales(i) * Y(:,i) - 2^scales(i) * X(:,i) cannot overflow.
 *
 * @param[in] m Number of rows of Y, X.
 * @param[in] num_rhs Number of columns of Y, X.
 * @param[in] Y  m-by-num_rhs dense matrix.
 * @param[in] ldY The leading dimension of Y. ldY >= max(1, m).
 * @param[in] X m-by-num_rhs dense matrix.
 * @param[in] ldX The leading dimension of X. ldX >= max(1, m).
 * @param[in] lambda_type  Array of length num_rhs. Describes if the i-th entry is a real
 *                         (lambda_type[i] == 0) or a complex eigenvalue
 *                         (lambda_type[i] == 1).
 * @param[out] scales      Array of length num_rhs. On exit, the scales[i] gives
 *                         the appropriate scaling factor for the i-th column of
 *                         the matrix difference. A complex eigenvalue spanning
 *                         two entries has a duplicate norm entry.
 */
int protect_matrix_difference(
    int m, int num_rhs,
    const double *restrict const Y, int ldY,
    const double *restrict const X, int ldX,
    const int *restrict const lambda_type,
    scaling_t *restrict const scales);




/**
 * @brief Computes scaling such that the update y := y - t x cannot overflow.
 *
 * If the return type is of type double, this routine
 * returns a scaling alpha such that y := (alpha * y) - t * (alpha * x)
 * cannot overflow.
 *
 * If the return type is of type int, this routine
 * returns a scaling alpha such that y := (2^alpha * y) - t * (2^alpha * x)
 * cannot overflow.
 *
 * Assume 0 <= t, x, y <= Omega.
 *
 * Credits: Carl Christian Kjelgaard Mikkelsen.
 */
scaling_t protect_update(double tnorm, double xnorm, double ynorm);





/**
 * @brief Computes scaling such that the update Y(:,i) := Y(:,i) - T X(:,i)
 * cannot overflow.
 *
 * This routine wraps multiple calls to protect_update().
 *
 * @returns Flag that indicates if rescaling is necessary (status == RESCALE)
 *          or not (status == NO_RESCALE) to survive the multi-rhs linear update
 *
 * Credits: Carl Christian Kjelgaard Mikkelsen.
 */
int protect_multi_rhs_update(
    const double *restrict const Xnorms, int num_rhs,
    const double tnorm,
    const double *restrict const Ynorms,
    const int *restrict const lambda_type,
    scaling_t *restrict const scales);





/**
 * @brief Solves (t - lambda) * ? = x robustly.
 *
 * If the type of scale is double, the routine solves (scale * x) / (t - lambda)
 * whereas, if the type of scale is int, the routine solves
 * (2^scale * x) / (t - lambda) such that no overflow occurs.
 *
 * @param[in]      t       Real scalar t.
 * @param[in]      lambda  Real scalar lambda.
 * @param[in, out] x       On entry, the scale rhs. On exit, the real solution x
 *                         in (scale * x) / (t - lambda) respectively
 *                         in (2^scale * x) / (t - lambda)
 * @param[out]     scale   Scalar scaling factor of x.
 */
void solve_1x1_real_system(
    double t, double lambda, double *x, scaling_t *scale);






/**
 * @brief Solves the complex-valued system
 * (t - lambda_re - lambda_im) * ? = x_re + i * x_im robustly.
 *
 * If the type of scale is double, the routine solves (scale * x) / (t - lambda)
 * whereas, if the type of scale is int, the routine solves
 * (2^scale * x) / (t - lambda) such that no overflow occurs. The complex
 * division is executed in real arithmetic.
 *
 * @param[in]      t         Real scalar t.
 * @param[in]      lambda_re Real part of the scalar complex eigenvalue.
 * @param[in]      lambda_im Imaginary part of the scalar complex eigenvalue.
 * @param[in, out] x_re      On entry, the real part of the right-hand side. On
 *                           exit, the real part of the solution.
 * @param[in, out] x_im      On entry, the imaginary part of the rhs. On exit,
 *                           the imaginary part of the solution.
 * @param[out]     scale     Joints scalar scaling factor for the real and
 *                           imaginary part of the solution x.
 */
void solve_1x1_cmplx_system(double t, double lambda_re, double lambda_im,
    double* x_re, double *x_im, scaling_t *scale);






/**
 * @brief Solves a real-valued 2-by-2 system robustly.
 *
 * Solves the real-valued system
 *        [ t11-lambda  t12        ] * [ x1 ] = [ b1 ]
 *        [ t21         t22-lambda ]   [ x2 ]   [ b2 ]
 * such that if cannot overflow.
 *
 * @param[in]      T       Real 2-by-2 matrix T.
 * @param[in]      ldT     The leading dimension of T. ldT >= 2.
 * @param[in]      lambda  Real eigenvalue.
 * @param[in, out] b       Real vector of length 2. On entry, the right-hand
 *                         side. On exit, the solution.
 * @param[out]     scale   Scalar scaling factor of the solution x.
 * 
 * Credits: Carl Christian Kjelgaard Mikkelsen.
 */
void solve_2x2_real_system(
    const double *restrict const T, int ldT,
    double lambda,
    double *restrict const b, scaling_t *restrict const scale);





/**
 * @brief Solves a complex-valued 2-by-2 system robustly.
 *
 * Let lambda := lambda_re + i * lambda_im. Solves the complex-valued system
 *        [ t11-lambda_re   t12        ] * [ x1 ] = [ b_re1 ] + i * [b_im1]
 *        [ t21             t22-lambda ]   [ x2 ]   [ b_re2 ]       [b_im2]
 * such that if cannot overflow. The solution x1 and x2 is complex-valued.
 *
 * @param[in]      T       Real 2-by-2 matrix T.
 * @param[in]      ldT     The leading dimension of T. ldT >= 2.
 * @param[in]      lambda_re  Real part of the eigenvalue.
 * @param[in]      lambda_im Imaginary part of the eigenvalue.
 * @param[in, out] b_re    Vector of length 2. On entry, the real part of the
 *                         rhs. On exit, the real part of the solution.
 * @param[in, out] b       Vector of length 2. On entry, the imaginary part of
 *                         the rhs. On exit, the imaginary part of the solution.
 * @param[out]     scale   Joint scalar scaling factor of the solution x.
 * 
 * Credits: Carl Christian Kjelgaard Mikkelsen.
 */
void solve_2x2_cmplx_system(const double *restrict const T, int ldT,
    double lambda_re, double lambda_im,
    double *restrict const b_re, double *restrict const b_im,
    scaling_t *restrict const scale);



#endif

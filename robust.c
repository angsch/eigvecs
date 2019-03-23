#include "robust.h"
#include "globals.h"
#include "utils.h"
#include "defines.h"

#include <math.h>
#include <stdio.h>
#include <float.h>
#include <math.h>


int MIN_EXP = DBL_MIN_EXP - 1; // -1022
int MAX_EXP = DBL_MAX_EXP - 1; //  1023

////////////////////////////////////////////////////////////////////////////////
// protect real division
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Compute scaling such that the division b / t cannot overflow
 * where b, t are real-valued.
 *
 * If the return type is double-prevision, this routine returns a scaling alpha
 * such that x = (alpha * b) / t cannot overflow.
 *
 * If the return type is int, this routine returns a scaling alpha such that
 * x = (2^alpha * b) / t cannot overflow.
 *
 * Assume |b|, |t| are bounded by Omega.
 *
 * Credits: Carl Christian Kjelgaard Mikkelsen.
 */
static double protect_real_division(double b, double t)
{
    // Initialize scaling factor.
    double scale = 1.0;

    // Find scaling alpha such that x = (alpha * b) / t cannot overflow.
    if (fabs(t) < g_omega_inv) {
        if (fabs(b) > fabs(t) * g_omega) {
            // Please observe that scales will be strictly less than 1.
            scale = (fabs(t) * g_omega) / fabs(b);
        }
    }
    else { // fabs(t) >= g_omega_inv
        // Exploit short circuiting, i.e., the left side is evaluated first.
        // If 1.0 > abs(t) holds, then it is safe to compute
        // fabs(t) * g_omega.
        if (1.0 > fabs(t) && fabs(b) > fabs(t) * g_omega) {
            scale = 1.0 / fabs(b);
        }
    }

    return scale;
}



////////////////////////////////////////////////////////////////////////////////
// protect sum
////////////////////////////////////////////////////////////////////////////////

// Returns scaling such that sum := (alpha * x) + (alpha * y) cannot overflow.
static double protect_sum(double x, double y)
{
    double scale = 1.0;

    // Protect against overflow if x and y have the same sign.
    if ((x > 0 && y > 0) || (x < 0 && y < 0))
        if (fabs(x) > g_omega - fabs(y))
            scale = 0.5;

    return scale;
}


////////////////////////////////////////////////////////////////////////////////
// protect matrix difference
////////////////////////////////////////////////////////////////////////////////

#ifdef INTSCALING
int protect_matrix_difference(
    int m, int num_rhs,
    const double *restrict const Y, int ldY,
    const double *restrict const X, int ldX,
    const int *restrict const lambda_type,
    scaling_t /* == int*/ *restrict const scales)
{
#define Y(i,j) Y[(i) + (j) * ldY]
#define X(i,j) X[(i) + (j) * ldX]

    // Status flag to indicate if rescaling is necessary.
    int status = NO_RESCALE;

    // Workspace.
    double tmp_scales[num_rhs];

    for (int k = num_rhs - 1; k >= 0; k--) {
        // Compute scaling factor for the k-th column.
        tmp_scales[k] = 1.0;

        if (lambda_type[k] == CMPLX) {
            // Find entry in eigenvector spanning the columns k and k - 1 that
            // grows most.
            for (int i = 0; i < m; i++) {
                // Simulate subtraction of imaginary parts.
                double s = protect_sum(Y(i,k), -X(i,k));
                tmp_scales[k] = minf(tmp_scales[k], s);

                // Simulate subtraction of real parts.
                s = protect_sum(Y(i,k - 1), -X(i,k - 1));
                tmp_scales[k] = minf(tmp_scales[k], s);
            }

            if (tmp_scales[k] != 1.0)
                status = RESCALE;

            // We have only one scaling factor per complex conjugate pair.
            tmp_scales[k - 1] = tmp_scales[k];

            // Skip the next entry.
            k--;
        }
        else { // lambda_type[k] == REAL
            // Find entry in k-th column vector that grows most.
            for (int i = 0; i < m; i++) {
                const double s = protect_sum(Y(i,k), -X(i,k));
                tmp_scales[k] = minf(tmp_scales[k], s);
            }

            if (tmp_scales[k] != 1.0)
                status = RESCALE;
        }
    }

    // Convert double-precision scaling factors to int-scaling factors.
    for (int k = num_rhs - 1; k >= 0; k--)
        scales[k] = ilogb(tmp_scales[k]);

#undef Y
#undef X

    return status;
}

#else

// Protect Y - X column-wise.
int protect_matrix_difference(
    int m, int num_rhs,
    const double *restrict const Y, int ldY,
    const double *restrict const X, int ldX,
    const int *restrict const lambda_type,
    scaling_t /* == double*/ *restrict const scales)
{
#define Y(i,j) Y[(i) + (j) * ldY]
#define X(i,j) X[(i) + (j) * ldX]

    // Status flag to indicate if rescaling is necessary.
    int status = NO_RESCALE;

    for (int k = num_rhs - 1; k >= 0; k--) {
        // Compute scaling factor for the k-th column.
        scales[k] = 1.0;

        if (lambda_type[k] == CMPLX) {
            // Find entry in eigenvector spanning the columns k and k - 1 that
            // grows most.
            for (int i = 0; i < m; i++) {
                // Simulate subtraction of imaginary parts.
                double s = protect_sum(Y(i,k), -X(i,k));
                scales[k] = minf(scales[k], s);

                // Simulate subtraction of real parts.
                s = protect_sum(Y(i,k - 1), -X(i,k - 1));
                scales[k] = minf(scales[k], s);
            }

            if (scales[k] != 1.0)
                status = RESCALE;

            // We have only one scaling factor per complex conjugate pair.
            scales[k - 1] = scales[k];

            // Skip the next entry.
            k--;
        }
        else { // lambda_type[k] == REAL
            // Find entry in k-th column vector that grows most.
            for (int i = 0; i < m; i++) {
                const double s = protect_sum(Y(i,k), -X(i,k));
                scales[k] = minf(scales[k], s);
            }

            if (scales[k] != 1.0)
                status = RESCALE;
        }
    }

#undef Y
#undef X

    return status;
}

#endif

////////////////////////////////////////////////////////////////////////////////
// protect multiplication (internal)
////////////////////////////////////////////////////////////////////////////////

// Returns scaling alpha such that y := t * (alpha * x) cannot overflow.
static double protect_mul(double tnorm, double xnorm)
{
    // Initialize scaling factor.
    double scale = 1.0;

    // Process simplified decision tree of protect_update().
    if (fabs(xnorm) <= 1.0) {
        if (fabs(tnorm) * fabs(xnorm) > g_omega) {
            scale = 0.5;
        }
    }
    else { // xnorm > 1.0
        if (fabs(tnorm) > g_omega / fabs(xnorm)) {
            scale = 0.5 / fabs(xnorm);
        }
    }

    return scale;
}


////////////////////////////////////////////////////////////////////////////////
// protect update
////////////////////////////////////////////////////////////////////////////////

#ifdef INTSCALING
scaling_t /* == int*/ protect_update(double tnorm, double xnorm, double ynorm)
{
    // Initialize scaling factor.
    double scale = 1.0;

    // Process decision tree.
    if (xnorm <= 1.0) {
        if (tnorm * xnorm > g_omega - ynorm) {
            scale = 0.5;
        }
    }
    else { // xnorm > 1.0
        if (tnorm > (g_omega - ynorm) / xnorm) {
            scale = 0.5 / xnorm;
        }
    }

    return ilogb(scale);
}

#else

// Returns scaling alpha such that y := (alpha * y) - t * (alpha * x) cannot
// overflow.
scaling_t /* == double*/ protect_update(double tnorm, double xnorm, double ynorm)
{
    // Initialize scaling factor.
    double scale = 1.0;

    // Process decision tree.
    if (xnorm <= 1.0) {
        if (tnorm * xnorm > g_omega - ynorm) {
            scale = 0.5;
        }
    }
    else { // xnorm > 1.0
        if (tnorm > (g_omega - ynorm) / xnorm) {
            scale = 0.5 / xnorm;
        }
    }

    return scale;
}

#endif

////////////////////////////////////////////////////////////////////////////////
// protect update scalar
////////////////////////////////////////////////////////////////////////////////

static double protect_update_scalar(double t, double x, double y)
{
    double scale = 1.0;

    // Protect p = x * y.
    double alpha1 = protect_mul(x, t);
    double p = t * (alpha1 * x);
    if (fabs(ilogb(y) - ilogb(p)) > 52) {
        // The factors are far apart. Either y or p is the final result.
        if (ilogb(p) > ilogb(y))
            scale = alpha1;
    }
    else {
        // Scale y consistently.
        y = y / alpha1;
        double alpha2 = protect_sum(y, -p);
        scale = alpha1 * alpha2;
    }

    return scale;
}



////////////////////////////////////////////////////////////////////////////////
// protect multi-rhs update
////////////////////////////////////////////////////////////////////////////////

#ifdef INTSCALING
int protect_multi_rhs_update(
    const double *restrict const Xnorms, int num_rhs,
    const double tnorm,
    const double *restrict const Ynorms,
    const int *restrict const lambda_type,
    scaling_t /* == int*/ *restrict const scales)
{
    // Status flag to indicate if rescaling is necessary.
    int status = NO_RESCALE;

    for (int k = num_rhs - 1; k >= 0; k--) {
        // Compute scaling factor for the k-th eigenvector.
        scales[k] = protect_update(tnorm, Xnorms[k], Ynorms[k]);

        if (lambda_type[k] == CMPLX) {
            // We have only one scaling factor per complex conjugate pair.
            scales[k - 1] = scales[k];

            // Skip the next entry.
            k--;
        }

        if (scales[k] != 0)
            status = RESCALE;
    }

    return status;
}

#else

int protect_multi_rhs_update(
    const double *restrict const Xnorms, int num_rhs,
    const double tnorm,
    const double *restrict const Ynorms,
    const int *restrict const lambda_type,
    scaling_t /* == double*/ *restrict const scales)
{
    // Status flag to indicate if rescaling is necessary.
    int status = NO_RESCALE;

    for (int k = num_rhs - 1; k >= 0; k--) {
        // Compute scaling factor for the k-th eigenvector.
        scales[k] = protect_update(tnorm, Xnorms[k], Ynorms[k]);

        if (lambda_type[k] == CMPLX) {
            // We have only one scaling factor per complex conjugate pair.
            scales[k - 1] = scales[k];

            // Skip the next entry.
            k--;
        }

        if (scales[k] != 1.0)
            status = RESCALE;
    }

    return status;
}

#endif


////////////////////////////////////////////////////////////////////////////////
// solve 1x1 real system
////////////////////////////////////////////////////////////////////////////////

#ifdef INTSCALING
void solve_1x1_real_system(double t, double lambda, double *x, scaling_t /* == int*/ *scale)
{
    // Compute csr := t + (-lambda) robustly. Note that the scaling contributes
    // as reciprocal to the global scaling.
    double s = protect_sum(t, -lambda);
    double csr = (s * t) - (s * lambda);

    // Compute a scaling to survive the real-valued division.
    double alpha = protect_real_division(x[0], csr);

    // Execute the division safely.
    x[0] = (alpha * x[0]) / csr;

    // Return scaling factor.
    scale[0] = ilogb(alpha / s);
}

#else

/// Solves the real 1x1 system (t - lambda) x = b robustly. 
/// x = x / (t - lambda)
void solve_1x1_real_system(double t, double lambda, double *x, scaling_t /* == double*/ *scale)
{
    // Compute csr := t + (-lambda) robustly. Note that the scaling contributes
    // as reciprocal to the global scaling.
    double s = protect_sum(t, -lambda);
    double csr = (s * t) - (s * lambda);

    // Compute a scaling to survive the real-valued division.
    double alpha = protect_real_division(x[0], csr);

    // Execute the division safely.
    x[0] = (alpha * x[0]) / csr;

    // Return scaling factor.
    scale[0] = alpha / s;
}

#endif


////////////////////////////////////////////////////////////////////////////////
// complex division in real arithmetic
////////////////////////////////////////////////////////////////////////////////

static void dladiv2(double a, double b, double c, double d, double r, double t,
    double *ret, double *scale)
{
    volatile double res;
    double alpha = 1.0;

    if (r != 0.0) {
        // Since r is in [0, 1], the multiplication is safe to execute.
        volatile double br = b * r;

        if (br != 0.0) {
            // res = (a + br) * t
            double s = protect_sum(a, br);
            res = (s * a) + (s * br);
            alpha = s * alpha;

            // WARNING: If optimization flags activate associative math, the
            // brackets in the computation of res is ignored. This problem has
            // been observed with -Ofast (GCC) and -O3 (Intel). The computation
            // overflows and produces NaNs in the solution.
            // The crude fix is as follows:
            // volatile double sres = s * res;
            // res = sres * t;
            s = protect_mul(fabs(t), fabs(res));
            res = (s * res) * t;
            alpha = s * alpha;
        }
        else {
            // res = a * t + (b * t) * r
            // Left term.
            double s1 = protect_mul(fabs(t), fabs(a));
            volatile double tmp1 = (s1 * a) * t;

            // Right term.
            double s2 = protect_mul(fabs(t), fabs(b));
            volatile double tmp2 = (s2 * b) * t;
            // The multiplication with r is safe.
            tmp2 = tmp2 * r;

            // Scale summands consistently.
            double smin = fmin(s1, s2);
            tmp1 = tmp1 * (s1 / smin);
            tmp2 = tmp2 * (s2 / smin);
            alpha = smin * alpha;

            // Add both terms.
            double s = protect_sum(tmp1, tmp2);
            res = (s * tmp1) + (s * tmp2);
            alpha = s * alpha;
        }
    }
    else {
        // res = (a + d * (b / c)) * t
        // tmp = b / c
        double s1 = protect_real_division(b, c);
        alpha = s1 * alpha;
        volatile double tmp = (s1 * b) / c;

        // tmp = d * tmp
        double s2 = protect_mul(fabs(d), fabs(tmp));
        alpha = s2 * alpha;
        tmp = d * (s2 * tmp);

        // Apply scaling to left term 'a' in the sum so that both summands
        // are consistently scaled.
        a = (s1 * s2) * alpha;

        // tmp = a + tmp
        double s = protect_sum(a, tmp);
        alpha = s * alpha;
        tmp = (s * a) + (s * tmp);

        // res = tmp * t
        s = protect_mul(fabs(tmp), fabs(t));
        alpha = s * alpha;
        res = (s * tmp) * t;
    }

    // Return augmented vector (alpha, res).
    *scale = alpha;
    *ret = res;
}



static void dladiv1(double a, double b, double c, double d, 
    double *p, double *q, double *scale)
{
    //           a + ib
    // p + i q = -------
    //           c + id

    // Since |d| < |c|, this division is safe to execute.
    volatile double r = d / c;

    // t = 1 / (c + d * r)
    // Since r is in [0, 1], the multiply is safe.
    volatile double dr = d * r;

    double s1 = protect_sum(c, dr);
    volatile double sum = (s1 * c) + (s1 * dr);

    double s2 = protect_real_division(1.0, sum);
    volatile double t = 1.0 / (s2 * sum);
    volatile double alpha = 1.0 / (s1 * s2);

    // Introduce local scaling factors for dladiv2.
    double beta1 = 1.0, beta2 = 1.0;

    // Compute (beta1, p).
    dladiv2(a, b, c, d, r, t, p, &beta1);

    // Compute (beta2, q).
    dladiv2(b, -a, c, d, r, t, q, &beta2);

    // Scale real and imaginary part consistently.
    double beta = 1.0;
    if ((beta1 > 1.0 && beta2 < 1.0) || (beta1 < 1.0 && beta2 > 1.0)) {
        printf("ERROR: The scalings cannot be consolidated without overflow or underflow.\n");
        // A complex eigenvector has a real part that under/overflowed and an
        // imaginary part that over/underflowed. LAPACK cannot capture this
        // case either (they, too, have only one scaling factor per eigenvector)
    }
    else {
        // Find the more extreme scaling factor.
        beta = fmin(beta1, beta2);

        // Apply scaling.
        *p = (*p) * (beta / beta1);
        *q = (*q) * (beta / beta2);
    }

    // Record global scaling factor.
    *scale = alpha * beta;
}


static void dladiv(double a, double b, double c, double d,
    double *x_re, double *x_im, double *scale)
{
    //                 a + ib
    // x_re + i x_im = -------
    //                 c + id
    if (fabs(d) < fabs(c)) {
        dladiv1(a, b, c, d, x_re, x_im, scale);
    }
    else {
        dladiv1(b, a, d, c, x_re, x_im, scale);
        *x_im = -(*x_im);
    }
}


#ifdef INTSCALING

void solve_1x1_cmplx_system(double t, double lambda_re, double lambda_im,
    double* x_re, double *x_im, scaling_t /* == int*/ *scale)
{
    // Solve (t - (lambda_re + i * lambda_im)) (p + i * q) = x_re + i * x_im.

    // Compute csr := (t + (-lambda_re)) robustly.
    double s = protect_sum(t, -lambda_re);
    double csr = (s * t) - (s * lambda_re);

    // Scale consistently csi := s * (-lambda_im).
    double csi = s * (-lambda_im);

    // Note that the scaling is applied to the rhs (x_re + i * x_im) after
    // the complex division.

    // The check If | C | < SMINI, use C = SMINI in LAPACK is unnecessary
    // because we assume to not have multiplicites in eigenvalues.

    // The scaling check for X = B / C in LAPACK is covered in protect_division.

    // Local scaling factor generated in the process of the complex division.
    double alpha = 1.0;

    // Compute the complex division in real arithmetic.
    //                 a + ib
    // x_re + i x_im = -------
    //                 c + id
    double a = *x_re;
    double b = *x_im;
    double c = csr;
    double d = csi;
    dladiv(a, b, c, d, x_re, x_im, &alpha);

    // Combine scaling factors and convert to int scaling factor.
    *scale = ilogb((1.0 / s) * (alpha));
}

#else

void solve_1x1_cmplx_system(double t, double lambda_re, double lambda_im,
    double* x_re, double *x_im, scaling_t /* == double*/ *scale)
{
    // Solve (t - (lambda_re + i * lambda_im)) (p + i * q) = x_re + i * x_im.

    // Compute csr := (t + (-lambda_re)) robustly.
    double s = protect_sum(t, -lambda_re);
    double csr = (s * t) - (s * lambda_re);

    // Scale consistently csi := s * (-lambda_im).
    double csi = s * (-lambda_im);

    // Note that the scaling is applied to the rhs (x_re + i * x_im) after
    // the complex division.

    // The check If | C | < SMINI, use C = SMINI in LAPACK is unnecessary
    // because we assume to not have multiplicites in eigenvalues.

    // The scaling check for X = B / C in LAPACK is covered in protect_division.

    // Local scaling factor generated in the process of the complex division.
    double alpha = 1.0;

    // Compute the complex division in real arithmetic.
    //                 a + ib
    // x_re + i x_im = -------
    //                 c + id
    double a = *x_re;
    double b = *x_im;
    double c = csr;
    double d = csi;
    dladiv(a, b, c, d, x_re, x_im, &alpha);

    // Combine scaling factors.
    *scale = (1.0 / s) * (alpha);
}

#endif



////////////////////////////////////////////////////////////////////////////////
// solve 2x2 real system
////////////////////////////////////////////////////////////////////////////////


// Credits: Carl Christian Kjelgaard Mikkelsen
static double backsolve_real_2x2_system(double *T, int ldT, double *b)
{
#define T(i,j) T[(i) + (j) * ldT]

    // Global scaling factor.
    double alpha = 1.0;

    double xnorm = max(fabs(b[0]), fabs(b[1]));

    double s = protect_real_division(b[1], T(1,1));
    if (s != 1.0) {
        // Apply scaling to right-hand side.
        b[0] = s * b[0];
        b[1] = s * b[1];

        // Update global scaling.
        alpha = s * alpha;

        // Update the infinity norm of the solution.
        xnorm = s * xnorm;
    }

    // Execute the division.
    b[1] = b[1] / T(1,1);

#ifdef INTSCALING
    s = ldexp(1.0, protect_update(fabs(T(0,1)), fabs(b[1]), xnorm));
#else
    s = protect_update(fabs(T(0,1)), fabs(b[1]), xnorm);
#endif

    if (s != 1.0) {
        // Apply scaling to right-hand side.
        b[0] = s * b[0];
        b[1] = s * b[1];

        // Update global scaling.
        alpha = s * alpha;
    }

    // Execute the linear update.
    b[0] = b[0] - b[1] * T(0,1);

    // Recompute norm.
    xnorm = max(fabs(b[0]), fabs(b[1]));

    s = protect_real_division(b[0], T(0,0));
    if (s != 1.0) {
        // Apply scaling to right-hand side.
        b[0] = s * b[0];
        b[1] = s * b[1];

        // Update global scaling.
        alpha = s * alpha;

        // Update the infinity norm of the solution.
        xnorm = s * xnorm;
    }

    // Execute the division.
    b[0] = b[0] / T(0,0);

    return alpha;

#undef T
}

// Swap row 0 and row 1.
static void swap_rows(int n, double *C)
{
#define C(i,j) C[(i) + (j) * 2]

    // Swap row 0 and row 1.
    for (int j = 0; j < n; j++) {
        double swap = C(0,j);
        C(0,j) = C(1,j);
        C(1,j) = swap;
    }

#undef C
}


static void find_real_pivot(double *C, int *pivot_row, int *pivot_col)
{
#define C(i,j) C[(i) + (j) * 2]

    // Find the coordinates of the pivot element.
    int row = 0;
    int col = 0;
    double cmax = 0.0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            double lmax = fabs(C(i,j));
            if (lmax > cmax) {
                row = i;
                col = j;
                cmax = lmax;
            }
        }
    }

    *pivot_row = row;
    *pivot_col = col;

#undef C
}


// Complete pivoting.
static void solve_2x2_real_system_internal(
    const double *restrict const T, int ldT, 
    double lambda,
    double *restrict const b, double *restrict const scale)
{
#define T(i,j) T[(i) + (j) * ldT]
#define C(i,j) C[(i) + (j) * 2]

    // Solve
    // (T - lambda I)x = b.

    // C = [(T - lambda * I) | b]
    double C[2 * 3];

    // Compute t + (-lambda) robustly. Recall the diagonals in the the 2-by-2
    // T block are equal, so that we have to protect only one subtraction.
    double s = protect_sum(T(0,0), -lambda);
    double csr = (s * T(0,0)) - (s * lambda);

    // Apply scaling to T. Note that scaling of b is not safe. Therefore s is
    // incorporated into the global scaling at the very end of this routine.
    // C := [s * (T - lambda I) | b].
    C(0,0) = csr;         C(0,1) = s * T(0,1);  C(0,2) = b[0];
    C(1,0) = s * T(1,0);  C(1,1) = csr;         C(1,2) = b[1];

    ////////////////////////////////////////////////////////////////////////////
    // Transform A to echelon form with complete pivoting.
    ////////////////////////////////////////////////////////////////////////////

    // Find pivot element in entire matrix.
    int pivot_row = 0, pivot_col = 0;
    find_real_pivot(C, &pivot_row, &pivot_col);

    // Permute pivot to the top-left corner.
    if (pivot_row == 1) {
        // Swap row 0 and row 1.
        swap_rows(3, C);
    }
    if (pivot_col == 1) {
        // Swap column 0 and column 1.
        for (int i = 0; i < 2; i++) {
            double swap = C(i,0);
            C(i,0) = C(i,1);
            C(i,1) = swap;
        }
    }

    // Compute multiplier, the reciprocal of the pivot.
    double ur11r = 1.0 / C(0,0);

    // Multiply first row with reciprocal of C(0,0).
    {
    C(0,0) = 1.0;
    C(0,1) = C(0,1) * ur11r; // Safe multiplication.

    // Treat rhs.
    double beta = protect_mul(C(0,2), ur11r);
    *scale = beta;
    C(0,2) = C(0,2) * beta;
    C(1,2) = C(1,2) * beta;
    C(0,2) = C(0,2) * ur11r;
    }

    // Second row - CR(1,0) * first_row.
    {
    C(1,1) = C(1,1) - C(1,0) * C(0,1); // Safe update.

    // Treat rhs.
    double beta = protect_update_scalar(C(1,0), C(0,2), C(1,2));
    *scale = (*scale) * beta;
    C(0,2) = C(0,2) * beta;
    C(1,2) = C(1,2) * beta;
    C(1,2) = C(1,2) - C(1,0) * C(0,2);

    // (1,0) has been annihilated.
    C(1,0) = 0.0;
    }

    // The system is now in upper triangular form.

    ////////////////////////////////////////////////////////////////////////////
    // Backward substitution.
    ////////////////////////////////////////////////////////////////////////////

    double alpha = backsolve_real_2x2_system(&C(0,0), 2, &C(0,2));
    *scale = (*scale) * alpha;

    // Copy the solution back.
    if (pivot_col == 1) {
        b[0] = C(1,2);
        b[1] = C(0,2);
    }
    else {
        b[0] = C(0,2);
        b[1] = C(1,2);
    }

#undef T
#undef C
}


#ifdef INTSCALING

void solve_2x2_real_system(
    const double *restrict const T, int ldT, 
    double lambda,
    double *restrict const b, int *restrict const scale)
{
    // Local scaling factor.
    double phi = 1.0;

    solve_2x2_real_system_internal(T, ldT, lambda, b, &phi);

    // Convert double-precision scaling factor to int scaling factor.
    *scale = ilogb(phi);
}

#else

void solve_2x2_real_system(
    const double *restrict const T, int ldT, 
    double lambda,
    double *restrict const b, double *restrict const scale)
{
    solve_2x2_real_system_internal(T, ldT, lambda, b, scale);
}

#endif

// Partial pivoting only.
// Credits: Carl Christian Kjelgaard Mikkelsen
/*void solve_2x2_real_system(
    const double *restrict const A, int ldA, double *restrict const b)
{
#define A(i,j) A[(i) + (j) * ldA]
#define C(i,j) C[(i) + (j) * 2]

    // Solve
    // [ a00 a01 ] [x0] = alpha * [b0]
    // [ a10 a11 ] [x1]           [b1].

    double C[2 * 3];

    // C = [A b].
    C(0,0) = A(0,0); C(0,1) = A(0,1); C(0,2) = b[0];
    C(1,0) = A(1,0); C(1,1) = A(1,1); C(1,2) = b[1];

    ////////////////////////////////////////////////////////////////////////////
    // Transform A to triangular form.
    ////////////////////////////////////////////////////////////////////////////

    // Find pivot element in first column.
    if (fabs(C(1,0)) > fabs(C(0,0))) {
        // Pivoting is necessary. Swap row 0 and row 1.
        swap_rows(3, C);
    }

    // Compute multipliers.
    // By virtue of the pivoting all multipliers are bounded by 1
    // Hence overflow is a not an issue when Omega is at least 1.
    C(1,0) = C(1,0) / C(0,0);

    // Do linear update.
    // These scalings do not change the solution.
    // They prevent overflow in the intermediate calculations.
    double tb = fabs(C(1,0));
    double xb = max(fabs(C(0,1)), fabs(C(0,2)));
    double bb = max(fabs(C(1,1)), fabs(C(1,2)));
    // Calculate scaling to prevent overflow in row update.
    double beta = protect_update(tb, xb, bb);

    // Do scaled row update.
    C(1,1) = (beta * C(1,1)) - C(1,0) * (beta * C(0,1));
    C(1,2) = (beta * C(1,2)) - C(1,0) * (beta * C(0,2));
    C(1,0) = 0.0;


    ////////////////////////////////////////////////////////////////////////////
    // Backsolve triangular system.
    ////////////////////////////////////////////////////////////////////////////

    printf("%.6e %.6e %.6e\n", C(0,0), C(0,1), C(0,2));
    printf("%.6e %.6e %.6e\n", C(1,0), C(1,1), C(1,2));

    // Solve
    // [ c00 c01 ] [x0] = alpha * [c02]
    // [   0 c11 ] [x1]           [c12].

    double scale = backsolve_real_2x2_system(C, 2, &C(0,2));

    printf("%.6e %.6e %.6e\n", C(0,0), C(0,1), C(0,2));
    printf("%.6e %.6e %.6e\n", C(1,0), C(1,1), C(1,2));
    printf("%.6e\n", scale);

    // Copy back solution.
    b[0] = C(0,2);
    b[1] = C(1,2);

#undef A
}*/



////////////////////////////////////////////////////////////////////////////////
// solve 2x2 complex system
////////////////////////////////////////////////////////////////////////////////


// Credits: Carl Christian Kjelgaard Mikkelsen
static void find_pivot(double *CR, double *CI, int *pivot_row, int *pivot_col)
{
#define CR(i,j) CR[(i) + (j) * 2]
#define CI(i,j) CI[(i) + (j) * 2]
#define cr(i,j) cr[(i) + (j) * 2]
#define ci(i,j) ci[(i) + (j) * 2]

    double cr[2 * 2];
    double ci[2 * 2];

    // Copy CR, CI.
    cr(0,0) = CR(0,0); cr(0,1) = CR(0,1);
    cr(1,0) = CR(1,0); cr(1,1) = CR(1,1);
    ci(0,0) = CI(0,0); ci(0,1) = CI(0,1);
    ci(1,0) = CI(1,0); ci(1,1) = CI(1,1);

    // Scalings done here are applied only locally.

    // Find smallest scaling factor.
    double smin = 1.0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            double s = protect_sum(fabs(cr(i,j)), fabs(ci(i,j)));
            if (s < smin)
                smin = s;
        }
    }

    // Scale all entries, if necessary.
    if (smin != 1.0) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                cr(i,j) = smin * cr(i,j);
                ci(i,j) = smin * ci(i,j);
            }
        }
    }

    // Now it is safe to find the coordinates of the pivot element.
    int row = 0;
    int col = 0;
    double cmax = 0.0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            double lmax = fabs(cr(i,j)) + fabs(ci(i,j));
            if (lmax > cmax) {
                row = i;
                col = j;
                cmax = lmax;
            }
        }
    }

    *pivot_row = row;
    *pivot_col = col;

#undef CR
#undef CI
#undef cr
#undef ci
}


// Return the more constraining scaling factor.
static double choose_scaling(double alpha, double beta)
{
    if (alpha >= 1.0 && beta >= 1.0) {
        return fmax(alpha, beta);
    }
    else if (alpha <= 1.0 && beta <= 1.0) {
        return fmin(alpha, beta);
    }
    else {
        printf("ERROR: Scaling factors cannot be consolidated\n");
        return 0.0;
    }
}



static void solve_2x2_cmplx_system_internal(const double *restrict const T, int ldT,
    double lambda_re, double lambda_im,
    double *restrict const b_re, double *restrict const b_im,
    double *restrict const scale)
{
#define CR(i,j) CR[(i) + (j) * 2]
#define CI(i,j) CI[(i) + (j) * 2]
#define T(i,j) T[(i) + (j) * ldT]

    // Solve
    // (T - lambda I) x = b.

    // CR = [(T - lambda_re * I) | b_re], CI = [(-lambda_im * I) | b_im].
    double CR[2 * 3];
    double CI[2 * 3];

    // Compute t + (-lambda_re) robustly. Recall the diagonals in the 2-by-2
    // T block are equal, so that we have to protect only one subtraction.
    double s = protect_sum(T(0,0), -lambda_re);
    double csr = (s * T(0,0)) - (s * lambda_re);

    // Apply scaling to T. Note that scaling of b is not safe. Therefore s is
    // incorporated into the global scaling at the very end of this routine.
    // CR + i * CI := s * (T - lambda I).
    CR(0,0) = csr;          CR(0,1) = s * T(0,1);   CR(0,2) = b_re[0];
    CR(1,0) = s * T(1,0);   CR(1,1) = csr;          CR(1,2) = b_re[1];

    CI(0,0) = s * (-lambda_im);CI(0,1) = 0.0;             CI(0,2) = b_im[0];
    CI(1,0) = 0.0;             CI(1,1) = s * (-lambda_im);CI(1,2) = b_im[1];

    ////////////////////////////////////////////////////////////////////////////
    // Transform A to echelon form with complete pivoting.
    ////////////////////////////////////////////////////////////////////////////

    // Find pivot element in entire matrix.
    int pivot_row = 0, pivot_col = 0;
    find_pivot(CR, CI, &pivot_row, &pivot_col);

    // Permute pivot to the top-left corner.
    if (pivot_row == 1) {
        // Swap row 0 and row 1.
        swap_rows(3, CR);
        swap_rows(3, CI);
    }
    if (pivot_col == 1) {
        // Swap column 0 and column 1.
        for (int i = 0; i < 2; i++) {
            double swap = CR(i,0);
            CR(i,0) = CR(i,1);
            CR(i,1) = swap;
            swap = CI(i,0);
            CI(i,0) = CI(i,1);
            CI(i,1) = swap;
        }
    }

    // Recall that (T-lambda I) has form [ a b; c a ]. With pivoting, there
    // are three cases.
    // 1) a is pivot. a is complex-valued, i.e. CI(0,0) != 0 or CI(1,1) != 0.
    // 2) b is pivot. After column pivoting, real values are on the diagonal.
    // 3) c is pivot. After row pivoting, real values are on the diagonal.

    if (CI(0,0) != 0 || CI(1,1) != 0) { // CHECK: CI(1,1)!=0 should be redundant
        // The pivot element is complex. As a consequence, the off-diagonals
        // are real.

        //printf("complex pivot\n");

        // Compute multipliers.
        volatile double temp;
        volatile double ur11r, ui11r;
        // Compute reciprocal of CR(0,0) + i CI(0,0) as
        //         1              CR(0,0) - i CI(0,0)
        // ------------------- = ---------------------
        // CR(0,0) + i CI(0,0)   CR(0,0)^2 + CI(0,0)^2
        if (fabs(CR(0,0)) > fabs(CI(0,0))) {
            //  CR(0,0) - i CI(0,0)         1 - i CI(0,0)/CR(0,0)
            // --------------------- = ----------------------------
            // CR(0,0)^2 + CI(0,0)^2    CR(0,0) + CI(0,0)^2/CR(0,0)
            //
            //      1 - i CI(0,0)/CR(0,0)
            // = -----------------------------------
            //   CR(0,0) * (1 + (CI(0,0)/CR(0,0))^2)
            temp = CI(0,0) / CR(0,0);
            // temp is in [0, 1). Then (1.0 + temp * temp) is in [1, 2).
            // The multiplication cannot overflow (safe mantissa growth).
            // As CR(0,0) is representable, the division is safe, too.
            ur11r = 1.0 / (CR(0,0) * (1.0 + temp * temp) );
            // Safe multiplication because temp is in [0, 1).
            ui11r = -temp * ur11r;
        }
        else { // fabs(CR(0,0)) < fabs(CI(0,0))
            // Note |CR(0,0) < CI(0,0)| (and not <=) because we assume
            // no multiplicites of eigenvalues. The safety of all instructions
            // follows as in the if case.
            temp = CR(0,0) / CI(0,0);
            ui11r = -1.0 / ( CI(0,0)*( 1.0 + temp * temp) );
            ur11r = -temp * ui11r;
        }

        // Multiply first row with reciprocal of CR(0,0) + i CI(0,0).
        {
        CI(0,1) = CR(0,1) * ui11r; // Safe multiplication.
        CR(0,1) = CR(0,1) * ur11r;

        // Treat rhs.
        // Prevent data race.
        temp = CR(0,2);

        // CR(0,2) = CR(0,2) * ur11r - CI(0,2) * ui11r;
        // CI(0,2) = temp * ui11r + CI(0,2) * ur11r;
        // Investigate multiplications and apply most constraining scaling.
        double beta1 = protect_mul(CR(0,2), ur11r);
        double beta2 = protect_mul(CI(0,2), ui11r);
        double beta3 = protect_mul(temp, ui11r);
        double beta4 = protect_mul(CI(0,2), ur11r);
        double beta = fmin(fmin(beta1, beta2), fmin(beta3, beta4));
        *scale = (*scale) * beta;
        CR(0,2) = CR(0,2) * beta; CI(0,2) = CI(0,2) * beta;
        CR(1,2) = CR(1,2) * beta; CI(1,2) = CI(1,2) * beta;
        temp = temp * beta;
        volatile double tmp1 = CR(0,2) * ur11r;
        volatile double tmp2 = - CI(0,2) * ui11r;
        volatile double tmp3 = temp * ui11r;
        volatile double tmp4 = CI(0,2) * ur11r;
        beta = fmin(protect_sum(tmp1, tmp2), protect_sum(tmp3, tmp4));
        *scale = (*scale) * beta;

        CR(0,2) = beta * tmp1 + beta * tmp2;
        CI(0,2) = beta * tmp3 + beta * tmp4;

        // (0,0) has been normalized.
        CR(0,0) = 1.0;
        CI(0,0) = 0.0;
        }

        // Second row - CR(1,0) * first row.
        {
        // Treat rhs.
        // Use more extreme scaling factor.
        double beta1 = protect_update_scalar(CR(1,0), CR(0,2), CR(1,2));
        double beta2 = protect_update_scalar(CR(1,0), CI(0,2), CI(1,2));
        double beta = choose_scaling(beta1, beta2);
        CR(0,2) = CR(0,2) * beta; CI(0,2) = CI(0,2) * beta;
        CR(1,2) = CR(1,2) * beta; CI(1,2) = CI(1,2) * beta;
        *scale = (*scale) * beta;

        CR(1,2) = CR(1,2) - CR(1,0) * CR(0,2);
        CI(1,2) = CI(1,2) - CR(1,0) * CI(0,2);

        // Treat (1,1).
        //   CR11 + i CI11 - CR10 * (CR01 + i CI01)
        // = [CR11 + i CI11] * [1 - CR10 / (CR11 + i CI11) * (CR01 + i CI01)]
        // As (1,0) and (0,1) have opposite signs, cancellation cannot occur.

        // Temporarily use (1,0) to compute CR10 / (CR11 + i CI11).
        // Reuse reciprocal since (1,1) and (0,0) are identical.
        CI(1,0) = CR(1,0) * ui11r;
        CR(1,0) = CR(1,0) * ur11r;

        // Compute 1 - CR10 / (CR11 + i CI11) * (CR01 + i CI01).
        const volatile double tr = 1.0 - CR(1,0) * CR(0,1) + CI(1,0) * CI(0,1);
        const volatile double ti = - CR(1,0) * CI(0,1) - CI(1,0) * CR(0,1);

        // Compute final multiplication with [CR11 + i CI11].
        // Precent data race.
        temp = CR(1,1);
        CR(1,1) = CR(1,1) * tr - CI(1,1) * ti;
        CI(1,1) = CI(1,1) * tr + temp * ti;

        // (1,0) has been annihilated.
        CR(1,0) = 0.0;
        CI(1,0) = 0.0;
        }
    }
    else {
        //printf("real pivot\n");
        // The pivot element is real. The off-diagonals are complex.

        // Multiply first row with 1/CR(0,0). This multiplication is safe for
        // [ CR(0,0)    CR(0,1)+i*CI(0,1) ] when Omega is at least 1.
        {
        CR(0,1) = (1.0 / CR(0,0)) * CR(0,1);
        CI(0,1) = (1.0 / CR(0,0)) * CI(0,1);

        // Threat rhs.
        //CR(0,2) = (1.0 / CR(0,0)) * CR(0,2);
        //CI(0,2) = (1.0 / CR(0,0)) * CI(0,2);
        // Investigate multiplications and apply most constraining scaling.
        double beta1 = protect_mul(fabs(1.0 / CR(0,0)), fabs(CR(0,2)));
        double beta2 = protect_mul(fabs(1.0 / CR(0,0)), fabs(CI(0,2)));
        double beta = fmin(beta1, beta2);
        *scale = (*scale) * beta;
        CR(0,2) = CR(0,2) * beta; CI(0,2) = CI(0,2) * beta;
        CR(1,2) = CR(1,2) * beta; CI(1,2) = CI(1,2) * beta;
        CR(0,2) = (1.0 / CR(0,0)) * CR(0,2);
        CI(0,2) = (1.0 / CR(0,0)) * CI(0,2);

        // Entry (0,0) has been normalized.
        CR(0,0) = 1.0;
        CI(0,0) = 0.0; // may be redundant
        }

        // Eliminate C(1,0): second row - C(1,0) * first row.
        {
        // C11 is real-valued.
        CR(1,1) = CR(1,1) - CR(1,0) * CR(0,1) + CI(1,0) * CI(0,1);
        CI(1,1) = - CI(1,0) * CR(0,1) - CR(1,0) * CI(0,1);

        // Treat rhs. (compare 4 scalings?)
        //CR(1,2) = CR(1,2) - CR(1,0) * CR(0,2) + CI(1,0) * CI(0,2);
        //CI(1,2) = CI(1,2) - CI(1,0) * CR(0,2) - CR(1,0) * CI(0,2);
        // Investigate multiplications and apply most constraining scaling.
        double beta1 = protect_mul(CR(1,0), CI(0,2));
        double beta2 = protect_mul(CI(1,0), CI(0,2));
        double beta3 = protect_mul(CI(1,0), CR(0,2));
        double beta4 = protect_mul(CR(1,0), CI(0,2));
        double beta = fmin(fmin(beta1, beta2), fmin(beta3, beta4));
        *scale = (*scale) * beta;
        CR(0,2) = CR(0,2) * beta; CI(0,2) = CI(0,2) * beta;
        CR(1,2) = CR(1,2) * beta; CI(1,2) = CI(1,2) * beta;
        double max1 = fmax(fabs(CR(1,2)), fabs(CR(1,0) * CR(0,2)));
        double max2 = fmax(fabs(CI(1,0) * CI(0,2)), fabs(CI(1,2)));
        double max3 = fmax(fabs(CI(1,0) * CR(0,2)), fabs(CR(1,0) * CI(0,2)));
        max1 = fmax(max1, max2);
        max1 = fmax(max1, max3);
        beta = protect_sum(max1, 2 * max1);
        *scale = (*scale) * beta;
        CR(0,2) = CR(0,2) * beta; CI(0,2) = CI(0,2) * beta;
        CR(1,2) = CR(1,2) * beta; CI(1,2) = CI(1,2) * beta;
        CR(1,2) = CR(1,2) - CR(1,0) * CR(0,2) + CI(1,0) * CI(0,2);
        CI(1,2) = CI(1,2) - CI(1,0) * CR(0,2) - CR(1,0) * CI(0,2);
        }

    }

    // The system is now in upper triangular form.

    ////////////////////////////////////////////////////////////////////////////
    // Backward substitution.
    ////////////////////////////////////////////////////////////////////////////
    // up to here the scaling factor is ok. After division crap.

    double xr1, xi1, xr2, xi2;
    double beta = 1.0;
    dladiv(CR(1,2), CI(1,2), CR(1,1), CI(1,1), &xr2, &xi2, &beta);

    *scale = (*scale) * beta;
    xr1 = CR(0,2);
    xi1 = CI(0,2);
    xr1 = xr1 - CR(0,1) * xr2 + CI(0,1) * xi2;
    xi1 = xi1 - CI(0,1) * xr2 - CR(0,1) * xi2;

    if (pivot_col == 1) {
        b_re[0] = xr2; b_im[0] = xi2;
        b_re[1] = xr1; b_im[1] = xi1;
    }
    else {
        b_re[0] = xr1; b_im[0] = xi1;
        b_re[1] = xr2; b_im[1] = xi2;
    }


#undef CR
#undef CI
#undef T
}

#ifdef INTSCALING

void solve_2x2_cmplx_system(const double *restrict const T, int ldT,
    double lambda_re, double lambda_im,
    double *restrict const b_re, double *restrict const b_im,
    int *restrict const scale)
{
    // Local scaling factor.
    double phi = 1.0;

    solve_2x2_cmplx_system_internal(
        T, ldT, lambda_re, lambda_im, b_re, b_im, &phi);

    // Convert double-precision scaling factor to int scaling factor.
    *scale = ilogb(phi);
}

#else

void solve_2x2_cmplx_system(const double *restrict const T, int ldT,
    double lambda_re, double lambda_im,
    double *restrict const b_re, double *restrict const b_im,
    double *restrict const scale)
{
    solve_2x2_cmplx_system_internal(
        T, ldT, lambda_re, lambda_im, b_re, b_im, scale);
}

#endif

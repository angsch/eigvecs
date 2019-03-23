#include "eigenvectors.h"
#include "utils.h"
#include "random.h"
#include "defines.h"
#include "partition.h"
#include "timing.h"
#include "IO.h"
#include "robust.h"
#include "norm.h"
#include "backtransform-task.h"
#include "backsolve-task.h"
#include "solve-task.h"

#include <stdlib.h>
#include <stdio.h>
#include <mm_malloc.h>
#include <string.h>
#include <math.h>

#include <omp.h>




// diag_type: 1x1 or 2x2 blocks on the diagonal of T
void solve(
    int n,
    const double *restrict const T, int ldT, const double tnorm,
    const int *restrict const diag_type,
    double *restrict const X, int ldX,
    int num_rhs,
    scaling_t *restrict const scales, double *restrict const Xnorms,
    const double *restrict const lambda, const int *restrict const lambda_type,
    const int *restrict const selected)
{
#define T(i,j) T[(i) + (j) * ldT]
#ifndef NDEBUG
    printf("solve: thread id = %d\n", omp_get_thread_num());
#endif

    int num_selected = count_selected(num_rhs, selected);

    // The i-th selected eigenvalue.
    int si = num_selected - 1;

    // Loop over eigenvalues.
    for (int k = num_rhs - 1; k >= 0; k--) {
        if (!selected[k]) {
            // Proceed with the next eigenvalue.
            if (lambda_type[k] == CMPLX) {
                // A complex conjugate pair of eigenvalues is not selected, so skip the next diagonal entry.
                k--;
            }
        }
        else { // k-th eigenvalue is selected
            if (lambda_type[k] == REAL) {
                // Locate eigenvector to k-th eigenvalue.
                double *X_re = X + si * ldX;

                // Locate corresponding scaling factor.
                scaling_t *beta = scales + si;

                // Compute norm of entire vector.
                double norm = vector_infnorm(n, X_re); // CHECK: Is this an *input* or does it have to be computed?
                // norm should be an input (recomputed after linear update)
                // double norm = Xnorms[si];
 
                // Loop over columns of T.
                for (int j = n - 1; j >= 0; j--) {
                    // if next block is 1-by-1 diagonal block:
                    if (diag_type[j] == REAL) {
                        scaling_t phi;
                        init_scaling_factor(1, &phi);
                        solve_1x1_real_system(T(j,j), lambda[k], X_re + j, &phi);

                        // Scale remaining parts of vector.
                        scale(j, X_re, &phi);
                        scale(n - (j + 1), X_re + (j + 1), &phi);

                        // Update global scaling factor.
                        update_global_scaling(beta, phi);

                        // Update norm of vector.
                        update_norm(&norm, phi);

                        // Protect against overflow in the linear update.
                        phi = protect_update(tnorm, fabs(X_re[j]), norm);

                        // Apply the scaling to the whole eigenvector.
                        scale(n, X_re, &phi);

                        // Update global scaling factor.
                        update_global_scaling(beta, phi);

                        // Now it is safe to execute the linear update.
                        for (int i = 0; i < j; i++) {
                            X_re[i] = X_re[i] - T(i,j) * X_re[j];
                        }

                        // Recompute norm without current last entry j.
                        norm = vector_infnorm(j, X_re);
                    }
                    else {
                        // if next block is 2-by-2 block:
                        scaling_t phi;
                        init_scaling_factor(1, &phi);
                        solve_2x2_real_system(&T(j-1,j-1), ldT, lambda[k], &X_re[j-1], &phi);

                        // Scale remaining parts of vector.
                        scale(j - 1, X_re, &phi);
                        scale(n - (j + 1), X_re + (j + 1), &phi);

                        // Update global scaling factor.
                        update_global_scaling(beta, phi);

                        // Update norm.
                        update_norm(&norm, phi);

                        // Update.
                        //for (int i = 0; i < j - 1; i++) {
                        //    X_re[i] = X_re[i] - T(i,j-1) * X_re[j-1];
                        //    X_re[i] = X_re[i] - T(i,j)   * X_re[j];
                        //}

                        // Protect against overflow in the first linear update.
                        phi = protect_update(tnorm, fabs(X_re[j-1]), norm);

                        // Apply the scaling to the whole eigenvector.
                        scale(n, X_re, &phi);

                        // Update global scaling.
                        update_global_scaling(beta, phi);

                        // Now it is safe to execute the first linear udpate.
                        for (int i = 0; i < j - 1; i++)
                            X_re[i] = X_re[i] - T(i,j-1) * X_re[j-1];

                        // Recompute norm.
                        norm = vector_infnorm(j, X_re);

                        // Protect against overflow in the second linear update.
                        phi = protect_update(tnorm, fabs(X_re[j]), norm);

                        // Apply the scaling to the whole eigenvector.
                        scale(n, X_re, &phi);

                        // Update global scaling.
                        update_global_scaling(beta, phi);

                        // Now it is safe to execute the second lienar update.
                        for (int i = 0; i < j - 1; i++)
                            X_re[i] = X_re[i] - T(i,j)   * X_re[j];

                        // Recompute norm without current last entry j - 1.
                        norm = vector_infnorm(j - 1, X_re);

                        // We processed a 2-by-2 block, so skip the next diagonal entry.
                        j--;
                    }
                }

                // The k-th real eigenvector has been computed. Recmopute norm.
                Xnorms[si] = vector_infnorm(n, X_re);

                // This eigenvalue spans 1 column. Update selected counter.
                si--;
            }
            else {
                // lambda_type[k] == CMPLX

                // Locate real and imaginary part of complex eigenvector for complex
                // conjugate pair of eigenvalues (k, k-1).
                double *X_re = X + (si - 1) * ldX;
                double *X_im = X + si * ldX;

                // Extract the eigenvalue as lambda = lambda_re + i * lambda_im.
                double lambda_re = lambda[k - 1];
                double lambda_im = fabs(lambda[k]); // fabs not necessary

                // Locate corresponding scaling factor.
                scaling_t *beta = scales + si;

                // Compute norm of entire vector.
                double norm = vector_cmplx_infnorm(n, X_re, X_im);

                // Loop over columns of T.
                for (int j = n - 1; j >= 0; j--) {

                    // if the next block is 1-by-1 diagonal block:
                    if (diag_type[j] == REAL) {
                        // Update si-th column. Column (si-1) is computed later.
                        scaling_t phi;
                        init_scaling_factor(1, &phi);
                        solve_1x1_cmplx_system(T(j,j), lambda_re, lambda_im, X_re + j, X_im + j, &phi); // X_im overflows in this routine.

                        // Scale the remaining parts of the vector.
                        scale(j, X_re, &phi);
                        scale(n - (j + 1), X_re + (j + 1), &phi);
                        scale(j, X_im, &phi);
                        scale(n - (j + 1), X_im + (j + 1), &phi);

                        // Update global scaling factor.
                        update_global_scaling(beta, phi);

                        // Update norm.
                        update_norm(&norm, phi);

                        // Update.
                        //for (int i = 0; i < j; i++) {
                        //    X_re[i] = X_re[i] - T(i,j) * X_re[j];
                        //    X_im[i] = X_im[i] - T(i,j) * X_im[j];
                        //}

                        // Protect against overflow in the linear update.
                        double absmax = maxf(fabs(X_re[j]), fabs(X_im[j]));
                        phi = protect_update(tnorm, absmax, norm);

                        // Apply scaling to the whole eigenvector.
                        scale(n, X_re, &phi);
                        scale(n, X_im, &phi);

                        // Update global scaling factor.
                        update_global_scaling(beta, phi);

                        // Now it is safe to execute the linear update.
                        for (int i = 0; i < j; i++) {
                            X_re[i] = X_re[i] - T(i,j) * X_re[j];
                            X_im[i] = X_im[i] - T(i,j) * X_im[j];
                        }

                        // Recompute norm without current last entry h.
                        norm = vector_cmplx_infnorm(j, X_re, X_im);
                    }
                    else {
                        // if next block is 2-by-2 diagonal block:

                        // Only si-th column is used. (si+1) is computed later.
                        scaling_t phi;
                        init_scaling_factor(1, &phi);
                        solve_2x2_cmplx_system(&T(j-1,j-1), ldT, lambda_re, lambda_im, X_re + j - 1, X_im + j - 1, &phi);

                        // Scale remaining parts of vector.
                        scale(j - 1, X_re, &phi);
                        scale(n - (j + 1), X_re + (j + 1), &phi);
                        scale(j - 1, X_im, &phi);
                        scale(n - (j + 1), X_im + (j + 1), &phi);

                        // Update global scaling factor.
                        update_global_scaling(beta, phi);

                        // Update norm of vector.
                        update_norm(&norm, phi);

                        // Update.
                        //for (int i = 0; i < j - 1; i++) {
                        //    X_re[i] = X_re[i] - T(i,j-1) * X_re[j-1];
                        //    X_im[i] = X_im[i] - T(i,j-1) * X_im[j-1];
                        //    X_re[i] = X_re[i] - T(i,j)   * X_re[j];
                        //    X_im[i] = X_im[i] - T(i,j)   * X_im[j];
                        //}

                        // Protect against overflow in the first linear update.
                        double absmax = maxf(fabs(X_re[j-1]), fabs(X_im[j-1]));
                        phi = protect_update(tnorm, absmax, norm);

                        // Apply scaling to the whole eigenvector.
                        scale(n, X_re, &phi);
                        scale(n, X_im, &phi);

                        // Update global scaling factor.
                        update_global_scaling(beta, phi);

                        // Now it is safe to execute the first linear update.
                        for (int i = 0; i < j - 1; i++) {
                            X_re[i] = X_re[i] - T(i,j-1) * X_re[j-1];
                            X_im[i] = X_im[i] - T(i,j-1) * X_im[j-1];
                        }

                        // Recompute norm.
                        norm = vector_cmplx_infnorm(j + 1, X_re, X_im);

                        // Protect against overflow in the second linear update.
                        absmax = maxf(fabs(X_re[j]), fabs(X_im[j]));
                        phi = protect_update(tnorm, absmax, norm);

                        // Apply scaling to the whole eigenvector.
                        scale(n, X_re, &phi);
                        scale(n, X_im, &phi);

                        // Update global scaling factor.
                        update_global_scaling(beta, phi);

                        // Now it is safe to execute the second linear update.
                        for (int i = 0; i < j - 1; i++) {
                            X_re[i] = X_re[i] - T(i,j)   * X_re[j];
                            X_im[i] = X_im[i] - T(i,j)   * X_im[j];
                        }

                        // Recompute norm.
                        norm = vector_cmplx_infnorm(j - 1, X_re, X_im);

                        // We processed a 2-by-2 block, so skip the next diagonal entry.
                        j--;
                    }
                }

                // Note that the second column of a complex conjugate pair is
                // never allocated or computed. Obtaining it is left to the
                // user. If the positions si - 1, si mark a 2-by-2 block, then
                // the eigenvector corresponding to lambda = alpha + i beta
                // is X(:, si - 1) + i * X(:, si).
                // The complex conjugate eigenvector corresponding to
                // lambda = alpha - i beta can be derived as
                // conj(X) := X(:, si -1) - i * X(:, si).

                // Copy global scaling factor.
                scales[si - 1] = scales[si];

                // The ki-th complex eigenvector has been computed. Recompute
                // and record norm for real and imaginary part.
                Xnorms[si] = vector_cmplx_infnorm(n, X_re, X_im);
                Xnorms[si - 1] = Xnorms[si];

                // We processed a complex conjugate pair of eigenvalues, so skip the next eigenvalue entry.
                k--;

                // This eigenvalue spans 2 columns. Update selected counter.
                si -= 2;
            }
        }
    }

#undef T
}


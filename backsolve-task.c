#include "backsolve-task.h"
#include "eigenvectors.h"
#include "utils.h"
#include "random.h"
#include "defines.h"
#include "partition.h"
#include "timing.h"
#include "IO.h"
#include "robust.h"
#include "norm.h"

#include <stdlib.h>
#include <stdio.h>
#include <mm_malloc.h>
#include <string.h>
#include <math.h>

#include <omp.h>



void backsolve(
    int n,
    const double *restrict const T, int ldT, const double tnorm,
    double *restrict const X, int ldX,
    scaling_t *restrict const scales, double *restrict const Xnorms,
    const int *restrict const lambda_type,
    const int *restrict const selected)
{
#define T(i,j) T[(i) + (j) * ldT]

#ifndef NDEBUG
    printf("backsolve: thread id = %d, n = %d \n", omp_get_thread_num(), n);
#endif
    // dtrevc computes the eigenvectors in a workspace and then copies it to
    // the final location. We omit the workspace and directly write to the final
    // location. Hence, the location of X, X_re, X_im is related to si, the
    // counter for selected eigenvalues, and not ki, the counter over all eigen-
    // values.

    int num_selected = count_selected(n, selected);

    // The i-th selected eigenvalue.
    int si = num_selected - 1;

    // Loop over eigenvalues from bottom to top.
    for (int ki = n - 1; ki >= 0; ki--) {
        if (!selected[ki]) {
            // Proceed with the next eigenvalue.
            if (lambda_type[ki] == CMPLX) {
                // A complex conjugate pair of eigenvalues is not selected, so skip the next diagonal entry.
                ki--;
            }
        }
        else { // ki-th eigenvalue is selected
            if (lambda_type[ki] == REAL) {
                // Compute a real right eigenvector.
                double lambda = T(ki,ki);

                // Locate eigenvector to ki-th eigenvalue.
                double *X_re = X + si * ldX;

                // Locate corresponding scaling factor.
                scaling_t *beta = scales + si;

                // Form right-hand side and compute norm.
                X_re[ki] = 1.0;
                for (int i = 0; i < ki; i++) {
                    X_re[i] = -T(i, ki);
                }

                // Compute norm of entire vector.
                double norm = vector_infnorm(ki + 1, X_re);

                // Solve the upper quasi-triangular system.
                // (T(0:ki-1,0:ki-1) - lambda I) \ X.

                // Loop over triangular matrix above the eigenvalue.
                for (int j = ki - 1; j >= 0; j--) {

                    // if next block is 1-by-1 diagonal block:
                    if (lambda_type[j] == REAL) {
                        scaling_t phi;
                        init_scaling_factor(1, &phi);
                        solve_1x1_real_system(T(j,j), lambda, X_re + j, &phi);

                        // Scale remaining parts of the vector.
                        scale(j, X_re, &phi);
                        scale(ki - (j + 1), X_re + (j + 1), &phi);

                        // Update global scaling factor.
                        update_global_scaling(beta, phi);

                        // Update norm of vector.
                        update_norm(&norm, phi);
                        // Protect against overflow in the linear update.
                        phi = protect_update(tnorm, fabs(X_re[j]), norm);

                        // Apply the scaling to the whole eigenvector.
                        scale(ki + 1, X_re, &phi);

                        // Update global scaling factor.
                        update_global_scaling(beta, phi);

                        // Now it is safe to execute the linear update.
                        for (int i = 0; i < j; i++)
                            X_re[i] = X_re[i] - T(i,j) * X_re[j];

                        // Recompute norm without current last entry j.
                        norm = vector_infnorm(j, X_re);
                    }
                    else {
                        // if next block is 2-by-2 diagonal block:
                        scaling_t phi;
                        init_scaling_factor(1, &phi);
                        solve_2x2_real_system(&T(j-1,j-1), ldT, lambda, &X_re[j-1], &phi);

                        // Scale remaining parts of vector.
                        scale(j - 1, X_re, &phi);
                        scale(ki - (j + 1), X_re + (j + 1), &phi);

                        // Update global scaling factor.
                        update_global_scaling(beta, phi);

                        // Update norm of vector.
                        update_norm(&norm, phi);

                        // Update.
                        //for (int i = 0; i < j - 1; i++) {
                        //    X_re[i] = X_re[i] - T(i,j-1) * X_re[j-1];
                        //    X_re[i] = X_re[i] - T(i,j)   * X_re[j];
                        //}

                        // Protect against overflow in the first linear update.
                        phi = protect_update(tnorm, fabs(X_re[j-1]), norm);

                        // Apply the scaling to the whole eigenvector.
                        scale(ki + 1, X_re, &phi);

                        // Update global scaling factor.
                        update_global_scaling(beta, phi);

                        // Now it is safe to execute the first linear update.
                        for (int i = 0; i < j - 1; i++)
                            X_re[i] = X_re[i] - T(i,j-1) * X_re[j-1];

                        // Recompute norm.
                        norm = vector_infnorm(j, X_re);

                        // Protect against overflow in the second linear update.
                        phi = protect_update(tnorm, fabs(X_re[j]), norm);

                        // Apply the scaling to the whole eigenvector.
                        scale(ki + 1, X_re, &phi);

                        // Update global scaling factor.
                        update_global_scaling(beta, phi);

                        // Now it is safe to execute the second linear update.
                        for (int i = 0; i < j - 1; i++)
                            X_re[i] = X_re[i] - T(i,j) * X_re[j];

                        // Recompute norm without current last entry j - 1.
                        norm = vector_infnorm(j - 1, X_re);

                        // We processed a 2-by-2 block, so skip the next diagonal entry.
                        j--;
                    }
                }

                // The ki-th real eigenvector has been computed. Recompute norm.
                Xnorms[si] = vector_infnorm(ki + 1, X_re);

                // This eigenvector spans 1 column. Update selected counter.
                si--;
            }
            else {
                // lambda_type[ki] == COMPLEX

                // Locate real and imaginary part of complex eigenvector for complex
                // conjugate pair of eigenvalues (ki, ki-1).
                double *X_re = X + (si - 1) * ldX;
                double *X_im = X + si * ldX;

                // Locate corresponding scaling factor.
                scaling_t *beta = scales + si;

                // Compute the eigenvalue as lambda = lambda_re + i * lambda_im.
                // A 2-by-2 block in canonical Schur form has the shape
                // [ T(ki-1,ki-1) T(ki-1,ki) ] = [ a b ]  or [ a -b ]
                // [ T(ki, ki-1)  T(ki, ki)  ]   [-c a ]     [ c  a ].
                // To avoid overflow, apply sqrt before the multiplication.
                double lambda_re = T(ki,ki);
                double lambda_im = sqrt(fabs(T(ki,ki-1))) * sqrt(fabs(T(ki-1,ki)));

                // Form right-hand side.
                if (fabs(T(ki-1,ki)) >= fabs(T(ki,ki-1))) {
                    X_re[ki-1] = 1.0;
                    X_im[ki] = lambda_im / T(ki-1,ki);
                }
                else {
                    X_re[ki-1] = -lambda_im / T(ki,ki-1);
                    X_im[ki] = 1.0;
                }
                X_re[ki] = 0.0;
                X_im[ki-1] = 0.0;
                for (int i = 0; i < ki - 1; i++) {
                    X_re[i] = -X_re[ki-1] * T(i,ki-1);
                    X_im[i] = -X_im[ki] * T(i,ki);
                }

                // Compute norm of entire vector.
                double norm = vector_cmplx_infnorm(ki + 1, X_re, X_im);

                // Solve the upper quasi-triangular system.
                // (T(0:ki-2,0:ki-2) - (lambda_re + i * lambda_im) I) \ (X_re + i * X_im).

                // Loop over triangular matrix above the eigenvalue pair. Note ki-2!
                for (int j = ki - 2; j >= 0; j--) {

                    // If next block is 1-by-1 diagonal bock:
                    if (lambda_type[j] == REAL) {
                        scaling_t phi;
                        init_scaling_factor(1, &phi);
                        solve_1x1_cmplx_system(T(j,j), lambda_re, lambda_im, X_re + j, X_im + j, &phi);

                        // Scale the remaining parts of the vector.
                        scale(j, X_re, &phi);
                        scale(ki - (j + 1), X_re + (j + 1), &phi);
                        scale(j, X_im, &phi);
                        scale(ki - (j + 1), X_im + (j + 1), &phi);

                        // Update global scaling factor.
                        update_global_scaling(beta, phi);

                        // Update norm of vector.
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
                        scale(ki + 1, X_re, &phi);
                        scale(ki + 1, X_im, &phi);

                        // Update global scaling factor.
                        update_global_scaling(beta, phi);

                        // Now it is safe to execute the linear update.
                        for (int i = 0; i < j; i++) {
                            X_re[i] = X_re[i] - T(i,j) * X_re[j];
                            X_im[i] = X_im[i] - T(i,j) * X_im[j];
                        }

                        // Recompute norm without current last entry j.
                        norm = vector_cmplx_infnorm(j, X_re, X_im);
                    }
                    else {
                        // If next block is 2-by-2 diagonal block:
                        scaling_t phi;
                        init_scaling_factor(1, &phi);
                        solve_2x2_cmplx_system(&T(j-1,j-1), ldT, lambda_re, lambda_im, X_re + j - 1, X_im + j - 1, &phi);

                        // Scale remaining parts of vector.
                        scale(j - 1, X_re, &phi);
                        scale(ki - j, X_re + (j + 1), &phi);
                        scale(j - 1, X_im, &phi);
                        scale(ki - j, X_im + (j + 1), &phi);

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
                        scale(ki + 1, X_re, &phi);
                        scale(ki + 1, X_im, &phi);

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
                        scale(ki + 1, X_re, &phi);
                        scale(ki + 1, X_im, &phi);

                        // Update global scaling factor.
                        update_global_scaling(beta, phi);

                        // Now it is safe to execute the second linear update.
                        for (int i = 0; i < j - 1; i++) {
                            X_re[i] = X_re[i] - T(i,j) * X_re[j];
                            X_im[i] = X_im[i] - T(i,j) * X_im[j];
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
                Xnorms[si] = vector_cmplx_infnorm(ki + 1, X_re, X_im);
                Xnorms[si - 1] = Xnorms[si];

                // We processed a complex conjugate pair of eigenvalues, so skip the next diagonal entry.
                ki--;

                // This eigenvector spans 2 columns. Update selected counter.
                si -= 2;
            }
        }
    }
#undef T
}

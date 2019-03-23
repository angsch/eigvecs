#include "reference.h"
#include "datalayout.h"
#include "partition.h"
#include "IO.h"
#include "utils.h"
#include "defines.h"
#include "timing.h"

#include <stdio.h>
#include <string.h>
#include <math.h>


void dtrevc(
    const char side, const char howmny,
    int *select, int n, double *T, int ldT,
    double *VR, int ldVR, int mm)
{
    extern void dtrevc_(
        const char *side, const char *howmny,
        const int *select, const int *n, const double *T, const int *ldT,
        double *VL, const int *ldVL, double *VR, const int *ldVR,
        const int *mm, int *m,
        double *work, int *info);

    double *work = (double *) malloc(3 * n * sizeof(double));
    int info;
    int m;
    // Dummy variables. Will not be referenced.
    double *VL = NULL; int ldVL = 1;

    dtrevc_(&side, &howmny,
            select, &n, T, &ldT, VL, &ldVL, VR, &ldVR, &mm, &m, work, &info);

    free(work);
}

/*
void dtrevc3(
        const char side, const char howmny,
        const int *select, int n, double *T, int ldT,
        double *VR, int ldVR, int mm)
{
    extern void dtrevc3_(
        const char *side, const char *howmny,
        const int *select, const int *n, const double *T, const int *ldT,
        double *VL, const int *ldVL, double *VR, const int *ldVR,
        const int *mm, int *m,
        double *work, const int *lwork, int *info);

    int m;
    double worklength;
    int lwork;
    int info;

    // Dummy variables. Will not be referenced.
    double *VL = NULL; int ldVL = 1;

    // Query optimal workspace.
    lwork = -1;
    dtrevc3_(&side, &howmny, select, &n, T, &ldT, VL, &ldVL, 
             VR, &ldVR, &mm, &m, &worklength, &lwork, &info);
    lwork = worklength;

    // Allocate workspace.
    double *work = (double *) malloc(lwork * sizeof(double));

    // Compute eigenvectors.
    dtrevc3_(&side, &howmny, select, &n, T, &ldT, VL, &ldVL, 
             VR, &ldVR, &mm, &m, work, &lwork, &info);

    // Free workspace.
    free(work);
}
*/



// Compute || X_re ||^2 where || . || is the Frobenius norm.
static double real_vector_frobeniusnorm2(int n, const double *restrict const X)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += X[i] * X[i];
    }

    return sum;
}


// Compute || X_re + i * X_im ||^2 where || . || is the Frobenius norm.
static double cmplx_vector_frobeniusnorm2(
    int n, const double *restrict const X_re, const double *restrict const X_im)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += X_re[i] * X_re[i] + X_im[i] * X_im[i];
    }

    return sum;
}


void compute_eigenvectors_dtrevc3(
    int n, double ***restrict const T_blocks, int ldT,
    double *restrict const lambda, const int *restrict const lambda_type,
    const int *restrict const selected,
    const partitioning_t *restrict const p,
    const partitioning_t *restrict const p_rhs,
    double ***restrict const Q_blocks, int ldQ,
    memory_layout_t mem_layout)
{
    int num_blks = p->num_blks;
    const int *first_row = p->first_row;
    const int *first_col = p->first_col;
    const int *first_rhs_col = p_rhs->first_col;

    int num_selected = count_selected(n, selected);

    double *T, *Q, *X, *Y;
    int ldX;

    double tm_start, tm_end;

    // Compact memory representation of the eigenvalues.
    double *selected_lambda = (double *) malloc(num_selected * sizeof(double));
    int *selected_lambda_type = (int *) malloc(num_selected * sizeof(int));

    // Copy all selected eigenvalues to a compact memory representation.
    compact_eigenvalues(n, selected, lambda, lambda_type,
        selected_lambda, selected_lambda_type);

    // LAPACK overwrites selected. Take a copy.
    int *selected_lapack = (int *) malloc(n * sizeof(int));
    memcpy(selected_lapack, selected, n * sizeof(int));

    scaling_t *sscales = (scaling_t *) malloc(num_selected * sizeof(scaling_t));
    double *xxnorms = (double *) malloc(num_selected * sizeof(double));
    for (int i = 0; i < num_selected; i++)
        sscales[i] = 1.0;

    // Allocate workspace.
    double *A = (double *) malloc((size_t)n * n * sizeof(double));
    double *W = (double *) malloc((size_t)n * n * sizeof(double));

    if (mem_layout == COLUMN_MAJOR) {
        // Let T and Q point at the base address of their tiled counterparts.
        T = T_blocks[0][0];
        Q = Q_blocks[0][0];

        // Allocate space for the solution.
        ldX = ldQ;
        X = (double *) malloc((size_t)ldX * n * sizeof(double));
        Y = (double *) malloc((size_t)n * n * sizeof(double));

        // Save a copy of Q.
        memcpy(X, Q, (size_t)ldX * n * sizeof(double));

        // Compute solution with LAPACK.
        tm_start = get_time();
        if (num_selected == n) {
            tm_start = get_time();
            // Compute all right eigenvectors and backtransform them.
            dtrevc('R', 'B', selected_lapack, n, T, ldT, X, ldX, n);
            tm_end = get_time();

            // For validation copy Y := X.
            memcpy(Y, X, (size_t) ldX * n * sizeof(double));
        } else {
            tm_start = get_time();
            // Compute a few selected right eigenvectors of T.
            dtrevc('R', 'S', selected_lapack, n, T, ldT, X, ldX, n);

            // Compute Y := Q * X exploiting the structure of X.
            // Loop over block columns in Y.
            for (int j = 0; j < num_blks; j++) {
                // Loop over block rows in Y.
                for (int i = 0; i < num_blks; i++) {
                    // Compute the actual number of rows and column.
                    int num_rows = first_row[i + 1] - first_row[i];
                    int num_cols = first_rhs_col[j + 1] - first_rhs_col[j];
                    int num_inner = first_row[j + 1] - first_row[0];

                    // Yij := Qi: * X:j.
                    dgemm('N', 'N', num_rows, num_cols, num_inner,
                          1.0, Q + first_row[i], n,
                          X + first_rhs_col[j] * ldX, ldX,
                          0.0, Y + first_row[i] + first_rhs_col[j] * ldX, ldX);
                }
            }
            tm_end = get_time();
        }

        printf("...LAPACK time = %.2f s.\n", tm_end - tm_start);

        // Compute W := T * Q^T.
        dgemm('N', 'T', n, n, n, 1.0, T, ldT, Q, ldQ, 0.0, W, n);

        // Compute A := Q * W.
        dgemm('N', 'N', n, n, n, 1.0, Q, ldQ, W, n, 0.0, A, n);

        // Compute W := A * Y.
        dgemm('N', 'N', n, n, n, 1.0, A, n, Y, n, 0.0, W, n);
    }
    else { //         TILE_LAYOUT
        // Convert T to column major layout.
        T = (double *) malloc((size_t)n * n * sizeof(double));
        convert_to_column_major(T, n, T_blocks, first_row, first_col, num_blks);

        // Convert Q to column major layout.
        Q = (double *) malloc((size_t)n * n * sizeof(double));
        convert_to_column_major(Q, n, Q_blocks, first_row, first_col, num_blks);

        // Allocate space for the solution.
        ldX = n;
        X = (double *) malloc((size_t)ldX * n * sizeof(double));
        Y = (double *) malloc((size_t)ldX * n * sizeof(double));

        // Save a copy of Q.
        memcpy(X, Q, (size_t)n * n * sizeof(double));

        // Compute solution with LAPACK.
        if (num_selected == n) {
            tm_start = get_time();
            // Compute all right eigenvectors and backtransform them.
            dtrevc('R', 'B', selected_lapack, n, T, n, X, n, n);
            tm_end = get_time();

            // For validation copy Y := X.
            memcpy(Y, X, (size_t)n * n * sizeof(double));
        }
        else {
            tm_start = get_time();
            // Compute a few selected right eigenvectors of T.
            dtrevc('R', 'S', selected_lapack, n, T, n, X, n, n);

            // Compute Y := Q * X exploiting the structure of X.
            // Loop over block columns in Y.
            for (int j = 0; j < num_blks; j++) {
                // Loop over block rows in Y.
                for (int i = 0; i < num_blks; i++) {
                    // Compute the actual number of rows and column.
                    int num_rows = first_row[i + 1] - first_row[i];
                    int num_cols = first_rhs_col[j + 1] - first_rhs_col[j];
                    int num_inner = first_row[j + 1] - first_row[0];

                    // Yij := Qi: * X:j.
                    dgemm('N', 'N', num_rows, num_cols, num_inner,
                          1.0, Q + first_row[i], n,
                          X + first_rhs_col[j] * ldX, ldX,
                          0.0, Y + first_row[i] + first_rhs_col[j] * ldX, ldX);
                }
            }

            tm_end = get_time();
        }

        printf("LAPACK time = %.2f s.\n", tm_end - tm_start);

#ifndef NDEBUG
        printf("LAPACK Y = \n");
        print(n, num_selected, Y, n);
#endif

        // Compute W := T * Q^T.
        dgemm('N', 'T', n, n, n, 1.0, T, n, Q, n, 0.0, W, n);

        // Compute A := Q * W.
        dgemm('N', 'N', n, n, n, 1.0, Q, n, W, n, 0.0, A, n);

        // Compute W := A * Y.
        dgemm('N', 'N', n, n, n, 1.0, A, n, Y, n, 0.0, W, n);

        free(T);
        free(Q);
    }

    // Compute W := W - lambda * Y.
    for (int j = 0; j < num_selected; j++) {
        if (selected_lambda_type[j] == REAL) {
            double lambda_re = selected_lambda[j];

            // Subtract columns.
            for (int i = 0; i < n; i++) {
                W[i + j * n] = W[i + j * n] - lambda_re * Y[i + j * n];
            }
        }
        else if (selected_lambda_type[j] == CMPLX) {
            double lambda_re = selected_lambda[j];
            double lambda_im = selected_lambda[j + 1];

            // Locate real and imaginary part of complex eigenvector.
            double *Y_re = Y + j * n;
            double *Y_im = Y + (j + 1) * n;
            double *W_re = W + j * n;
            double *W_im = W + (j + 1) * n;
            
            // Subtract columns.
            for (int i = 0; i < n; i++) {
                W_re[i] = W_re[i] - lambda_re * Y_re[i] + lambda_im * Y_im[i];
                W_im[i] = W_im[i] - lambda_re * Y_im[i] - lambda_im * Y_re[i];
            }

            // We processed a complex conjugate pair of eigenvalues. Skip
            // the next column.
            j++;
        }
    }

    // Compute || W ||_F.
    double normW = 0.0;
    for (int j = 0; j < num_selected; j++) {
        if (selected_lambda_type[j] == REAL) {
            normW += real_vector_frobeniusnorm2(n, W + j * n);
        }
        else if (selected_lambda_type[j] == CMPLX) {
            double *W_re = W + j * n;
            double *W_im = W + (j + 1) * n;
            normW += cmplx_vector_frobeniusnorm2(n, W_re, W_im);
            j++;
        }
    }
    normW = sqrt(normW);

    printf("...||A y - lambda y||_F = %.2e\n", normW);

    // Clean up.
    free(X);
    free(W);
    free(A);
    free(Y);
    free(selected_lambda);
    free(selected_lambda_type);
    free(selected_lapack);
    free(sscales);
    free(xxnorms);
}

#include "validation.h"
#include "utils.h"
#include "partition.h"
#include "defines.h"
#include "norm.h"
#include "IO.h"

#include <mm_malloc.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "omp.h"


/** Compute A  :=   Q   *   T   *   Q^T.
 *       (n x n) (n x n) (n x n) (n x n) */
static void compute_input_matrix(int n,
    double ***restrict const T_blocks, int ldT,
    double ***restrict const Q_blocks, int ldQ,
    double ***restrict const A_blocks, int ldA,
    const partitioning_t *restrict const partitioning,
    memory_layout_t layout)
{
    int num_blks = partitioning->num_blks;
    const int *first_row = partitioning->first_row;
    const int *first_col = partitioning->first_col;

    // Allocate workspace.
    int ldW = n;
    double *W = (double *) _mm_malloc((size_t)ldW * n * sizeof(double), ALIGNMENT);

    double ***W_blocks = malloc(num_blks * sizeof(double **));
    for (int i = 0; i < num_blks; i++) {
        W_blocks[i] = malloc(num_blks * sizeof(double *));
    }

    partition_matrix(W, ldW, layout, partitioning, W_blocks);


    // Compute W := T * Q^T.
    // dgemm('N', 'T', n, n, n, 1.0, T, ldT, Q, ldQ, 0.0, W, ldW);
    #pragma omp parallel
    #pragma omp single nowait
    {
        for (int i = 0; i < num_blks; i++) {
            for (int j = 0; j < num_blks; j++) {
                // Compute Wij := sum_{k} Tik * Qjk^T.
                #pragma omp task shared(W,W_blocks)
                {
                    // Compute the actual number of rows and columns.
                    const int num_rows = first_row[i + 1] - first_row[i];
                    const int num_cols = first_col[j + 1] - first_col[j];

                    // Initialize Wij with zeros.
                    if (layout == COLUMN_MAJOR)
                        set_zero(num_rows, num_cols, W_blocks[i][j], ldW);
                    else //       TILE_LAYOUT
                        set_zero(num_rows, num_cols, W_blocks[i][j], num_rows);

                    // Compute matrix product.
                    for (int k = 0; k < num_blks; k++) {

                        // Compute inner dimension = cols of T/rows of Q.
                        const int num_inner = first_col[k + 1] - first_col[k];

                        // Wij := Wij + Tik * Qjk^T.
                        if (layout == COLUMN_MAJOR)
                            dgemm('N', 'T', num_rows, num_cols, num_inner, 
                                  1.0, T_blocks[i][k], ldT, Q_blocks[j][k], ldQ,
                                  1.0, W_blocks[i][j], ldW);
                        else //       TILE_LAYOUT
                            dgemm('N', 'T', num_rows, num_cols, num_inner, 
                                  1.0, T_blocks[i][k], num_rows, Q_blocks[j][k], num_cols,
                                  1.0, W_blocks[i][j], num_rows);
                    }
                }
            }
        }
    }

    // Compute A := Q * W.
    // dgemm('N', 'N', n, n, n, 1.0, Q, ldQ, W, ldQ, 0.0, A, ldA);
    #pragma omp parallel
    #pragma omp single nowait
    {
        for (int i = 0; i < num_blks; i++) {
            for (int j = 0; j < num_blks; j++) {
                // Compute Aij := sum_{k} Qik * Wkj.
                #pragma omp task shared(W,W_blocks)
                {
                    // Compute the actual number of rows and columns.
                    const int num_rows = first_row[i + 1] - first_row[i];
                    const int num_cols = first_col[j + 1] - first_col[j];

                    // Initialize Aij with zeros.
                    if (layout == COLUMN_MAJOR)
                        set_zero(num_rows, num_cols, A_blocks[i][j], ldA);
                    else //       TILE_LAYOUT
                        set_zero(num_rows, num_cols, A_blocks[i][j], num_rows);

                    // Compute matrix product.
                    for (int k = 0; k < num_blks; k++) {
                        // Compute inner dimension = cols of Q/rows of W.
                        const int num_inner = first_row[k + 1] - first_row[k];

                        // Aij := Aij + Qik * Wkj.
                        if (layout == COLUMN_MAJOR)
                            dgemm('N', 'N', num_rows, num_cols, num_inner,
                                  1.0, Q_blocks[i][k], ldQ, W_blocks[k][j], ldW,
                                  1.0, A_blocks[i][j], ldA);
                        else //       TILE_LAYOUT
                            dgemm('N', 'N', num_rows, num_cols, num_inner,
                                  1.0, Q_blocks[i][k], num_rows, W_blocks[k][j], num_inner,
                                  1.0, A_blocks[i][j], num_rows);
                    }
                }
            }
        }
    }

    // Free workspace.
    _mm_free(W);
    for (int i = 0; i < num_blks; i++) {
        free(W_blocks[i]);
    }
    free(W_blocks);
}


static void compute_tiled_gemv(double ***restrict const A_blocks, int ldA,
    double ***restrict const X_blocks, int ldX,
    double ***restrict const Y_blocks, int ldY,
    int num_blks,
    const int *restrict const first_row, const int *restrict const first_col,
    memory_layout_t layout)
{
    // Compute Y := A * X in tile format.
    // Loop over block columns in Y, X.
    #pragma omp parallel
    #pragma omp single nowait
    for (int j = 0; j < num_blks; j++) {
        // Loop over block rows in Y.
        for (int i = 0; i < num_blks; i++) {
            #pragma omp task
            {
                // Compute the actual number of rows and columns.
                const int num_rows = first_row[i + 1] - first_row[i];
                const int num_cols = first_col[j + 1] - first_col[j];

                // Set Yij to zero.
                if (layout == COLUMN_MAJOR)
                    set_zero(num_rows, num_cols, Y_blocks[i][j], ldY);
                else //       TILE_LAYOUT
                    set_zero(num_rows, num_cols, Y_blocks[i][j], num_rows);

                // Compute Yij := Yij + sum_{k} (Aik * Xkj)
                for (int k = 0; k < num_blks; k++) {
                    // Compute the actual number of columns.
                    const int num_inner = first_row[k + 1] - first_row[k];

                    // Yij := Yij + Aik * Xkj.
                    if (layout == COLUMN_MAJOR) {
                        dgemm('N', 'N', num_rows, num_cols, num_inner,
                              1.0, A_blocks[i][k], ldA,
                              X_blocks[k][j], ldX,
                              1.0, Y_blocks[i][j], ldY);
                    }
                    else { //     TILE_LAYOUT
                        dgemm('N', 'N', num_rows, num_cols, num_inner,
                              1.0, A_blocks[i][k], num_rows,
                              X_blocks[k][j], num_inner,
                              1.0, Y_blocks[i][j], num_rows);
                    }
                }
            }
        }
    }
}



static void subtract_tiled_matrices(
    double ***restrict const X_blocks, int ldX,
    double ***restrict const Y_blocks, int ldY,
    const double *restrict const lambda,
    const int *restrict lambda_type,
    int num_blks,
    const int *restrict const first_row, const int *restrict const first_col,
    memory_layout_t layout)
{
    // Y := Y - alpha * X.
    for (int blkj = 0; blkj < num_blks; blkj++) {
        for (int blki = 0; blki < num_blks; blki++) {
            // Locate base pointer of current blocks.
            double *X = X_blocks[blki][blkj];
            double *Y = Y_blocks[blki][blkj];

            // Compute actual number of rows and columns.
            const int num_rows = first_row[blki + 1] - first_row[blki];
            const int num_cols = first_col[blkj + 1] - first_col[blkj];

            // Compute the global column index.
            const int col = first_col[blkj];

            // Compute within block Yij := Yij - lambdaj Xij.
            if (layout == COLUMN_MAJOR) {
                for (int j = 0; j < num_cols; j++) {
                    if (lambda_type[col + j] == REAL) {
                        const double lambda_re = lambda[col + j];
                        for (int i = 0; i < num_rows; i++) {
                            Y[i + ldY * j] -= lambda_re * X[i + ldX * j];
                        }
                    }
                    else if (lambda_type[col + j] == CMPLX) {
                        double lambda_re = lambda[col + j];
                        double lambda_im = lambda[col + j + 1];

                        // Locate real and imaginary part of complex eigenvector.
                        double *X_re = X + j * ldX;
                        double *X_im = X + (j + 1) * ldX;
                        double *Y_re = Y + j * ldY;
                        double *Y_im = Y + (j + 1) * ldY;

                        // Y := Y - lambda * X.
                        for (int i = 0; i < num_rows; i++) {
                            // Let lambdaj = lambda_re + i * lambda_im.
                            // Let X = X_re + i * X_im. Then:
                            // lambdaj * X
                            //   =    lambda_re * X_re
                            //      - lambda_im * X_im
                            //      + i * lambda_re * X_im
                            //      + i * lambda_im * X_re
                            //
                            // Together Y := Y - lambdaj * X:
                            Y_re[i] = Y_re[i] 
                                - lambda_re * X_re[i]
                                + lambda_im * X_im[i];
                            Y_im[i] = Y_im[i]
                                - lambda_re * X_im[i]
                                - lambda_im * X_re[i];

                        }

                        // Note that the second column of a complex conjugate
                        // pair is never allocated or computed. Hence,
                        // lambda = lambda_re - i * lambda_im is never checked.

                        // We processed a 2-by-2 block. The first column
                        // contained the real part; the second the complex part.
                        // Skip the next column.
                        j++;
                    }
                }
            }
            else { //     TILE_LAYOUT
                for (int j = 0; j < num_cols; j++) {
                    if (lambda_type[col + j] == REAL) {
                        const double lambda_re = lambda[col + j];
                        for (int i = 0; i < num_rows; i++) {
                            Y[i + num_rows * j] -= 
                                lambda_re * X[i + num_rows * j];
                        }
                    }
                    else if (lambda_type[col + j] == CMPLX) {
                        double lambda_re = lambda[col + j];
                        double lambda_im = lambda[col + j + 1];

                        // Locate real and imaginary part of complex eigenvector.
                        double *X_re = X + j * num_rows;
                        double *X_im = X + (j + 1) * num_rows;
                        double *Y_re = Y + j * num_rows;
                        double *Y_im = Y + (j + 1) * num_rows;

                        // Y := Y - lambda * X.
                        for (int i = 0; i < num_rows; i++) {
                            // Let lambdaj = lambda_re + i * lambda_im.
                            // Let X = X_re + i * X_im. Then:
                            // lambdaj * X
                            //   =    lambda_re * X_re
                            //      - lambda_im * X_im
                            //      + i * lambda_re * X_im
                            //      + i * lambda_im * X_re
                            //
                            // Together Y := Y - lambdaj * X:
                            Y_re[i] = Y_re[i]
                                - lambda_re * X_re[i]
                                + lambda_im * X_im[i];
                            Y_im[i] = Y_im[i]
                                - lambda_re * X_im[i]
                                - lambda_im * X_re[i];
                        }

                        // Note that the second column of a complex conjugate
                        // pair is never allocated or computed. Hence,
                        // lambda = lambda_re - i * lambda_im is never checked.

                        // We processed a 2-by-2 block. The first column
                        // contained the real part; the second the complex part.
                        // Skip the next column.
                        j++;
                    }
                }
            }
        }
    }
}




void validate(int n, 
    double ***restrict const T_blocks, int ldT,
    double ***restrict const Q_blocks, int ldQ,
    double ***restrict const X_blocks, int ldX,
    double *restrict const lambda, const int *restrict const lambda_type,
    const int *restrict const selected,
    const partitioning_t *restrict const partitioning,
    const partitioning_t *restrict const partitioning_rhs,
    memory_layout_t layout)
{
    int num_blks = partitioning->num_blks;
    const int *first_row = partitioning->first_row;
    const int *first_col = partitioning->first_col;
    const int *first_rhs_col = partitioning_rhs->first_col;

    int ldA = n;
    double *A;
    A = (double *) _mm_malloc((size_t)ldA * n * sizeof(double), ALIGNMENT);

    double ***A_blocks = malloc(num_blks * sizeof(double **));
    for (int i = 0; i < num_blks; i++) {
        A_blocks[i] = malloc(num_blks * sizeof(double *));
    }

    // Partition A analogously to T and Q.
    partition_matrix(A, ldA, layout, partitioning, A_blocks);

    printf("...Compute A := Q * T * Q^T\n");
    compute_input_matrix(
        n, T_blocks, ldT, 
        Q_blocks, ldQ, 
        A_blocks, ldA,
        partitioning, layout);

#ifndef NDEBUG
    if (layout == COLUMN_MAJOR) {
        printf("A = \n");
        print(n, n, A, ldA);
    }
    else {
        double AA[ldA*n];
        convert_to_column_major(AA, ldA, A_blocks, first_row, first_col, num_blks);
        printf("A in column major\n");
        print(n, n, AA, ldA);
    }
#endif

    printf("...Verify eigenvectors A * x = lambda * x\n");

    int num_selected = count_selected(n, selected);

    // Allocate workspace.
    double *selected_lambda = (double *) malloc(num_selected * sizeof(double));
    int *selected_lambda_type = (int *) malloc(num_selected * sizeof(int));
    int ldY = ldX;
    double *Y = (double *) _mm_malloc((size_t) ldY * num_selected * sizeof(double), ALIGNMENT);
    double ***Y_blocks = malloc(num_blks * sizeof(double **));
    for (int i = 0; i < num_blks; i++) {
        Y_blocks[i] = malloc(num_blks * sizeof(double *));
    }

    partition_matrix(Y, ldY, layout, partitioning_rhs, Y_blocks);

    // Copy all selected eigenvalues to a compact memory representation.
    int idx = 0;
    for (int i = 0; i < n; i++) {
        if (selected[i]) {
            selected_lambda[idx] = lambda[i];
            selected_lambda_type[idx] = lambda_type[i];
            idx++;
        }
    }

    // Y := A * X
    compute_tiled_gemv(A_blocks, ldA,
        X_blocks, ldX,
        Y_blocks, ldY,
        num_blks, first_row, first_rhs_col, layout);


    #ifndef NDEBUG
    if (layout == COLUMN_MAJOR) {
        printf("Y = \n");
        print(n, num_selected, Y_blocks[0][0], ldY);
    }
    else {
        double YY[ldY*n];
        convert_to_column_major(YY, ldY, Y_blocks, first_row, first_rhs_col, num_blks);
        printf("Y in column major\n");
        print(n, num_selected, YY, ldY);
    }
    #endif

    // Y := Y - lambda * X.
    subtract_tiled_matrices(
        X_blocks, ldX,
        Y_blocks, ldY,
        selected_lambda, selected_lambda_type,
        num_blks, first_row, first_rhs_col, layout);

    double normA = real_matrix_frobeniusnorm(A_blocks, ldA, num_blks, num_blks,
        first_row, first_col, layout);
    double normY = matrix_frobeniusnorm(Y_blocks, ldY, selected_lambda_type,
        num_blks, num_blks, first_row, first_rhs_col, layout);
    printf("...||A y -lambda y||_F = %.2e\n", normY);
    printf("...||A y -lambda y||_F / ||A||_F = %.2e\n", normY / normA);


    #ifndef NDEBUG
    if (layout == COLUMN_MAJOR) {
        printf("Y = \n");
        print(n, num_selected, Y_blocks[0][0], ldY);
    }
    else {
        double YY[ldY*n];
        convert_to_column_major(YY, ldY, Y_blocks, first_row, first_rhs_col, num_blks);
        printf("Y in column major\n");
        print(n, num_selected, YY, ldY);
    }
    #endif

    // Free workspace.
    _mm_free(Y);
    free(selected_lambda);
    free(selected_lambda_type);

    // Clean up.
    _mm_free(A);
    for (int i = 0; i < num_blks; i++) {
        free(A_blocks[i]);
        free(Y_blocks[i]);
    }
    free(A_blocks);
    free(Y_blocks);
}

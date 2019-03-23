#include "norm.h"
#include "defines.h"
#include "utils.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>


// Credits: Björn Adlerborn
double vector_infnorm(int n, const double *x)
{
    double norm = 0;
    for (int i = 0; i < n; ++i) {
        double abs =  fabs(x[i]);
        if (abs > norm) {
            norm = abs;
        }
    }
    return norm;
}


double vector_cmplx_infnorm(int n, const double *x_re, const double *x_im)
{
    double norm = 0;
    for (int i = 0; i < n; ++i) {
        // Compute len = sqrt(x_re[i] * x_re[i] + x_im[i] * x_im[i]) robustly.
        double maxabs = maxf(fabs(x_re[i]), fabs(x_im[i]));
        double len = maxabs * sqrt(  (x_re[i] / maxabs) * (x_re[i] / maxabs)
                                   + (x_im[i] / maxabs) * (x_im[i] / maxabs));
        if (len > norm) {
            norm = len;
        }
    }
    return norm;
}


double vector_2norm(int n, const double *x)
{
    double norm = 0;
    for (int i = 0; i < n; ++i) {
        norm += x[i] * x[i];
    }
    return sqrt(norm);
}


// Credits: Björn Adlerborn, slightly modified.
double matrix_infnorm(int n, int m, const double *A, int ldA)
{
#define A(i,j) A[(i) + (j) * ldA]

    double *rowsums = calloc(n, sizeof(double));

    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; i++) {
            rowsums[i] += fabs(A(i,j)); 
        }
    }

    double norm = rowsums[0];
    for (int i = 1; i < n; i++) {
        if (rowsums[i] > norm) {
            norm = rowsums[i];
        }
    }

    free(rowsums);
    return norm;

#undef A
}


double matrix_tile_layout_infnorm(double ***A_blocks,
    int num_row_blks, int num_col_blks,
    const int *restrict const first_row,
    const int *restrict const first_col)
{
    // Find the row count of the matrix.
    const int n = first_row[num_row_blks];

    double *rowsums = calloc(n, sizeof(double));

    for (int j = 0; j < num_col_blks; j++) {
        for (int i = 0; i < num_row_blks; i++) {
            // Find block properties.
            double *A = A_blocks[i][j];
            const int num_rows = first_row[i + 1] - first_row[i];
            const int num_cols = first_col[j + 1] - first_col[j];

            // Loop over entries in block.
            for (int jj = 0; jj < num_cols; jj++) {
                for (int ii = 0; ii < num_rows; ii++) {
                    const int row = first_row[i] + ii;
                    rowsums[row] += fabs(A[ii + num_rows * jj]);
                }
            }
        }
    }

    double norm = rowsums[0];
    for (int i = 1; i < n; i++) {
        if (rowsums[i] > norm) {
            norm = rowsums[i];
        }
    }

    free(rowsums);
    return norm;

}


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



double real_matrix_frobeniusnorm(
    double ***X_blocks, int ldX,
    int num_row_blks, int num_col_blks,
    const int *restrict const first_row, const int *restrict const first_col,
    memory_layout_t layout)
{
    double norm = 0.0;

    for (int blkj = 0; blkj < num_col_blks; blkj++) {
        for (int blki = 0; blki < num_row_blks; blki++) {
            // Locate base pointer of current block.
            double *X = X_blocks[blki][blkj];

            // Compute actual number of rows and columns.
            const int m = first_row[blki + 1] - first_row[blki];
            const int n = first_col[blkj + 1] - first_col[blkj];

            // Compute squared Frobenius norm within block Xij.
            for (int j = 0; j < n; j++) {
                if (layout == COLUMN_MAJOR) {
                    norm += real_vector_frobeniusnorm2(m, X + j * ldX);
                }
                else { //     TILE_LAYOUT
                    norm += real_vector_frobeniusnorm2(m, X + j * m);
                }
            }
        }
    }

    return sqrt(norm);
}


double matrix_frobeniusnorm(
    double ***X_blocks, int ldX,
    const int *restrict const lambda_type,
    int num_row_blks, int num_col_blks,
    const int *restrict const first_row, const int *restrict const first_col,
    memory_layout_t layout)
{
    double norm = 0.0;

    for (int blkj = 0; blkj < num_col_blks; blkj++) {
        for (int blki = 0; blki < num_row_blks; blki++) {
            // Locate base pointer of current block.
            double *X = X_blocks[blki][blkj];

            // Compute actual number of rows and columns.
            const int m = first_row[blki + 1] - first_row[blki];
            const int n = first_col[blkj + 1] - first_col[blkj];

            // Compute squared Frobenius norm within block Xij.
            for (int j = 0; j < n; j++) {
                // Compute the global column index.
                const int col = first_col[blkj] + j;

                if (lambda_type[col] == REAL) {
                    if (layout == COLUMN_MAJOR) {
                        norm += real_vector_frobeniusnorm2(m, X + j * ldX);
                    }
                    else { //     TILE_LAYOUT
                        norm += real_vector_frobeniusnorm2(m, X + j * m);
                    }
                }
                else if (lambda_type[col] == CMPLX) {
                    if (layout == COLUMN_MAJOR) {
                        double *X_re = X + j * ldX;
                        double *X_im = X + (j + 1) * ldX;
                        norm += cmplx_vector_frobeniusnorm2(m, X_re, X_im);
                    }
                    else { //     TILE_LAYOUT
                        double *X_re = X + j * m;
                        double *X_im = X + (j + 1) * m;
                        norm += cmplx_vector_frobeniusnorm2(m, X_re, X_im);
                    }

                    // We processed a 2-by-2 block. The first column
                    // contained the real part; the second the complex part.
                    // Skip the next column.
                    j++;
                }
            }
        }
    }

    return sqrt(norm);
}

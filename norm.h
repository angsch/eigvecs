#ifndef NORM_H
#define NORM_H

#include "datalayout.h"

double vector_infnorm(int n, const double *x);
double vector_cmplx_infnorm(int n, const double *x_re, const double *x_im);
double vector_2norm(int n, const double *x);
double matrix_infnorm(int n, int m, const double *A, int ldA);
double matrix_tile_layout_infnorm(double ***A_blocks,
    int num_row_blks, int num_col_blks,
    const int *restrict const first_row,
    const int *restrict const first_col);


/// Compute the Frobenius norm of a real-valued matrix.
double real_matrix_frobeniusnorm(
    double ***X_blocks, int ldX,
    int num_row_blks, int num_col_blks,
    const int *restrict const first_row, const int *restrict const first_col,
    memory_layout_t layout);


/// Compute the Frobenius norm of a complex-valued matrix.
double matrix_frobeniusnorm(
    double ***X_blocks, int ldX,
    const int *restrict const lambda_type,
    int num_row_blks, int num_col_blks,
    const int *restrict const first_row, const int *restrict const first_col,
    memory_layout_t layout);

#endif

#include "random.h"
#include "norm.h"
#include "utils.h"
#include "defines.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#ifdef DISTRIBUTED
#include "mpi.h"
#endif


int random_integer (int low, int high)
{
    int size = high - low + 1;
    double x = random_double (0, 1);
    int k = (int) (x * size);
    if (k == size) {
        --k;
    }
    return low + k;
}


double random_double (double low, double high)
{
    double x = (double) rand () / RAND_MAX;
    return low + x * (high - low);
}


static void generate_diagonal_householder_block(
    int n, int ld, double *const H, const double *restrict const v)
{
#define H(i,j) H[(i) + (j) * ld]

    // H := I - 2 * v * v^T
    for(int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (i == j) {
                H(i,j) = 1.0 - 2.0 * v[i] * v[j];
            }
            else {
                H(i,j) = -2.0 * v[i] * v[j];
            }
        }
    }

#undef H
}

static void generate_offdiagonal_householder_block(
    int n, int m, int ld, double *const Hij, const double *restrict const vi,
    const double *restrict const vj)
{
#define Hij(i,j) Hij[(i) + (j) * ld]

    // Hij := -2 * vi * vj^T
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            Hij(i,j) = -2.0 * vi[i] * vj[j];
        }
    }

#undef Hij
}


// Credits: original implementation by Mirko Myllykoski
void generate_householder_matrix(
    int n,
    int ld, double ***H_blocks, int num_blks,
    const int *restrict const first_row, const int *restrict const first_col,
    memory_layout_t mem_layout)
{
    // Create v.
    double *v = (double *) malloc(n * sizeof(double));
    
    for (int i = 0; i < n; i++)
        v[i] = 2.0 * (1.0 * rand() / RAND_MAX) - 1.0;

    // Normalize vector.
    double norm = vector_2norm(n, v);
    for (int i = 0; i < n; i++) {
        v[i] = v[i] / norm;
    }
    

    // Compute H := I - 2 * v * v^T.
    #pragma omp parallel
    #pragma omp single nowait
    for (int i = 0; i < num_blks; i++) {
        for (int j = 0; j < num_blks; j++) {
            if (i == j) {
                // Compute Householder block Hii.
                #pragma omp task shared(H_blocks)
                {
                    // Locate subvector vi.
                    const double *vi = v + first_row[i];

                    // Compute dimension of diagonal block Hii.
                    const int num_rows = first_row[i + 1] - first_row[i];

                    // Select leading dimension according to memory layout.
                    int ldH;
                    if (mem_layout == TILE_LAYOUT)
                        ldH = num_rows;
                    else
                        ldH = ld;

                    // Hii := I - 2 * vi * vi^T.
                    generate_diagonal_householder_block(num_rows, ldH, H_blocks[i][i], vi);
                }
            }
            else {
                // Compute Householder block Hij.
                #pragma omp task shared(H_blocks)
                {
                    // Locate subvectors vi, vj.
                    const double *vi = v + first_row[i];
                    const double *vj = v + first_col[j];

                    // Compute dimension of block Hij.
                    const int num_rows = first_row[i + 1] - first_row[i];
                    const int num_cols = first_col[j + 1] - first_col[j];

                    // Select leading dimension according to memory layout.
                    int ldH;
                    if (mem_layout == TILE_LAYOUT)
                        ldH = num_rows;
                    else
                        ldH = ld;

                    // Hij := -2 * vi * vj^T.
                    generate_offdiagonal_householder_block(
                        num_rows, num_cols, ldH, H_blocks[i][j], vi, vj);
                }
            }
        }
    }

    free(v);
}


void generate_offdiagonal_householder_matrix(
    int first_blk_row, int num_blk_rows,
    int first_blk_col, int num_blk_cols,
    int ld, double ***H_blocks, const double *restrict const v,
    const int *restrict const first_row, const int *restrict const first_col,
    memory_layout_t mem_layout)
{
    #pragma omp parallel
    #pragma omp single nowait
    for (int i = first_blk_row; i < first_blk_row + num_blk_rows; i++) {
        for (int j = first_blk_col; j < first_blk_col + num_blk_cols; j++) {
            // Hij := -2 * vi * vj^T.
            #pragma omp task shared(H_blocks)
            {
                // Compute dimensions of block Hij.
                const int num_rows = first_row[i + 1] - first_row[i];
                const int num_cols = first_col[j + 1] - first_col[j];

                // Locate vi, vj.
                const double *vi = v + first_row[i];
                const double *vj = v + first_col[j];

                // Fill block.
                if (mem_layout == COLUMN_MAJOR)
                    generate_offdiagonal_householder_block(num_rows, num_cols,
                        ld, H_blocks[i][j], vi, vj);
                else //           TILE_LAYOUT
                    generate_offdiagonal_householder_block(num_rows, num_cols,
                        num_rows, H_blocks[i][j], vi, vj);
            }
        }
    }
}


static void generate_dense_block(int n, int m, double *restrict const A, int ld)
{
    const double low = 0.1;
    const double high = 1.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A[i + ld *j] = random_double(low, high);
        }
    }
}


static void generate_upper_quasi_triangular_block(
    int n, int ld, double *restrict const T,
    const double *restrict const lambda, const int *restrict const lambda_type)
{
#define T(i,j) T[(i) + ld * (j)]

    const double low = 0.1;
    const double high = 1.0;

    // Create random upper triangular matrix.
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) {
            T(i,j) = random_double(low, high);
        }
        for (int i = j + 1; i < n; i++) {
            T(i,j) = 0.0;
        }
    }
    // Position eigenvalues.
    for (int j = 0; j < n; j++) {
        if (lambda_type[j] == REAL) {
            // Place real eigenvalue as 1-by-1 block at T(j,j).
            T(j,j) = lambda[j];
        }
        else {
            // Place complex conjugate pair as 2-by-2 block at T(j:j+1, j:j+1).
            // [a -b]
            // [b  a]
            const double a = lambda[j];
            const double b = lambda[j+1];
            T(j  ,j) = a;  T(j  ,j+1) = -b;
            T(j+1,j) = b;  T(j+1,j+1) = a;
            
            // We processed a 2-by-2 block, so skip the next entry.
            j++;
        }
    }

#undef T
}


void generate_upper_triangular_matrix(
    int first_blk, int num_blks,
    double ***restrict const T_blocks, int ld,
    const int *restrict const first_row, const int *restrict const first_col,
    const double *restrict const lambda, const int *restrict const lambda_type,
    memory_layout_t mem_layout)
{
    // Fill upper quasi-triangular matrix T with random values and fill the
    // diagonal with the given eigenvalues.
    #pragma omp parallel
    #pragma omp single nowait
    {
        // Fill blocks Tij above diagonal randomly.
        for (int i = first_blk; i < first_blk + num_blks; i++) {
            for (int j = i + 1; j < first_blk + num_blks; j++) {
                #pragma omp task
                {
                    // Compute dimensions of block Tij.
                    const int num_rows = first_row[i + 1] - first_row[i];
                    const int num_cols = first_col[j + 1] - first_col[j];

                    // Fill block.
                    if (mem_layout == COLUMN_MAJOR)
                        generate_dense_block(num_rows, num_cols, T_blocks[i][j], ld);
                    else //           TILE_LAYOUT
                        generate_dense_block(num_rows, num_cols, T_blocks[i][j], num_rows);
                }
            }
        }

        // Fill diagonal blocks Tjj using given eigenvalues.
        for (int j = first_blk; j < first_blk + num_blks; j++) {
            #pragma omp task
            {
                const int num_cols = first_col[j + 1] - first_col[j];
                // Fill block.
                if (mem_layout == COLUMN_MAJOR)
                    generate_upper_quasi_triangular_block(
                        num_cols, ld, T_blocks[j][j], 
                        &lambda[first_col[j]], &lambda_type[first_col[j]]);
                else
                    generate_upper_quasi_triangular_block(
                        num_cols, num_cols, T_blocks[j][j], 
                        &lambda[first_col[j]], &lambda_type[first_col[j]]);
            }
        }


        // Zero out off-diagonal blocks Tij below diagonal.
        for (int i = first_blk; i < first_blk + num_blks; i++) {
            for (int j = first_blk; j < i; j++) {
                #pragma omp task
                {
                    // Compute dimensions of block Tij.
                    const int num_rows = first_row[i + 1] - first_row[i];
                    const int num_cols = first_col[j + 1] - first_col[j];

                    // Select leading dimension according to memory layout.
                    int ldT;
                    if (mem_layout == TILE_LAYOUT)
                        ldT = num_rows;
                    else
                        ldT = ld;

                    set_zero(num_rows, num_cols, T_blocks[i][j], ldT);
                }
            }
        }
    }
}




// Slaughtered one of Mirko's routines to only generate eigenvalues.
void generate_eigenvalues(const int n, double complex_ratio,
    double *restrict const lambda, int *restrict const lambda_type)
{
    int complex_count = complex_ratio * n / 2;
    int real_count = n - 2 * complex_count;

    // place the 1x1 blocks between the 2-by-2 blocks
    int spaces[complex_count+1];
    for (int i = 0; i < complex_count+1; i++)
        spaces[i] = 0;
    for (int i = 0; i < real_count; i++)
        spaces[rand() % (complex_count+1)]++;

    // TODO: n + i should become a (added to make diagonally dominant)
    // generate the 1-by-1 blocks
    for (int i = 0; i < n; i++) {
        lambda[i] = /*ldexp(i + 1.0, -1022);*/ 0.000000000000000000000001 * (1.0 + i);
        lambda_type[i] = REAL;
    }
    //lambda[n-1] = ldexp(n + 3.0, -1022);
    //lambda[n-2] = ldexp(n + 2.0, -1022);

    // generate the 2-by-2 blocks
    int i = 0;
    for (int j = 0; j < complex_count; j++) {
        i += spaces[j];
        // create real and imaginary part
        int grid_height = sqrt(2*complex_count)/2;
        double real = j / grid_height - grid_height + 0.5;
        double imag = j % grid_height + 1.0;
        // TODO: n + real should become real (added to make diagonally dominant)
        lambda[i] = 0.000000000000000000000000000000000001 * real;
        lambda[i+1] = 0.0000000000000000000000000000000000001 * imag;
        lambda_type[i] = CMPLX;
        lambda_type[i+1] = CMPLX;
        i += 2;
    }
}


static int select_block(int i, int n, int *restrict const lambda_type,
    int *restrict const selected)
{
    if (lambda_type[i] == REAL) {
        selected[i] = 1;
        return 1;
    }
    else {
        // Identify if the complex conjugate pair of eigenvalues spans
        // (i, i + 1) or (i - 1, i).
        if (i == 0 && lambda_type[i] == CMPLX) {
            selected[0] = 1;
            selected[1] = 1;
            return 2;
        }
        // Count complex eigenvalues.
        int num_cmplx = 0;
        for (int k = 0; k <= i; k++) {
            if (lambda_type[k] == CMPLX) {
                num_cmplx++;
            }
        }
        if (i > 0 && num_cmplx % 2 == 0) {
            // Span (i - 1, i).
            selected[i - 1] = 1;
            selected[i] = 1;
            return 2;
        }
        else {
            // Span (i, i + 1).
            selected[i] = 1;
            selected[i + 1] = 1;
            return 2;
        }
    }
}


int generate_selection(const int n, int *restrict const lambda_type,
    int *restrict const selected, double select_ratio)
{
    memset(selected, 0, n * sizeof(int));

    int num_selected = 0;
    while(num_selected < n * select_ratio) {
        int i = random_integer(0, n - 1);
        if (!selected[i]) {
            num_selected += select_block(i, n, lambda_type, selected);
        }
    }

    return num_selected;
}

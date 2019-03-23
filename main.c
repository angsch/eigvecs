#include "random.h"
#include "utils.h"
#include "eigenvectors.h"
#include "validation.h"
#include "defines.h"
#include "partition.h"
#include "datalayout.h"
#include "timing.h"
#include "IO.h"
#include "robust.h"
#include "preprocessing.h"
#include "reference.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mm_malloc.h>
#include <assert.h>
#include <time.h>

#include "omp.h"




static long long estimate_flops(int n, const int *restrict const selected)
{
    long long sum = 0;

    // Backsolve.
    //
    // If the i-th eigenvalue is selected, the eigenvector column has length
    // j := (i + 1). A standard backward substitution of a column of length j
    // requires j divisions, j subtractions, (j^2-j)/2 additions
    // and (j^2-j)/2 multiplications.
    // In total, j^2 + j flops. Sum up flops for each selected column.
    for (int i = 0; i < n; i++) {
        if (selected[i]) {
            // Correct zero indexing.
            const int j = i + 1;
            sum += j * j + j;
        }
    }

    // Base transform.
    //
    // To transform a real column of length j, the zeros-exploiting matrix-vector
    // multiply   Q   *   x
    //         (n x j) (j x 1)
    // yields n (2j - 1) flops.
    for (int i = 0; i < n; i++) {
        if (selected[i]) {
            // Correct zero indexing.
            const long long j = i + 1;
            sum += n * (2 * j - 1);
        }
    }

    return sum;
}




int main(int argc, char **argv)
{
    // Dimension of the system, block size, number of selected eigenvalues.
    int n, blksz, num_selected;
    // Ratio of complex/real eigenvalues, proportion of selected eigenvalues.
    double complex_ratio, select_ratio;
    // Seed for random number generator.
    unsigned int seed;

    // Matrices of the Schur decomposition A = Q * T * Q^T.
    double *Q, *T;
    int ldQ, ldT;

    if (argc != 6) {
        printf("Usage %s matrix-size blksz complex-ratio select-ratio\n", argv[0]);
        printf("matrix-size:   Size of the quasi-triangular matrix\n");
        printf("blksz:         The tile size\n");
        printf("complex-ratio: Ratio of complex/real eigenvalues\n");
        printf("select-ratio:  Proportion of selected eigenvalues\n");
        printf("seed:          Seed for the random number generator\n");
        return EXIT_FAILURE;
    }

    // Set inputs.
    n = atoi(argv[1]);
    blksz = atoi(argv[2]);
    complex_ratio = atof(argv[3]);
    select_ratio = atof(argv[4]);
    seed = (unsigned int)atoi(argv[5]);

    // Initialize the random number generator.
    srand(seed);

    assert(select_ratio > 0.0 && select_ratio <= 1.0);
    assert(complex_ratio >= 0.0 && complex_ratio <= 1.0);
    assert(blksz <= n);
    assert(n > 0);
    assert(blksz > 0);

    // Check blksz input parameter.
#if defined(__AVX512F__)
    if (blksz % 8 != 0) {
        blksz = 8 * ((blksz + 4)/8);
        printf("WARNING: To meet alignment constraints, I change blksz "
               "to %d.\n", blksz);
    }
#elif !defined(__AVX512F__) && defined (__AVX__)
    if (blksz % 4 != 0) {
        blksz = get_size_with_padding(blksz);
        printf("WARNING: To meet alignment constraints, I change blksz "
               "to %d.\n", blksz);
    }
#endif

    // Print configuration.
    printf("Configuration:\n");
    #pragma omp parallel
    #pragma omp single
    printf("  OpenMP threads = %d\n", omp_get_num_threads());
    printf("  n = %d\n", n);
    printf("  blksz = %d\n", blksz);
    printf("  Percentage of complex eigenvalues = %.6lf\n", complex_ratio);
    printf("  Percentage of selected eigenvalues = %.6lf\n", select_ratio);
    printf("  Seed = %u\n", seed);

    // Select appropriate leading dimensions.
    ldT = get_size_with_padding(n);
    ldQ = get_size_with_padding(n);

    // Allocate matrices.
    T = (double *) _mm_malloc((size_t)ldT * n * sizeof(double), ALIGNMENT);
    Q = (double *) _mm_malloc((size_t)ldQ * n * sizeof(double), ALIGNMENT);

    // Eigenvalues, actual values and type (real, complex) identifier.
    double *lambda = (double *) _mm_malloc(n * sizeof(double), ALIGNMENT);
    int *lambda_type = (int *) malloc(n * sizeof(int));
    generate_eigenvalues(n, complex_ratio, lambda, lambda_type);

    // Selection of eigenvalues.
    int *selected = (int *) malloc(n * sizeof(int));
    num_selected = generate_selection(n, lambda_type, selected, select_ratio);

    // Eigenvectors, split into two matrices for real and imaginary parts.
    int ldX = get_size_with_padding(n);
    double *X = (double *) _mm_malloc((size_t) ldX * num_selected * sizeof(double), ALIGNMENT);
    memset(X, 0.0, (size_t)ldX * num_selected * sizeof(double));

    // Compute the number of blocks.
    int num_blks = (n + blksz - 1) / blksz;

    // Map block index onto actual row/column index.
    int *first_row = (int *) malloc((num_blks + 1) * sizeof(int));
    int *first_col = (int *) malloc((num_blks + 1) * sizeof(int));
    int *first_rhs_col = (int *) malloc((num_blks + 1) * sizeof(int));

    // Set memory layout.
    memory_layout_t mem_layout = TILE_LAYOUT; // TILE_LAYOUT, COLUMN_MAJOR
    printf("  Used memory layout: ");
    if (mem_layout == TILE_LAYOUT)
        printf("tile layout\n");
    else
        printf("colum major\n");


    // Compute a partitioning that does not split pairs of complex conjugate
    // eigenvalues across blocks. This will be the OpenMP grid.
    partitioning_t fine_grid = {.num_blks = num_blks,
                                .first_row = first_row,
                                .first_col = first_col};
    partition(n,
              &fine_grid,
              lambda_type, blksz, num_blks);

    // Compute a partitioning of the right-hand sides that considers the selection.
    partitioning_t fine_grid_rhs = {.num_blks = num_blks,
                                    .first_row = first_row,
                                    .first_col = first_rhs_col};
    partition_selected(n, first_col, selected, first_rhs_col, num_blks);


    // Allocate pointers to all blocks in all relevant matrices.
    double ***Q_blocks = malloc(num_blks * sizeof(double **));
    double ***T_blocks = malloc(num_blks * sizeof(double **));
    double ***X_blocks = malloc(num_blks * sizeof(double **));
    for (int i = 0; i < num_blks; i++) {
        Q_blocks[i] = malloc(num_blks * sizeof(double *));
        T_blocks[i] = malloc(num_blks * sizeof(double *));
        X_blocks[i] = malloc(num_blks * sizeof(double *));
    }


    // Apply partitioning to matrices.
    partition_matrix(Q, ldQ, mem_layout, &fine_grid, Q_blocks);
    partition_matrix(T, ldT, mem_layout, &fine_grid, T_blocks);
    partition_matrix(X, ldX, mem_layout, &fine_grid_rhs, X_blocks);

    // Prepare scaling factors. One per column per block row.
    scaling_t *scales;
    double *Xnorms;
    scales = (scaling_t *) malloc(num_blks * num_selected * sizeof(scaling_t));
    init_scaling_factor(num_blks * num_selected, scales);

    Xnorms = (double *) malloc(num_blks * num_selected * sizeof(double));

#define scales(col, blkrow) scales[(col) + (blkrow) * num_selected]
#define Xnorms(col, blkrow) Xnorms[(col) + (blkrow) * num_selected]


    printf("Generate T, Q...\n");
    generate_upper_triangular_matrix(0, num_blks, T_blocks, ldT, 
        first_row, first_col, lambda, lambda_type, mem_layout);
    generate_householder_matrix(n, ldQ, Q_blocks, num_blks, first_row, first_col, mem_layout);


    // Prepare column majorants of T.
    double *Tnorms = (double *) malloc(num_blks * num_blks * sizeof(double));
    printf("Compute column majorants of T...\n");
    compute_majorants(T_blocks, ldT, Tnorms, &fine_grid, mem_layout);

    // Compute the eigenvectors.
    double tm_start = get_time();

    compute_eigenvectors_separate_phases_openmp(
        n, T_blocks, ldT,
        lambda, lambda_type, selected,
        &fine_grid, &fine_grid_rhs,
        Q_blocks, ldQ,
        X_blocks, ldX, scales, Tnorms, Xnorms,
        mem_layout);
    printf("Execution time = %.2f s.\n", get_time() - tm_start);

    long long flops = estimate_flops(n, selected);
    printf("Flops = %lld\n", flops);

    printf("Validate...\n");
    validate(n, T_blocks, ldT, Q_blocks, ldQ, X_blocks, ldX,
        lambda, lambda_type, selected,
        &fine_grid, &fine_grid_rhs,
        mem_layout);


/*
    if (myrank == 0) {
        printf("Compute reference solution with LAPACK...\n");
        compute_eigenvectors_dtrevc3(n, T_blocks, ldT,
            lambda, lambda_type, selected,
            &fine_grid, &fine_grid_rhs,
            Q_blocks, ldQ,
            mem_layout);
    }
*/

    // Clean up.
    _mm_free(T);
    _mm_free(Q);
    _mm_free(X);
    free(lambda_type);
    free(selected);
    _mm_free(lambda);
    free(first_row);
    free(first_col);
    free(first_rhs_col);
    for (int i = 0; i < num_blks; i++) {
        free(Q_blocks[i]);
        free(T_blocks[i]);
        free(X_blocks[i]);
    }
    free(X_blocks);
    free(Q_blocks);
    free(T_blocks);
    free(scales);
    free(Xnorms);
    free(Tnorms);

    return EXIT_SUCCESS;
}

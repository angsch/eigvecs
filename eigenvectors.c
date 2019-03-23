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
#include "update-task.h"

#include <stdlib.h>
#include <stdio.h>
#include <mm_malloc.h>
#include <string.h>
#include <math.h>

#include <omp.h>

static void find_max(
    int num_rows, int num_selected, int n,
    const double *restrict const X, int ldX,
    const int *restrict const lambda_type, const int *restrict const selected,
    double *restrict emax)
{
    int si = num_selected - 1;
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
                // Locate eigenvector to ki-th eigenvalue.
                const double *X_re = X + si * ldX;

                // Find max entry.
                emax[si] = vector_infnorm(num_rows, X_re);
                si--;
            }
            else { // COMPLEX
                // Locate real and imaginary part of complex eigenvector for complex
                // conjugate pair of eigenvalues (ki, ki-1).
                const double *X_re = X + (si - 1) * ldX;
                const double *X_im = X + si * ldX;

                // Find max entry.
                emax[si] = 0.0;
                for (int i = 0; i < num_rows; i++) {

                    if (fabs(X_re[i]) + fabs(X_im[i]) > emax[si]) {
                        emax[si] = fabs(X_re[i]) + fabs(X_im[i]);
                    }
                }


                // Duplicate max entry for real and imaginary column.
                emax[si - 1] = emax[si];

                ki--;
                si -= 2;
            }
        }
    }
}



static void unify_scaling(
    const partitioning_t *restrict const p_rhs,
    scaling_t *restrict const scales,
    double ***restrict const X_blocks, int ldX,
    const int *restrict const lambda_type, const int *restrict const selected,
    memory_layout_t layout)
{
    const int num_blks = p_rhs->num_blks;
    const int *first_row = p_rhs->first_row;
    const int *first_col = first_row;
    const int *first_rhs_col = p_rhs->first_col;

    const int num_selected = first_rhs_col[num_blks];

#define scales(col, blkrow) scales[(col) + (blkrow) * num_selected]

    ////////////////////////////////////////////////////////////////////////////
    // Compute the most constraining scaling factor.
    ////////////////////////////////////////////////////////////////////////////

    scaling_t *smin = (scaling_t *) malloc(num_selected * sizeof(scaling_t));

    // Find the minimum scaling factor for each column.
    const int first = first_rhs_col[0];
    const int last = first_rhs_col[num_blks];

    // Initialize the scaling factors for the columns I am responsible for.
    init_scaling_factor(last - first, smin);


    for (int blkj = 0; blkj < num_blks; blkj++) {
        for (int blki = 0; blki <= blkj; blki++) {
            for (int j = first_rhs_col[blkj]; j < first_rhs_col[blkj + 1]; j++) {
                // smin[j] = 1.0;
#ifdef INTSCALING
                smin[j] = min(smin[j], scales(j, blki));
#else
                smin[j] = minf(smin[j], scales(j, blki));
#endif
            }
        }
    }

    double *emax = (double *) malloc(num_selected * sizeof(double));
    memset(emax, 0.0, num_selected * sizeof(double));

    double *tmp = (double *) malloc(num_selected * sizeof(double));

    #pragma omp parallel
    #pragma omp single nowait
    for (int blkj = 0; blkj < num_blks; blkj++) {
        #pragma omp task
        {
            for (int blki = 0; blki <= blkj; blki++) {
                // Compute the dimensions of block (blki,blkj).
                const int num_rows = first_row[blki + 1] - first_row[blki];
                const int num_sel = first_rhs_col[blkj + 1] - first_rhs_col[blkj];

                // Compute the number of uncompressed columns.
                const int num_cols = first_col[blkj + 1] - first_col[blkj];

                // The current block with corresponding eigenvalues.
                double *X = X_blocks[blki][blkj];
                const int *lambda_type_X = lambda_type + first_col[blkj];
                const int *selected_X = selected + first_col[blkj];
                double *tmp_X = tmp + first_rhs_col[blkj];
                memset(tmp_X, 0.0, num_sel * sizeof(double));


                if (layout == COLUMN_MAJOR)
                    find_max(num_rows, num_sel, num_cols, X, ldX,
                        lambda_type_X, selected_X, tmp_X);
                else //       TILE_LAYOUT
                    find_max(num_rows, num_sel, num_cols, X, num_rows,
                        lambda_type_X, selected_X, tmp_X);


                // Reduce to maximum normalization factor.
                for (int j = first_rhs_col[blkj]; j < first_rhs_col[blkj + 1]; j++) {
                    // Compute normalization factor *simulating* consistent scaling.
                    double s = compute_upscaling(smin[j], scales(j, blki));

                    emax[j] = maxf(s * tmp[j], emax[j]);
                }
            }

            // Apply scaling.
            for (int blki = 0; blki <= blkj; blki++) {
                // Compute the dimensions of block (i,j).
                const int num_rows = first_row[blki + 1] - first_row[blki];
                const int num_cols = first_rhs_col[blkj + 1] - first_rhs_col[blkj];

                // The currenct column.
                double *X;

                for (int j = 0; j < num_cols; j++) {
                    if (layout == COLUMN_MAJOR)
                        X = X_blocks[blki][blkj] + j * ldX;
                    else   //     TILE_LAYOUT
                        X = X_blocks[blki][blkj] + j * num_rows;

                    // Compute absolute column index.
                    const int col = first_rhs_col[blkj] + j;

                    // Apply scaling.
                    double s = compute_upscaling(smin[col], scales(col, blki));

                    for (int i = 0; i < num_rows; i++) {
                        X[i] = (s * X[i]) / emax[col];
                    }
                }

            }
        }
    }

    free(smin);
    free(emax);
    free(tmp);

#undef scales
}


static void init_2D_lock_grid(int m, int n, omp_lock_t *lock/*, omp_lock_hint_t hint*/)
{
#define lock(i,j) lock[(i) + m * (j)]

    // gcc-5 does not implement locks with hints. Ignore hints for the time being.

    // omp_lock_hint_uncontended: Lock incurs few conflicts.
    // omp_lock_hint_contended: Lock incurs a lot of conflicts.
    // omp_lock_hint_nonspeculatitve: high conflict potential due to overlapping of the working sets of the threads
    // omp_lock_hint_speculative: optimistic locking: conflict potential is low // Speculative locks can benefit more with growing thread counts

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            omp_init_lock(&lock(i,j));
            //omp_init_lock_with_hint(&lock(i,j), hint);
        }
    }

#undef lock
}


void compute_eigenvectors_separate_phases_openmp(
    int n, double ***restrict const T_blocks, int ldT,
    double *restrict const lambda, int *restrict const lambda_type,
    const int *restrict const selected,
    const partitioning_t *restrict const p, const partitioning_t *restrict const p_rhs,
    double ***restrict const Q_blocks, int ldQ,
    double ***restrict const X_blocks, int ldX,
    scaling_t *restrict const scales,
    const double *restrict const Tnorms,
    double *restrict const Xnorms,
    memory_layout_t layout)
{
    int num_blks = p->num_blks;
    const int *first_row = p->first_row;
    const int *first_col = p->first_col;
    const int *first_rhs_col = p_rhs->first_col;

#define scales(col, blkrow) scales[(col) + (blkrow) * first_rhs_col[num_blks]] // num_selected
#define Xnorms(col, blkrow) Xnorms[(col) + (blkrow) * first_rhs_col[num_blks]]
#define Tnorms(i,j) Tnorms[(i) + (j) * num_blks]

    // Initialize Xnorms.
    for (int blkrow = 0; blkrow < num_blks; blkrow++)
        for (int col = 0; col < first_rhs_col[num_blks]; col++)
            Xnorms(col,blkrow) = 0.0;


    // Allocate workspace.
    double *Y;
    int ldY = get_size_with_padding(n);
    int m = count_selected(n, selected);
    Y = (double *) _mm_malloc((size_t)ldY * m * sizeof(double), ALIGNMENT);

    // Allocate pointers to all blocks in Y.
    double ***Y_blocks = malloc(num_blks * sizeof(double **));
    for (int i = 0; i < num_blks; i++) {
        Y_blocks[i] = malloc(num_blks * sizeof(double *));
    }

    // Partition Y analogously to X.
    partition_matrix(Y, ldY, layout, p_rhs, Y_blocks);

    // It is: num_rows = first_row[k + 1] - first_row[k].

    // Allocate locks for blocks in X to synchronize the updates.
    omp_lock_t lock[num_blks][num_blks];
    init_2D_lock_grid(num_blks, num_blks, &lock[0][0]/*, omp_lock_hint_contended | omp_lock_hint_speculative*/);


    // Copy all selected eigenvalue types to a compact memory representation.
    int *selected_lambda_type = (int *) malloc(m * sizeof(int));
    int idx = 0;
    for (int i = 0; i < n; i++) {
        if (selected[i]) {
            selected_lambda_type[idx] = lambda_type[i];
            idx++;
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    // Backsolve (T - lambda * I) X = 0.

    // Range of block right-hand sides.
    int first = 0;
    int last = num_blks;


    // Loop over right-hand sides.
    #pragma omp parallel
    {
    #pragma omp single nowait
    //for (int k = num_blks - 1; k >= 0; k--) {
    for (int k = last - 1; k >= first; k--) {
        // Insert tasks only if at least one eigenvalue is selected.
        const int width = first_col[k + 1] - first_col[k];
        int num_selected = count_selected(width, &selected[first_row[k]]);
        if (num_selected > 0) {
            // Loop over block columns in T.
            for (int j = k; j >= 0; j--) {
                if (k == j) {
                    // Backsolve (Tkk - lambda_kk I) \ Xkk.
                    #pragma omp task depend(out:X_blocks[k][k])
                    {
                        // Compute the number of rhs including non-selected.
                        const int num_rhs = first_col[k + 1] - first_col[k];

                        // Solve (Tkk - lambda I) \ Xkk using the eigenvalues in Tkk.
                        if (layout == COLUMN_MAJOR)
                            backsolve(num_rhs, T_blocks[k][k], ldT, Tnorms(k,k),
                                X_blocks[k][k], ldX,
                                &scales(first_rhs_col[k],k), &Xnorms(first_rhs_col[k],k),
                                &lambda_type[first_row[k]],
                                &selected[first_row[k]]);
                        else //       TILE_LAYOUT
                            backsolve(num_rhs, T_blocks[k][k], num_rhs, Tnorms(k,k),
                                X_blocks[k][k], num_rhs,
                                &scales(first_rhs_col[k],k), &Xnorms(first_rhs_col[k],k),
                                &lambda_type[first_row[k]],
                                &selected[first_row[k]]);
                    }
                }
                else { // k != j
                    // Solve (Tjj - lambda_kk I) \ Xjk.
                    #pragma omp task depend(in:X_blocks[j+1:num_blks][k]) \
                    depend(out:X_blocks[j][k])
                    {
                        // Compute the actual number of rows.
                        const int num_rows = first_row[j + 1] - first_row[j];

                        // Compute the number of rhs including non-selected.
                        const int num_rhs = first_col[k + 1] - first_col[k];

                        // Solve (Tjj - lambda I) \ Xjk using the eigenvalues in Tkk.
                        if (layout == COLUMN_MAJOR)
                            solve(num_rows, 
                                T_blocks[j][j], ldT, Tnorms(j,j),
                                &lambda_type[first_row[j]],
                                X_blocks[j][k], ldX,
                                num_rhs,
                                &scales(first_rhs_col[k],j), &Xnorms(first_rhs_col[k],j),
                                &lambda[first_row[k]], &lambda_type[first_col[k]],
                                &selected[first_row[k]]);
                        else //       TILE_LAYOUT
                            solve(num_rows,
                                T_blocks[j][j], num_rows, Tnorms(j,j),
                                &lambda_type[first_row[j]],
                                X_blocks[j][k], num_rows,
                                num_rhs,
                                &scales(first_rhs_col[k],j), &Xnorms(first_rhs_col[k],j),
                                &lambda[first_row[k]], &lambda_type[first_col[k]],
                                &selected[first_row[k]]);
                    }
                }

                // Loop over block rows above diagonal.
                for (int i = 0; i < j; i++) {
                    #pragma omp task depend(in:X_blocks[j][k]) \
                    depend(inout:X_blocks[i][k]) shared(lock)
                    {
                        // Compute the actual number of rows.
                        const int num_rows = first_row[i + 1] - first_row[i];

                        // Compute the actual number of columns.
                        const int num_cols = first_col[j + 1] - first_col[j];

                        // Compute the actual number of right-hand sides (only selected).
                        const int num_rhs = first_rhs_col[k + 1] - first_rhs_col[k];

                        // Execute linear update Xik := Xik -    Tij         *         Xjk.
                        //       (num_rows x num_rhs)   (num_rows x num_cols) (num_cols x num_rhs)
                        if (layout == COLUMN_MAJOR)
                            update(num_rows, num_cols,
                                T_blocks[i][j], ldT, Tnorms(i,j),
                                num_rhs, &lock[i][k],
                                X_blocks[i][k], ldX,
                                &scales(first_rhs_col[k],i), &Xnorms(first_rhs_col[k],i),
                                X_blocks[j][k], ldX,
                                &scales(first_rhs_col[k],j), &Xnorms(first_rhs_col[k],j),
                                &selected_lambda_type[first_rhs_col[k]]);
                        else //       TILE_LAYOUT
                            update(num_rows, num_cols,
                                T_blocks[i][j], num_rows, Tnorms(i,j),
                                num_rhs, &lock[i][k],
                                X_blocks[i][k], num_rows,
                                &scales(first_rhs_col[k],i), &Xnorms(first_rhs_col[k],i),
                                X_blocks[j][k], num_cols,
                                &scales(first_rhs_col[k],j), &Xnorms(first_rhs_col[k],j),
                                &selected_lambda_type[first_rhs_col[k]]);
                    }
                }
            }
        }
    }
    }

    unify_scaling(p_rhs, scales, X_blocks, ldX,
        lambda_type, selected, layout);

    // Y follows the 1D vertical block distribution of X.


    // Transform back to original basis:
    // Y := Q * X <=> (Y_real, Y_imag) = Q * (X_real, Y_imag).
    #pragma omp parallel
    #pragma omp single nowait
    // Loop over block columns in Y.
    for (int j = first; j < last; j++) {
        // Loop over block rows in Y.
        for (int i = 0; i < num_blks; i++) {
            // Yij := Qi: * X:j.
            #pragma omp task
            backtransform(p_rhs,
                i,j,
                Q_blocks, ldQ,
                Y_blocks, ldY,
                X_blocks, ldX,
                layout);
        }
    }

    copy_submatrix(0, num_blks, first, last - first, Y_blocks, ldY, X_blocks, ldX, first_row, first_rhs_col, layout);


    // Clean up.
    for (int i = 0; i < num_blks; i++) {
        free(Y_blocks[i]);
        for (int k = 0; k < num_blks; k++) {
            omp_destroy_lock(&lock[i][k]);
        }
    }
    free(selected_lambda_type);
    free(Y_blocks);
    _mm_free(Y);
}

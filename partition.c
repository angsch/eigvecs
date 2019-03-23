#include "partition.h"
#include "defines.h"
#include "utils.h"

#include <assert.h>
#include <stdio.h>


void partition(
    int n,
    partitioning_t *const p,
    //int *first_row, int *first_col,
    const int *restrict const lambda_type,
    int blksz, int num_blks)
{
    int *first_row = p->first_row;
    int *first_col = p->first_col;

    // Compute candidate partitioning for T, X.
    for (int i = 0; i < num_blks; i++) {
        for (int j = 0; j < num_blks; j++) {
            first_row[i] = i * blksz;
            first_col[j] = j * blksz;
        }
    }

    // Fill pad so that #rows = first_row[k+1]-first_row[k].
    first_row[num_blks] = n;
    first_col[num_blks] = n;

    // Loop over diagonal blocks. Check how eigenvalues are split across blocks.
    int num_cmplx = 0;

    // Absolute column index in T.
    int first_idx = 0;
    int last_idx = min(first_idx + blksz, n);

    for (int k = 0; k < num_blks; k++) {
        // Count complex eigenvalues in this block.
        for (int i = first_idx; i < last_idx; i++) {
            if (lambda_type[i] == CMPLX) {
                num_cmplx++;
            }
        }
        if ((num_cmplx % 2) == 0) {
            // This block respects pairs of complex eigenvalues. Proceed.
            first_idx = last_idx;
            last_idx = min(first_idx + blksz, n);
        }
        else {
            // (num_cmplx % 2) == 1

            printf("Split complex eigenvalue\n");

            // Pair of complex eigenvalues is split across blocks. Adapt sizes.
            {
                // If this block is the last one, skip the remaining computation.
                if (k == num_blks - 1) {
                    continue;
                }

                // Let next block start one entry later.
                first_row[k+1]++;
                first_col[k+1]++;
            }

            // Advance to the next block.
            first_idx = last_idx + 1;
            last_idx = min(first_idx + blksz - 1, n);
            num_cmplx = 0;
        }
    }
}


int count_selected(int n, const int *restrict const selected)
{
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (selected[i]) {
            count++;
        }
    }
    return count;
}


void partition_selected(
    int n,
    const int *restrict const first_col,
    const int *restrict const selected,
    int *restrict const first_rhs_col,
    int num_blks)
{
    int num_selected = count_selected(n, selected);

    first_rhs_col[0] = 0;
    first_rhs_col[num_blks] = num_selected;

    for (int i = 1; i < num_blks; i++) {
        // Shrink width of block column to number of selected.
        int width = first_col[i] - first_col[i - 1];
        int quantity = count_selected(width, selected + first_col[i - 1]);
        first_rhs_col[i] = first_rhs_col[i - 1] + quantity;
    }
}


void partition_matrix(
    const double *restrict const A, int ldA,
    memory_layout_t layout,
    const partitioning_t *restrict const p,
    double ***restrict const A_blocks)
{
    // Extract row and column partitioning.
    const int *first_row = p->first_row;
    const int *first_col = p->first_col;
    const int num_blks = p->num_blks;

    switch (layout) {
    case COLUMN_MAJOR:
    {
        #define A(i,j) A[(i) + (j) * ldA]
        for (int i = 0; i < num_blks; i++) {
            for (int j = 0; j < num_blks; j++) {
                A_blocks[i][j] = &A(first_row[i], first_col[j]);
            }
        }
        #undef A
    }
    break;

    case TILE_LAYOUT: {
        // Use column major order to store blocks.
        for (int i = 0; i < num_blks; i++) {
            for (int j = 0; j < num_blks; j++) {
                A_blocks[i][j] 
                    = A + first_row[num_blks] * first_col[j] // Full block cols to the left of us.
                        + first_row[i] * (first_col[j+1] - first_col[j]); // Offset in our block column.
            }
        }
    }
    break;

    default:
    {
        assert(0);
    }
    break;
    }
}


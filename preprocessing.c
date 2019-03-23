#include "preprocessing.h"
#include "norm.h"



void compute_majorants(
    double ***restrict const T_blocks, int ldT,
    double *restrict const T_norms,
    partitioning_t *p, memory_layout_t mem_layout)
{
    // Extract fine grid.
    const int num_blks = p->num_blks;
    const int* first_row = p->first_row;
    const int* first_col = p->first_col;

#define T_norms(i,j) T_norms[(i) + (j) * num_blks]

    // Traverse blocks within tile.
    for (int ii = 0; ii < num_blks; ii++) {
        for (int jj = 0; jj < num_blks; jj++) {
            // Compute dimensions of block Tij.
            const int m = first_row[ii + 1] - first_row[ii];
            const int n = first_col[jj + 1] - first_col[jj];

            // Compute column majorants of Tij.
            double *Tij = T_blocks[ii][jj];
            if (mem_layout == COLUMN_MAJOR) {
                T_norms(ii,jj) = matrix_infnorm(m, n, Tij, ldT);
            }
            else {//          TILE_LAYOUT
                T_norms(ii,jj) = matrix_infnorm(m, n, Tij, m);
            }
        }
    }

#undef T_norms
}


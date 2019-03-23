#include "datalayout.h"


void convert_to_column_major(
    double *restrict const Aout, int ldA,
    double ***A_blocks,
    const int *restrict const first_row, const int *restrict const first_col,
    int num_blks)
{

    #define Aout(i,j) Aout[(i) + (j) * ldA]
    // Loop over blocks in tile layout.
    for (int j = 0; j < num_blks; j++) {
        for (int i = 0; i < num_blks; i++) {
            // Convert block (i,j).
            const int num_rows = first_row[i+1] - first_row[i];
            const int num_cols = first_col[j+1] - first_col[j];
            // Loop over entries within block.
            for (int jj = 0; jj < num_cols; jj++) {
                for (int ii = 0; ii < num_rows; ii++) {
                    // Compute indices in column major format.
                    const int colmajor_i = first_row[i] + ii;
                    const int colmajor_j = first_col[j] + jj;
                    Aout(colmajor_i, colmajor_j) = A_blocks[i][j][ii + num_rows * jj];
                }
            }
        }
    }

    #undef Aout
}

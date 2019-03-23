#ifndef DATALAYOUT_H
#define DATALAYOUT_H

#include "typedefs.h"

/**
 * Converts a square matrix from tile layout to column major layout.
 * 
 * In order to generalize this routine to non-square matrices, change
 * num_blks to num_blk_rows, num_blk_cols.
 * 
 * @param[out] Aout      On exit, the matrix in column major format.
 * @param[in]  ldA       Leading dimension of Aout.
 * @param[in]  A_blocks  Matrix in tile layout.
 * @param[in]  first_rol Array with num_blks entries. first_row[i] holds the
 *                       start row of the i-th block row in tile layout.
 * @param[in]  first_col Array with num_blks entries. first_col[j] holds the
 *                       start column of the j-th block column in tile layout.
 * @param[in]  num_blks  Number of blocks.
 * */
void convert_to_column_major(
    double *restrict const Aout, int ldA,
    double ***A_blocks,
    const int *restrict const first_row, const int *restrict const first_col,
    int num_blks);

#endif

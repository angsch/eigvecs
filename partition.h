#ifndef PARTITION_H
#define PARTITION_H

#include "datalayout.h"
#include "typedefs.h"

/**
 * Generate partitioning that does not split 2x2 blocks.
 *
 * @param[in]  n           Dimension of the system
 * @param[out] p           Partitioning of n-by-n system into regular grid with
 *                         tiles of size approximately blksz that respects 
 *                         complex eigenvalues. On exit, first_row[i] holds
 *                         the start row of the i-th block row; first_col[j]
 *                         holds the start column of the j-th block column.
 * @param[in]  lambda_type Array of length n. Describes if T(i,i) is a real
 *                         or complex eigenvalue
 * @param[in]  blksz       Blocksize
 * @param[in]  num_blksz   Number of blocks
 */
void partition(
    int n,
    partitioning_t *const p,
    const int *restrict const lambda_type,
    int blksz, int num_blks);


/**
 * Apply partitioning to a matrix.
 * 
 * @param[in]  A         The matrix to be partitioned.
 * @param[in]  ldA       The leading dimension of A.
 * @param[in]  layout    COLUMN_MAJOR or TILE_LAYOUT.
 * @param[in]  p         Partitioning of A that contains row and column indices
 *                       of all blocks.
 * @param[out] A_blocks  On entry, a num_blks-by-num_blks array of pointers. On
 *                       exit, A_blocks[i][j] holds the base pointer to the 
 *                       block (i,j) according to the partitioning.
 */
void partition_matrix(
    const double *restrict const A, int ldA,
    memory_layout_t layout,
    const partitioning_t *restrict const p,
    double ***restrict const A_blocks);



/**
 * Compact partitioning to match selected eigenvalues.
 *
 * @param[in]  n             Dimension of the system
 * @param[in]  first_col     Array with num_blks entries. first_col[j]
 *                           holds the start column of the j-th block column
 * @param[in]  selected      Array with n entries. Marks selected eigenvalues.
 * @param[out] first_rhs_col Array with #selected entries.
 * @param[in]  num_blks      Number of blocks.
 */
void partition_selected(
    int n,
    const int *restrict const first_col,
    const int *restrict const selected,
    int *restrict const first_rhs_col,
    int num_blks);


/**
 * Counts the entries set to one.
 *
 * @param[in] n        Length of selected.
 * @param[in] selected Boolean array.
 * @returns Number of ones.
 */
int count_selected(int n, const int *restrict const selected);

#endif

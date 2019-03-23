#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "datalayout.h"

/** Computes the column majorants of all blocks of T.
 * 
 * The eigenvectors require a matrix (T - lambda I), not a matrix T.
 * Here we compute an upper bound for each tile of T, not of (T - lambda I).
 *
 * The column majorants computed here are only used in linear updates.
 * The linear updates do not use diagonal entries (these are used in
 * divisions). Hence it it safe to not consider lambda in this routine.
 *
 * @param[in]  T_blocks      The upper quasi-triangular matrix partitioned into
 *                           blocks.
 * @param[in]  ldT           Leading dimension of T
 * @param[out] T_norms       Upper bounds for blocks in T.
 * @param[in]  p             Partitioning (OpenMP grid) of T.
 * @param[in]  mem_layout    COLUMN_MAJOR or TILE_LAYOUT.
 */
void compute_majorants(
    double ***restrict const T_blocks, int ldT,
    double *restrict const T_norms,
    partitioning_t *p, memory_layout_t mem_layout);


#endif

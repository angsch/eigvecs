#ifndef REFERENCE_H
#define REFERENCE_H

#include "datalayout.h"

void compute_eigenvectors_dtrevc3(
    int n, double ***restrict const T_blocks, int ldT,
    double *restrict const lambda, const int *restrict const lambda_type,
    const int *restrict const selected,
    const partitioning_t *restrict const p,
    const partitioning_t *restrict const p_rhs,
    double ***restrict const Q_blocks, int ldQ,
    memory_layout_t mem_layout);

#endif

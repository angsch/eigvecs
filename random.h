#ifndef RANDOM_H
#define RANDOM_H

#include "defines.h"
#include "datalayout.h"
#include "typedefs.h"

int random_integer(int low, int high);
double random_double(double low, double high);

void generate_eigenvalues(const int n, double complex_ratio,
    double *restrict const lambda, int *restrict const lambda_type);

int generate_selection(const int n, int *restrict const lambda_type,
    int *restrict const selected, double select_ratio);

/*void generate_upper_triangular_matrix(
    const int n, double *T, const int ldT,
    double complex_ratio);*/

void generate_householder_matrix(
    int n,
    int ld, double ***H_blocks, int num_blks,
    const int *restrict const first_row, const int *restrict const first_col,
    memory_layout_t mem_layout);

void generate_upper_triangular_matrix(
    int first_blk, int num_blks,
    double ***restrict const T_blocks, int ld,
    const int *restrict const first_row, const int *restrict const first_col,
    const double *restrict const lambda, const int *restrict const lambda_type,
    memory_layout_t mem_layout);

/*
void generate_distributed_upper_triangular_matrix(
    double ***restrict const T_blocks, int ldT,
    partitioning_t *p, memory_layout_t mem_layout,
    const double *restrict const lambda, const int *restrict const lambda_type,
    two_dim_block_t *dist2D, int myrank);



void generate_upper_triangular_matrix(
    int first_blk, int num_blks,
    double ***restrict const T_blocks, int ld,
    const int *restrict const first_row, const int *restrict const first_col,
    const double *restrict const lambda, const int *restrict const lambda_type,
    memory_layout_t mem_layout);


void generate_distributed_householder_matrix(
    int n, double ***restrict const H_blocks, int ld,
    partitioning_t *p, memory_layout_t mem_layout,
    two_dim_block_t *dist2D, int myrank);


void generate_householder_matrix(
    int first_blk, int num_blks,
    int ld, double ***H_blocks,
    const double *restrict const v,
    const int *restrict const first_row, const int *restrict const first_col,
    memory_layout_t mem_layout);*/

#endif

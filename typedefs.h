#ifndef TYPEDEF_H
#define TYPEDEF_H

enum memory_layout {COLUMN_MAJOR, TILE_LAYOUT};
typedef enum memory_layout memory_layout_t;

typedef struct {
    int num_blks; // num_blks + 1 exists.
    int *first_row;
    int *first_col;
} partitioning_t;



#ifdef INTSCALING
typedef int scaling_t;
#else
typedef double scaling_t;
#endif

#endif

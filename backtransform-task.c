#include "backtransform-task.h"
#include "utils.h"

#include <stdio.h>
#include <mm_malloc.h>




//    Y    := first_write * Y +    Q   *   X       where first_write is {0, 1}
// (n x m)                      (n x k) (k x m)
void basetransform(
    int n, int m, int k, const double *restrict Q, int ldQ,
    omp_lock_t *lock, int *restrict const cnt_adds,
    double *restrict const Y, int ldY,
    double *restrict const X, int ldX)
{
#define Y(i,j) Y[(i) + (j) * ldY]
#define W(i,j) W[(i) + (j) * ldW]

    // Allocate workspace.
    int ldW = get_size_with_padding(n);
    double *W = (double *) _mm_malloc((size_t)ldW * m * sizeof(double),
        ALIGNMENT);

    // Compute W := Q * X.
    dgemm('N', 'N',
          n, m, k,
          1.0, Q, ldQ,
          X, ldX,
          0.0, W, ldW);

    // Wait until lock becomes available. Blocking call is cheaper than taskyield.
    omp_set_lock(lock);

    // Update.
    if ((*cnt_adds) == 0) {
        // No update has been processed so far, write Y := Q * X.
        for (int j = 0; j < m; j++)
            for (int i = 0; i < n; i++)
                Y(i,j) = W(i,j);

    }
    else {
        // Some updates have been processed, write Y := Y + Q * X.
        for (int j = 0; j < m; j++)
            for (int i = 0; i < n; i++)
                Y(i,j) = Y(i,j) + W(i,j);
    }

    // Increment counter atomically - we are in a lock-protected region.
    (*cnt_adds)++;

    omp_unset_lock(lock);

    // Free workspace.
    _mm_free(W);

#undef Y
#undef W
}


//   Yij    :=    Qi:  *   X:j
// (n x m)      (n x k)  (k x m)
void backtransform(
    const partitioning_t *restrict p_rhs,
    int i, int j,
    double ***restrict Q_blocks, int ldQ,
    double ***restrict Y_blocks, int ldY,
    double ***restrict X_blocks, int ldX,
    memory_layout_t layout)
{
    const int *first_row = p_rhs->first_row;
    const int *first_rhs_col = p_rhs->first_col;

    // Compute the actual number of rows and columns.
    const int num_rows = first_row[i + 1] - first_row[i];
    const int num_cols = first_rhs_col[j + 1] - first_rhs_col[j];

    if (layout == COLUMN_MAJOR) {
        // Compute the actual number of columns of Q/rows of X.
        const int num_inner = first_row[j + 1] - first_row[0];

        dgemm('N', 'N', num_rows, num_cols, num_inner,
              1.0, Q_blocks[i][0], ldQ,
              X_blocks[0][j], ldX,
              0.0, Y_blocks[i][j], ldY);
    }
    else { //     TILE_LAYOUT
        // Initialize block Yij with zeros.
        set_zero(num_rows, num_cols, Y_blocks[i][j], num_rows);

        for (int k = 0; k <= j; k++) {
            // Compute the actual number of columns of Q/rows of X.
            const int num_inner = first_row[k + 1] - first_row[k];

            //          Yij         :=       Qik * Xkj.
            // (num_rows x num_cols)
            //         (num_rows x num_inner) (num_inner x num_cols)
            dgemm('N', 'N', num_rows, num_cols, num_inner,
                  1.0, Q_blocks[i][k], num_rows,
                  X_blocks[k][j], num_inner,
                  1.0, Y_blocks[i][j], num_rows);
        }
    }
}

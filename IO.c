#include "IO.h"
#include "datalayout.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>


void print(int n, int m, const double *restrict const A, int ldA)
{
    #define A(i,j) A[(i) + (j) * ldA]

    if (n == 0 || m == 0) {
        printf("[]\n");
    }
    else {
            for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                printf("%.8e, ", A(i,j));
            }
            printf(";\n");
        }
        printf("\n");
    }

    #undef A
}


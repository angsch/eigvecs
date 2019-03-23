#include "eigenvectors.h"
#include "utils.h"
#include "defines.h"
#include "timing.h"
#include "IO.h"
#include "robust.h"
#include "norm.h"
#include "update-task.h"

#include <stdlib.h>
#include <stdio.h>
#include <mm_malloc.h>
#include <string.h>
#include <math.h>





//    Y           := Y -   T   *   X
// (n x num_rhs)        (n x m) (m x num_rhs)
void update(
    int n, int m,
    const double *restrict const T, int ldT, const double tnorm,
    int num_rhs, omp_lock_t *lock,
    double *restrict const Y, int ldY,
    scaling_t *restrict const Yscales, double *restrict const Ynorms,
    double *restrict const Xin, int ldX,
    const scaling_t *restrict const Xscales, const double *restrict const Xinnorms,
    const int *restrict const lambda_type)
{
#define Y(i,j) Y[(i) + (j) * ldY]
#define W(i,j) W[(i) + (j) * ldW]

#ifndef NDEBUG
    printf("update (%dx%d)(%dx%d)\n", n, m, m, num_rhs);
    printf("update thread id = %d\n", omp_get_thread_num());
#endif

    // Pointer to X - either a copy or the original memory.
    double *X;

    // Pointer to norms of X - either a copy or the original memory.
    double *Xnorms;

    // Status flag if X has to be rescaled.
    int rescale_X = 0;

    // Status flag if Y has to be rescaled.
    int rescale_Y = 0;

    // Indicator if xinnorms has to be rescaled. Note that this only affects
    // consistency scaling and not overflow protection.
    int rescale_xnorms = 0;

    // Workspace to store locally computed scaling factors.
    scaling_t tmp_scales[num_rhs];

    ////////////////////////////////////////////////////////////////////////////
    // Compute scaling factor.
    ////////////////////////////////////////////////////////////////////////////

    // CHECK: Is task suspension really faster than a blocking call (in the 
    // non-robust version this was not the case)?
    while (!omp_test_lock(lock)) {
        #pragma omp taskyield
        ;
    }

    for (int k = 0; k < num_rhs; k++) {
        if (Yscales[k] < Xscales[k]) {
            rescale_xnorms = 1;
            break;
        }
    }

    if (rescale_xnorms) {
        // As X is read-only, copy xnorms.
        Xnorms = (double *) _mm_malloc(num_rhs * sizeof(double), ALIGNMENT);
        memcpy(Xnorms, Xinnorms, num_rhs * sizeof(double));

        // Simulate the consistency scaling.
        for (int k = 0; k < num_rhs; k++) {
            if (Yscales[k] < Xscales[k]) {
                // The common scaling factor is Yscales[k].
                const double s = compute_upscaling(Yscales[k], Xscales[k]);

                // Mark X for scaling. Physical rescaling is deferred.
                rescale_X = 1;

                // Update norm.
                Xnorms[k] = s * Xinnorms[k];
            }
            else if (Xscales[k] < Yscales[k]) {
                // The common scaling factor is Xscales[k].
                const double s = compute_upscaling(Xscales[k], Yscales[k]);

                // Mark Y for scaling. Physical rescaling is deferred.
                rescale_Y = 1;

                // Update norm: norm(s * Y) = s * norm(Y).
                Ynorms[k] = s * Ynorms[k];
            }
        }
    }
    else { // !rescale_xnorms.
        // No changes to Xinnorms necessary. Operate on original memory.
        Xnorms = Xinnorms;

        // Xnorms does not need scaling, but Ynorms may do.
        for (int k = 0; k < num_rhs; k++) {
            if (Xscales[k] < Yscales[k]) {
                // The common scaling factor is Xscales[k].
                const double s = compute_upscaling(Xscales[k], Yscales[k]);

                // Mark Y for scaling. Phyiscal rescaling is deferred.
                rescale_Y = 1;

                // Update norm: norm(s * Y) = s * norm(Y).
                Ynorms[k] = s * Ynorms[k];
            }
            // No other case distinctions necessary. Yscales[k] < Xscales[k]
            // cannot occur because !rescale_xnorms.
        }
    }


    ////////////////////////////////////////////////////////////////////////////
    // Apply scaling.
    ////////////////////////////////////////////////////////////////////////////

    // Status flag if update is safe to execute or if rescaling is required.
    int status;

    // Compute scaling factors needed to survive the linear update.
    status = protect_multi_rhs_update(
        Xnorms, num_rhs, tnorm, Ynorms, lambda_type, tmp_scales);

    if (status == RESCALE)
        rescale_X = 1;

    // If X has to be rescaled, take a copy of X and do scaling on the copy.
    if (rescale_X) {
        X = (double *) _mm_malloc((size_t)ldX * num_rhs * sizeof(double), ALIGNMENT);

        for (int k = 0; k < num_rhs; k++) {
            if (Yscales[k] < Xscales[k]) {
                // Copy X and simultaneously rescale.
                // The common scaling factor is Yscales[k]. Combine with
                // robust update scaling factor.
                const double s = compute_combined_upscaling(
                                    Yscales[k], Xscales[k], tmp_scales[k]);
                for (int i = 0; i < m; i++)
                    X[i + ldX * k] = s * Xin[i + ldX * k];
            }
            else if (Xscales[k] < Yscales[k]) {
                // Copy X and simultaneously rescale with robust update factor.
                const double s = convert_scaling(tmp_scales[k]);
                for (int i = 0; i < m; i++)
                    X[i + ldX * k] = s * Xin[i + ldX * k];
            }
            else {
                // Xscales[k] == Yscales[k].

                // Copy X and simultaneously rescale with robust update factor.
                const double s = convert_scaling(tmp_scales[k]);
                for (int i = 0; i < m; i++)
                    X[i + ldX * k] = s * Xin[i + ldX * k];
            }
        }
    }
    else { // !rescale_X
        // No changes to X necessary. Operate on original memory.
        X = Xin;
    }


    // If Y has to be rescaled, directly modify Y.
    if (rescale_Y) {
        for (int k = 0; k < num_rhs; k++) {
            if (Yscales[k] < Xscales[k]) {
                // The common scaling factor is Yscales[k]. Rescale Y with
                // robust update factor, if necessary.
                scale(n, Y + ldY * k, tmp_scales + k);
            }
            else if (Xscales[k] < Yscales[k]) {
                // The common scaling factor is Xscales[k]. Combine with
                // robust update scaling factor.
                const double s = compute_combined_upscaling(
                                    Xscales[k], Yscales[k], tmp_scales[k]);
                for (int i = 0; i < n; i++)
                    Y[i + ldY * k] = s * Y[i + ldY * k];
            }
            else {
                // Xscales[k] == Yscales[k].
                scale(n, Y + ldY * k, tmp_scales + k);
            }
        }
    }

    // Update global scaling of Y. Recall that Yscales has note been updated
    // during the consistency scaling.
#ifdef INTSCALING
    for (int k = 0; k < num_rhs; k++) {
        Yscales[k] = min(Yscales[k], Xscales[k]) + tmp_scales[k];
    }
#else
    for (int k = 0; k < num_rhs; k++) {
        Yscales[k] = minf(Yscales[k], Xscales[k]) * tmp_scales[k];
    }
#endif

    ////////////////////////////////////////////////////////////////////////////
    // Compute update.
    ////////////////////////////////////////////////////////////////////////////

    // Y := Y - T * X.
    dgemm('N', 'N',
          n, num_rhs, m,
          -1.0, T, ldT,
          X, ldX,
          1.0, Y, ldY);

    ////////////////////////////////////////////////////////////////////////////
    // Recompute norms.
    ////////////////////////////////////////////////////////////////////////////

    for (int k = 0; k < num_rhs; k++) {
        if (lambda_type[k] == CMPLX) {
            // We store only one scaling factor per complex eigenvector pair.
            // So interpret columns as real and imaginary part.
            const double *Y_re = Y + k * ldY;
            const double *Y_im = Y + (k + 1) * ldY;

            Ynorms[k] = vector_cmplx_infnorm(n, Y_re, Y_im);

            // Duplicate norm for real and imaginary column.
            Ynorms[k + 1] = Ynorms[k];

            k++;
        }
        else { // lambda_type[k] == REAL
            Ynorms[k] = vector_infnorm(n, Y + k * ldY);
        }
    }

    omp_unset_lock(lock);

    ////////////////////////////////////////////////////////////////////////////
    // Free workspace.
    ////////////////////////////////////////////////////////////////////////////

    if (rescale_X) {
        _mm_free(X);
    }

#undef Y
#undef W
}


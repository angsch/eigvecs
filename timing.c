#include <sys/time.h>
#include <stdlib.h>

#include "timing.h"


double get_time (void)
{
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

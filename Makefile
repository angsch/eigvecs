# Default build:
# make all

TARGET := all
DEFINES := -DINTSCALING -DNDEBUG

# Select mode: release, profile
MODE := release

# Select compiler and linker: intel, gnu
COMPILER := gnu

# Set global defines
DEFINES += -DALIGNMENT=64

# Define dynamically linked libraries
LIBS := -lrt -lm

# ------------------------------------------------------------------------------
# Selection of flags
# ------------------------------------------------------------------------------

# Compiler-specific optimisation, reporting and linker flags
ifeq ($(COMPILER), intel)
	CC               := icc
	DEFINES          +=
	AGGRESSIVE_FLAGS := #-Ofast -unroll-aggressive
	SAFE_FLAGS       := -O2 -xHost -malign-double -qopt-prefetch
	CFLAGS           := -Wall -std=gnu99 $(SAFE_FLAGS) $(AGGRESSIVE_FLAGS) -qopenmp -ipo
	LDFLAGS          := -ipo -qopt-report=1 -qopt-report-phase=vec -qopenmp
	LIBS             += -mkl
else # gnu
	CC               := gcc
	DEFINES          +=
	DIAGNOSTICS      := -Wall -Werror=implicit-function-declaration -Werror=incompatible-pointer-types #-fopt-info
	AGGRESSIVE_FLAGS := #-Ofast
	SAFE_FLAGS       := -O3 -march=native -funroll-loops -fprefetch-loop-arrays -malign-double -LNO:prefetch -g
	CFLAGS           := -Wall -std=gnu99 $(SAFE_FLAGS) $(AGGRESSIVE_FLAGS) -pipe -fopenmp $(DIAGNOSTICS)
	LDFLAGS          := -flto -O3 -fopenmp -g
	LIBS             += -lopenblas -fopenmp
endif



# Enable profiling (-p) and line information (-g), if necessary
ifeq ($(MODE),profile)
	CFLAGS  += -pg
	FCFLAGS += -pg
	# Sorry, we have to forbid inlining when doing lto.
	LDFLAGS += -fno-inline
	LIBS    += -pg
endif


# Select all C source files
SRCS := $(wildcard *.c)
OBJS := $(SRCS:.c=.o)


# ------------------------------------------------------------------------------
# Makefile rules and targets
# ------------------------------------------------------------------------------

.SUFFIXES: .c .o

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $(TARGET) $(LIBS)

%.o : %.c
	$(CC) $(CFLAGS) $(DEFINES) -c $< -o $@

clean: 
	rm -f all *.o ipo_out.optrpt


.PHONY: clean

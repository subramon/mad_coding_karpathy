# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

CFLAGS := -g 
# CFLAGS= -O3
# CFLAGS= -Ofast
# the most basic way of building that is most likely to work on most systems
CFLAGS += -DLLM_EXPT

INCS = -I./codegen/
.PHONY: run

run: run.c \
	codegen/rmsnorm.c  \
	codegen/rope1.c  \
	codegen/rope2.c 
	gcc ${INCS} ${CFLAGS} -c run.c -o run.o 
	gcc ${INCS} ${CFLAGS} -c codegen/rmsnorm.c -mavx2 -mfma -o rmsnorm.o
	gcc ${INCS} ${CFLAGS} -c codegen/rope1.c -mavx2 -mfma -o rope1.o
	gcc ${INCS} ${CFLAGS} -c codegen/rope2.c -mavx2 -mfma -o rope2.o
	$(CC) run.o \
		rmsnorm.o \
		rope1.o \
		rope2.o \
		-o run -lm

rundebug: run.c 
	$(CC) -g -o run run.c -lm
	$(CC) -g -o runq runq.c -lm
# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
# https://simonbyrne.github.io/notes/fastmath/
# -Ofast enables all -O3 optimizations.
# Disregards strict standards compliance.
# It also enables optimizations that are not valid for all standard-compliant programs.
# It turns on -ffast-math, -fallow-store-data-races and the Fortran-specific
# -fstack-arrays, unless -fmax-stack-var-size is specified, and -fno-protect-parens.
# It turns off -fsemantic-interposition.
# In our specific application this is *probably* okay to use
.PHONY: runfast
runfast: run.c
	$(CC) -Ofast -o run run.c -lm
	$(CC) -Ofast -o runq runq.c -lm

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: run.c
	$(CC) -Ofast -fopenmp -march=native run.c  -lm  -o run
	$(CC) -Ofast -fopenmp -march=native runq.c  -lm  -o runq

.PHONY: win64
win64:
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o run.exe -I. run.c win.c
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o runq.exe -I. runq.c win.c

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: rungnu
rungnu:
	$(CC) -Ofast -std=gnu11 -o run run.c -lm
	$(CC) -Ofast -std=gnu11 -o runq runq.c -lm

.PHONY: runompgnu
runompgnu:
	$(CC) -Ofast -fopenmp -std=gnu11 run.c  -lm  -o run
	$(CC) -Ofast -fopenmp -std=gnu11 runq.c  -lm  -o runq

# run all tests
.PHONY: test
test:
	pytest

# run only tests for run.c C implementation (is a bit faster if only C code changed)
.PHONY: testc
testc:
	pytest -k runc

# run the C tests, without touching pytest / python
# to increase verbosity level run e.g. as `make testcc VERBOSITY=1`
VERBOSITY ?= 0
.PHONY: testcc
testcc:
	$(CC) -DVERBOSITY=$(VERBOSITY) -O3 -o testc test.c -lm
	./testc

.PHONY: clean
clean:
	rm -f run
	rm -f runq
	rm -f *.o

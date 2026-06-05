# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

CFLAGS := -g 
# CFLAGS= -O3
# CFLAGS= -Ofast
# Exactly one of these should be defined
# CFLAGS += -DKARPATHY
# CFLAGS += -DGEMINI
CFLAGS += -DDEEPSEEK

INCS = -I./codegen/
.PHONY: run

all : run

swiglu.o : codegen/swiglu.c 
	gcc ${INCS} ${CFLAGS} -c codegen/swiglu.c -mavx2 -mfma -o swiglu.o

dotprod.o : codegen/dotprod.c 
	gcc ${INCS} ${CFLAGS} -c codegen/dotprod.c -mavx2 -mfma -o dotprod.o

saxpy.o : codegen/saxpy.c 
	gcc ${INCS} ${CFLAGS} -c codegen/saxpy.c -mavx2 -mfma -o saxpy.o

vvincr.o : codegen/vvincr.c 
	gcc ${INCS} ${CFLAGS} -c codegen/vvincr.c -mavx2 -mfma -o vvincr.o

rmsnorm.o : codegen/rmsnorm.c 
	gcc ${INCS} ${CFLAGS} -c codegen/rmsnorm.c -mavx2 -mfma -o rmsnorm.o

softmax.o : codegen/softmax.c 
	gcc ${INCS} ${CFLAGS} -c codegen/softmax.c -mavx2 -mfma -o softmax.o

vecmatmul.o : codegen/vecmatmul.c 
	gcc ${INCS} ${CFLAGS} -c codegen/vecmatmul.c -mavx2 -mfma -o vecmatmul.o

run: run.c \
	rmsnorm.o \
	softmax.o  \
	vvincr.o  \
	saxpy.o  \
	dotprod.o  \
	swiglu.o  \
	vecmatmul.o  \
	codegen/rope1.c  \
	codegen/rope2.c 
	gcc ${INCS} ${CFLAGS} -c run.c -o run.o 
	gcc ${INCS} ${CFLAGS} -c codegen/rope1.c -mavx2 -mfma -o rope1.o
	gcc ${INCS} ${CFLAGS} -c codegen/rope2.c -mavx2 -mfma -o rope2.o
	$(CC) run.o \
		rmsnorm.o \
		softmax.o \
		vvincr.o  \
		saxpy.o  \
		dotprod.o  \
		swiglu.o  \
		vecmatmul.o \
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

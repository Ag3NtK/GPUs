NVCC = nvcc

CFLAGS = -O3 -I.

SRC = main.o matrix_mul.o
	
# Ahora NVCC compila los archivos .c
%.o: %.c
	$(NVCC) $(CFLAGS) -c -o $@ $<

# NVCC compila los archivos .cu
%.o: %.cu
	$(NVCC) $(CFLAGS) -c -o $@ $<

# NVCC une todo
matrix_mul: $(SRC) 
	$(NVCC) -o matrix_mult $(SRC) $(CFLAGS)

clean:
	del -f *.o matrix_mult matrix_mult.exe matrix_mult.lib matrix_mult.exp
	
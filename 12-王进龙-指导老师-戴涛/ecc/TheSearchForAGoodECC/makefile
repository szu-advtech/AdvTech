CC=gcc
THREAD_POW=6
N=96
K=32
CFLAGS=-O3 -Wall -pthread -DTHREAD_POW=$(THREAD_POW) -DN=$(N) -DK=$(K)
OBJ=assess_matrix.o
EXE=assess_matrix

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)
    
$(EXE): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f $(OBJ) $(EXE)
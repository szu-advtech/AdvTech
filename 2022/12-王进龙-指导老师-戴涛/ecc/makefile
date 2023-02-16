CC=gcc
THREAD_POW=6
CFLAGS=-O3 -Wall -pthread -DTHREAD_POW=$(THREAD_POW)
OBJ=decoder.o
DEPS=decoder_default_matrix.h
EXE=decoder

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
    
$(EXE): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f $(OBJ) $(EXE)
#include <stdio.h>
#include <stdlib.h>
#include <argp.h>
#include <pthread.h>

// The defines 'THREAD_POW', 'N', and 'K' must be defined when compiling (see the makefile)
#define LEN(arr) ((long long) (sizeof (arr) / sizeof (arr)[0]))
#define NTHREADS 1<<THREAD_POW
#ifndef __SIZEOF_INT128__
    printf("ERROR: Must be run on a machine that can support the __int128 data type.\n");
    exit 1;
#endif

// DECLARATIONS AND GLOBALS
__int128 generator_matrix[K];
unsigned long long weights_per_thread[NTHREADS][N+1];
unsigned long long iterations_per_thread = (1LL<<(K-THREAD_POW));
static __int128 convert_string_to_int128(char* str)
{
    char c = str[0];
    int col_count = 0;
    __int128 result = 0;
    while (c == '0' || c == '1')
    {
        __int128 val = c - '0';
        result |= val<<col_count;
        c = str[++col_count];
    }
    return result;
}

// PER-THREAD CODE
void *find_weights(void *args)
{
    int thread_id = *(int*)args;
    long long counter = thread_id * iterations_per_thread;
    long long message = counter ^ (counter >> 1); // The start-th gray code value

    // Since this thread is starting in the middle of the list of message, we need to manually
    // calculate the corresponding codeword. This is a vector-matrix multiplication (message * generator matrix)
    // But it's easier to just add up all the rows of the generator matrix that won't be set to zero.
    __int128 codeword = 0;
    for (int i=0; i<K; i++)
    {
        if (((1LL<<i) & message))
        {
            codeword = codeword ^ generator_matrix[i];
        }
    }

    // Loop over this thread's subset of messages
    for (long long i=0; i<iterations_per_thread; i++)
    {
        // Look at the weight (popcount) of the current codeword
        // Weight is up to 96 (N)
        // (Have to call popcount twice since it's expecting a 64 bit number)
        int weight = __builtin_popcountll(codeword) + __builtin_popcountll(codeword >> 64);
        weights_per_thread[thread_id][weight]++;

        // And then increment to the next codeword
        // The 'addition' in this case is bitwise xor
        // Take the row at index location, and add it to our current codeword
        // Index goes up to K
        counter += 1;
        int index = __builtin_ctzll(counter);
        codeword = codeword ^ generator_matrix[index];
    }
    return NULL;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("This is an error correcting code assessment program. It is compiled to use %i threads.\n", NTHREADS);
        printf("usage: %s matrix_filename\n", argv[0]);
        printf("   matrix_filename: The generator matrix to assess. The program will read the file and use the contained matrix.\n");   
        return 0;
    }
    
    char* filename = argv[1];
    FILE* file = fopen(filename, "r");
    if (file == NULL)
    {
        printf("ERROR: Couldn't find or open file %s\n", filename);
        return 1;
    }
    
    char buffer[1024];
    int line_count = 0;
    while (fgets(buffer, 1024, file))
    {
        generator_matrix[line_count++%K] = convert_string_to_int128(buffer);
    }
    fclose(file);
    if (line_count != K)
    {
        printf("ERROR: Read the matrix in file %s but read %i lines, and was expecting %i\n", filename, line_count, K);
        return 1;
     }
    
    // Spawn threads
    int thread_ids[NTHREADS];
    pthread_t threads[NTHREADS];
    for (int i=0; i<NTHREADS; i++)
    {
        thread_ids[i] = i;
        if (pthread_create(&threads[i], NULL, find_weights, (void *) &thread_ids[i]) != 0)
        {
            printf("Error: Couldn't spawn threads.");
            return 1;
        }
    }
    // Join all the threads
    for (int i=0; i<NTHREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }
    // Combine results
    for (int i=1; i<NTHREADS; i++)
    {
        for (int j=0; j<N+1; j++)
        {
            weights_per_thread[0][j] += weights_per_thread[i][j];
        }        
    }
    // Report results in csv format
    printf("Weight,Count\n");
    for (int i=0; i<N+1; i++)
    {
        printf("%i,%llu\n", i, weights_per_thread[0][i]);
    }

  return 0;
}

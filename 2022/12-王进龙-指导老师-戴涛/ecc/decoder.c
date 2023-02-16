#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <argp.h>
#include <unistd.h>
#include "decoder_default_matrix.h"

#define LEN(arr) ((long long) (sizeof (arr) / sizeof (arr)[0]))
#define NTHREADS 1<<THREAD_POW
#define N 96
#define K 32
#ifndef __SIZEOF_INT128__
    printf("ERROR: Must be run on a machine that can support the __int128 data type.\n");
    exit 1;
#endif

// DECLARATIONS AND GLOBALS
struct decoding
{
    int distance;
    long long message;
};

__int128 target;
struct decoding decodings[NTHREADS];
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
void *find_best_decoding(void *args)
{
    int thread_id = *(int*)args;
    long long counter = thread_id * iterations_per_thread;
    
    // Exclude the all-0 message
    if (counter == 0) { counter += 1; }

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
    int best_distance = N;
    long long best_message = 0;
    // Loop over this thread's subset of messages
    for (long long i=0; i<iterations_per_thread; i++)
    {
        // Look at the weight (popcount) of the current codeword
        // Weight is up to 96 (N)
        // (Have to call popcount twice since it's expecting a 64 bit number)
        __int128 difference = target ^ codeword;
        int distance = __builtin_popcountll(difference) + __builtin_popcountll(difference >> 64);
        if (distance < best_distance)
        {
            best_distance = distance;
            best_message = message;
        }

        // And then increment to the next codeword
        // The 'addition' in this case is bitwise xor
        // Take the row at index location, and add it to our current codeword
        // Index goes up to K
        counter += 1;
        int index = __builtin_ctzll(counter);
        codeword = codeword ^ generator_matrix[index];
        message = message ^ (1LL<<index);
    }
    
    // Save into the decodings array
    decodings[thread_id].distance = best_distance;
    decodings[thread_id].message = best_message;
    return NULL;
}

// MAIN
int main(int argc, char *argv[])
{
    if (argc <= 1 || argc > 3)
    {
        printf("This is an error correcting code decoder program. It is compiled to use %i threads.\n", NTHREADS);
        printf("usage: %s string_to_decode [matrix_filename]\n", argv[0]);
        printf("   string_to_decode: The codeword or corrupted codeword that you would like to decode into a valid message.");
        printf("   matrix_filename: An optional argument. If provided, the program will read the file and use the contained matrix. Otherwise it will use the built-in matrix.");   
        return 0;
    }
    
    if (argc == 3)
    {
        char* filename = argv[2];
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
    }
    
    char* rcvd_codeword = argv[1];
    if (strlen(rcvd_codeword) != N)
    {
        printf("Error: Codeword <%s> is %li bits long (need %i)\n", rcvd_codeword, strlen(rcvd_codeword), N);
        return 1;
    }

    target = convert_string_to_int128(rcvd_codeword);

    // Spawn threads
    int thread_ids[NTHREADS];
    pthread_t threads[NTHREADS];
    for (int i=0; i<NTHREADS; i++)
    {
        thread_ids[i] = i;
        if (pthread_create(&threads[i], NULL, find_best_decoding, (void *) &thread_ids[i]) != 0)
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
    // Combine results from all threads 
    struct decoding best_decoding = decodings[0];
    for (int i=1; i<NTHREADS; i++)
    {
         if (decodings[i].distance < best_decoding.distance)
         {
                best_decoding = decodings[i];
         }
    }
    
    printf("{\n");
    printf("\"message\": \"");
    for (int i=0; i<K; i++) {
        printf("%lli", (best_decoding.message>>i)&1);
    }
    printf("\",\n");
    printf("\"distance\": %i\n", best_decoding.distance);
    printf("}\n");

    return 0;
}

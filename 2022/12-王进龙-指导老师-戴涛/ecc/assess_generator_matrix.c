#include <stdio.h>
#include <stdlib.h>
#include <argp.h>

#define LEN(arr) ((long long) (sizeof (arr) / sizeof (arr)[0]))
#define N 128

int main(int argc, char *argv[])
{
  char* filename = argv[1];
  FILE* file = fopen(filename, "r");
  if (file == NULL) {
    printf("No such file");
    return 0;
  }
  char buffer[1024];
  __int128 gen_matrix_vals[N];
  int row_count = 0;
  int col_count = 0;
  while (fgets(buffer, 1024, file)){
    // We expect each line to be a row of the matrix, with numbers
    // separated by spaces
    char c = buffer[0];
    col_count = 0;
    __int128 result = 0;
    while (c == '0' || c == '1') {
      int val = c - '0';
      result += val * (((__int128)1)<<col_count);
      c = buffer[++col_count];
    }
    gen_matrix_vals[row_count++] = result;
  }
  fclose(file);

  // And now let's see how we can cycle through a gray code
  long long counter = 1LL; // Counter got 1 through 2**K
  __int128 codeword = 0; // Codewords potentially go up to 2**N
  long long  weights[N+1]; // Weights go from 0 through N (96)
  for (int i=0; i<LEN(weights); i++) {
    weights[i] = 0LL;
  }
  // Loop over all messages (2**K)
  for (long long i=0LL; i<(1LL<<row_count); i++){
    // Look at the weight (popcount) of the current codeword
    // Weight is up to 96 (N)
    // Have to call popcount twice since it's expecting a 64 bit number
    int weight = __builtin_popcountll(codeword) + __builtin_popcountll(codeword >> 64);
    // Increment the appropriate bucket
    weights[weight]++;
    // And then increment to the next codeword
    // The 'addition' in this case is bitwise xor
    // Take the row at index location, and add it to our current codeword
    // Index goes up to K
    int index = __builtin_ctzll(counter++);
    codeword = codeword ^ gen_matrix_vals[index];
  }

  printf("Weight,Count\n");
  for (int i=0; i<=col_count; i++) {
    printf("%i,%i\n", i, weights[i]);
  }

  return 0;
}

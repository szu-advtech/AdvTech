#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <malloc.h>

int main()
{
  void *test[10];
  test[0] = malloc(100);
  memset(test[0], 'a', 100);
  printf("length: %d\n", malloc_usable_size(test[0]));
}
#include<stdlib.h>
#include<stdio.h>
#include<dfc.h>
const int N = 4;
int matrixA[ N][ N];
int matrixB[ N][ N];
int matrixC[ N][ N];



int DF_fanout_DF_output_A = 0;
int DF_persize_DF_output_A;
DF_AD DF_AD_DF_output_A;

int DF_output_A[17];

int DF_fanout_DF_Source_D1 = 1;
int DF_persize_DF_Source_D1;
DF_AD DF_AD_DF_Source_D1;

int DF_fanout_DF_Source_C1 = 1;
int DF_persize_DF_Source_C1;
DF_AD DF_AD_DF_Source_C1;

int DF_fanout_DF_Source_D = 1;
int DF_persize_DF_Source_D;
DF_AD DF_AD_DF_Source_D;

int DF_fanout_DF_Source_C = 1;
int DF_persize_DF_Source_C;
DF_AD DF_AD_DF_Source_C;

int DF_fanout_DF_Source_B1 = 1;
int DF_persize_DF_Source_B1;
DF_AD DF_AD_DF_Source_B1;

int DF_fanout_DF_Source_A1 = 1;
int DF_persize_DF_Source_A1;
DF_AD DF_AD_DF_Source_A1;

int DF_fanout_DF_Source_B = 1;
int DF_persize_DF_Source_B;
DF_AD DF_AD_DF_Source_B;

int DF_fanout_DF_Source_A = 1;
int DF_persize_DF_Source_A;
DF_AD DF_AD_DF_Source_A;



DF_FN DF_FN_FUNS;
DF_FN DF_FN_SOURCED1;
DF_FN DF_FN_SOURCEC1;
DF_FN DF_FN_SOURCED;
DF_FN DF_FN_SOURCEC;
DF_FN DF_FN_SOURCEB1;
DF_FN DF_FN_SOURCEA1;
DF_FN DF_FN_SOURCEB;
DF_FN DF_FN_SOURCEA;
DF_TFL DF_TFL_TABLE;

FILE *fp_sche = NULL;

pthread_mutex_t sched_info_mutex;

// pid_t gettid() { return syscall(SYS_gettid); }

struct timeval program_start;


void SOURCEA(/* DF-C function */)
{
  int DF_count;

int DF_Source_A;

  DF_persize_DF_Source_A = sizeof(DF_Source_A);
  DF_SOURCE_Get_And_Update(&DF_FN_SOURCEA, &DF_count);
int DF_FN_item_index=use_funcname_to_get_item_index(&DF_TFL_TABLE,"SOURCEA");
{
  DF_Source_A = matrixA[(DF_count - 1) / N][0];
  if (DF_count == N * N + 1)
    {
      DF_Source_Stop(&DF_TFL_TABLE, DF_FN_item_index);
    }
}
  DF_AD_UpData(DF_count,&DF_TFL_TABLE, &DF_FN_SOURCEA, &DF_Source_A, DF_persize_DF_Source_A);

}


void SOURCEB(/* DF-C function */)
{
  int DF_count;

int DF_Source_B;

  DF_persize_DF_Source_B = sizeof(DF_Source_B);
  DF_SOURCE_Get_And_Update(&DF_FN_SOURCEB, &DF_count);
int DF_FN_item_index=use_funcname_to_get_item_index(&DF_TFL_TABLE,"SOURCEB");
{
  DF_Source_B = matrixA[(DF_count - 1) / N][1];
  if (DF_count == N * N + 1)
    {
      DF_Source_Stop(&DF_TFL_TABLE, DF_FN_item_index);
    }
}
  DF_AD_UpData(DF_count,&DF_TFL_TABLE, &DF_FN_SOURCEB, &DF_Source_B, DF_persize_DF_Source_B);

}


void SOURCEA1(/* DF-C function */)
{
  int DF_count;

int DF_Source_A1;

  DF_persize_DF_Source_A1 = sizeof(DF_Source_A1);
  DF_SOURCE_Get_And_Update(&DF_FN_SOURCEA1, &DF_count);
int DF_FN_item_index=use_funcname_to_get_item_index(&DF_TFL_TABLE,"SOURCEA1");
{
  DF_Source_A1 = matrixA[(DF_count - 1) / N][2];
  if (DF_count == N * N + 1)
    {
      DF_Source_Stop(&DF_TFL_TABLE, DF_FN_item_index);
    }
}
  DF_AD_UpData(DF_count,&DF_TFL_TABLE, &DF_FN_SOURCEA1, &DF_Source_A1, DF_persize_DF_Source_A1);

}


void SOURCEB1(/* DF-C function */)
{
  int DF_count;

int DF_Source_B1;

  DF_persize_DF_Source_B1 = sizeof(DF_Source_B1);
  DF_SOURCE_Get_And_Update(&DF_FN_SOURCEB1, &DF_count);
int DF_FN_item_index=use_funcname_to_get_item_index(&DF_TFL_TABLE,"SOURCEB1");
{
  DF_Source_B1 = matrixA[(DF_count - 1) / N][3];
  if (DF_count == N * N + 1)
    {
      DF_Source_Stop(&DF_TFL_TABLE, DF_FN_item_index);
    }
}
  DF_AD_UpData(DF_count,&DF_TFL_TABLE, &DF_FN_SOURCEB1, &DF_Source_B1, DF_persize_DF_Source_B1);

}


void SOURCEC(/* DF-C function */)
{
  int DF_count;

int DF_Source_C;

  DF_persize_DF_Source_C = sizeof(DF_Source_C);
  DF_SOURCE_Get_And_Update(&DF_FN_SOURCEC, &DF_count);
int DF_FN_item_index=use_funcname_to_get_item_index(&DF_TFL_TABLE,"SOURCEC");
{
  DF_Source_C = matrixB[0][(DF_count - 1) % N];
  if (DF_count == N * N + 1)
    {
      DF_Source_Stop(&DF_TFL_TABLE, DF_FN_item_index);
    }
}
  DF_AD_UpData(DF_count,&DF_TFL_TABLE, &DF_FN_SOURCEC, &DF_Source_C, DF_persize_DF_Source_C);

}


void SOURCED(/* DF-C function */)
{
  int DF_count;

int DF_Source_D;

  DF_persize_DF_Source_D = sizeof(DF_Source_D);
  DF_SOURCE_Get_And_Update(&DF_FN_SOURCED, &DF_count);
int DF_FN_item_index=use_funcname_to_get_item_index(&DF_TFL_TABLE,"SOURCED");
{
  DF_Source_D = matrixB[1][(DF_count - 1) % N];
  if (DF_count == N * N + 1)
    {
      DF_Source_Stop(&DF_TFL_TABLE, DF_FN_item_index);
    }
}
  DF_AD_UpData(DF_count,&DF_TFL_TABLE, &DF_FN_SOURCED, &DF_Source_D, DF_persize_DF_Source_D);

}


void SOURCEC1(/* DF-C function */)
{
  int DF_count;

int DF_Source_C1;

  DF_persize_DF_Source_C1 = sizeof(DF_Source_C1);
  DF_SOURCE_Get_And_Update(&DF_FN_SOURCEC1, &DF_count);
int DF_FN_item_index=use_funcname_to_get_item_index(&DF_TFL_TABLE,"SOURCEC1");
{
  DF_Source_C1 = matrixB[2][(DF_count - 1) % N];
  if (DF_count == N * N + 1)
    {
      DF_Source_Stop(&DF_TFL_TABLE, DF_FN_item_index);
    }
}
  DF_AD_UpData(DF_count,&DF_TFL_TABLE, &DF_FN_SOURCEC1, &DF_Source_C1, DF_persize_DF_Source_C1);

}


void SOURCED1(/* DF-C function */)
{
  int DF_count;

int DF_Source_D1;

  DF_persize_DF_Source_D1 = sizeof(DF_Source_D1);
  DF_SOURCE_Get_And_Update(&DF_FN_SOURCED1, &DF_count);
int DF_FN_item_index=use_funcname_to_get_item_index(&DF_TFL_TABLE,"SOURCED1");
{
  DF_Source_D1 = matrixB[3][(DF_count - 1) % N];
  if (DF_count == N * N + 1)
    {
      DF_Source_Stop(&DF_TFL_TABLE, DF_FN_item_index);
    }
}
  DF_AD_UpData(DF_count,&DF_TFL_TABLE, &DF_FN_SOURCED1, &DF_Source_D1, DF_persize_DF_Source_D1);

}


void FUNS(/* DF-C function */)
{
int DF_Source_D1;

  DF_persize_DF_Source_D1 = sizeof(DF_Source_D1);
int DF_Source_C1;

  DF_persize_DF_Source_C1 = sizeof(DF_Source_C1);
int DF_Source_D;

  DF_persize_DF_Source_D = sizeof(DF_Source_D);
int DF_Source_C;

  DF_persize_DF_Source_C = sizeof(DF_Source_C);
int DF_Source_B1;

  DF_persize_DF_Source_B1 = sizeof(DF_Source_B1);
int DF_Source_A1;

  DF_persize_DF_Source_A1 = sizeof(DF_Source_A1);
int DF_Source_B;

  DF_persize_DF_Source_B = sizeof(DF_Source_B);
int DF_Source_A;

  DF_persize_DF_Source_A = sizeof(DF_Source_A);
int DF_output_A;

  DF_persize_DF_output_A = sizeof(DF_output_A);
  int DF_count = DF_AD_GetData(&DF_FN_FUNS, &DF_Source_D1, DF_persize_DF_Source_D1, &DF_Source_C1, DF_persize_DF_Source_C1, &DF_Source_D, DF_persize_DF_Source_D, &DF_Source_C, DF_persize_DF_Source_C, &DF_Source_B1, DF_persize_DF_Source_B1, &DF_Source_A1, DF_persize_DF_Source_A1, &DF_Source_B, DF_persize_DF_Source_B, &DF_Source_A, DF_persize_DF_Source_A);
{
  DF_output_A = DF_Source_A * DF_Source_C + DF_Source_B * DF_Source_D;
  printf("total: -- %d\n", DF_output_A);
  matrixC[(DF_count - 1) / N][(DF_count - 1) % N] = DF_output_A;
}
  DF_AD_UpData(DF_count,&DF_TFL_TABLE, &DF_FN_FUNS, &DF_output_A, DF_persize_DF_output_A);

}


int __original_main()
{
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
        {
          matrixA[i][j] = i + j;
          matrixB[i][j] = i - j;
        }
    }
  DF_Run(&DF_TFL_TABLE, 17);
  for (int i = 0; i < N; i++)
    {
      for (int j = 0; j < N; j++)
        {
          printf("%d ", matrixC[i][j]);
        }
      printf("\n");
    }
  return (0);
}

/* DF-C-generated main() */
int main(int argc, char **argv)
{
  gettimeofday(&program_start, NULL);
  fp_sche = fopen("./sched_info.txt", "w+");
  pthread_mutex_init(&sched_info_mutex, NULL);
  DF_ADInit(&DF_AD_DF_output_A, sizeof(int), DF_fanout_DF_output_A);
  DF_ADInit(&DF_AD_DF_Source_D1, sizeof(int), DF_fanout_DF_Source_D1);
  DF_ADInit(&DF_AD_DF_Source_C1, sizeof(int), DF_fanout_DF_Source_C1);
  DF_ADInit(&DF_AD_DF_Source_D, sizeof(int), DF_fanout_DF_Source_D);
  DF_ADInit(&DF_AD_DF_Source_C, sizeof(int), DF_fanout_DF_Source_C);
  DF_ADInit(&DF_AD_DF_Source_B1, sizeof(int), DF_fanout_DF_Source_B1);
  DF_ADInit(&DF_AD_DF_Source_A1, sizeof(int), DF_fanout_DF_Source_A1);
  DF_ADInit(&DF_AD_DF_Source_B, sizeof(int), DF_fanout_DF_Source_B);
  DF_ADInit(&DF_AD_DF_Source_A, sizeof(int), DF_fanout_DF_Source_A);

  DF_FNInit1(&DF_FN_FUNS, &FUNS, "FUNS", 8, &DF_AD_DF_Source_D1, &DF_AD_DF_Source_C1, &DF_AD_DF_Source_D, &DF_AD_DF_Source_C, &DF_AD_DF_Source_B1, &DF_AD_DF_Source_A1, &DF_AD_DF_Source_B, &DF_AD_DF_Source_A);
  DF_FNInit2(&DF_FN_FUNS, 1, &DF_AD_DF_output_A);
  DF_FNInit1(&DF_FN_SOURCED1, &SOURCED1, "SOURCED1", 0);
  DF_FNInit2(&DF_FN_SOURCED1, 1, &DF_AD_DF_Source_D1);
  DF_FNInit1(&DF_FN_SOURCEC1, &SOURCEC1, "SOURCEC1", 0);
  DF_FNInit2(&DF_FN_SOURCEC1, 1, &DF_AD_DF_Source_C1);
  DF_FNInit1(&DF_FN_SOURCED, &SOURCED, "SOURCED", 0);
  DF_FNInit2(&DF_FN_SOURCED, 1, &DF_AD_DF_Source_D);
  DF_FNInit1(&DF_FN_SOURCEC, &SOURCEC, "SOURCEC", 0);
  DF_FNInit2(&DF_FN_SOURCEC, 1, &DF_AD_DF_Source_C);
  DF_FNInit1(&DF_FN_SOURCEB1, &SOURCEB1, "SOURCEB1", 0);
  DF_FNInit2(&DF_FN_SOURCEB1, 1, &DF_AD_DF_Source_B1);
  DF_FNInit1(&DF_FN_SOURCEA1, &SOURCEA1, "SOURCEA1", 0);
  DF_FNInit2(&DF_FN_SOURCEA1, 1, &DF_AD_DF_Source_A1);
  DF_FNInit1(&DF_FN_SOURCEB, &SOURCEB, "SOURCEB", 0);
  DF_FNInit2(&DF_FN_SOURCEB, 1, &DF_AD_DF_Source_B);
  DF_FNInit1(&DF_FN_SOURCEA, &SOURCEA, "SOURCEA", 0);
  DF_FNInit2(&DF_FN_SOURCEA, 1, &DF_AD_DF_Source_A);

  DF_SourceInit(&DF_TFL_TABLE, 8, &DF_FN_SOURCED1, &DF_FN_SOURCEC1, &DF_FN_SOURCED, &DF_FN_SOURCEC, &DF_FN_SOURCEB1, &DF_FN_SOURCEA1, &DF_FN_SOURCEB, &DF_FN_SOURCEA);
  DF_Init(&DF_TFL_TABLE, 9, &DF_FN_FUNS, &DF_FN_SOURCED1, &DF_FN_SOURCEC1, &DF_FN_SOURCED, &DF_FN_SOURCEC, &DF_FN_SOURCEB1, &DF_FN_SOURCEA1, &DF_FN_SOURCEB, &DF_FN_SOURCEA);
  DF_OutputInit(&DF_TFL_TABLE, 1, &DF_AD_DF_output_A);
  int DF_original_main_ret = (int) __original_main();
  fclose(fp_sche);
  void** result = DF_Result(&DF_TFL_TABLE);
  return(DF_original_main_ret);
}


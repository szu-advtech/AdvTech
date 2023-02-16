
const int N  = 4;
int matrixA[N][N];
int matrixB[N][N];
int matrixC[N][N];


void SOURCEA(;int DF_Source_A)
{
        DF_Source_A = matrixA[(DF_count-1)/N][0];
        if (DF_count == N*N+1)
        {
            DF_Source_Stop();
        }
        // printf("COUNT: -- %d\n", DF_count);
}
void SOURCEB(;int DF_Source_B)
{
        DF_Source_B = matrixA[(DF_count-1)/N][1];
        if (DF_count == N*N+1)
        {
            DF_Source_Stop();
        }
}
void SOURCEA1(;int DF_Source_A1)
{
        DF_Source_A1 = matrixA[(DF_count-1)/N][2];
        if (DF_count == N*N+1)
        {
            DF_Source_Stop();
        }
        // printf("COUNT: -- %d\n", DF_count);
}
void SOURCEB1(;int DF_Source_B1)
{
        DF_Source_B1 = matrixA[(DF_count-1)/N][3];
        if (DF_count == N*N+1)
        {
            DF_Source_Stop();
        }
}
void SOURCEC(;int DF_Source_C)
{
        DF_Source_C = matrixB[0][(DF_count-1)%N];
        if (DF_count == N*N+1)
        {
            DF_Source_Stop();
        }
}
void SOURCED(;int DF_Source_D)
{
        DF_Source_D = matrixB[1][(DF_count-1)%N];
        if (DF_count == N*N+1)
        {
            DF_Source_Stop();
        }
}
void SOURCEC1(;int DF_Source_C1)
{
        DF_Source_C1 = matrixB[2][(DF_count-1)%N];
        if (DF_count == N*N+1)
        {
            DF_Source_Stop();
        }
}
void SOURCED1(;int DF_Source_D1)
{
        DF_Source_D1 = matrixB[3][(DF_count-1)%N];
        if (DF_count == N*N+1)
        {
            DF_Source_Stop();
        }
}

void FUNS(int DF_Source_A, int DF_Source_B, int DF_Source_A1, int DF_Source_B1,
        int DF_Source_C, int DF_Source_D,int DF_Source_C1, int DF_Source_D1; int DF_output_A)
{
        DF_output_A = DF_Source_A * DF_Source_C+ DF_Source_B * DF_Source_D;
        printf("total: -- %d\n", DF_output_A);
        matrixC[(DF_count-1)/N][(DF_count-1)%N] = DF_output_A;
}
int main()
{
        for (int i = 0; i < N; i++)
        {
                for (int j = 0; j < N; j++)
                {
                        matrixA[i][j] = i+j;
                        matrixB[i][j] = i-j;
                }
        }
        
        DF_Run(17);
        for (int i = 0; i < N; i++)
        {
                for (int j = 0; j < N; j++)
                {
                        printf("%d ", matrixC[i][j]);
                }
                printf("\n");
        }
        
        return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <sys/stat.h>

// 4139823 = 4.0M
#define MAX_PAYLOAD 1024 // 2kb
char *newName = "/home/pi/Desktop/VMAC-Exp/videos/newTest.flv";
struct package
{
    uint16_t seq;
    uint16_t times;
    uint16_t size;
    uint8_t payload[MAX_PAYLOAD];
};

int main()
{

    // 建立、绑定udpsocket
    int sockListen;
    if ((sockListen = socket(AF_INET, SOCK_DGRAM, 0)) == -1)
    {
        printf("socket fail\n");
        return -1;
    }
    int set = 1;
    setsockopt(sockListen, SOL_SOCKET, SO_REUSEADDR, &set, sizeof(int));
    struct sockaddr_in recvAddr;
    memset(&recvAddr, 0, sizeof(struct sockaddr_in));
    recvAddr.sin_family = AF_INET;
    recvAddr.sin_port = htons(8008);
    recvAddr.sin_addr.s_addr = INADDR_ANY;
    // 必须绑定，否则无法监听
    if (bind(sockListen, (struct sockaddr *)&recvAddr, sizeof(struct sockaddr)) == -1)
    {
        printf("bind fail\n");
        return -1;
    }
    else
    {
        printf("bind successful\n");
    }

    // 结构体数组存储接收数据
    FILE *newFp = fopen(newName, "wb+");
    uint16_t seq = 0;
    int times = 1;
    // 通过recv接收,seq = times就停
    int recvbytes;
    int addrLen = sizeof(struct sockaddr_in);
    int flag = 1; // 标志是否第一次传送
    // void *newData[2000];
    struct package *recv;
    struct package *recvbuf = NULL;
    recvbuf = (struct package *)malloc(sizeof(struct package));

    // storage result file
    FILE *resultFp = fopen("recfv_result.txt", "a");
    // record loss package
    double loss = 0;
    //float loss_rate;

    // times = total seq
    do
    {
        // 每次将buf清零;
        memset(recvbuf, 0, sizeof(struct package));

        // 接收头部 sizeof(struct package)
        if ((recvbytes = recvfrom(sockListen, recvbuf, sizeof(struct package), 0,
                                  (struct sockaddr *)&recvAddr, &addrLen)) != -1)
        {
            printf("receive a broadCast: seq = %d times = %d \n", recvbuf->seq, recvbuf->times);
            fprintf(resultFp, "receive a broadCast: seq = %d times = %d \n", recvbuf->seq, recvbuf->times);
            if (flag == 1)
            {
                // printf("flag = 1\n");
                // set total times
                times = recvbuf->times;
                // 直接分配所有内存（另一个做法：每次单独分配内存）
                // 维护链表数据结构，每次收到一个数据，分配内存，并插入链表，最后对链表排序
                recv = (struct package *)malloc((sizeof(struct package)) * times);
                for (int i = 0; i < times; i++)
                {
                    recv[i].size = 0;
                }
                // printf("recvbuf->times = %d", recvbuf->times);
            }

            recv[recvbuf->seq].seq = recvbuf->seq;
            recv[recvbuf->seq].size = recvbuf->size;
            recv[recvbuf->seq].times = recvbuf->times;
            memcpy(recv[recvbuf->seq].payload, recvbuf->payload, recvbuf->size);
        }
        else
        {
            printf("recvfrom fail\n");
        }

        if (flag == 1)
        {
            flag = 0;
        }

        seq = recvbuf->seq;
        // printf("seq: %d\n", seq);
    } while (seq < times - 1);

    // 遍历循环，合成接受的信息
    for (int i = 0; i < times; i++)
    {
        if (recv[i].size != 0)
        {
            fwrite(recv[i].payload, recv[i].size, 1, newFp);
        }
        else
        {
            // 记录丢包
            loss++;
        }
    }
    fprintf(resultFp, "loss rate = %f\n", ((double)loss / (double)times));
    fprintf(resultFp, "\n--------------------------------------------\n");
    // 关闭socket，释放内存空间
    fclose(newFp);
    fclose(resultFp);
    close(sockListen);
    free(recvbuf);
    free(recv);
}

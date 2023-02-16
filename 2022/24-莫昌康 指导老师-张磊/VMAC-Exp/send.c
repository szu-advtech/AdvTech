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

char *name = "/home/pi/Desktop/VMAC-Exp/videos/videoTest.flv";
struct package
{
    uint16_t seq;
    uint16_t times; // 总共包的数量
    uint16_t size;  // 数据大小
};

int main()
{
    // 建立udpsocket
    int brdcFd;
    if ((brdcFd = socket(PF_INET, SOCK_DGRAM, 0)) == -1)
    {
        printf("socket fail\n");
        return -1;
    }
    int optval = 1; // 这个值一定要设置，否则可能导致sendto()失败
    setsockopt(brdcFd, SOL_SOCKET, SO_BROADCAST | SO_REUSEADDR, &optval, sizeof(int));
    struct sockaddr_in theirAddr;
    memset(&theirAddr, 0, sizeof(struct sockaddr_in));
    theirAddr.sin_family = AF_INET;
    theirAddr.sin_addr.s_addr = inet_addr("192.168.0.255");
    theirAddr.sin_port = htons(8008);
    int sendBytes;
    // 文件分片
    FILE *fp = NULL;
    int times;
    int left;

    fp = fopen(name, "rb");

    struct stat sb;
    if (stat(name, &sb) == -1)
    {
        perror("stat");
        exit(EXIT_FAILURE);
    }
    // 切割多少份
    times = sb.st_size / MAX_PAYLOAD;
    left = sb.st_size % MAX_PAYLOAD;

    // file: 0 1 2 3 .... times
    void *newData[times];
    int seq = 0;
    // 存储结果
    FILE *resultFp = NULL;
    resultFp = fopen("send_result.txt", "a");
    for (int i = 0; i < times; i++)
    {
        // 存储到数据结构中，每次传一个struct
        struct package p;
        memset(&p, 0, sizeof(struct package));
        int len = 0;

        if (i != times - 1)
        {
            newData[i] = malloc(MAX_PAYLOAD + sizeof(struct package));
            fread(newData[i] + sizeof(struct package), MAX_PAYLOAD, 1, fp);
            len = MAX_PAYLOAD;
        }
        else
        {
            newData[i] = malloc(left + sizeof(struct package));
            fread(newData[i] + sizeof(struct package), left, 1, fp);
            len = left;
        }
        p.times = times;
        p.seq = seq;
        p.size = len;
        memcpy(newData[i], &p, sizeof(struct package));

        // p.data = (uint8_t *)newData[i];
        seq += 1;
        // 通过sendto传送
        if ((sendBytes = sendto(brdcFd, newData[i], sizeof(struct package) + len, 0,
                                (struct sockaddr *)&theirAddr, sizeof(struct sockaddr))) == -1)
        {
            printf("sendto fail, errno=%d\n", errno);
            return -1;
        }
        // 显示传送消息
        printf("seq = %d times = %d ,msgLen=%ld, sendBytes=%d\n",
               p.seq, p.times, (len + sizeof(struct package)), sendBytes);
        // 将信息输出到txt中

        if (resultFp != NULL)
        {
            fprintf(resultFp, "seq = %d times = %d ,msgLen=%ld, sendBytes=%d\n",
                    p.seq, p.times, (len + sizeof(struct package)), sendBytes);
        }
    }

    fprintf(resultFp, "\n--------------------------------------------\n");
    // 关闭socket,关文件，释放内存
    close(brdcFd);
    fclose(fp);
    fclose(resultFp);
    for (int i = 0; i < times; i++)
    {
        free(newData[i]);
    }
    return 0;
}
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include "vmac-usrsp.h"
#include <sys/stat.h>
#include <malloc.h>

/**
 * DOC: Introdcution
 * This document defines a code to implement VMAC sender or receiver
 */

/**
 * DOC : Using VMAC sender
 * This receiver is a standard C executable. Simply compile and run
 * Eg: gcc stress-test.c csiphash.c vmac-usrsp.c -pthread -lm
 *
 *  ./a.out p  --> p signfies this is the sender/producer
 */

/**
 * DOC : Using VMAC receiver
 * This receiver is a standard C executable. Simply compile and run
 * Eg: gcc stress-test.c csiphash.c vmac-usrsp.c -pthread -lm
 *
 *  ./a.out c  --> c signfies this is the receiver/consumer
 */

/**
 * DOC : Warning
 * In a standard test run, always run the sender BEFORE
 * you run the receiver as the sender waits for an interest from the receiver.
 * Do not use both -p and -c arguments while running the code. Use either one
 */

// defines sender thread, used by pthread_create()
pthread_t sendth;
// volatile means the variable maybe is shared in threads
volatile int running2 = 0;
volatile int total;
volatile int consumer = 0;
volatile int producer = 0;
int times = 0;
// (unused)FILE* 指针作为文件句柄，是文件访问的唯一标识，它由fopen函数创建，fopen打开文件成功，则返回一个有效的FILE*指针，否则返回空指针NULL
FILE *sptr, *cptr, *fptr;
double loss = 0.0;
int c;
// tm:time, we can get time and date by using struct tm
struct tm *loc;
unsigned int count = 0;
long ms;
// time_t 用来存储从1970年到现在经过了多少秒
time_t s;
struct timespec spec;
int window[1500];
double intTime;
unsigned int ifReceived = 0;
// 文件
char *name = "/home/pi/Desktop/VMAC-Exp/videos/videoTest.flv";
char *newName = "/home/pi/Desktop/VMAC-Exp/videos/newTest.flv";
char *recordAlpha = "/home/pi/Desktop/VMAC-Exp/record/alpha.txt";
// 将接收的信息存储到文件中
FILE *newFp = NULL;
// 存储对应seq的指针数组
char *all_data[3000];
struct package_info current_package;

#define INTERSET_LENGTH 70      /* the length of interest frame */
#define SEND_DATA_RATE 54.0     /* the rate when sending data frame */
#define FRAME_PAYLOAD_SIZE 1024 /* 1024 Bytes per-frame */
#define FRAMES_PER_RUN 2295     /* the number of frames needing to be send in per-run */
#define DEBUG 1

// 读第一个文件的内容，保证视频能运行
char *first_data;

/**
 * vmac_send_interest  - Sends interest packet
 *
 * Creates an interest frame and sends it. Run as a C thread process
 *
 * Arguments : @tid : Thread id which is a automatically created when calling
 * pthread_create. Do NOT set this manually
 *
 * @param      tid   thread ID.
 *
 * @return     void
 */
void *vmac_send_interest(void *tid)
{
    int i;
    // TODO: dataname current is not used?
    char *dataname = "chat";
    uint16_t name_len = strlen(dataname);
    char buffer[INTERSET_LENGTH] = "buffer";
    total = 0;
    struct vmac_frame frame;
    struct meta_data meta;

    while (1)
    {
        // if (ifReceived == 1)
        // {
        //     break;
        // }
        newFp = fopen(newName, "wb+");
        printf("open newFp\n");
        total = 0;
        frame.buf = buffer;
        frame.len = INTERSET_LENGTH;
        frame.InterestName = dataname;
        frame.name_len = name_len;
        meta.type = VMAC_FC_INTEREST;
        meta.rate = 6.5; /* interest 的发送速率不用太高，6.5Mbps */
        send_vmac(&frame, &meta);
        clock_gettime(CLOCK_REALTIME, &spec);
        s = spec.tv_sec;
        ms = round(spec.tv_nsec / 1.0e6);
        intTime = spec.tv_sec;
        intTime += spec.tv_nsec / 1.0e9;
        if (ms > 999)
        {
            s++;
            ms = 0;
        }
        printf("=====================\n=====================NEW RUN=====================\n=====================");
        printf("Sent @ timestamp=%lu %" PRIdMAX ".%03ld\n", (unsigned long)time(NULL), (intmax_t)s, ms);
        // 每 100s 发送一次兴趣帧，但是应该需要取消
        sleep(35);

        printf("current_package.pack_num: %d\n", current_package.pack_num);
        if (current_package.pack_num != 0)
        {
            for (int i = 0; i < current_package.pack_num; i++)
            {
                printf("i: %d all_data[i]: %p\n", i, all_data[i]);
                if (i == 0 && all_data[i] == NULL)
                {
                    fwrite(first_data, FRAME_PAYLOAD_SIZE, 1, newFp);
                }
                if (all_data[i] != NULL)
                {
                    // printf("i: %d, length: %d\n", i, malloc_usable_size(all_data[i]));
                    int len = FRAME_PAYLOAD_SIZE;
                    if (i == current_package.pack_num - 1)
                    {
                        len = 425;
                    }
                    fwrite(all_data[i] + sizeof(struct package_info), len, 1, newFp);
                }
            }

            for (int i = 0; i < current_package.pack_num; i++)
            {
                if (all_data[i] != NULL)
                {
                    free(all_data[i]);
                }
            }
        }
        memset(all_data, 0, sizeof(all_data));
        fclose(newFp);
        printf("close newFp\n");

        sleep(30);
        // ifReceived = 1; /* 已经接收过帧了 */
    }
}

/**
 *  vmac_send_data - VMAC producer
 *
 *  Creates data frames and sends them to receiver(s). Run as a C thread process
 *
 *  Arguments :
 *  @tid : Thread id which is a automatically created when calling pthread_create. In this case
 *  not run as thread. Default value of 0 to be used
 */

void *vmac_send_data(void *tid)
{
    char *dataname = "chat";
    int i = 0, j = 0;
    uint16_t name_len = strlen(dataname);
    uint16_t len = FRAME_PAYLOAD_SIZE - 1;
    struct vmac_frame frame;
    struct meta_data meta;
    running2 = 1;
    printf("Sleeping for 15 seconds\n");
    sleep(15);
    printf("Sending no.%d\n", times++);
    meta.type = VMAC_FC_DATA;
    meta.rate = SEND_DATA_RATE;
    // meta.rate = 60.0;
    FILE *fp = fopen(name, "rb");
    int pack_num;
    int left;
    struct stat sb;
    if (stat(name, &sb) == -1)
    {
        perror("stat");
        exit(EXIT_FAILURE);
    }

    // 切割多少份
    pack_num = sb.st_size / FRAME_PAYLOAD_SIZE;
    left = sb.st_size % FRAME_PAYLOAD_SIZE;

#ifdef DEBUG
    printf("ready to step in loop");
#endif

    for (i = 0; i < pack_num; i++)
    {
        // 将数据信息存到结构体中
        char *newData = NULL;
        // 复制控制信息
        struct package_info pinfo;
        memset(&pinfo, 0, sizeof(struct package_info));
        pinfo.pack_num = pack_num;

        // 每次从文件取2KB数据
        int len = 0;
        if (i != pack_num - 1)
        {
            newData = malloc(FRAME_PAYLOAD_SIZE + sizeof(struct package_info));
            memcpy(newData, &pinfo, sizeof(struct package_info));
            fread(newData + sizeof(struct package_info), FRAME_PAYLOAD_SIZE, 1, fp); // 指针p加上数值a是指p所指向的地址往后偏移a字节
            len = FRAME_PAYLOAD_SIZE + sizeof(struct package_info);
        }
        else
        {
            newData = malloc(left + sizeof(struct package_info));
            memcpy(newData, &pinfo, sizeof(struct package_info));
            fread(newData + sizeof(struct package_info), left, 1, fp);
            len = left + sizeof(struct package_info);
        }

        // seq放到meta中
        meta.seq = i;
        // 组帧
        frame.len = len;     // data_len放到frame中
        frame.buf = newData; // buf 是一个地址，指向了真实数据，在堆里面
        frame.InterestName = dataname;
        frame.name_len = 4;
        printf("sending vmac frame: %d  %d\n ", meta.seq, frame.len);
        int result = send_vmac(&frame, &meta);
        if (result != 0)
        {
            perror("result error\n");
            free(newData);
        }
        free(newData);
        // sleep(1);
    }
    // 关闭文件，释放内存
    fclose(fp);
    running2 = 0;
}

/**
 * recv_frame - VMAC recv frame function
 *
 * @param      frame  struct containing frame buffer, interestname (if available), and their lengths respectively.
 * @param      meta   The meta meta information about frame currently: type, seq, encoding, and rate,
 */
void callbacktest(struct vmac_frame *frame, struct meta_data *meta)
{
    uint8_t type = meta->type;
    uint64_t enc = meta->enc;
    double goodput;
    char *buff = frame->buf;
    uint16_t len = frame->len;
    uint16_t seq = meta->seq;
    double frameSize = 0.008928; /* in megabits 1116 bytes after V-MAC and 802.11 headers*/
    uint16_t interestNameLen = frame->name_len;
    double timediff;
    double waittime = 3; /* 15 seconds waiting/sleep for all interests to come in */
    struct package_info pinfo;
    clock_gettime(CLOCK_REALTIME, &spec);
    s = spec.tv_sec;
    ms = round(spec.tv_nsec / 1.0e6);
    if (ms > 999)
    {
        s++;
        ms = 0;
    }
    timediff = spec.tv_sec;
    timediff += spec.tv_nsec / 1.0e9;
    timediff = timediff - intTime;
    uint8_t *newData;

    FILE *fp;

    // 如果是 producer 收到 Interest 帧
    if (type == VMAC_FC_INTEREST && producer == 1 && running2 == 0)
    {
        printf("type:%u and seq=%d and count=%u @%" PRIdMAX ".%03ld\n", type, seq, count, (intmax_t)s, ms);
        pthread_create(&sendth, NULL, vmac_send_data, (void *)0);
    }
    // 如果是 consumer 收到 data 帧
    else if (type == VMAC_FC_DATA && consumer)
    {
        total++;
        loss = ((double)(FRAMES_PER_RUN - total) / FRAMES_PER_RUN) * (double)100;
        goodput = (double)(total * frameSize) / (timediff - waittime);
        printf("type:%u | seq=%d | loss=%f | length=%d |T= %f\n",
               type, seq, loss, frame->len, timediff - waittime);
        // printf("content= %s \n length =%d\n", frame->buf, frame->len);
        if (seq > 49757)
        {
            printf("open file");
            fp = fopen(recordAlpha, "a");
            fprintf(fp, "type:%u | seq=%d | loss=%f | length=%d |T= %f\n",
                    type, seq, loss, frame->len, timediff - waittime);
            fclose(fp);
            printf("close file");
        }
        // 拆分frame = package_info + data, 将每个数据对应的seq存储到all_data对应下标内存中
        memset(&pinfo, 0, sizeof(struct package_info));
        memcpy(&pinfo, frame->buf, sizeof(struct package_info));

        all_data[seq] = malloc(frame->len);
        memcpy(all_data[seq], frame->buf, frame->len);

        current_package.pack_num = pinfo.pack_num;
    }
    free(frame);
    free(meta);
}

/**
 *  run_vmac  - Decides if sender or receiver.
 *
 *  Decides if VMAC sender of receiver
 *
 *  Arguments :
 *  @weare: 0 - Sender, 1 - Receiver
 *
 */
void run_vmac(int weare)
{
    uint8_t type;
    uint16_t len, name_len;
    uint8_t flags = 0;
    pthread_t consumerth;
    char dataname[1600];
    char choice;

    choice = weare;
    if (choice == 0)
    {
        printf("We are producer\n");
        producer = 1;
    }
    else if (choice == 1)
    {
        printf("We are consumer\n");
        running2 = 1;
        producer = 0;
        consumer = 1;
        pthread_create(&consumerth, NULL, vmac_send_interest, (void *)0);
        // 设置第一个文件内容块
        FILE *fp = fopen(name, "rb");
        struct stat sb;
        if (stat(name, &sb) == -1)
        {
            perror("stat");
            exit(EXIT_FAILURE);
        }

        // 将数据信息存到结构体中
        first_data = NULL;

        // 从文件头取2KB数据
        first_data = malloc(FRAME_PAYLOAD_SIZE);
        fread(first_data, FRAME_PAYLOAD_SIZE, 1, fp); // 指针p加上数值a是指p所指向的地址往后偏移a字节

        // 关闭文件，释放内存
        fclose(fp);
    }
}

/**
 * main - Main function
 *
 * Function registers, calls run_vmac
 *
 * Arguments
 * p or c (Look at DOC Using VMAC sender or VMAC receiver
 */
int main(int argc, char *argv[])
{
    int weare = 0;
    // void(*ptr), void表示该指针变量ptr可以指向返回值为void的函数, 后面的括号表示该函数的参数
    void (*ptr)(struct vmac_frame *, struct meta_data *) = &callbacktest;
    vmac_register(ptr);
    if (argc < 2)
    {
        printf("%s\n", "please input a mode(p or c)");
        return -1;
    }

    if (strncmp(argv[1], "p", sizeof(argv[1])) == 0)
    {
        weare = 0;
    }
    else if (strncmp(argv[1], "c", sizeof(argv[1])) == 0)
    {
        weare = 1;
    }
    else
    {
        printf("%s\n", "error input mode, supposed to be 'p' or 'c'");
        return -1;
    }

    memset(all_data, 0, sizeof(all_data));
    memset(&current_package, 0, sizeof(current_package));
    current_package.pack_num = 0;
    run_vmac(weare);
    // 保持程序运行的同时使该线程让出CPU
    while (1)
    {
        sleep(1);
    }
    return 1;
}
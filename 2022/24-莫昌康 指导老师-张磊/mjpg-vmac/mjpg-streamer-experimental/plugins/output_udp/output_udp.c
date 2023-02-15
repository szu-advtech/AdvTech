/*******************************************************************************
#                                                                              #
#      MJPG-streamer allows to stream JPG frames from an input-plugin          #
#      to several output plugins                                               #
#                                                                              #
#      Copyright (C) 2007 Tom Stöveken                                         #
#                                                                              #
# This program is free software; you can redistribute it and/or modify         #
# it under the terms of the GNU General Public License as published by         #
# the Free Software Foundation; version 2 of the License.                      #
#                                                                              #
# This program is distributed in the hope that it will be useful,              #
# but WITHOUT ANY WARRANTY; without even the implied warranty of               #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                #
# GNU General Public License for more details.                                 #
#                                                                              #
# You should have received a copy of the GNU General Public License            #
# along with this program; if not, write to the Free Software                  #
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA    #
#                                                                              #
*******************************************************************************/

/*
  This output plugin is based on code from output_file.c
  Writen by Dimitrios Zachariadis
  Version 0.1, May 2010

  It provides a mechanism to take snapshots with a trigger from a UDP packet.
  The UDP msg contains the path for the snapshot jpeg file
  It echoes the message received back to the sender, after taking the snapshot
*/
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <signal.h>
#include <sys/socket.h>
#include <resolv.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <getopt.h>
#include <pthread.h>
#include <fcntl.h>
#include <time.h>
#include <syslog.h>
#include <linux/netlink.h>
#include <jpeglib.h> // 记得编译的时候-ljpeg sudo apt install libjpeg-dev
#include "../../utils.h"
#include "../../mjpg_streamer.h"
#include "vmac-usrsp.h"

#define OUTPUT_PLUGIN_NAME "UDP output plugin"
#define VMAC_SOCKET_TYPE SOCK_DGRAM
#define VMAC_PORT 8080
#define VMAC_SOCKET_PROTOCOL 0
#define VMAC_SOCKET_FAMILY AF_INET
#define VMAC_UDP_USER 0x1f
// #define MAX_PAYLOAD 0x32000 /* 200KB max payload per-frame */
#define MAX_PAYLOAD 0x00400 /* 1K max payload per-frame */

static pthread_t worker;
static globals *pglobal;
static int delay, max_frame_size, mode;
static char *folder = "/tmp";
static unsigned char *frame = NULL;
static char *command = NULL;
static int input_number = 0;

// UDP port
static int port = 8080;

/******************************************************************************
Description.: print a help message
Input Value.: -
Return Value: -
******************************************************************************/
void help(void)
{
    fprintf(stderr, " ---------------------------------------------------------------\n"
                    " Help for output plugin..: " OUTPUT_PLUGIN_NAME "\n"
                    " ---------------------------------------------------------------\n"
                    " The following parameters can be passed to this plugin:\n\n"
                    " [-f | --folder ]........: folder to save pictures\n"
                    " [-d | --delay ].........: delay after saving pictures in ms\n"
                    " [-c | --command ].......: execute command after saveing picture\n"
                    " [-p | --port ]..........: UDP port to listen for picture requests. UDP message is the filename to save\n\n"
                    " [-i | --input ].......: read frames from the specified input plugin (first input plugin between the arguments is the 0th)\n\n"
                    " ---------------------------------------------------------------\n");
}

/******************************************************************************
Description.: clean up allocated resources
Input Value.: unused argument
Return Value: -
******************************************************************************/
void worker_cleanup(void *arg)
{
    static unsigned char first_run = 1;

    if (!first_run)
    {
        DBG("already cleaned up resources\n");
        return;
    }

    first_run = 0;
    OPRINT("cleaning up resources allocated by worker thread\n");

    if (frame != NULL)
    {
        free(frame);
    }
    // 可以在这里关闭socket
    close(vmac_priv.sock_fd);
}

// convert mjpeg frame to RGB24
int MJPEG2RGB(uint8_t *data_frame, int bytesused)
{
    // variables:

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    unsigned int width, height;
    // data points to the mjpeg frame received from v4l2.
    unsigned char *data = data_frame;
    size_t data_size = bytesused;

    // all the pixels after conversion to RGB.
    unsigned char *pixels; // to store RBG 存放RGB结果
    int pixel_size = 0;    // size of one pixel
    if (data == NULL || data_size <= 0)
    {
        printf("Empty data!\n");
        return -1;
    }
    uint8_t h1 = 0xFF;
    uint8_t h2 = 0xD8; // jpg的头部两个字节

    //	if(*(data)!=h1 || *(data+1)==h2)
    //	{
    //		// error header
    //		printf("wrong header %d\n ",cnt);
    //		return -2;
    //	}
    // ... In the initialization of the program:
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, data, data_size);
    int rc = jpeg_read_header(&cinfo, TRUE);
    if (!(1 == rc))
    {
        printf("Not a jpg frame.\n");
        return -2;
    }
    jpeg_start_decompress(&cinfo);
    width = cinfo.output_width;
    height = cinfo.output_height;
    pixel_size = cinfo.output_components; // 3
    int bmp_size = width * height * pixel_size;
    pixels = (unsigned char *)malloc(bmp_size);

    // ... Every frame:

    while (cinfo.output_scanline < cinfo.output_height)
    {
        unsigned char *temp_array[] = {pixels + (cinfo.output_scanline) * width * pixel_size};
        jpeg_read_scanlines(&cinfo, temp_array, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    // Write the decompressed bitmap out to a ppm file, just to make sure
    // 保存为PPM6格式（P6	Pixmap	Binary）

    char fname[25] = {0}; // file name
    // fname = "output_udp.ppm";
    sprintf(fname, "output_udp.ppm"); // cnt 是用来计算的全局变量
    char buf[50];                     // for header
    rc = sprintf(buf, "P6 %d %d 255\n", width, height);
    FILE *fd = fopen(fname, "w");
    fwrite(buf, rc, 1, fd);
    fwrite(pixels, bmp_size, 1, fd);
    fflush(fd);
    fclose(fd);

    free(pixels); // free

    return 0;
}

/******************************************************************************
Description.: this is the main worker thread
              it loops forever, grabs a fresh frame and stores it to file
Input Value.:
Return Value:
******************************************************************************/
void *worker_thread(void *arg)
{
    int ok = 1, frame_size = 0, rc = 0, current_round = 0;
    unsigned char *tmp_framebuffer = NULL;
    /* vmac_send_data_init */
    char *dataname = "chat";
    int i = 0, j = 0;
    uint16_t name_len = strlen(dataname);
    uint16_t len = MAX_PAYLOAD - 1;
    struct vmac_frame my_frame;
    struct meta_data meta;
    memset(&my_frame, 0, sizeof(struct vmac_frame));
    memset(&meta, 0, sizeof(struct meta_data));
    /* 设置消息控制字段 */
    my_frame.InterestName = "chat";
    my_frame.name_len = strlen(my_frame.InterestName);
    meta.type = 0x01;
    meta.rate = 54.0;

    /* 线程结束时的清理函数 set cleanup handler to cleanup allocated resources */
    pthread_cleanup_push(worker_cleanup, NULL);

    // listen frames
    while (ok >= 0 && !pglobal->stop)
    {
        // DBG("waiting for fresh frame\n");
        pthread_mutex_lock(&pglobal->in[input_number].db);
        pthread_cond_wait(&pglobal->in[input_number].db_update, &pglobal->in[input_number].db);

        /* read buffer */
        frame_size = pglobal->in[input_number].size;

        /* check if buffer for frame is large enough, increase it if necessary */
        if (frame_size > max_frame_size)
        {
            // DBG("increasing buffer size to %d\n", frame_size);

            max_frame_size = frame_size + (1 << 16);
            if ((tmp_framebuffer = realloc(frame, max_frame_size)) == NULL)
            {
                pthread_mutex_unlock(&pglobal->in[input_number].db);
                LOG("not enough memory\n");
                return NULL;
            }

            frame = tmp_framebuffer;
        }

        /* copy frame to our local buffer now */
        memcpy(frame, pglobal->in[input_number].buf, frame_size);
        printf("got frame (size: %d kB)\n", frame_size / 1024);

        /* allow others to access the global buffer again */
        pthread_mutex_unlock(&pglobal->in[input_number].db);

        // send data to kernel by netlink
        struct udp_image_control message_control;
        meta.seq = current_round;
        my_frame.buf = malloc(MAX_PAYLOAD + sizeof(struct udp_image_control));
        // 数据太大了，则需要分片处理
        if (frame_size > 10)
        {
            int times = (frame_size / 1024) + 1;
            // printf("the frame needs to be divided by %d times\n", times);
            for (int i = 0; i < times; i++)
            {
                message_control.seq = i;
                if (i == times - 1)
                {
                    my_frame.len = frame_size - (i)*MAX_PAYLOAD + sizeof(struct udp_image_control);
                }
                else
                {
                    my_frame.len = MAX_PAYLOAD + sizeof(struct udp_image_control);
                }
                memcpy(my_frame.buf, &message_control, sizeof(struct udp_image_control));
                memcpy(my_frame.buf + sizeof(struct udp_image_control),
                       frame + (i * MAX_PAYLOAD), my_frame.len - sizeof(struct udp_image_control));
                // sendmsg(vmac_priv.sock_fd, &vmac_priv.msg, 0);
                int result = send_vmac(&my_frame, &meta);
                if (result != 0)
                {
                    perror("result error\n");
                }
            }
        }
        // printf("saving rpg\n");
        // MJPEG2RGB(frame, frame_size);

        /* if specified, wait now */
        if (delay > 0)
        {
            // usleep(1000000 * delay * 2);
            sleep(3);
        }

        current_round++;
        free(my_frame.buf);
    }

    /* cleanup now */
    pthread_cleanup_pop(1);

    return NULL;
}

/* receive message from vmac kernel */
void callback()
{
    printf("here is callback!\n");
}

/* register the netlink socket and bind */
int vmac_register_udp(void(*cf))
{
    int socketfd;
    int err;
    char keys[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf};
    struct sched_param params; /* Data structure to describe a process' schedulability.  */
    // vmac_priv.key = "\000\001\002\003\004\005\006\a\b\t\n\v\f\r\016\017";
    memcpy(vmac_priv.key, keys, sizeof(keys));
    vmac_priv.msgy[0] = 'a';
    vmac_priv.cb = cf;

    // 建立 socket
    vmac_priv.sock_fd = socket(PF_NETLINK, SOCK_RAW, VMAC_UDP_USER);
    if (vmac_priv.sock_fd < 0)
    {
        printf("create socket failed. %d\n", vmac_priv.sock_fd);
        return -1;
    }

    // src_addr 初始化
    memset(&vmac_priv.src_addr, 0, sizeof(vmac_priv.src_addr));
    vmac_priv.src_addr.nl_family = AF_NETLINK;
    vmac_priv.src_addr.nl_pid = getpid(); /* Netlink的通信依据是一个对应于进程的标识，一般定为该进程的 ID */

    // 和指定协议进行绑定
    err = bind(vmac_priv.sock_fd, (struct sockaddr *)&vmac_priv.src_addr, sizeof(vmac_priv.src_addr));
    if (err < 0)
    {
        printf("bind socket failed. %d\n", vmac_priv.sock_fd);
        return err;
    }

    // dest_addr 初始化
    memset(&vmac_priv.dest_addr, 0, sizeof(vmac_priv.dest_addr));
    vmac_priv.dest_addr.nl_family = AF_NETLINK;
    vmac_priv.dest_addr.nl_pid = 0;    /* For Linux Kernel */
    vmac_priv.dest_addr.nl_groups = 0; /* unicast */

    /* 数组和结构体会先调用 memset() 初始化后再赋值 */

    // 设置发送 netlink
    vmac_priv.nlh = (struct nlmsghdr *)malloc(NLMSG_SPACE(MAX_PAYLOAD));
    memset(vmac_priv.nlh, 0, NLMSG_SPACE(MAX_PAYLOAD));
    vmac_priv.nlh->nlmsg_len = NLMSG_SPACE(MAX_PAYLOAD);
    vmac_priv.nlh->nlmsg_pid = getpid(); // self pid
    vmac_priv.nlh->nlmsg_flags = 0;
    vmac_priv.nlh->nlmsg_type = 255; /* 可自定义 type */

    /* 拷贝信息到发送缓冲中, 前 1024 个为0, just a register */
    memset(vmac_priv.msgy, 0, 1024);
    /** NLMSG_DATA 用于取得消息的数据部分的首地址，设置和读取消息数据部分时需要使用该宏,
     返回的是传入的 nlmsghdr 结构体的地址 + 固定的 nlmsghdr 长度，即头部之后就是消息体 */
    memcpy(NLMSG_DATA(vmac_priv.nlh), vmac_priv.msgy, strlen(vmac_priv.msgy));

    /* 构造发送信息结构体 */
    vmac_priv.iov.iov_base = (void *)vmac_priv.nlh;
    vmac_priv.iov.iov_len = vmac_priv.nlh->nlmsg_len;
    vmac_priv.msg.msg_name = (void *)&vmac_priv.dest_addr;
    vmac_priv.msg.msg_namelen = sizeof(vmac_priv.dest_addr);
    vmac_priv.msg.msg_iov = &vmac_priv.iov;
    /* 缓冲区地址指针长度设置成 1？ 因为记录的是 iovec 的个数，允许一次传递多个 buff */
    vmac_priv.msg.msg_iovlen = 1;

    // 设置接收 netlink
    vmac_priv.nlh2 = (struct nlmsghdr *)malloc(NLMSG_SPACE(MAX_PAYLOAD));
    memset(vmac_priv.nlh2, 0, NLMSG_SPACE(MAX_PAYLOAD));
    vmac_priv.nlh2->nlmsg_len = NLMSG_SPACE(MAX_PAYLOAD);
    vmac_priv.iov2.iov_base = (void *)vmac_priv.nlh2;
    vmac_priv.iov2.iov_len = vmac_priv.nlh2->nlmsg_len;
    vmac_priv.msg2.msg_name = (void *)&vmac_priv.dest_addr;
    vmac_priv.msg2.msg_namelen = sizeof(vmac_priv.dest_addr);
    vmac_priv.msg2.msg_iov = &vmac_priv.iov2;
    vmac_priv.msg2.msg_iovlen = 1;

    // // 如果是 producer，正常发送消息试一试
    // if (mode == 1)
    // {
    // }
    // // 如果是 consumer, 给 producer 发送兴趣帧
    // else if (mode == 2)
    // {
    //     vmac_priv.nlh->nlmsg_type = 254;
    // }

    err = sendmsg(vmac_priv.sock_fd, &vmac_priv.msg, 0);
    if (err < 0)
    {
        printf("send msg err: %d\n", err);
        return err;
    }

    return 0;
}

/*** plugin interface functions ***/
/******************************************************************************
Description.: this function is called first, in order to initialise
              this plugin and pass a parameter string
Input Value.: parameters
Return Value: 0 if everything is ok, non-zero otherwise
******************************************************************************/
int output_init(output_parameter *param)
{
    int i, err;
    delay = 6;

    param->argv[0] = OUTPUT_PLUGIN_NAME;

    /* show all parameters for DBG purposes */
    for (i = 0; i < param->argc; i++)
    {
        DBG("argv[%d]=%s\n", i, param->argv[i]);
    }

    // init 时必须要先调用一次 reset_getopt();
    reset_getopt();
    while (1)
    {
        /* no_argument或0表示此选项不带参数，required_argument或1表示此选项带参数，optional_argument或2表示是一个可选选项 */
        int option_index = 0, c = 0;
        static struct option long_options[] = {
            {"h", no_argument, 0, 0},
            {"help", no_argument, 0, 0},
            {"f", required_argument, 0, 0},
            {"folder", required_argument, 0, 0},
            {"d", required_argument, 0, 0},
            {"delay", required_argument, 0, 0},
            {"c", required_argument, 0, 0},
            {"command", required_argument, 0, 0},
            {"p", required_argument, 0, 0},
            {"port", required_argument, 0, 0},
            {"i", required_argument, 0, 0},
            {"input", required_argument, 0, 0},
            {0, 0, 0, 0}};

        c = getopt_long_only(param->argc, param->argv, "", long_options, &option_index);

        /* no more options to parse */
        if (c == -1)
            break;

        /* unrecognized option */
        if (c == '?')
        {
            help();
            return 1;
        }

        switch (option_index)
        {
            /* h, help */
        case 0:
        case 1:
            DBG("case 0,1\n");
            help();
            return 1;
            break;

            /* f, folder */
        case 2:
        case 3:
            DBG("case 2,3\n");
            folder = malloc(strlen(optarg) + 1);
            strcpy(folder, optarg);
            if (folder[strlen(folder) - 1] == '/')
                folder[strlen(folder) - 1] = '\0';
            break;

            /* d, delay */
        case 4:
        case 5:
            DBG("case 4,5\n");
            delay = atoi(optarg);
            break;

            /* c, command */
        case 6:
        case 7:
            DBG("case 6,7\n");
            command = strdup(optarg);
            break;

            /* p, port */
        case 8:
        case 9:
            DBG("case 8,9\n");
            port = atoi(optarg);
            break;
            /* i, input */
        case 10:
        case 11:
            DBG("case 10,11\n");
            input_number = atoi(optarg);
            break;
        }
    }

    pglobal = param->global;
    if (!(input_number < pglobal->incnt))
    {
        OPRINT("ERROR: the %d input_plugin number is too much only %d plugins loaded\n", input_number, pglobal->incnt);
        return 1;
    }
    OPRINT("input plugin.....: %d: %s\n", input_number, pglobal->in[input_number].plugin);
    OPRINT("output folder.....: %s\n", folder);
    OPRINT("delay after save..: %d\n", delay);
    OPRINT("command...........: %s\n", (command == NULL) ? "disabled" : command);
    if (port > 0)
    {
        OPRINT("UDP port..........: %d\n", port);
    }
    else
    {
        OPRINT("UDP port..........: %s\n", "disabled");
    }

    // register vmac socket
    void (*ptr)() = &callback;
    err = vmac_register(ptr);

    return err;
}

/******************************************************************************
Description.: calling this function stops the worker thread
Input Value.: -
Return Value: always 0
******************************************************************************/
int output_stop(int id)
{
    DBG("will cancel worker thread\n");
    pthread_cancel(worker);
    return 0;
}

/******************************************************************************
Description.: calling this function creates and starts the worker thread
Input Value.: -
Return Value: always 0
******************************************************************************/
int output_run(int id)
{
    DBG("launching worker thread\n");

    pthread_create(&worker, 0, worker_thread, NULL);
    pthread_detach(worker);
    return 0;
}
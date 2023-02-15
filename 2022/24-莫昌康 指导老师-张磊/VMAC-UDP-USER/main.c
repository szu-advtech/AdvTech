#include "main.h"
/**
 * client.c 客户端收集信息发送
 * 1. 创建基础 socket 套接字
 * 2. 按照UDP sendmsg 套接字 API 手动创建UDP信息包（在 socket 里面 / 独立）
 * 3. UDP 信息包里面存在一个 name 的基础信息，并约定好位置
 * 4. 调用 vmac API 发送信息
 *
 * sendto: char *buff, -> struct msghdr -> skb_buff(udp header + udp data)
 * 思路1：因为在 __sys_sendto 系统调用中创建了用来描述要发送的数据的结构体 struct msghdr 并调用了sock_sendmsg来执行实际的发送，
 * 所以可以考虑增加一个内核模块，并暴露出一个系统调用，该系统调用模仿 sendto 逻辑，最后并不发送数据，而是返回这种类型的数据
 * (缺点：里面很多函数，只要调用其他层函数的时候都得改，并且有些是考虑 IP 层逻辑的，例如调了某些函数后会考虑上一次的状态，例如等待队列是否满、是否需要分片等)
 *
 * 问题1：这个数据是否就是最终传给 IP 层的数据？还需要经过 udp_sendmsg 的哪些逻辑封装吗？这个问题到底需不需要考虑，因为如何对接 UDP 这里很模糊
 * 答：在 udp_sendmsg 中会调用 udp_send_skb 创建真正的 UDP 数据包：udphdr + udpdata(struct msghdr)，然后发给 ip 层
 *
 * 思路2：按照 VMAC 的代码逻辑，VMAC 实现用户-内核通信的方式是创建一个新的 “协议族VMAC_USER”，可以在 vmac module 中添加一个逻辑专门处理 “如何将数据处理成 UDP 数据”
 * 因为本身 vmac module 已经是内核中了，所以调用内核相关的数据结构会方便
 * （得考虑能不能直接copy代码直接调用函数的问题）
 *
 * 思路3：由问题1，因为对接的 udp 数据包的形式很模糊，直接在用户层可以设置数据包的关键数据结构：struct msghdr / skb_buff( skb = ip_make_skb() ) 但这个函数有ip的分片逻辑
 * 那么直接将需要发送数据设置成这个格式再传给 vmac
 *
 */

#define VMAC_SOCKET_FAMILY AF_INET
#define VMAC_SOCKET_TYPE SOCK_DGRAM
// 先设置为0让内核自动选择，也有可能需要设置为17
#define VMAC_SOCKET_PROTOCOL 0
#define VMAC_PORT 8008

#define VMAC_UDP_USER 0x1f
#define MAX_PAYLOAD 0x800 /* 2KB max payload per-frame */

void vmac_sendto(char *buff) {}

/* udp_sendmsg */
void vmac_sendmsg()
{
}

/**
 * create_vmac_udp 以字符串形式接收需要发送的信息，封装好 udp 数据包返回
 */
char *create_vmac_udp(int socketfd, char *buff)
{
  return "";
}

void callback()
{
  printf("here is callback!\n");
}

int vmac_register(void(*cf))
{
  int socketfd;
  int size;
  char keys[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf};
  struct sched_param params; /* Data structure to describe a process' schedulability.  */
  // vmac_priv.key = "\000\001\002\003\004\005\006\a\b\t\n\v\f\r\016\017";
  memcpy(vmac_priv.key, keys, sizeof(keys));
  vmac_priv.msgy[0] = 'a';
  vmac_priv.cb = cf;
  vmac_priv.sock_fd = socket(PF_NETLINK, SOCK_RAW, VMAC_UDP_USER);
  if (vmac_priv.sock_fd < 0)
  {
    printf("create socket failed. %d\n", vmac_priv.sock_fd);
    return 0;
  }
  // size = 101, because only initial the 'a' in msgy[0], the rest are null char
  size = strlen(vmac_priv.msgy) + 100; /* seg fault occurs if size < 100 */
  memset(&vmac_priv.src_addr, 0, sizeof(vmac_priv.src_addr));
  /* sockaddr_nl => sockaddr_netlink, 源地址是一个 netlink 的地址格式 */
  vmac_priv.src_addr.nl_family = AF_NETLINK;
  vmac_priv.src_addr.nl_pid = getpid(); /* Netlink的通信依据是一个对应于进程的标识，一般定为该进程的 ID */
  bind(vmac_priv.sock_fd, (struct sockaddr *)&vmac_priv.src_addr, sizeof(vmac_priv.src_addr));
  memset(&vmac_priv.dest_addr, 0, sizeof(vmac_priv.dest_addr));
  vmac_priv.dest_addr.nl_family = AF_NETLINK;
  vmac_priv.dest_addr.nl_pid = 0;
  vmac_priv.dest_addr.nl_groups = 0;
  // netlink header, 其大小暂时分配为 2kb，整个信息(头+数据)的大小是 2kb, 其成员都是一些 unsigned int，固定长度
  vmac_priv.nlh = (struct nlmsghdr *)malloc(NLMSG_SPACE(MAX_PAYLOAD));
  vmac_priv.nlh2 = (struct nlmsghdr *)malloc(NLMSG_SPACE(MAX_PAYLOAD));
  /* 发现数组和结构体会先调用 memset() 初始化后再赋值 */
  memset(vmac_priv.nlh, 0, NLMSG_SPACE(MAX_PAYLOAD));
  memset(vmac_priv.nlh2, 0, NLMSG_SPACE(MAX_PAYLOAD));
  // 信息(头+数据)的长度直接赋值为最大PAYLOAD, nlh 是发送信息, nlh2 是接收信息
  vmac_priv.nlh2->nlmsg_len = NLMSG_SPACE(MAX_PAYLOAD);
  vmac_priv.nlh->nlmsg_len = size;
  vmac_priv.nlh->nlmsg_pid = getpid();
  vmac_priv.nlh->nlmsg_flags = 0;
  vmac_priv.nlh->nlmsg_type = 4;
  /* 接收缓冲区内容指针，这里指向了 nlh2 的地址，并且缓冲区大小也设置为上面设置的 2kb */
  vmac_priv.iov2.iov_base = (void *)vmac_priv.nlh2;
  vmac_priv.iov2.iov_len = vmac_priv.nlh2->nlmsg_len;
  /* name 和 len 一起出现基本都是设置对应变量的地址，以及该读多少长度才能完整读出该变量 */
  vmac_priv.msg2.msg_name = (void *)&vmac_priv.dest_addr;
  vmac_priv.msg2.msg_namelen = sizeof(vmac_priv.dest_addr);
  /* 缓冲区地址 */
  vmac_priv.msg2.msg_iov = &vmac_priv.iov2;
  /* 缓冲区地址指针长度设置成 1？ 因为记录的是 iovec 的个数，允许一次传递多个 buff */
  vmac_priv.msg2.msg_iovlen = 1;
  /* 发送缓冲区指针 */
  vmac_priv.iov.iov_base = (void *)vmac_priv.nlh;
  vmac_priv.iov.iov_len = vmac_priv.nlh->nlmsg_len;
  vmac_priv.msg.msg_name = (void *)&vmac_priv.dest_addr;
  vmac_priv.msg.msg_namelen = sizeof(vmac_priv.dest_addr);
  vmac_priv.msg.msg_iov = &vmac_priv.iov;
  vmac_priv.msg.msg_iovlen = 1;

  vmac_priv.nlh->nlmsg_type = 255; /* 可自定义 type */
  /* char[2000] 只设置了前1024个 */
  memset(vmac_priv.msgy, 0, 1024);
  // 因为是一个建立连接消息，所以先设置初始值, digest64: long long 是 8bytes 的
  vmac_priv.digest64 = 0;
  // 这里是指先给消息的数据部分前8字节设置成0，下面一条是指给后面部分设置成 vmac_priv.msgy（有可能是因为前面留着name）
  /** NLMSG_DATA 用于取得消息的数据部分的首地址，设置和读取消息数据部分时需要使用该宏,
   返回的是传入的 nlmsghdr 结构体的地址 + 固定的 nlmsghdr 长度，即头部之后就是消息体 */
  memcpy(NLMSG_DATA(vmac_priv.nlh), &vmac_priv.digest64, 8);
  memcpy(NLMSG_DATA(vmac_priv.nlh) + 8, vmac_priv.msgy, strlen(vmac_priv.msgy));
  size = strlen(vmac_priv.msgy) + 100;
  sendmsg(vmac_priv.sock_fd, &vmac_priv.msg, 0);

  return 0;
}

int main(void)
{
  void (*ptr)() = &callback;
  vmac_register(ptr);

  return 0;
}
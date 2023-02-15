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

#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/net.h>
#include <linux/file.h>
#include <linux/security.h>
#include <linux/socket.h>
#include <linux/uaccess.h>
#include <linux/audit.h>
#include <uapi/linux/in.h>
#include <net/sock.h>
#include <linux/skbuff.h>
#include <linux/udp.h>
#include <linux/ip.h>
#include <net/cfg80211.h>
#include <linux/netlink.h>
#include "main.h"

#define VMAC_SOCKET_FAMILY AF_INET
#define VMAC_SOCKET_TYPE SOCK_DGRAM
// 先设置为0让内核自动选择，也有可能需要设置为17
#define VMAC_SOCKET_PROTOCOL 0
#define VMAC_PORT 8008
#define REGISTER_TYPE 255 /* 注册 netlink 时发的VMAC type, 无其他意义 */

#define VMAC_UDP_USER 0x1f

struct control
{
  char type[1];
  char rate[1];
  char enc[8];
  char seq[2];
  char bwsg;
  char rate_idx;
  char signal;
};
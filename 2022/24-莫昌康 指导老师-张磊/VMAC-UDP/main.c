#include "vmac_udp.h"

// 全局socket描述符
int socketfd;
// 32位的地址
struct socket *vmac_sock;
// net_link
struct sock *nl_sk = NULL;
// just a hack to build and integrate within kernel...
const struct cfg80211_ops mac80211_config_ops = {};
// sockaddr_in as udp
struct sockaddr_in server_addr;
int pidt;

/// @brief 构建一个udp报文并发送
/// @param pkt     payload
/// @param pkt_len length of payload
/// @param smac  source mac address, 48 bits
/// @param dmac  dest mac address, 48 bits   u_char *smac, u_char *dmac,
/// @param sip   source ip address, 32 bits
/// @param dip   dest ip address, 32 bits
/// @param sport source port, 16 bits
/// @param dport dest port, 16 bits
/// @return
int create_vmac_udp(u_char *pkt, int pkt_len, uint32_t sip, uint32_t dip, uint16_t sport, uint16_t dport)
{
  uint16_t len; /* 数据段长度 skb->data ~ skb->tail */
  struct sk_buff *skb = NULL;
  struct udphdr *udph = NULL;
  // struct iphdr *iph = NULL;
  // struct ethhdr *ethdr = NULL;
  u_char *pdata = NULL;
  int nret = 1;
  int i = 0;

  // 创建一个skb
  len = pkt_len;
  skb = alloc_skb(len, GFP_ATOMIC);
  if (NULL == skb)
    goto out;

  // 为skb预留空间，方便后面skb_buff协议封装
  skb_reserve(skb, pkt_len + sizeof(struct udphdr)); /* 2 + sizeof(struct iphdr) + sizeof(struct ethhdr) */
  // pr_info("pkt_len + sizeof(struct udphdr): %d\n", pkt_len + sizeof(struct udphdr));

  /****************** skb字节填充 ******************/
  // skb->dev = br->dev;
  skb->pkt_type = PACKET_OTHERHOST;
  skb->protocol = __constant_htons(ETH_P_IP);
  skb->ip_summed = CHECKSUM_NONE;
  skb->priority = 0;

  /**
   * 数据包封装
   * 分别压入应用层，传输层，网络层，链路层栈帧
   * skb_push由后面往前面，与skb_put不同
   */
  pdata = skb_push(skb, pkt_len);
  udph = (struct udphdr *)skb_push(skb, sizeof(struct udphdr));
  skb_reset_transport_header(skb);

  // iph = (struct iphdr *)skb_push(skb, sizeof(struct iphdr));
  // skb_reset_network_header(skb);

  // ethdr = (struct ethhdr *)skb_push(skb, sizeof(struct ethhdr));
  // skb_reset_mac_header(skb);

  /****************** 应用层数据填充 ******************/
  memcpy(pdata, pkt, pkt_len);

  /****************** 传输层udp数据填充 ******************/
  memset(udph, 0, sizeof(struct udphdr));
  udph->source = sport;
  udph->dest = dport;
  udph->len = htons(sizeof(struct udphdr) + pkt_len); // 主机字节序转网络字节序
  udph->check = 0;                                    // skb_checksum之前必须置0.协议规定

  /****************** 网络层数据填充 ******************/
  // iph->version = 4;
  // iph->ihl = sizeof(struct iphdr) >> 2;
  // iph->frag_off = 0;
  // iph->protocol = IPPROTO_UDP;
  // iph->tos = 0;
  // iph->daddr = dip;
  // iph->saddr = sip;
  // iph->ttl = 0x40;
  // iph->tot_len = __constant_htons(pkt_len + sizeof(struct udphdr) + sizeof(struct iphdr));
  // iph->check = 0;
  // iph->check = ip_fast_csum((unsigned char *)iph, iph->ihl); // 计算校验和

  // TODO: 先不计算校验和
  // skb->csum = skb_checksum(skb, iph->ihl * 4, skb->len - iph->ihl * 4, 0); // skb校验和计算
  // udph->check = csum_tcpudp_magic(sip, dip, skb->len - iph->ihl * 4, IPPROTO_UDP, skb->csum); // udp和tcp伪首部校验和

  /****************** 链路层数据填充 ******************/
  // memcpy(ethdr->h_dest, dmac, ETH_ALEN);
  // memcpy(ethdr->h_source, smac, ETH_ALEN);
  // ethdr->h_proto = __constant_htons(ETH_P_IP);

  /****************** 调用VMAC函数，发送数据包 ******************/
  pr_warn("skb->data: %s\n", skb->data + 8);
  pr_warn("skb->len: %d\n", skb->len - 8);

  nret = 0; /* 这里是必须的 */
  printk("send vmac correct\n");

out:
  if (0 != nret && NULL != skb) /* 这里前面的nret判断是必须的，不然必定死机 */
  {
    // dev_put(br->dev); // 减少设备的引用计数
    kfree_skb(skb);
  }

  return nret;
}

// recv from userspace
void nl_recv(struct sk_buff *skb)
{
  struct nlmsghdr *nlh;
  struct msghdr *msg_header;
  struct sk_buff *recv_skb;
  struct control rxc;
  u8 type;
  int size;
  int err;
  nlh = nlmsg_hdr(skb);
  type = nlh->nlmsg_type;
  size = nlh->nlmsg_len;
  if (pidt != nlh->nlmsg_pid)
    pidt = nlh->nlmsg_pid;

  // WARN_ON(skb);
  pr_info("nlh->nlmsg_len: %d kB\n", size / 1024);
  // printk("receive data from user process: %s\n", (unsigned char *)NLMSG_DATA(nlh)); // 打印接收的数据内容

  if (nlh->nlmsg_type != REGISTER_TYPE)
  {
    create_vmac_udp((unsigned char *)nlmsg_data(nlh), size, htonl(INADDR_ANY), htonl(INADDR_ANY), htons(VMAC_PORT), htons(VMAC_PORT));
  }
  else
  {
    pr_info("REGISTER_TYPE\n");
  }
}

static int vmac_udp_init(void)
{
  int err;
  struct netlink_kernel_cfg cfg = {
      .input = nl_recv};

  pidt = -1;

  // 绑定端口
  // bzero(&server_addr, sizeof(server_addr));
  server_addr.sin_family = VMAC_SOCKET_FAMILY;
  /* 监听0.0.0.0地址 socket只绑定端口让路由表决定传到哪个ip */
  server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  server_addr.sin_port = htons(VMAC_PORT);
  memset(&(server_addr.sin_zero), 0, 8);

  nl_sk = netlink_kernel_create(&init_net, VMAC_UDP_USER, &cfg);
  if (!nl_sk)
  {
    printk(KERN_ALERT "VMAC FAILED ERROR: nl_sk is %p\n", nl_sk);
    return -1;
  }

  pr_info("vmac_udp_init.\n");
  return 0;
}

static void vmac_udp_exit(void)
{
  netlink_kernel_release(nl_sk);
  pr_notice("vmac_udp_exit\n");
}

module_init(vmac_udp_init);
module_exit(vmac_udp_exit);

MODULE_AUTHOR("MOC");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_DESCRIPTION("For basic udp-vmac transmission!/n");
MODULE_ALIAS("A new UDP-VMAC layer");

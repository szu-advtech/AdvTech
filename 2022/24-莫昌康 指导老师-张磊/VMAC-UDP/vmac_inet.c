#include "vmac_udp.h"
#include <net/inet_sock.h>
#include <net/flow.h>
#include <net/ip.h>
#include <net/udp.h>
#include <net/udplite.h>
#include <net/flow.h>
#include <net/icmp.h>
#include <net/ip_fib.h>
#include <linux/inetdevice.h>
#include <linux/icmp.h>
#include <linux/errqueue.h>

int vmac_inet_bind(struct socket *sock, struct sockaddr *uaddr, int addr_len)
{
  return inet_bind(sock, uaddr, addr_len);
}

static inline int ip_select_ttl(struct inet_sock *inet, struct dst_entry *dst)
{
  int ttl = inet->uc_ttl;
  if (ttl < 0)
    ttl = ip4_dst_hoplimit(dst);
  return ttl;
}

static void ip_cork_release(struct inet_cork *cork)
{
  cork->flags &= ~IPCORK_OPT;
  kfree(cork->opt);
  cork->opt = NULL;
  dst_release(cork->dst);
  cork->dst = NULL;
}

static void ip_copy_addrs(struct iphdr *iph, const struct flowi4 *fl4)
{
  BUILD_BUG_ON(offsetof(typeof(*fl4), daddr) !=
               offsetof(typeof(*fl4), saddr) + sizeof(fl4->saddr));
  memcpy(&iph->saddr, &fl4->saddr,
         sizeof(fl4->saddr) + sizeof(fl4->daddr));
}

static void ip_local_error_vmac(struct sock *sk, int err, __be32 daddr, __be16 port, u32 info)
{
  struct inet_sock *inet = inet_sk(sk);
  struct sock_exterr_skb *serr;
  struct iphdr *iph;
  struct sk_buff *skb;

  if (!inet->recverr)
    return;

  skb = alloc_skb(sizeof(struct iphdr), GFP_ATOMIC);
  if (!skb)
    return;

  skb_put(skb, sizeof(struct iphdr));
  skb_reset_network_header(skb);
  iph = ip_hdr(skb);
  iph->daddr = daddr;

  serr = SKB_EXT_ERR(skb);
  serr->ee.ee_errno = err;
  serr->ee.ee_origin = SO_EE_ORIGIN_LOCAL;
  serr->ee.ee_type = 0;
  serr->ee.ee_code = 0;
  serr->ee.ee_pad = 0;
  serr->ee.ee_info = info;
  serr->ee.ee_data = 0;
  serr->addr_offset = (u8 *)&iph->daddr - skb_network_header(skb);
  serr->port = port;

  __skb_pull(skb, skb_tail_pointer(skb) - skb->data);
  skb_reset_transport_header(skb);

  if (sock_queue_err_skb(sk, skb))
    kfree_skb(skb);
}

int ip_generic_getfrag_vmac(void *from, char *to, int offset, int len, int odd, struct sk_buff *skb)
{
  struct msghdr *msg = from;
  // memcpy(to + offset, from, len);

  if (skb->ip_summed == CHECKSUM_PARTIAL)
  {
    memcpy(to, &msg->msg_iter, len);
  }

  return 0;
}

void ip_rt_get_source_vmac(u8 *addr, struct sk_buff *skb, struct rtable *rt)
{
  __be32 src;

  if (rt_is_output_route(rt))
    src = ip_hdr(skb)->saddr;
  else
  {
    struct fib_result res;
    struct flowi4 fl4;
    struct iphdr *iph;

    iph = ip_hdr(skb);

    memset(&fl4, 0, sizeof(fl4));
    fl4.daddr = iph->daddr;
    fl4.saddr = iph->saddr;
    fl4.flowi4_tos = RT_TOS(iph->tos);
    fl4.flowi4_oif = rt->dst.dev->ifindex;
    fl4.flowi4_iif = skb->dev->ifindex;
    fl4.flowi4_mark = skb->mark;

    rcu_read_lock();
    if (fib_lookup(dev_net(rt->dst.dev), &fl4, &res, 0) == 0)
    {
      pr_alert("todo here: fib_lookup(dev_net)");
      src = 0;
      // src = FIB_RES_PREFSRC(dev_net(rt->dst.dev), res);
    }
    else
      src = inet_select_addr(rt->dst.dev,
                             rt_nexthop(rt, iph->daddr),
                             RT_SCOPE_UNIVERSE);
    rcu_read_unlock();
  }
  memcpy(addr, &src, 4);
}

void ip_options_build_vmac(struct sk_buff *skb, struct ip_options *opt,
                           __be32 daddr, struct rtable *rt, int is_frag)
{
  unsigned char *iph = skb_network_header(skb);

  memcpy(&(IPCB(skb)->opt), opt, sizeof(struct ip_options));
  memcpy(iph + sizeof(struct iphdr), opt->__data, opt->optlen);
  opt = &(IPCB(skb)->opt);

  if (opt->srr)
    memcpy(iph + opt->srr + iph[opt->srr + 1] - 4, &daddr, 4);

  if (!is_frag)
  {
    if (opt->rr_needaddr)
      ip_rt_get_source_vmac(iph + opt->rr + iph[opt->rr + 2] - 5, skb, rt);
    if (opt->ts_needaddr)
      ip_rt_get_source_vmac(iph + opt->ts + iph[opt->ts + 2] - 9, skb, rt);
    if (opt->ts_needtime)
    {
      __be32 midtime;

      midtime = inet_current_timestamp();
      memcpy(iph + opt->ts + iph[opt->ts + 2] - 5, &midtime, 4);
    }
    return;
  }
  if (opt->rr)
  {
    memset(iph + opt->rr, IPOPT_NOP, iph[opt->rr + 1]);
    opt->rr = 0;
    opt->rr_needaddr = 0;
  }
  if (opt->ts)
  {
    memset(iph + opt->ts, IPOPT_NOP, iph[opt->ts + 1]);
    opt->ts = 0;
    opt->ts_needaddr = opt->ts_needtime = 0;
  }
}

static int ip_setup_cork(struct sock *sk, struct inet_cork *cork,
                         struct ipcm_cookie *ipc, struct rtable **rtp)
{
  struct ip_options_rcu *opt;
  struct rtable *rt;

  rt = *rtp;
  if (unlikely(!rt))
  {
    pr_alert("rt err %p\n", rt);
    return -EFAULT;
  }

  /*
   * setup for corking.
   */
  opt = ipc->opt;
  if (opt)
  {
    if (!cork->opt)
    {
      cork->opt = kmalloc(sizeof(struct ip_options) + 40,
                          sk->sk_allocation);
      if (unlikely(!cork->opt))
        return -ENOBUFS;
    }
    memcpy(cork->opt, &opt->opt, sizeof(struct ip_options) + opt->opt.optlen);
    cork->flags |= IPCORK_OPT;
    cork->addr = ipc->addr;
  }

  cork->fragsize = ip_sk_use_pmtu(sk) ? dst_mtu(&rt->dst) : READ_ONCE(rt->dst.dev->mtu);

  if (!inetdev_valid_mtu(cork->fragsize))
    return -ENETUNREACH;

  cork->gso_size = ipc->gso_size;

  cork->dst = &rt->dst;
  /* We stole this route, caller should not release it. */
  *rtp = NULL;

  cork->length = 0;
  cork->ttl = ipc->ttl;
  cork->tos = ipc->tos;
  cork->priority = ipc->priority;
  cork->transmit_time = ipc->sockc.transmit_time;
  cork->tx_flags = 0;
  sock_tx_timestamp(sk, ipc->sockc.tsflags, &cork->tx_flags);

  return 0;
}

/*
 *	Maintain the counters used in the SNMP statistics for outgoing ICMP
 */
void icmp_out_count(struct net *net, unsigned char type)
{
  ICMPMSGOUT_INC_STATS(net, type);
  ICMP_INC_STATS(net, ICMP_MIB_OUTMSGS);
}

/*
 *	Throw away all pending data on the socket.
 */
static void __ip_flush_pending_frames(struct sock *sk,
                                      struct sk_buff_head *queue,
                                      struct inet_cork *cork)
{
  struct sk_buff *skb;

  while ((skb = __skb_dequeue_tail(queue)) != NULL)
    kfree_skb(skb);

  ip_cork_release(cork);
}

// /*
//  * copy saddr and daddr, possibly using 64bit load/stores
//  * Equivalent to :
//  *   iph->saddr = fl4->saddr;
//  *   iph->daddr = fl4->daddr;
//  */

/*
 *	Combined all pending IP fragments on the socket as one IP datagram
 *	and push them out.
 */
struct sk_buff *__ip_make_skb_vmac(struct sock *sk,
                                   struct flowi4 *fl4,
                                   struct sk_buff_head *queue,
                                   struct inet_cork *cork)
{
  struct sk_buff *skb, *tmp_skb;
  struct sk_buff **tail_skb;
  struct inet_sock *inet = inet_sk(sk);
  struct net *net = sock_net(sk);
  struct ip_options *opt = NULL;
  struct rtable *rt = (struct rtable *)cork->dst;
  struct iphdr *iph;
  __be16 df = 0;
  __u8 ttl;

  skb = __skb_dequeue(queue);
  if (!skb)
    goto out;
  tail_skb = &(skb_shinfo(skb)->frag_list);

  /* move skb->data to ip header from ext header */
  if (skb->data < skb_network_header(skb))
    __skb_pull(skb, skb_network_offset(skb));
  while ((tmp_skb = __skb_dequeue(queue)) != NULL)
  {
    __skb_pull(tmp_skb, skb_network_header_len(skb));
    *tail_skb = tmp_skb;
    tail_skb = &(tmp_skb->next);
    skb->len += tmp_skb->len;
    skb->data_len += tmp_skb->len;
    skb->truesize += tmp_skb->truesize;
    tmp_skb->destructor = NULL;
    tmp_skb->sk = NULL;
  }

  /* Unless user demanded real pmtu discovery (IP_PMTUDISC_DO), we allow
   * to fragment the frame generated here. No matter, what transforms
   * how transforms change size of the packet, it will come out.
   */
  skb->ignore_df = ip_sk_ignore_df(sk);

  /* DF bit is set when we want to see DF on outgoing frames.
   * If ignore_df is set too, we still allow to fragment this frame
   * locally. */
  if (inet->pmtudisc == IP_PMTUDISC_DO ||
      inet->pmtudisc == IP_PMTUDISC_PROBE ||
      (skb->len <= dst_mtu(&rt->dst) &&
       ip_dont_fragment(sk, &rt->dst)))
    df = htons(IP_DF);

  if (cork->flags & IPCORK_OPT)
    opt = cork->opt;

  if (cork->ttl != 0)
    ttl = cork->ttl;
  else if (rt->rt_type == RTN_MULTICAST)
    ttl = inet->mc_ttl;
  else
    ttl = ip_select_ttl(inet, &rt->dst);

  iph = ip_hdr(skb);
  iph->version = 4;
  iph->ihl = 5;
  iph->tos = (cork->tos != -1) ? cork->tos : inet->tos;
  iph->frag_off = df;
  iph->ttl = ttl;
  iph->protocol = sk->sk_protocol;
  ip_copy_addrs(iph, fl4);
  ip_select_ident(net, skb, sk);

  if (opt)
  {
    iph->ihl += opt->optlen >> 2;
    ip_options_build_vmac(skb, opt, cork->addr, rt, 0);
  }

  skb->priority = (cork->tos != -1) ? cork->priority : sk->sk_priority;
  skb->mark = sk->sk_mark;
  skb->tstamp = cork->transmit_time;
  /*
   * Steal rt from cork.dst to avoid a pair of atomic_inc/atomic_dec
   * on dst refcount
   */
  cork->dst = NULL;
  skb_dst_set(skb, &rt->dst);

  if (iph->protocol == IPPROTO_ICMP)
    icmp_out_count(net, ((struct icmphdr *)
                             skb_transport_header(skb))
                            ->type);

  ip_cork_release(cork);
out:
  return skb;
}

static int __ip_append_data_vmac(struct sock *sk,
                                 struct flowi4 *fl4,
                                 struct sk_buff_head *queue,
                                 struct inet_cork *cork,
                                 struct page_frag *pfrag,
                                 int getfrag(void *from, char *to, int offset,
                                             int len, int odd, struct sk_buff *skb),
                                 void *from, int length, int transhdrlen,
                                 unsigned int flags)
{
  struct inet_sock *inet = inet_sk(sk);
  struct sk_buff *skb;

  struct ip_options *opt = cork->opt;
  int hh_len;
  int exthdrlen;
  int mtu;
  int copy;
  int err;
  int offset = 0;
  unsigned int maxfraglen, fragheaderlen, maxnonfragsize;
  int csummode = CHECKSUM_NONE;
  struct rtable *rt = (struct rtable *)cork->dst;
  unsigned int wmem_alloc_delta = 0;
  u32 tskey = 0;
  bool paged;

  skb = skb_peek_tail(queue);

  exthdrlen = !skb ? rt->dst.header_len : 0;
  mtu = cork->gso_size ? IP_MAX_MTU : cork->fragsize;
  paged = !!cork->gso_size;

  if (cork->tx_flags & SKBTX_ANY_SW_TSTAMP &&
      sk->sk_tsflags & SOF_TIMESTAMPING_OPT_ID)
    tskey = sk->sk_tskey++;

  hh_len = LL_RESERVED_SPACE(rt->dst.dev);

  fragheaderlen = sizeof(struct iphdr) + (opt ? opt->optlen : 0);
  maxfraglen = ((mtu - fragheaderlen) & ~7) + fragheaderlen;
  maxnonfragsize = ip_sk_ignore_df(sk) ? 0xFFFF : mtu;

  if (cork->length + length > maxnonfragsize - fragheaderlen)
  {
    pr_err("ip_local_error: has been exceed length!\n");
    ip_local_error_vmac(sk, EMSGSIZE, fl4->daddr, inet->inet_dport,
                        mtu - (opt ? opt->optlen : 0));
    return -EMSGSIZE;
  }

  /*
   * transhdrlen > 0 means that this is the first fragment and we wish
   * it won't be fragmented in the future.
   */
  if (transhdrlen &&
      length + fragheaderlen <= mtu &&
      rt->dst.dev->features & (NETIF_F_HW_CSUM | NETIF_F_IP_CSUM) &&
      (!(flags & MSG_MORE) || cork->gso_size) &&
      (!exthdrlen || (rt->dst.dev->features & NETIF_F_HW_ESP_TX_CSUM)))
    csummode = CHECKSUM_PARTIAL;

  cork->length += length;

  /* So, what's going on in the loop below?
   *
   * We use calculated fragment length to generate chained skb,
   * each of segments is IP fragment ready for sending to network after
   * adding appropriate IP header.
   */

  if (!skb)
    goto alloc_new_skb;

  while (length > 0)
  {
    /* Check if the remaining data fits into current packet. */
    copy = mtu - skb->len;
    if (copy < length)
      copy = maxfraglen - skb->len;
    if (copy <= 0)
    {
      char *data;
      unsigned int datalen;
      unsigned int fraglen;
      unsigned int fraggap;
      unsigned int alloclen;
      unsigned int pagedlen = 0;
      struct sk_buff *skb_prev;
    alloc_new_skb:
      skb_prev = skb;
      if (skb_prev)
        fraggap = skb_prev->len - maxfraglen;
      else
        fraggap = 0;

      /*
       * If remaining data exceeds the mtu,
       * we know we need more fragment(s).
       */
      datalen = length + fraggap;
      if (datalen > mtu - fragheaderlen)
        datalen = maxfraglen - fragheaderlen;
      fraglen = datalen + fragheaderlen;

      if ((flags & MSG_MORE) &&
          !(rt->dst.dev->features & NETIF_F_SG))
        alloclen = mtu;
      else if (!paged)
        alloclen = fraglen;
      else
      {
        alloclen = min_t(int, fraglen, MAX_HEADER);
        pagedlen = fraglen - alloclen;
      }

      alloclen += exthdrlen;

      /* The last fragment gets additional space at tail.
       * Note, with MSG_MORE we overallocate on fragments,
       * because we have no idea what fragment will be
       * the last.
       */
      if (datalen == length + fraggap)
        alloclen += rt->dst.trailer_len;

      if (transhdrlen)
      {
        skb = sock_alloc_send_skb(sk,
                                  alloclen + hh_len + 15,
                                  (flags & MSG_DONTWAIT), &err);
      }
      else
      {
        skb = NULL;
        if (refcount_read(&sk->sk_wmem_alloc) + wmem_alloc_delta <=
            2 * sk->sk_sndbuf)
          skb = alloc_skb(alloclen + hh_len + 15,
                          sk->sk_allocation);
        if (unlikely(!skb))
          err = -ENOBUFS;
      }
      if (!skb)
        goto error;

      /*
       *	Fill in the control structures
       */
      skb->ip_summed = csummode;
      skb->csum = 0;
      skb_reserve(skb, hh_len);

      /* only the initial fragment is time stamped */
      skb_shinfo(skb)->tx_flags = cork->tx_flags;
      cork->tx_flags = 0;
      skb_shinfo(skb)->tskey = tskey;
      tskey = 0;

      /*
       *	Find where to start putting bytes.
       */
      data = skb_put(skb, fraglen + exthdrlen - pagedlen);
      skb_set_network_header(skb, exthdrlen);
      skb->transport_header = (skb->network_header +
                               fragheaderlen);
      data += fragheaderlen + exthdrlen;

      if (fraggap)
      {
        skb->csum = skb_copy_and_csum_bits(
            skb_prev, maxfraglen,
            data + transhdrlen, fraggap, 0);
        skb_prev->csum = csum_sub(skb_prev->csum,
                                  skb->csum);
        data += fraggap;
        pskb_trim_unique(skb_prev, maxfraglen);
      }

      copy = datalen - transhdrlen - fraggap - pagedlen;
      // pr_warn("datalen: %d\n", datalen);
      // pr_warn("transhdrlen: %d\n", transhdrlen);
      // pr_warn("skb: %u\n", skb->ip_summed);
      // pr_warn("pagedlen: %d\n", pagedlen);
      // pr_warn("copy: %d\n", copy);
      if (copy > 0 && getfrag(from, data + transhdrlen, offset, copy, fraggap, skb) < 0)
      {
        err = -EFAULT;
        pr_alert("getfrag err %d\n", getfrag(from, data + transhdrlen, offset, copy, fraggap, skb));
        kfree_skb(skb);
        goto error;
      }

      offset += copy;
      length -= copy + transhdrlen;
      transhdrlen = 0;
      exthdrlen = 0;
      csummode = CHECKSUM_NONE;

      if ((flags & MSG_CONFIRM) && !skb_prev)
        skb_set_dst_pending_confirm(skb, 1);

      /*
       * Put the packet on the pending queue.
       */
      if (!skb->destructor)
      {
        skb->destructor = sock_wfree;
        skb->sk = sk;
        wmem_alloc_delta += skb->truesize;
      }
      __skb_queue_tail(queue, skb);
      continue;
    }

    if (copy > length)
      copy = length;

    if (!(rt->dst.dev->features & NETIF_F_SG) &&
        skb_tailroom(skb) >= copy)
    {
      unsigned int off;

      off = skb->len;
      if (getfrag(from, skb_put(skb, copy),
                  offset, copy, off, skb) < 0)
      {
        __skb_trim(skb, off);
        err = -EFAULT;
        pr_alert("__skb_trim err %p\n", rt);
        goto error;
      }
    }
    else
    {
      int i = skb_shinfo(skb)->nr_frags;

      err = -ENOMEM;
      if (!sk_page_frag_refill(sk, pfrag))
        goto error;

      if (!skb_can_coalesce(skb, i, pfrag->page,
                            pfrag->offset))
      {
        err = -EMSGSIZE;
        if (i == MAX_SKB_FRAGS)
          goto error;

        __skb_fill_page_desc(skb, i, pfrag->page,
                             pfrag->offset, 0);
        skb_shinfo(skb)->nr_frags = ++i;
        get_page(pfrag->page);
      }
      copy = min_t(int, copy, pfrag->size - pfrag->offset);
      if (getfrag(from,
                  page_address(pfrag->page) + pfrag->offset,
                  offset, copy, skb->len, skb) < 0)
      {
        pr_alert("error_efault: %d \n", getfrag(from,
                                                page_address(pfrag->page) + pfrag->offset,
                                                offset, copy, skb->len, skb));
        goto error_efault;
      }

      pfrag->offset += copy;
      skb_frag_size_add(&skb_shinfo(skb)->frags[i - 1], copy);
      skb->len += copy;
      skb->data_len += copy;
      skb->truesize += copy;
      wmem_alloc_delta += copy;
    }
    offset += copy;
    length -= copy;
  }

  if (wmem_alloc_delta)
    refcount_add(wmem_alloc_delta, &sk->sk_wmem_alloc);
  return 0;

error_efault:
  err = -EFAULT;
error:
  cork->length -= length;
  IP_INC_STATS(sock_net(sk), IPSTATS_MIB_OUTDISCARDS);
  refcount_add(wmem_alloc_delta, &sk->sk_wmem_alloc);
  return err;
}

int ip_append_data_vmac(struct sock *sk, struct flowi4 *fl4,
                        int getfrag(void *from, char *to, int offset, int len,
                                    int odd, struct sk_buff *skb),
                        void *from, int length, int transhdrlen,
                        struct ipcm_cookie *ipc, struct rtable **rtp,
                        unsigned int flags)
{
  struct inet_sock *inet = inet_sk(sk);
  int err;

  if (flags & MSG_PROBE)
    return 0;

  if (skb_queue_empty(&sk->sk_write_queue))
  {
    err = ip_setup_cork(sk, &inet->cork.base, ipc, rtp);
    if (err)
      return err;
  }
  else
  {
    transhdrlen = 0;
  }

  return __ip_append_data_vmac(sk, fl4, &sk->sk_write_queue, &inet->cork.base,
                               sk_page_frag(sk), getfrag,
                               from, length, transhdrlen, flags);
}

struct sk_buff *ip_make_skb_vmac(struct sock *sk,
                                 struct flowi4 *fl4,
                                 int getfrag(void *from, char *to, int offset,
                                             int len, int odd, struct sk_buff *skb),
                                 void *from, int length, int transhdrlen,
                                 struct ipcm_cookie *ipc, struct rtable **rtp,
                                 struct inet_cork *cork, unsigned int flags)
{
  struct sk_buff_head queue;
  int err;

  if (flags & MSG_PROBE)
    return NULL;

  __skb_queue_head_init(&queue);

  cork->flags = 0;
  cork->addr = 0;
  cork->opt = NULL;
  err = ip_setup_cork(sk, cork, ipc, rtp);
  if (err)
    return ERR_PTR(err);

  err = __ip_append_data_vmac(sk, fl4, &queue, cork,
                              &current->task_frag, getfrag,
                              from, length, transhdrlen, flags);
  if (err)
  {
    __ip_flush_pending_frames(sk, &queue, cork);
    return ERR_PTR(err);
  }

  return __ip_make_skb_vmac(sk, fl4, &queue, cork);
}

/**
 * udp_sendmsg 通过调用 udp_send_skb 函数将 skb 送到下一网络层，在本文中是 IP 协议层。 这个函数做了一些重要的事情：
    1. 向 skb 添加 UDP 头
    2. 处理校验和：软件校验和，硬件校验和或无校验和（如果禁用）
    3. 调用 ip_send_skb 将 skb 发送到 IP 协议层
    4. 更新发送成功或失败的统计计数器
*/
static int udp_send_skb(struct sk_buff *skb, struct flowi4 *fl4,
                        struct inet_cork *cork)
{
  struct sock *sk = skb->sk;
  struct inet_sock *inet = inet_sk(sk);
  struct udphdr *uh;
  int err;
  int is_udplite = IS_UDPLITE(sk);
  int offset = skb_transport_offset(skb);
  int len = skb->len - offset;
  int datalen = len - sizeof(*uh);
  __wsum csum = 0;

  /*
   * Create a UDP header
   */
  uh = udp_hdr(skb);
  uh->source = inet->inet_sport;
  uh->dest = fl4->fl4_dport;
  uh->len = htons(len);
  uh->check = 0;

  if (cork->gso_size)
  {
    const int hlen = skb_network_header_len(skb) +
                     sizeof(struct udphdr);

    if (hlen + cork->gso_size > cork->fragsize)
    {
      kfree_skb(skb);
      return -EINVAL;
    }
    if (skb->len > cork->gso_size * UDP_MAX_SEGMENTS)
    {
      kfree_skb(skb);
      return -EINVAL;
    }
    if (sk->sk_no_check_tx)
    {
      kfree_skb(skb);
      return -EINVAL;
    }
    if (skb->ip_summed != CHECKSUM_PARTIAL || is_udplite ||
        dst_xfrm(skb_dst(skb)))
    {
      kfree_skb(skb);
      return -EIO;
    }

    if (datalen > cork->gso_size)
    {
      skb_shinfo(skb)->gso_size = cork->gso_size;
      skb_shinfo(skb)->gso_type = SKB_GSO_UDP_L4;
      skb_shinfo(skb)->gso_segs = DIV_ROUND_UP(datalen,
                                               cork->gso_size);
    }
    goto csum_partial;
  }

  /* 处理校验和 */
  if (is_udplite) /*     UDP-Lite  首先处理 UDP-Lite 校验和     */
    csum = udplite_csum(skb);

  else if (sk->sk_no_check_tx)
  { /* UDP csum off 如果 socket 校验和选项被关闭（setsockopt 带 SO_NO_CHECK 参数），它将被标记为校验和关闭  */

    skb->ip_summed = CHECKSUM_NONE;
    goto send;
  }
  else if (skb->ip_summed == CHECKSUM_PARTIAL)
  { /* UDP hardware csum 如果硬件支持 UDP 校验和，则将调用 udp4_hwcsum 来设置它  */
  csum_partial:

    udp4_hwcsum(skb, fl4->saddr, fl4->daddr);
    goto send;
  }
  else
    csum = udp_csum(skb);

  /* add protocol-dependent pseudo-header 添加了伪头 */
  uh->check = csum_tcpudp_magic(fl4->saddr, fl4->daddr, len,
                                sk->sk_protocol, csum);

  /* 如果校验和为 0，则根据 RFC 768，校验为全 1 */
  if (uh->check == 0)
    uh->check = CSUM_MANGLED_0;

// TODO
send:
  // err = ip_send_skb_vmac(sock_net(sk), skb);
  // if (err)
  // {
  //   if (err == -ENOBUFS && !inet->recverr)
  //   {
  //     UDP_INC_STATS(sock_net(sk),
  //                   UDP_MIB_SNDBUFERRORS, is_udplite);
  //     err = 0;
  //   }
  // }
  // else
  //   UDP_INC_STATS(sock_net(sk),
  //                 UDP_MIB_OUTDATAGRAMS, is_udplite);
  return err;
}

/* udp_sendmsg logic */
int udp_sendmsg(struct sock *sk, struct msghdr *msg, size_t len)
{
  /*套接字的网络层表示转换成INET套接字的表示*/
  struct inet_sock *inet = inet_sk(sk);
  /*套接字的网络层表示转换成UDP套接字的表示*/
  struct udp_sock *up = udp_sk(sk);
  DECLARE_SOCKADDR(struct sockaddr_in *, usin, msg->msg_name);
  struct flowi4 fl4_stack;
  struct flowi4 *fl4;
  int ulen = len;
  struct ipcm_cookie ipc;
  struct rtable *rt = NULL;
  int free = 0;
  int connected = 0;
  __be32 daddr, faddr, saddr;
  __be16 dport;
  u8 tos;
  int err, is_udplite = IS_UDPLITE(sk);
  /* 如果 up->corkflag> 0，则将套接字标记为 UDP-Lite,后半段判断是否还需要发送更多消息 */
  int corkreq = up->corkflag || msg->msg_flags & 0x8000;
  int (*getfrag)(void *, char *, int, int, int, struct sk_buff *);
  struct sk_buff *skb;
  struct ip_options_data opt_copy;

  /* UDP数据报最长为64KB */
  if (len > 0xFFFF)
    return -EMSGSIZE;

  /*
   *	Check the flags. UDP不支持发送带外数据(加速数据)，如果发送标志中设置了MSG_OOB，则返回
   */
  if (msg->msg_flags & MSG_OOB) /* Mirror BSD error message compatibility */
    return -EOPNOTSUPP;

  /*
   * udp和udplite使用不同的getfrag。根据is_udplite标志来指定"从用户态复制数据到UDP分片"的函数，UDP和轻量级UDP的实现共用了一套函数，
   * 只是在计算校验和上有点区别。轻量级UDP可以在发送前(而不是在复制数据到分片中时)对数据前部指定数目的字节或全部数据执行校验和。而UDP如果是由软件执行校验和(当网卡硬件支持udp checksum offload
   * 并开启相关功能后，校验和由硬件执行)，则在复制数据到分片中时对数据包中的全部数据执行校验和。所以，轻量级UDP和UDP使用不同的"getfrag"函数。
   */
  getfrag = is_udplite ? udplite_getfrag : ip_generic_getfrag_vmac;

  fl4 = &inet->cork.fl.u.ip4;
  pr_info("if corking: %d\n", up->pending); /* 0 */
  /* 检查 up->pending 以确定 socket 当前是否已被塞住(corked)，如果是，则直接跳到 do_append_data 进行数据追加(append)。 */
  if (up->pending)
  {
    /*
     * There are pending frames. The socket lock must be held while it's corked. 当前的sock有等待发送的数据，直接将数据追加
     */
    lock_sock(sk);
    /* likely和unlikely对程序逻辑没影响，likely提示编译器括号内的内容为真的概率更大，unlikely相反 */
    if (likely(up->pending))
    {
      /*
       * 为什么再判断一次?为了提升效率。
       * 因为大部分情况下pending标记是没有的，这样的话就不会进入到这里，就可以省掉一个lock_sock(比较复杂、耗时)，仅当设置了pending后，
       * 才加锁并再检查一次，这样就能在大部分情况下不用锁，少数情况下加锁，这种方法是内核中常用的提升效率的策略。
       */
      if (unlikely(up->pending != AF_INET))
      { /*pending既不是0，又不是AF_INET，那就是有问题了*/
        release_sock(sk);
        return -EINVAL;
      }
      /* up->pending为AF_INET时候，直接跳转到数据发送，这里进行的了第一次数据发送后的数据发送 */
      /* 利用当前sock发送队列中的原有的skb发送数据，将新数据附加到相应skb中的数据区即可 */
      goto do_append_data;
    }
    release_sock(sk);
  }

  /* 接下来的都是第一次发包时的操作，UDP数据报长度，包括UDP data + UDP header */
  ulen += sizeof(struct udphdr);

  /*
  *	Get and verify the address.
    接下来获取目的 IP 地址和端口，有两个可能的来源：
    1. 如果之前 socket 已经建立连接，那 socket 本身就存储了目标地址
    2. 地址通过辅助结构（struct msghdr）传入，通常为调用 sendto 发送UDP数据
 */
  pr_info("usin: %p\n", usin); /* exist */
  if (usin)
  {
    /*判断长度*/
    if (msg->msg_namelen < sizeof(*usin))
      return -EINVAL;
    /*判断地址族，必须为AF_INET*/
    if (usin->sin_family != AF_INET)
    {
      if (usin->sin_family != AF_UNSPEC)
        return -EAFNOSUPPORT;
    }
    daddr = usin->sin_addr.s_addr;
    dport = usin->sin_port;
    /* 目的端口不能为零 */
    if (dport == 0)
      return -EINVAL;
  }
  else
  {
    /* msg没有目的地址的情况：通常为先调用了connect,然后调用send发送UDP数据，UDP套接字调用connetc之后，UDP传输控制块状态为TCP_ESTABLISHED */
    if (sk->sk_state != TCP_ESTABLISHED)
      /*即没有指明目的地址，又没有建立connect连接，则返错。*/
      return -EDESTADDRREQ;
    daddr = inet->inet_daddr;
    dport = inet->inet_dport;
    /* Open fast path for connected socket.
       Route will not be used, if at least one option is set.
     */
    /* 对于已连接的UDP套接口，设置connected标志，在后续查找路由时用，可以根据此做快速处理 */
    connected = 1;
  }

  /* 获取存储在 socket 上的源地址、发送网络设备索引（device index）和时间戳选项 */
  ipcm_init_sk(&ipc, inet);
  /* 保留了当套接字被解锁时创建 UDP 标头的信息 */
  ipc.gso_size = READ_ONCE(up->gso_size);

  // msg中控制信息处理, 处理辅助消息
  if (msg->msg_controllen)
  {
    pr_alert("have msg_controllen: %d, but not be deal with.\n", msg->msg_controllen);
    err = 0;
    // /*如果msg_controllen辅助缓冲区中有数据，则消息发送*/
    // err = udp_cmsg_send(sk, msg, &ipc.gso_size);
    // if (err > 0)
    //   /*调用ip_cmsg_send处理控制信息，包括IP选项等...*/
    //   err = ip_cmsg_send(sk, msg, &ipc,
    //                      sk->sk_family == AF_INET6);
    // if (unlikely(err < 0))
    // {
    //   kfree(ipc.opt);
    //   return err;
    // }
    // /* 如果ipc中存在IP选项，则设置free标记，表示需要在处理完成后释放。因为此时的ipc->opt肯定是在ip_cmsg_send中分配的 */
    // if (ipc.opt)
    //   free = 1;
    // /*这里表示不进行路由*/
    // connected = 0;
  }
  /* 如果发送数据中的控制信息中没有IP选项信息，则从inet_sock结构中获取 */
  if (!ipc.opt)
  {
    struct ip_options_rcu *inet_opt;

    rcu_read_lock();
    inet_opt = rcu_dereference(inet->inet_opt);
    if (inet_opt)
    {
      memcpy(&opt_copy, inet_opt,
             sizeof(*inet_opt) + inet_opt->opt.optlen);
      ipc.opt = &opt_copy.opt;
    }
    rcu_read_unlock();
  }

  if (cgroup_bpf_enabled_vmac(11) && !connected)
  {
    err = BPF_CGROUP_RUN_PROG_UDP4_SENDMSG_LOCK(sk, (struct sockaddr *)usin, &ipc.addr);
    if (err)
      goto out_free;
    if (usin)
    {
      if (usin->sin_port == 0)
      {
        /* BPF program set invalid port. Reject it. */
        err = -EINVAL;
        goto out_free;
      }
      daddr = usin->sin_addr.s_addr;
      dport = usin->sin_port;
    }
  }

  /*由于控制信息需要保存目的地址，因此将源地址保存*/
  saddr = ipc.addr;
  ipc.addr = faddr = daddr;

  /*
   * 如果存在宽松或严格源站选路的IP选项，则不能根据目的地址选路，而需要将IP选项中下一站地址作为目的地址来选路，
   * 因此从IP选项中提取下一站地址，供后续选路时作为目的地址使用。另外，因为需要重新选路，所以清除connected标记
   */
  if (ipc.opt && ipc.opt->opt.srr)
  {
    if (!daddr)
    {
      err = -EINVAL;
      goto out_free;
    }
    faddr = ipc.opt->opt.faddr;
    connected = 0;
  }
  tos = get_rttos(&ipc, inet);
  if (sock_flag(sk, SOCK_LOCALROUTE) ||
      (msg->msg_flags & MSG_DONTROUTE) ||
      (ipc.opt && ipc.opt->opt.is_strictroute))
  {
    tos |= RTO_ONLINK;
    connected = 0;
  }

  pr_warn("if multicast: %d\n", ipv4_is_multicast(daddr)); /* 0 */
  if (ipv4_is_multicast(daddr))
  {
    if (!ipc.oif || netif_index_is_l3_master(sock_net(sk), ipc.oif))
      ipc.oif = inet->mc_index;
    if (!saddr)
      saddr = inet->mc_addr;
    connected = 0;
  }
  else if (!ipc.oif)
  {
    ipc.oif = inet->uc_index;
  }
  else if (ipv4_is_lbcast(daddr) && inet->uc_index)
  {
    /* oif is set, packet is to local broadcast and uc_index is set. oif is most likely set
     * by sk_bound_dev_if. If uc_index != oif check if the oif is an L3 master and uc_index is an L3 slave.
     * If so, we want to allow the send using the uc_index.
     */
    if (ipc.oif != inet->uc_index &&
        ipc.oif == l3mdev_master_ifindex_by_index(sock_net(sk), inet->uc_index))
    {
      ipc.oif = inet->uc_index;
    }
  }

  if (connected)
    rt = (struct rtable *)sk_dst_check(sk, 0);

  if (!rt)
  {
    struct net *net = sock_net(sk);
    __u8 flow_flags = inet_sk_flowi_flags(sk);

    fl4 = &fl4_stack;

    flowi4_init_output(fl4, ipc.oif, ipc.sockc.mark, tos,
                       RT_SCOPE_UNIVERSE, sk->sk_protocol,
                       flow_flags,
                       faddr, saddr, dport, inet->inet_sport,
                       sk->sk_uid);

    security_sk_classify_flow(sk, flowi4_to_flowi(fl4));
    rt = ip_route_output_flow(net, fl4, sk);
    if (IS_ERR(rt))
    {
      err = PTR_ERR(rt);
      rt = NULL;
      if (err == -ENETUNREACH)
        IP_INC_STATS(net, IPSTATS_MIB_OUTNOROUTES);
      goto out;
    }

    err = -EACCES;
    if ((rt->rt_flags & RTCF_BROADCAST) &&
        !sock_flag(sk, SOCK_BROADCAST))
      goto out;
    if (connected)
      sk_dst_set(sk, dst_clone(&rt->dst));
  }

  if (msg->msg_flags & MSG_CONFIRM)
    goto do_confirm;

back_from_confirm:

  saddr = fl4->saddr;
  if (!ipc.addr)
    daddr = ipc.addr = fl4->daddr;

  /* Lockless fast path for the non-corking case. */
  pr_warn("corkreq: %d\n", corkreq); /* 0，走快速路径 */
  if (!corkreq)
  {
    struct inet_cork cork;

    /**
     * ip_make_skb 函数将创建一个 skb，其中需要考虑到很多的事情，例如：
        MTU
        UDP corking（如果启用）
        UDP Fragmentation Offloading（UFO）
        Fragmentation（分片）：如果硬件不支持 UFO，但是要传输的数据大于 MTU，需要软件做分片
    */
    /*
     * 将sock相应的skb队列中的所有skb合并成一个数据报文(skb)，实际使用skb_shinfo->frag_list将所有skb连接起来。
     * 为什么要这样?这里将所有skb都合并后，可能导致这个包的size大于mtu，那到IP层的时候还会进行进一步分片?
     * 原因是:udp是面向数据报文的，报文必须完整，属于同一个报文的数据必须要放到同一个skb中，否则对端无法知道这是同一个报文。
     * 那IP层怎么处理呢?就不考虑同一个报文的问题?IP层分片会携带相关头信息，对端会根据这些信息进行重组，重组后对传输层来说就是一个报文。
     * 分片其实就是IP层应该负责的。此时IP分片实际就是将原来skb->frag_list中的skb摘出来，不会做其它的操作，效率很高。
     */
    skb = ip_make_skb_vmac(sk, fl4, getfrag, msg, ulen,
                           sizeof(struct udphdr), &ipc, &rt,
                           &cork, msg->msg_flags);
    err = PTR_ERR(skb);
    if (!IS_ERR_OR_NULL(skb))
      /*直接发送数据*/
      err = udp_send_skb(skb, fl4, &cork);
    goto out;
  }

  /* 要往skb中添加数据了并访问sock中的相关数据了，此时需要加锁了，因为同一个socket可能被多个进程在多个CPU上同时访问 */
  lock_sock(sk);
  if (unlikely(up->pending))
  {
    /* The socket is already corked while preparing it. */
    /* ... which is an evident application bug. --ANK */
    release_sock(sk);

    net_dbg_ratelimited("socket already corked\n");
    err = -EINVAL;
    goto out;
  }
  /*
   *	Now cork the socket to pend data.
   *  缓存目的地址、目的端口、源地址和源端口信息，便于在发送处理时方便获取信息
   */
  fl4 = &inet->cork.fl.u.ip4;
  fl4->daddr = daddr;
  fl4->saddr = saddr;
  fl4->fl4_dport = dport;
  fl4->fl4_sport = inet->inet_sport;
  /* 设置AF_INET标记，表明正在处理UDP数据包 */
  up->pending = AF_INET;

/*
 * 运行到这里，有两种情况:1.最初进入函数时pending=AF_INET，即有udp数据包正在处理中；2.设置了cork，
 * 则表明需要阻塞，使用原有的skb一起发送数据.
 */
do_append_data:
  /* 增加包长 */
  up->len += ulen;
  /*
   * 调用IP层接口函数ip_append_data，进入IP层处理，主要工作为:
   * 将数据拷贝到适合的skb(利用发送队列中现有的或新创建)中，可能有两种情况: 1. 放入skb的线性
   * 区(skb->data)中，或者放入skb_shared_info的分片(frag)中，同时还需要考虑MTU对skb数据进行分割。
   */
  err = ip_append_data_vmac(sk, fl4, getfrag, msg, ulen,
                            sizeof(struct udphdr), &ipc, &rt,
                            corkreq ? msg->msg_flags | MSG_MORE : msg->msg_flags);
  if (err)
    /*出错则清空所有pending的数据帧，并清空pending标记*/
    udp_flush_pending_frames(sk);
  /*未设置cork(如果设置了cork，则需要等待组成64k大小的UDP数据报后再发送)，则直接发送数据到IP层*/
  else if (!corkreq)
    err = udp_push_pending_frames(sk);
  else if (unlikely(skb_queue_empty(&sk->sk_write_queue)))
    /*如果发送队列为空，则说明没有数据正在处理了，则复位pending标记*/
    up->pending = 0;
  release_sock(sk);

out:
  ip_rt_put(rt);
out_free:
  if (free)
    kfree(ipc.opt);
  if (!err)
    return len;
  /*
   * ENOBUFS = no kernel mem, SOCK_NOSPACE = no sndbuf space.  Reporting
   * ENOBUFS might not be good (it's not tunable per se), but otherwise
   * we don't have a good statistic (IpOutDiscards but it can be too many
   * things).  We could add another new stat but at least for now that
   * seems like overkill.
   */
  if (err == -ENOBUFS || test_bit(SOCK_NOSPACE, &sk->sk_socket->flags))
  {
    UDP_INC_STATS(sock_net(sk),
                  UDP_MIB_SNDBUFERRORS, is_udplite);
  }
  return err;

do_confirm:
  if (msg->msg_flags & MSG_PROBE)
    dst_confirm_neigh(&rt->dst, &fl4->daddr);
  if (!(msg->msg_flags & MSG_PROBE) || len)
    goto back_from_confirm;
  err = 0;
  goto out;
}

int vmac_inet_sendmsg(struct socket *sock, struct msghdr *msg, size_t size)
{
  // 如果没有绑定端口, 这里自动绑定端口
  struct sock *sk = sock->sk;
  int err;

  sock_rps_record_flow(sk);

  // /* We may need to bind the socket. */
  // if (!inet_sk(sk)->inet_num && !sk->sk_prot->no_autobind &&
  //     inet_autobind(sk))
  //   return -EAGAIN;

  pr_info("call udp_sendmsg\n");
  err = udp_sendmsg(sk, msg, size);
  return err;
}
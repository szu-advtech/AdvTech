#include "vmac-usrsp.h"
#include "uthash.h"

#define SEG_FAULT_LIMIT 100 /* seg fault occurs if size < 100 */

struct vmac_lib_priv vmac_priv;

/**
 * @brief      maps data rate to rix index for MAC and PHY coordination
 *
 * @param[in]  rate  The rate requested by tx
 *
 * @return     rix of internal firmware index
 */
uint8_t getrix(double rate)
{
  int i;
  uint8_t ret = 0; /* default to 1 Mbps if cannot find rate */
  for (i = 0; i < RATES_NUM; i++)
  {
    if (rates[i].rate == rate)
    {
      ret = rates[i].rix;
    }
  }
  return ret;
}

/**
 * @brief      Reception thread
 *
 * @param      tid   The thread id (empty for now, not used)
 *
 * @return     Functions runs indefinitely waiting to receive any frame, parse and extract information then pass to callback
 */
void *recvvmac(void *tid)
{
  uint64_t enc;
  uint16_t type;
  uint16_t seq, intnamelen;
  char *intname;
  int send;
  char *buffer;
  struct control rxc;
  struct hash *s = NULL;
  struct vmac_frame *frame;
  struct meta_data *meta;

  while (1)
  {
    recvmsg(vmac_priv.sock_fd, &vmac_priv.msg2, 0);

    /* allocate structs for callback function */
    frame = malloc(sizeof(struct vmac_frame));
    meta = malloc(sizeof(struct meta_data));
    /* Process received frame */
    buffer = malloc(vmac_priv.nlh2->nlmsg_len - SEG_FAULT_LIMIT);
    memcpy(&rxc, NLMSG_DATA(vmac_priv.nlh2), sizeof(struct control));
    memcpy(&buffer[0], NLMSG_DATA(vmac_priv.nlh2) + sizeof(struct control), vmac_priv.nlh2->nlmsg_len - SEG_FAULT_LIMIT);
    type = (*(uint16_t *)(rxc.type));
    seq = (*(uint16_t *)(rxc.seq));
    enc = (*(uint64_t *)(rxc.enc));

    frame->buf = buffer;
    frame->len = vmac_priv.nlh2->nlmsg_len - SEG_FAULT_LIMIT;
    frame->InterestName = intname;
    frame->name_len = intnamelen;
    meta->type = type;
    meta->seq = seq;
    meta->enc = enc;
    (*vmac_priv.cb)(frame, meta);
  }
}

/**
 * @brief      Register process with kernel module and create rx reception thread
 *
 * @param[in]  cf    callback function pointer
 *
 * @return     0 on success
 */
int vmac_register(void (*cf)())
{
  int size;
  char keys[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf};
  struct sched_param params;
  // array assignment directly copys the address of original array,but memcpy() copys data from first mem to next mem
  memcpy(vmac_priv.key, keys, sizeof(keys));
  vmac_priv.msgy[0] = 'a';
  vmac_priv.cb = cf;
  // Returns a file descriptor(int) for the new socket, notes: __protocol = VMAC_USER
  vmac_priv.sock_fd = socket(PF_NETLINK, SOCK_RAW, VMAC_USER);
  // seg fault occurs if size < 100, thus add it to upper 100kb(ie. 102kb total)
  size = strlen(vmac_priv.msgy) + SEG_FAULT_LIMIT;
  memset(&vmac_priv.src_addr, 0, sizeof(vmac_priv.src_addr));
  vmac_priv.src_addr.nl_family = AF_NETLINK;
  vmac_priv.src_addr.nl_pid = getpid();
  bind(vmac_priv.sock_fd, (struct sockaddr *)&vmac_priv.src_addr, sizeof(vmac_priv.src_addr));
  memset(&vmac_priv.dest_addr, 0, sizeof(vmac_priv.dest_addr));
  vmac_priv.dest_addr.nl_family = AF_NETLINK;
  vmac_priv.dest_addr.nl_pid = 0;
  vmac_priv.dest_addr.nl_groups = 0;
  /* register 时分配好内存，最大是 2kb，之后就直接赋值，不用再重新分配内存 */
  vmac_priv.nlh = (struct nlmsghdr *)malloc(NLMSG_SPACE(MAX_PAYLOAD));
  vmac_priv.nlh2 = (struct nlmsghdr *)malloc(NLMSG_SPACE(MAX_PAYLOAD));
  memset(vmac_priv.nlh, 0, NLMSG_SPACE(MAX_PAYLOAD));
  memset(vmac_priv.nlh2, 0, NLMSG_SPACE(MAX_PAYLOAD));
  vmac_priv.nlh2->nlmsg_len = NLMSG_SPACE(MAX_PAYLOAD);
  vmac_priv.nlh->nlmsg_len = size;
  vmac_priv.nlh->nlmsg_pid = getpid();
  vmac_priv.nlh->nlmsg_flags = 0;
  vmac_priv.nlh->nlmsg_type = 4;
  vmac_priv.iov2.iov_base = (void *)vmac_priv.nlh2;
  vmac_priv.iov2.iov_len = vmac_priv.nlh2->nlmsg_len;
  vmac_priv.msg2.msg_name = (void *)&vmac_priv.dest_addr;
  vmac_priv.msg2.msg_namelen = sizeof(vmac_priv.dest_addr);
  vmac_priv.msg2.msg_iov = &vmac_priv.iov2;
  vmac_priv.msg2.msg_iovlen = 1;
  vmac_priv.iov.iov_base = (void *)vmac_priv.nlh;
  vmac_priv.iov.iov_len = vmac_priv.nlh->nlmsg_len;
  vmac_priv.msg.msg_name = (void *)&vmac_priv.dest_addr;
  vmac_priv.msg.msg_namelen = sizeof(vmac_priv.dest_addr);
  vmac_priv.msg.msg_iov = &vmac_priv.iov;
  vmac_priv.msg.msg_iovlen = 1;
  params.sched_priority = sched_get_priority_max(SCHED_FIFO);
  pthread_setschedparam(vmac_priv.thread, SCHED_FIFO, &params);
  pthread_create(&vmac_priv.thread, NULL, recvvmac, (void *)0);
  vmac_priv.nlh->nlmsg_type = 255;
  memset(vmac_priv.msgy, 0, 1024);
  vmac_priv.digest64 = 0;
  memcpy(NLMSG_DATA(vmac_priv.nlh), &vmac_priv.digest64, 8);
  memcpy(NLMSG_DATA(vmac_priv.nlh) + 8, vmac_priv.msgy, strlen(vmac_priv.msgy));
  size = strlen(vmac_priv.msgy) + SEG_FAULT_LIMIT;
  // 这里sendmsg的内容有 msg_name, msg_iov
  sendmsg(vmac_priv.sock_fd, &vmac_priv.msg, 0);
  return 0;
}

/**
 * @brief      Sends a vmac frame to V-MAC kernel module.
 *
 * @param[in]  frame  contains data and interest buffers with their lengths, respectively.
 * @param      meta   contains meta data to be passed to kernel (e.g., type of frame, rate, sequence if applicable)
 *
 * @return     0 on success.
 */
int send_vmac(struct vmac_frame *frame, struct meta_data *meta)
{
  struct control txc;
  struct hash *s;
  char *name;
  uint8_t ratesh = getrix(meta->rate);
  vmac_priv.digest64 = siphash24(frame->InterestName, frame->name_len, vmac_priv.key);
  vmac_priv.nlh->nlmsg_type = (uint16_t)meta->type;

  memcpy(&txc.type[0], &meta->type, sizeof(uint8_t));
  memcpy(&txc.enc[0], &vmac_priv.digest64, sizeof(uint64_t));
  memcpy(&txc.seq[0], &meta->seq, sizeof(uint16_t));
  memcpy(&txc.rate[0], &ratesh, sizeof(uint8_t));
  memcpy(NLMSG_DATA(vmac_priv.nlh), &txc, sizeof(struct control));

  if (frame->len != 0)
  {
    memcpy(NLMSG_DATA(vmac_priv.nlh) + sizeof(struct control), frame->buf, frame->len);
  }

  vmac_priv.nlh->nlmsg_len = frame->len + SEG_FAULT_LIMIT;
  vmac_priv.iov.iov_len = frame->len + SEG_FAULT_LIMIT;
  // transmit data to kernel
  sendmsg(vmac_priv.sock_fd, &vmac_priv.msg, 0);
  return 0;
}

/**
 * @brief      Adds Interest name to userspace hashmap/
 *
 * @param      InterestName  The interest name
 * @param[in]  name_len      The name length
 * NOTE: DEPRACATED
 */
void add_name(char *InterestName, uint16_t name_len)
{
  struct hash *s;
  char *name;
  HASH_FIND(hh, vmac_priv.names, &vmac_priv.digest64, sizeof(uint64_t), s);
  vmac_priv.digest64 = siphash24(InterestName, name_len, vmac_priv.key);
  HASH_FIND(hh, vmac_priv.names, &vmac_priv.digest64, sizeof(uint64_t), s);

  if (s == NULL)
  {
    name = malloc(name_len);
    memcpy(name, InterestName, name_len);
    s = (struct hash *)malloc(sizeof(struct hash));
    s->id = vmac_priv.digest64;
    s->name = name;
    HASH_ADD(hh, vmac_priv.names, id, sizeof(uint64_t), s);
  }
}

/**
 * @brief      Delete interest name from userspace hashmap.
 *
 * @param      InterestName  The interest name
 * @param[in]  name_len      The name length
 * NOTE: DEPRACATED
 */
void del_name(char *InterestName, uint16_t name_len)
{
  struct hash *s;
  char *name;
  vmac_priv.digest64 = siphash24(InterestName, name_len, vmac_priv.key);
  HASH_FIND(hh, vmac_priv.names, &vmac_priv.digest64, sizeof(uint64_t), s);

  if (s != NULL)
  {
    HASH_DEL(vmac_priv.names, s);
    free(s);
  }
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <linux/netlink.h>
#include <syslog.h>
#include <pthread.h>
#include <setjmp.h>
#include <sched.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>

struct vmac_lib_priv
{
  struct hash *names;
  struct sockaddr_nl src_addr, dest_addr;
  /* TX structs */
  struct nlmsghdr *nlh;
  struct iovec iov;
  struct msghdr msg;

  /* RX structs */
  struct nlmsghdr *nlh2;
  struct iovec iov2;
  struct msghdr msg2;

  uint64_t digest64;
  uint8_t fixed_rate;
  void (*cb)();
  char msgy[2000]; /* buffer to store frame */
  int sock_fd;
  pthread_t thread;
  char key[16];
};

struct vmac_lib_priv vmac_priv;
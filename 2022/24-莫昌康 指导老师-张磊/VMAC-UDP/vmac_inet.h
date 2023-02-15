#include <linux/netdev_features.h>
#include <net/inet_common.h>

int vmac_inet_bind(struct socket *sock, struct sockaddr *uaddr, int addr_len);

int vmac_inet_sendmsg(struct socket *sock, struct msghdr *msg, size_t size);

#define cgroup_bpf_enabled_vmac(atype) (0)
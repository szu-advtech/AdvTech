int ip_send_skb_vmac(struct net *, struct sk_buff *);
int sendto_vmac(char __user *buff_name, void __user *buff, size_t len, unsigned int flags, struct sockaddr __user *addr, int addr_len);
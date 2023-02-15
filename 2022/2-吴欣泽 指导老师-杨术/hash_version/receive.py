#!/usr/bin/env python
import sys
import struct
import os
import time
from scapy.all import sniff, sendp, hexdump, get_if_list, get_if_hwaddr
from scapy.all import Packet, IPOption
from scapy.all import ShortField, IntField, LongField, BitField, FieldListField, FieldLenField
from scapy.all import IP, TCP, UDP, Raw
from scapy.all import *
from scapy.layers.inet import _IPOption_HDR

def get_if():
    ifs=get_if_list()
    iface=None
    for i in get_if_list():
        if "eth0" in i:
            iface=i
            break;
    if not iface:
        print "Cannot find eth0 interface"
        exit(1)
    return iface

class IPOption_MRI(IPOption):
    name = "MRI"
    option = 31
    fields_desc = [ _IPOption_HDR,
                    FieldLenField("length", None, fmt="B",
                                  length_of="swids",
                                  adjust=lambda pkt,l:l+4),
                    ShortField("count", 0),
                    FieldListField("swids",
                                   [],
                                   IntField("", 0),
                                   length_from=lambda pkt:pkt.count*4) ]
def handle_pkt(pkt):
    ifaces = filter(lambda i: 'eth' in i, os.listdir('/sys/class/net/'))
    iface = ifaces[0]
    data=''
    try:
	data = pkt[Raw].load
    except:
	pass
    if "hand2" in data or "hand3" in data:
	print data
    elif TCP in pkt and pkt[TCP].flags == 2: #监听到第一次握手，发送第二次握手
	print "got hand 1"
	time.sleep(1)
	pkt.show2()
	pkt2= Ether(src=get_if_hwaddr(iface), dst=pkt[Ether].src)
	pkt2 = pkt2 /IP(src=pkt[IP].dst, dst=pkt[IP].src)/TCP(dport=pkt[TCP].sport,sport=pkt[TCP].dport,seq=random.randint(100,200),ack=pkt[TCP].seq + 1,flags=18) /"hand2"
	sendp(pkt2, iface=iface, verbose=False)
	print "\n ack sending!!!"
	pkt2.show2()
    elif TCP in pkt:
        print "got a packet"
        pkt.show2()
    #    hexdump(pkt)
    else:
	pass
    sys.stdout.flush()


def main():
    ifaces = filter(lambda i: 'eth' in i, os.listdir('/sys/class/net/'))
    iface = ifaces[0]
    print "sniffing on %s" % iface
    sys.stdout.flush()
    sniff(iface = iface,
          prn = lambda x: handle_pkt(x))

if __name__ == '__main__':
    main()

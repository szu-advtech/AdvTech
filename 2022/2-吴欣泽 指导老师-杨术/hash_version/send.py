#!/usr/bin/env python
import argparse
import sys
import socket
import random
import struct
import time

from scapy.all import sendp, send, get_if_list, get_if_hwaddr
from scapy.all import Packet
from scapy.all import Ether, IP, UDP, TCP,sr1
from scapy.all import *


def get_if():
    ifs=get_if_list()
    iface=None # "h1-eth0"
    for i in get_if_list():
        if "eth0" in i:
            iface=i
            break;
    if not iface:
        print "Cannot find eth0 interface"
        exit(1)
    return iface

def src_ip_str():
    return str(random.randint(120, 150)) + '.' + str(random.randint(1, 254)) + \
            '.' + str(random.randint(1, 254)) + '.' + \
            str(random.randint(1, 254))

def send(src_ip=src_ip_str(),tseq=random.randint(0,100),tack=0,tdport=1234,tsport=80,tflags="S",data="-1",TTL=64): 
    
    sys.argv.append('10.0.0.20')
    sys.argv.append('hello'+str(src_ip))
    print(sys.argv)
    if len(sys.argv)<3:
        print 'pass 2 arguments: <destination> "<message>"'
        exit(1)

    addr = socket.gethostbyname(sys.argv[1])
    iface = get_if()
    print "sending on interface %s to %s" % (iface, str(addr))
    pkt =  Ether(src=get_if_hwaddr(iface), dst='ff:ff:ff:ff:ff:ff')
    #pkt = pkt /IP(src=src_ip, dst=addr) / TCP(dport=1, sport=random.randint(1, 65535)) / sys.argv[2]
    data= sys.argv[2] if data == '-1' else data
    pkt = pkt /IP(src=src_ip, dst='10.0.0.20',ttl=TTL) / TCP(seq=tseq, ack=tack, dport=tdport, sport=tsport, flags=tflags) / data
    pkt.show2()
    sendp(pkt, iface=iface, verbose=False)

def handle_pkt(pkt):
    iface = get_if()
    tmp_data=''
    try:
	tmp_data = pkt[Raw].load
    except:
	pass
    if "hand2" in tmp_data:
	data='hand3'
	for i in range(0,1): #第三次握手
	    time.sleep(1)
	    send(src_ip=pkt[IP].dst,tseq=pkt[TCP].ack,tack=pkt[TCP].seq+1,tdport=1234,tsport=80,tflags=24,data=data,TTL=63)
	for i in range(0,100): #第三次握手后继续发消息
	    time.sleep(1)
	    seqq=random.randint(200,300)
	    send(src_ip=pkt[IP].dst,tseq=seqq,tack=pkt[TCP].seq+1,tdport=1234,tsport=80,tflags=24,data='hand3 '+str(seqq),TTL=63)
	return True
    else:
	return False


def main():
    iface = get_if()
    send(src_ip='136.229.153.215', data="",TTL=63) #第一次握手，然后监听第二次握手
    sys.stdout.flush()
    sniff(iface = iface,
	  stop_filter = lambda x:handle_pkt(x),
         prn = lambda x: handle_pkt(x),
	  )


if __name__ == '__main__':
    main()

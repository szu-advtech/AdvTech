#!/usr/bin/env python
import argparse
import sys
import socket
import random
import struct

from scapy.all import *



def main():

	dst_ip = "10.0.0.20"
	dst_port = 80
	src_ip = str(random.randint(120, 150)) + '.' + str(random.randint(1, 254)) + \
		'.' + str(random.randint(1, 254)) + '.' + \
		str(random.randint(1, 254))
	src_port = 20001
	data = "hello from: "+src_ip
	##数据是我随意构造的，一个随便的http请求
	try:
	    spk1 = IP(dst=dst_ip)/TCP(dport=dst_port,sport=src_port,flags="S")
	    res1 = sr1(spk1)
	    ack1 = res1[TCP].ack
	    ack2 = res1[TCP].seq + 1
	    spk2 = IP(src=src_ip, dst=dst_ip)/TCP(dport=dst_port,sport=src_port,seq=ack1,ack=ack2,flags="A")
	    send(spk2)
	except Exception as e:
	    print(e)
	da1 = IP(src=src_ip,dst=dst_ip)/TCP(dport=dst_port,sport=src_port,seq=ack1,ack=ack2,flags=24)/data
	res2 = sr1(da1)





    src_ip = str(random.randint(120, 150)) + '.' + str(random.randint(1, 254)) + \
            '.' + str(random.randint(1, 254)) + '.' + \
            str(random.randint(1, 254))
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
    pkt = pkt /IP(src=src_ip, dst=addr) / TCP(dport=1234, sport=random.randint(49152,65535)) / sys.argv[2]
    #pkt = pkt /IP(dst='10.0.0.20') / TCP(dport=1234, sport=1234) / sys.argv[2] 
    pkt.show2()
    sendp(pkt, iface=iface, verbose=False)


if __name__ == '__main__':
    main()

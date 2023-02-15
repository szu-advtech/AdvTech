#!/usr/bin/python

# Copyright 2013-present Barefoot Networks, Inc. 
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mininet.net import Mininet
from mininet.topo import Topo
from mininet.log import setLogLevel, info
from mininet.cli import CLI
from mininet.link import TCLink

from p4_mininet import P4Switch, P4Host

import argparse
from time import sleep
import os
import subprocess

_THIS_DIR = os.path.dirname(os.path.realpath(__file__))
_THRIFT_BASE_PORT = 22222

parser = argparse.ArgumentParser(description='Mininet demo')
parser.add_argument('--behavioral-exe', help='Path to behavioral executable',
                    type=str, action="store", required=True)
parser.add_argument('--json', help='Path to JSON config file',
                    type=str, action="store", required=True)
parser.add_argument('--cli', help='Path to BM CLI',
                    type=str, action="store", required=True)
parser.add_argument('--thrift-port', help='Thrift server port for table updates',
                            type=int, action="store", required=True)


args = parser.parse_args()

class MyTopo(Topo):
    def __init__(self, sw_path, json_path, thrift_port, **opts):
        Topo.__init__(self,**opts)
        
        switch = self.addSwitch('s1',
                                sw_path = sw_path,
                                json_path = json_path,
                                thrift_port = thrift_port,
                                pcap_dump = True)
#internal host
        h1 = self.addHost('h1',
                          ip = "10.0.0.10",
                          mac = "00:04:00:00:00:10")
        self.addLink(h1,switch)
#external host
        h2 = self.addHost('h2',
          #                ip = "192.168.0.10",
                           ip = "10.0.0.20",
                           mac = "00:05:00:00:00:10")
        self.addLink(h2, switch)
                    

def main():

    topo = MyTopo(args.behavioral_exe,
                  args.json,
                  args.thrift_port)
 

    net = Mininet(topo = topo,
                  host = P4Host,
                  switch = P4Switch,
                  controller = None )
    net.start()

    sw_macs = ["00:aa:bb:00:00:04", "00:aa:bb:00:00:05"]
    
    sw_addrs = ["10.0.0.1", "192.168.0.1"]
    
    h_macs = ["00:05:00:00:00:10","00:04:00:00:00:10"]
    h_addrs = ['10.0.0.20','10.0.0.10']
    
    for n in xrange(2):
        h = net.get('h%d' %(n+1))
        h.setARP(h_addrs[n], h_macs[n])
  #      h.setDefaultRoute("dev eth0 via %s" % sw_addrs[n])
    
    for n in xrange(2):
        h = net.get('h%d' % (n+1))
        h.describe()

    sleep(1)
    
    cmd = [args.cli,"--json",  args.json, "--thrift-port",str(args.thrift_port)]

    print(cmd)
    with open("commands.txt", "r") as f:
        print " ".join(cmd)
        try:
            output = subprocess.check_output(cmd, stdin = f)
            print output
        except subprocess.CalledProcessError as e:
            print e
            print e.output


    sleep(1)


    print "Ready !"

    CLI( net )
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    main() 

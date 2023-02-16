#!/usr/bin/env python
import os
from mininet.net import Mininet
from mininet.node import Controller, RemoteController
from mininet.cli import CLI
from mininet.link import Intf
from mininet.log import setLogLevel, info

def ToRealnet():

    net = Mininet( topo=None, build=False)

    info( '*** Adding controller\n' )
    net.addController('c1', controller=RemoteController,ip='127.0.0.1', port=6653)

    info( '*** Add switches\n')
    s1 = net.addSwitch('s1',protocols="OpenFlow14")
    s2 = net.addSwitch('s2',protocols="OpenFlow14")
    
    info( '*** Add hosts\n')
    h1 = net.addHost('h1', ip='0.0.0.0',mac='00:00:00:00:00:01')
    h2 = net.addHost('h2', ip='0.0.0.0',mac='00:00:00:00:00:02')
    h3 = net.addHost('h3', ip='0.0.0.0',mac='00:00:00:00:00:03')
    h4 = net.addHost('h4', ip='0.0.0.0',mac='00:00:00:00:00:04')



    info( '*** Add links\n')
    net.addLink(h1, s1)
    net.addLink(h2, s1)
    net.addLink(h3, s2)
    net.addLink(h4, s2)
    net.addLink(s1, s2)
    
    info( '*** Starting network\n')
    net.start()
    os.popen('ovs-vsctl add-port s1 eth0')
    h1.cmdPrint('dhclient '+h1.defaultIntf().name)
    h2.cmdPrint('dhclient '+h2.defaultIntf().name)
    h3.cmdPrint('dhclient '+h3.defaultIntf().name)
    h4.cmdPrint('dhclient '+h4.defaultIntf().name)
    CLI(net)
    os.popen('ovs-vsctl del-port s1 eth0')
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    ToRealnet()



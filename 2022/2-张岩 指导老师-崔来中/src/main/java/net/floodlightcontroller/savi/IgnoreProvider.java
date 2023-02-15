package net.floodlightcontroller.savi;
import net.floodlightcontroller.devicemanager.SwitchPort;
import net.floodlightcontroller.routing.IRoutingDecision.RoutingAction;
public class IgnoreProvider extends ReactiveProvider {
	@Override
	protected RoutingAction process(SwitchPort switchPort, Ethernet eth) {
		MacAddress macAddress = eth.getSourceMACAddress();
		if(securityPort.contains(switchPort) || !topologyService.isEdge(switchPort.getSwitchDPID(), switchPort.getPort())) {
			return RoutingAction.FORWARD_OR_FLOOD;
		}
		if(eth.getEtherType() == EthType.IPv4) {
			IPv4 ipv4 = (IPv4)eth.getPayload();
			IPv4Address address = ipv4.getSourceAddress();
			if(this.manager.check(switchPort, macAddress, address)) {
				return RoutingAction.FORWARD_OR_FLOOD;
			}
			else {
				return RoutingAction.NONE;
			}
		}
		else if(eth.getEtherType() == EthType.IPv6) {
			IPv6 ipv6 = (IPv6)eth.getPayload();
			IPv6Address address = ipv6.getSourceAddress();
			for(int i = 0 ; i < list.size() ; i++){
				log.info(list.get(i).getAddress().toString());
			if(this.manager.check(switchPort, macAddress, address)) {
				return RoutingAction.FORWARD_OR_FLOOD;
			}
			else {
				if (ipv6.getNextHeader() == IpProtocol.IPv6_ICMP) {
					ICMPv6 icmPv6= (ICMPv6) ipv6.getPayload();
					if (icmPv6.getICMPv6Type() == ICMPv6.ROUTER_ADVERTSEMENT) {
						System.out.println("IgnoreProvider 55  RA报文被丢弃了");
					}
				}
				return RoutingAction.NONE;
			}
		}
		else if(eth.getEtherType() == EthType.ARP) {
			ARP arp = (ARP)eth.getPayload();
			IPv4Address address = arp.getSenderProtocolAddress();
			if(this.manager.check(switchPort, address)) {
				return RoutingAction.FORWARD_OR_FLOOD;
			}
			else {
				return RoutingAction.NONE;
			}
		}
		return RoutingAction.NONE;
	}
}

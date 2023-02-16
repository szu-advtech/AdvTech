package net.floodlightcontroller.savi;
import java.util.ArrayList;
import java.util.List;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.action.OFAction;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.EthType;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.TableId;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.devicemanager.SwitchPort;
import net.floodlightcontroller.forwarding.Forwarding;
import net.floodlightcontroller.packet.ARP;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.packet.IPv4;
import net.floodlightcontroller.packet.IPv6;
import net.floodlightcontroller.routing.IRoutingDecision.RoutingAction;
public class ReactiveProvider extends Provider {
	{
	}
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
				Match.Builder mb = OFFactories.getFactory(OFVersion.OF_14).buildMatch();
				mb.setExact(MatchField.IN_PORT, switchPort.getPort());
				mb.setExact(MatchField.ETH_TYPE, EthType.IPv4);
				mb.setExact(MatchField.IPV4_SRC, ipv4.getSourceAddress());
				List<OFAction> actions = new ArrayList<>();
				doFlowMod(switchPort.getSwitchDPID(), TableId.of(0), mb.build(), actions, null, Forwarding.FLOWMOD_DEFAULT_PRIORITY+1,0,5);
				return RoutingAction.NONE;
			}
		}
		else if(eth.getEtherType() == EthType.IPv6) {
			IPv6 ipv6 = (IPv6)eth.getPayload();
			IPv6Address address = ipv6.getSourceAddress();
			if(this.manager.check(switchPort, macAddress, address)) {
				return RoutingAction.FORWARD_OR_FLOOD;
			}
			else {
				Match.Builder mb = OFFactories.getFactory(OFVersion.OF_14).buildMatch();
				mb.setExact(MatchField.IN_PORT, switchPort.getPort());
				mb.setExact(MatchField.ETH_TYPE, EthType.IPv6);
				mb.setExact(MatchField.IPV6_SRC, ipv6.getSourceAddress());
				List<OFAction> actions = new ArrayList<>();
				doFlowMod(switchPort.getSwitchDPID(), TableId.of(0), mb.build(), actions, null, Forwarding.FLOWMOD_DEFAULT_PRIORITY+1,0,5);
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
				Match.Builder mb = OFFactories.getFactory(OFVersion.OF_14).buildMatch();
				mb.setExact(MatchField.IN_PORT, switchPort.getPort());
				mb.setExact(MatchField.ETH_TYPE, EthType.ARP);
				mb.setExact(MatchField.ARP_SPA, address);
				List<OFAction> actions = new ArrayList<>();
				doFlowMod(switchPort.getSwitchDPID(), TableId.of(0), mb.build(), actions, null, Forwarding.FLOWMOD_DEFAULT_PRIORITY+1,0,5);
				return RoutingAction.NONE;
			}
		}
		return null;
	}
	@Override
	public void switchAdded(DatapathId switchId) {
		super.switchAdded(switchId);
	}
	@Override
	public void switchRemoved(DatapathId switchId) {
		 super.switchRemoved(switchId);
	}
	@Override
	public void startUp(FloodlightModuleContext context) throws FloodlightModuleException {
		floodlightProvider.addOFMessageListener(OFType.PACKET_IN, this);
		floodlightProvider.addOFMessageListener(OFType.ERROR, this);
		switchService.addOFSwitchListener(this);
		restApiService.addRestletRoutable(new SAVIRestRoute());
		super.startUp(context);
	}
}

package net.floodlightcontroller.savi.module;
import java.util.ArrayList;
import java.util.List;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.types.EthType;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IpProtocol;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.TransportPort;
import net.floodlightcontroller.devicemanager.SwitchPort;
import net.floodlightcontroller.packet.DHCP;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.packet.IPv4;
import net.floodlightcontroller.packet.UDP;
import net.floodlightcontroller.routing.IRoutingDecision.RoutingAction;
import net.floodlightcontroller.savi.action.Action;
import net.floodlightcontroller.savi.action.Action.ActionFactory;
import net.floodlightcontroller.savi.action.ClearIPv4BindingAction;
import net.floodlightcontroller.savi.action.ClearSwitchBindingAction;
import net.floodlightcontroller.savi.binding.Binding;
import net.floodlightcontroller.savi.binding.BindingPool;
import net.floodlightcontroller.savi.binding.BindingStatus;
public class DHCPService extends SAVIBaseService {
	protected BindingPool<IPv4Address> pool;
	@Override
	public void startUpService() {
		pool = new BindingPool<>();
	}
	protected  boolean isDHCP(Ethernet eth) {
		if(eth.getEtherType() == EthType.IPv4){
			IPv4 ipv4 = (IPv4)eth.getPayload();
			if(ipv4.getProtocol().equals(IpProtocol.UDP)){
				UDP udp = (UDP)ipv4.getPayload();
				if(udp.getDestinationPort().getPort() == 68|| udp.getDestinationPort().getPort() == 67) {
					return true;
				}
			}
		}
		return false;
	}
	protected RoutingAction processDHCP(SwitchPort switchPort,Ethernet eth){
		IPv4 ipv4 = (IPv4)eth.getPayload();
		UDP udp = (UDP)ipv4.getPayload();
		DHCP dhcp = (DHCP)udp.getPayload();
		MacAddress mac = eth.getSourceMACAddress();
		if(!pool.isContain(mac)){
			pool.addHardwareBinding(mac, switchPort);
		}
		switch(dhcp.getPacketType()){
		case DHCPDISCOVER:
			return processDiscover(switchPort, eth);
		case DHCPOFFER:
			return processOffer(switchPort, eth);
		case DHCPREQUEST:
			return processRequest(switchPort, eth);
		case DHCPACK:
			return processAck(switchPort, eth);
		case DHCPNAK:
			return processNack(switchPort, eth);
		case DHCPDECLINE:
			return processDecline(switchPort, eth);
		case DHCPRELEASE:
			break;
		default:
			break;
		}
		return null;
	}
	protected RoutingAction processDiscover(SwitchPort switchPort,Ethernet eth){
		List<Action> actions = new ArrayList<>();
		actions.add(Action.ActionFactory.getFloodAction(switchPort.getSwitchDPID(), switchPort.getPort(), eth));
		saviProvider.pushActions(actions);
		if(pool.getSwitchPort(eth.getSourceMACAddress()) == null) {
			pool.addHardwareBinding(eth.getSourceMACAddress(), switchPort);
		}
		return RoutingAction.NONE;
	}
	protected RoutingAction processOffer(SwitchPort switchPort,Ethernet eth){
		List<Action> actions = new ArrayList<>();
		MacAddress dstMac = eth.getDestinationMACAddress();
		MacAddress srcMac = eth.getSourceMACAddress();
		IPv4 ipv4 = (IPv4)eth.getPayload();
		IPv4Address ipv4Address = ipv4.getSourceAddress();
		if(!pool.isContain(ipv4Address)){
			Binding<IPv4Address> binding = new Binding<>();
			binding.setAddress(ipv4Address);
			binding.setMacAddress(srcMac);
			binding.setStatus(BindingStatus.BOUND);
			binding.setSwitchPort(switchPort);
			pool.addBinding(ipv4Address, binding);
			actions.add(ActionFactory.getBindIPv4Action(binding));
		}
		if(pool.getSwitchPort(dstMac) == null) {
			actions.add(ActionFactory.getFloodAction(switchPort, eth));
		}
		else {
			actions.add(ActionFactory.getPacketOutAction(eth, 
													 pool.getSwitchPort(dstMac),
													 OFPort.CONTROLLER));
		}
		saviProvider.pushActions(actions);
		return RoutingAction.NONE;
	}
	protected RoutingAction processRequest(SwitchPort switchPort,Ethernet eth){
		List<Action>  actions = new ArrayList<>();
		IPv4 ipv4 = (IPv4)eth.getPayload();
		UDP udp = (UDP)ipv4.getPayload();
		DHCP dhcp = (DHCP)udp.getPayload();
		MacAddress macAddress = eth.getSourceMACAddress();
		IPv4Address ipv4Address = dhcp.getRequestIP();
		log.info("REQUEST"+ipv4Address.toString());
		if(pool.isContain(ipv4Address)){
			if(pool.check(ipv4Address, macAddress)){
				Binding<IPv4Address> binding = pool.getBinding(ipv4Address);
				if(binding.getStatus() == BindingStatus.BOUND) {
					binding.setStatus(BindingStatus.REBINDING);
					return RoutingAction.FORWARD_OR_FLOOD;
				}
				else {
					actions.add(ActionFactory.getFloodAction(switchPort.getSwitchDPID(), switchPort.getPort(), eth));
					saviProvider.pushActions(actions);
					return RoutingAction.NONE;
				}
			}
			else{
				return RoutingAction.NONE;
			}
		}
		else{
			Binding<IPv4Address> binding = new Binding<>();
			binding.setAddress(ipv4Address);
			binding.setMacAddress(macAddress);
			binding.setStatus(BindingStatus.REQUESTING);
			binding.setAddress(ipv4Address);
			binding.setTransactionId(dhcp.getTransactionId());
			binding.setSwitchPort(switchPort);
			pool.addBinding(ipv4Address, binding);
			actions.add(ActionFactory.getFloodAction(switchPort.getSwitchDPID(), switchPort.getPort(), eth));
			saviProvider.pushActions(actions);
			return RoutingAction.NONE;
		}
	}
	protected RoutingAction  processAck(SwitchPort switchPort,Ethernet eth){
		List<Action> actions = new ArrayList<>();
		IPv4 ipv4 = (IPv4)eth.getPayload();
		UDP udp = (UDP)ipv4.getPayload();
		DHCP dhcp = (DHCP)udp.getPayload();
		MacAddress macAddress = eth.getDestinationMACAddress();
		IPv4Address ipv4Address = dhcp.getYourIPAddress();
		if(pool.isContain(ipv4Address)){
			Binding<IPv4Address> binding = pool.getBinding(ipv4Address);
			if(binding.getStatus() == BindingStatus.REQUESTING){
				binding.setBindingTime();
				binding.setStatus(BindingStatus.BOUND);
				binding.setLeaseTime(dhcp.getSeconds());
				actions.add(ActionFactory.getPacketOutAction(eth, pool.getSwitchPort(macAddress), OFPort.CONTROLLER));
				actions.add(ActionFactory.getBindIPv4Action(binding));			
			}
			else if(binding.getStatus() == BindingStatus.REBINDING){
				binding.setBindingTime();
				binding.setStatus(BindingStatus.BOUND);
				binding.setLeaseTime(dhcp.getSeconds());
				actions.add(ActionFactory.getPacketOutAction(eth, pool.getSwitchPort(macAddress), OFPort.CONTROLLER));
			}
			saviProvider.pushActions(actions);
		}
		return RoutingAction.NONE;
	}
	protected RoutingAction processNack(SwitchPort switchPort,Ethernet eth){
		List<Action> actions = new ArrayList<>();
		IPv4 ipv4 = (IPv4)eth.getPayload();
		UDP udp = (UDP)ipv4.getPayload();
		DHCP dhcp = (DHCP)udp.getPayload();
		MacAddress macAddress = eth.getDestinationMACAddress();
		IPv4Address ipv4Address = dhcp.getYourIPAddress();
		if(pool.isContain(ipv4Address)){
			Binding<IPv4Address> binding = pool.getBinding(ipv4Address);
			actions.add(ActionFactory.getPacketOutAction(eth, pool.getSwitchPort(macAddress), OFPort.CONTROLLER));
			if(binding.getStatus()!=BindingStatus.BOUND){
				pool.delBinding(ipv4Address);
			}
		}
		return RoutingAction.NONE;
	}
	protected RoutingAction processDecline(SwitchPort switchPort,Ethernet eth){
		List<Action> actions = new ArrayList<>();
		IPv4 ipv4 = (IPv4)eth.getPayload();
		UDP udp = (UDP)ipv4.getPayload();
		DHCP dhcp = (DHCP)udp.getPayload();
		MacAddress macAddress = eth.getDestinationMACAddress();
		IPv4Address ipv4Address = dhcp.getYourIPAddress();
		if(pool.isContain(ipv4Address)){
			Binding<IPv4Address> binding = pool.getBinding(ipv4Address);
			if (binding.getStatus() == BindingStatus.BOUND) {
			}
			actions.add(ActionFactory.getPacketOutAction(eth, pool.getSwitchPort(macAddress), OFPort.CONTROLLER));
			pool.delBinding(ipv4Address);
		}
		return RoutingAction.NONE;
	}
	@Override
	protected void doClearIPv4BindingAction(ClearIPv4BindingAction action){
		pool.delBinding(action.getIPv4Address());
	}
	@Override
	protected void doClearSwitchBindingAction(ClearSwitchBindingAction action){
		pool.delSwitch(action.getSwitchId());
	}
	@Override
	public boolean match(Ethernet eth) {
		return isDHCP(eth);
	}
	@Override
	public List<Match> getMatches() {
		List<Match> array = new ArrayList<>();
		Match.Builder mb = OFFactories.getFactory(OFVersion.OF_14).buildMatch();
		mb.setExact(MatchField.ETH_TYPE, EthType.IPv4);
		mb.setExact(MatchField.IP_PROTO, IpProtocol.UDP);
		mb.setExact(MatchField.UDP_DST, TransportPort.of(67));
		mb.setExact(MatchField.UDP_SRC, TransportPort.of(68));
		array.add(mb.build());
		mb = OFFactories.getFactory(OFVersion.OF_14).buildMatch();
		mb.setExact(MatchField.ETH_TYPE, EthType.IPv4);
		mb.setExact(MatchField.IP_PROTO, IpProtocol.UDP);
		mb.setExact(MatchField.UDP_DST, TransportPort.of(68));
		mb.setExact(MatchField.UDP_SRC, TransportPort.of(67));
		array.add(mb.build());
		return array;
	}
	@Override
	public RoutingAction process(SwitchPort switchPort, Ethernet eth) {
		return processDHCP(switchPort, eth);
	}
	@Override
	public void handlePortDown(SwitchPort switchPort) {
	}
	@Override
	public void checkDeadline(){
		List<Action> actions = new ArrayList<>();
		for(Binding<IPv4Address> binding:pool.getAllBindings()){
			if(binding.isLeaseExpired()){
				actions.add(ActionFactory.getUnbindIPv4Action(binding.getAddress(), binding));
				pool.delBinding(binding.getAddress());
			}
		}
		if(actions.size()>0){
			saviProvider.pushActions(actions);
		}
	}
}

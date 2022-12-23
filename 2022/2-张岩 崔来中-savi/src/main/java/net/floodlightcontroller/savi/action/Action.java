package net.floodlightcontroller.savi.action;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFPort;
import net.floodlightcontroller.devicemanager.SwitchPort;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.savi.binding.Binding;
public abstract class Action {
	static Map<ActionType, Class<? extends Action>> actionStore;
	static{
		actionStore = new HashMap<>();
		actionStore.put(ActionType.FLOOD, FloodAction.class);
		actionStore.put(ActionType.BIND_IPv4, BindIPv4Action.class);
		actionStore.put(ActionType.BIND_IPv6, BindIPv6Action.class);
		actionStore.put(ActionType.PACKET_OUT, PacketOutAction.class);
		actionStore.put(ActionType.UNBIND_IPv4, UnbindIPv4Action.class);
		actionStore.put(ActionType.UNBIND_IPv6, UnbindIPv6Action.class);
		actionStore.put(ActionType.CLEAR_IPv4_BINDING, ClearIPv4BindingAction.class);
		actionStore.put(ActionType.CLEAR_IPv6_BINDING, ClearIPv6BindingAction.class);
		actionStore.put(ActionType.CLEAR_PORT_BINDING, ClearPortBindingAction.class);
		actionStore.put(ActionType.CLEAR_SWITCH_BINDING, ClearSwitchBindingAction.class);
	}
	ActionType 	type;
	public ActionType getType() {
		return type;
	}
	public void setType(ActionType type){
		this.type = type;
	}
	public static class ActionFactory {
		public static FloodAction getFloodAction(DatapathId switchId,OFPort inPort, Ethernet eth){
			return new FloodAction(switchId, inPort, eth);
		}
		public static FloodAction getFloodAction(SwitchPort switchPort, Ethernet eth) {
			return new FloodAction(switchPort.getSwitchDPID(),switchPort.getPort(),eth);
		}
		public static BindIPv4Action getBindIPv4Action(Binding<IPv4Address> binding) {
			return new BindIPv4Action(binding);
		}
		public static BindIPv6Action getBindIPv6Action(Binding<IPv6Address> binding) {
			return new BindIPv6Action(binding);
		}
		public static PacketOutAction getPacketOutAction(Ethernet eth, DatapathId switchId, OFPort inPort, List<OFPort> outPorts){
			return new PacketOutAction(eth, switchId, inPort, outPorts);
		}
		public static PacketOutAction getPacketOutAction(Ethernet eth,SwitchPort switchPort, OFPort inPort){
			List<OFPort> ports = new ArrayList<>();
			ports.add(switchPort.getPort());
			return new PacketOutAction(eth,switchPort.getSwitchDPID(),inPort,ports);
		}
		public static UnbindIPv4Action getUnbindIPv4Action(IPv4Address ipv4Address, Binding<IPv4Address> binding){
			return new UnbindIPv4Action(ipv4Address, binding);
		}
		public static UnbindIPv6Action getUnbindIPv6Action(IPv6Address ipv6Address, Binding<IPv6Address> binding) {
			return new UnbindIPv6Action(ipv6Address, binding);
		}
		public static ClearIPv4BindingAction getClearIPv4BindingAction(IPv4Address ipv4Address) {
			return new ClearIPv4BindingAction(ipv4Address);
		}
		public static ClearIPv6BindingAction getClearIPv6BindingAction(IPv6Address ipv6Address) {
			return new ClearIPv6BindingAction(ipv6Address);
		}
		public static ClearSwitchBindingAction getClearSwitchBindingAction(DatapathId switchId) {
			return new ClearSwitchBindingAction(switchId);
		}
		public static ClearPortBindingAction getClearPortBindingAction(List<SwitchPort> switchPorts) {
			return new ClearPortBindingAction(switchPorts);
		}
		public static CheckIPv4BindingAction getCheckIPv4Binding(SwitchPort switchPort, MacAddress macAddress, IPv4Address ipv4Address){
			return new CheckIPv4BindingAction(switchPort, macAddress, ipv4Address);
		}
		public static CheckIPv6BindingAction getCheckIPv6Binding(SwitchPort switchPort, MacAddress macAddress, IPv6Address ipv6Address){
			return new CheckIPv6BindingAction(switchPort, macAddress, ipv6Address);
		}
	}
}

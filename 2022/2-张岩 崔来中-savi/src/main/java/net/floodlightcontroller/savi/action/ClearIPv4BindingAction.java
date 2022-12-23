package net.floodlightcontroller.savi.action;
import org.projectfloodlight.openflow.types.IPv4Address;
public class ClearIPv4BindingAction extends Action {
	IPv4Address ipv4Address;
	public ClearIPv4BindingAction() {
		this.type = ActionType.CLEAR_IPv4_BINDING;
	}
	public ClearIPv4BindingAction(IPv4Address ipv4Address) {
		this.type = ActionType.CLEAR_IPv4_BINDING;
		this.ipv4Address = ipv4Address;
	}
	public IPv4Address getIPv4Address() {
		return ipv4Address;
	}
	public void setIPv4Address(IPv4Address ipv4Address) {
		this.ipv4Address = ipv4Address;
	}
}

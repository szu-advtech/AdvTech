package net.floodlightcontroller.savi.action;
import org.projectfloodlight.openflow.types.IPv6Address;
public class ClearIPv6BindingAction extends Action {
	IPv6Address ipv6Address;
	public ClearIPv6BindingAction() {
		this.type = ActionType.CLEAR_IPv6_BINDING;
	}
	public ClearIPv6BindingAction(IPv6Address ipv6Address) {
		this.type = ActionType.CLEAR_IPv6_BINDING;
		this.ipv6Address = ipv6Address;
	}
	public IPv6Address getIpv6Address() {
		return ipv6Address;
	}
	public void setIpv6Address(IPv6Address ipv6Address) {
		this.ipv6Address = ipv6Address;
	}
}

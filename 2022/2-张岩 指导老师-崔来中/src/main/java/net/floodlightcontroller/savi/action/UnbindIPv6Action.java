package net.floodlightcontroller.savi.action;
import org.projectfloodlight.openflow.types.IPv6Address;
import net.floodlightcontroller.savi.binding.Binding;
public class UnbindIPv6Action extends Action {
	IPv6Address 			ipv6Address;
	Binding<IPv6Address>	binding;
	public UnbindIPv6Action() {
		this.type = ActionType.UNBIND_IPv6;
		this.ipv6Address = null;
		this.binding = null;
	}
	public UnbindIPv6Action(IPv6Address ipv6Address, Binding<IPv6Address> binding) {
		this.type = ActionType.UNBIND_IPv6;
		this.ipv6Address = ipv6Address;
		this.binding = binding;
	}
	public IPv6Address getIPv6Address() {
		return ipv6Address;
	}
	public void setIPv6Address(IPv6Address ipv6Address) {
		this.ipv6Address = ipv6Address;
	}
	public Binding<IPv6Address> getBinding() {
		return binding;
	}
	public void setBinding(Binding<IPv6Address> binding) {
		this.binding = binding;
	}
}

package net.floodlightcontroller.savi.action;
import org.projectfloodlight.openflow.types.IPv4Address;
import net.floodlightcontroller.savi.binding.Binding;
public class UnbindIPv4Action extends Action {
	IPv4Address 		 ipv4Address;
	Binding<IPv4Address> binding;
	public UnbindIPv4Action() {
		this.type = ActionType.UNBIND_IPv4;
		this.ipv4Address = null;
		this.binding = null;
	}
	public UnbindIPv4Action(IPv4Address ipv4Address, Binding<IPv4Address> binding){
		this.type = ActionType.UNBIND_IPv4;
		this.ipv4Address = ipv4Address;
		this.binding = binding;
	}
	public IPv4Address getIpv4Address() {
		return ipv4Address;
	}
	public void setIpv4Address(IPv4Address ipv4Address) {
		this.ipv4Address = ipv4Address;
	}
	public Binding<IPv4Address> getBinding() {
		return binding;
	}
	public void setBinding(Binding<IPv4Address> binding) {
		this.binding = binding;
	}
}

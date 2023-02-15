package net.floodlightcontroller.savi.action;
import org.projectfloodlight.openflow.types.IPv6Address;
import net.floodlightcontroller.savi.binding.Binding;
public class BindIPv6Action extends Action {
	Binding<IPv6Address> binding;
	public BindIPv6Action(){
		this.type = ActionType.BIND_IPv6;
		this.binding = null;
	}
	public BindIPv6Action(Binding<IPv6Address> binding){
		this.type = ActionType.BIND_IPv6;
		this.binding = binding;
	}
	public Binding<IPv6Address> getBinding() {
		return binding;
	}
	public void setBinding(Binding<IPv6Address> binding) {
		this.binding = binding;
	}
}

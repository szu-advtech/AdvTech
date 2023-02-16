package net.floodlightcontroller.savi.action;
import org.projectfloodlight.openflow.types.IPv4Address;
import net.floodlightcontroller.savi.binding.Binding;
public class BindIPv4Action extends Action {
	protected Binding<IPv4Address> binding;
	public BindIPv4Action() {
		this.type = ActionType.BIND_IPv4;
		binding = null;
	}
	public BindIPv4Action(Binding<IPv4Address> binding){
		this.type = ActionType.BIND_IPv4;
		this.binding = binding;
	}
	public Binding<IPv4Address> getBinding() {
		return binding;
	}
	public void setBinding(Binding<IPv4Address> binding) {
		this.binding = binding;
	}
}

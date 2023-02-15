package net.floodlightcontroller.savi.action;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.MacAddress;
import net.floodlightcontroller.devicemanager.SwitchPort;
public class CheckIPv6BindingAction extends Action {
	MacAddress macAddress;
	IPv6Address ipv6Address;
	SwitchPort switchPort;
	public CheckIPv6BindingAction() {
		this.type = ActionType.CHECK_IPv6_BINDING;
		this.macAddress = null;
		this.switchPort = null;
		this.macAddress = null;
	}
	public CheckIPv6BindingAction(SwitchPort switchPort, MacAddress macAddress, IPv6Address ipv6Address) {
		this.type = ActionType.CHECK_IPv6_BINDING;
		this.switchPort = switchPort;
		this.macAddress = macAddress;
		this.ipv6Address = ipv6Address;
	}
	public MacAddress getMacAddress() {
		return macAddress;
	}
	public void setMacAddress(MacAddress macAddress) {
		this.macAddress = macAddress;
	}
	public IPv6Address getIPv6Address() {
		return ipv6Address;
	}
	public void setIPv6Address(IPv6Address ipv6Address) {
		this.ipv6Address = ipv6Address;
	}
	public SwitchPort getSwitchPort() {
		return switchPort;
	}
	public void setSwitchPort(SwitchPort switchPort) {
		this.switchPort = switchPort;
	}
}

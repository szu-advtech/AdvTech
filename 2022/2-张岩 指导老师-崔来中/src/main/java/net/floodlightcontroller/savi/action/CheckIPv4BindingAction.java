package net.floodlightcontroller.savi.action;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.MacAddress;
import net.floodlightcontroller.devicemanager.SwitchPort;
public class CheckIPv4BindingAction extends Action {
	SwitchPort switchPort;
	MacAddress macAddress;
	IPv4Address ipv4Address;
	public CheckIPv4BindingAction() {
		this.type = ActionType.CHECK_IPv4_BINDING;
	}
	public CheckIPv4BindingAction(SwitchPort switchPort, MacAddress macAddress, IPv4Address ipv4Address){
		this.type = ActionType.CHECK_IPv4_BINDING;
		this.macAddress = macAddress;
		this.switchPort = switchPort;
		this.ipv4Address = ipv4Address;
	}
	public SwitchPort getSwitchPort() {
		return switchPort;
	}
	public void setSwitchPort(SwitchPort switchPort) {
		this.switchPort = switchPort;
	}
	public MacAddress getMacAddress() {
		return macAddress;
	}
	public void setMacAddress(MacAddress macAddress) {
		this.macAddress = macAddress;
	}
	public IPv4Address getIPv4Address() {
		return ipv4Address;
	}
	public void setIPv4Address(IPv4Address ipv4Address) {
		this.ipv4Address = ipv4Address;
	}
}

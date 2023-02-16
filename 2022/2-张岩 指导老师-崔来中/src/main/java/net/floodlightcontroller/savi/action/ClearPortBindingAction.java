package net.floodlightcontroller.savi.action;
import java.util.List;
import net.floodlightcontroller.devicemanager.SwitchPort;
public class ClearPortBindingAction extends Action {
	List<SwitchPort> switchPorts;
	public ClearPortBindingAction() {
		this.type = ActionType.CLEAR_PORT_BINDING;
	}
	public ClearPortBindingAction(List<SwitchPort> switchPorts) {
		this.type = ActionType.CLEAR_PORT_BINDING;
		this.switchPorts = switchPorts;
	}
	public List<SwitchPort> getSwitchPorts() {
		return switchPorts;
	}
	public void setSwitchPorts(List<SwitchPort> switchPorts) {
		this.switchPorts = switchPorts;
	}
}

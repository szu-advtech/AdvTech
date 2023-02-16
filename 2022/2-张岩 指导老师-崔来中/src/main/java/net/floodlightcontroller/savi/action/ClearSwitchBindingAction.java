package net.floodlightcontroller.savi.action;
import org.projectfloodlight.openflow.types.DatapathId;
public class ClearSwitchBindingAction extends Action {
	DatapathId switchId;
	public ClearSwitchBindingAction() {
		this.type = ActionType.CLEAR_SWITCH_BINDING;
	}
	public ClearSwitchBindingAction(DatapathId switchId){
		this.type = ActionType.CLEAR_SWITCH_BINDING;
		this.switchId = switchId;
	}
	public DatapathId getSwitchId() {
		return switchId;
	}
	public void setSwitchId(DatapathId switchId) {
		this.switchId = switchId;
	}
}

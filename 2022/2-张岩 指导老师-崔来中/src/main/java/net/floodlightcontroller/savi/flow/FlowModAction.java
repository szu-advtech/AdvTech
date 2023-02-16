package net.floodlightcontroller.savi.flow;
import java.util.List;
import org.projectfloodlight.openflow.protocol.action.OFAction;
import org.projectfloodlight.openflow.protocol.instruction.OFInstruction;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.TableId;
public class FlowModAction extends FlowAction{
	DatapathId switchId;
	TableId tableId;
	Match match;
	List<OFAction> actions;
	List<OFInstruction> instructions;
	int priority;
	int hardTimeout;
	int idleTimeout;
	public FlowModAction(){
		this.type=FlowActionType.MOD;
	}
	public FlowModAction(DatapathId switchId,TableId tableId,
			Match match,
			List<OFAction> actions,
			List<OFInstruction> instructions,
			int priority,int hardTimeout,int idleTimeout){
		this.type=FlowActionType.MOD;
		this.switchId=switchId;
		this.tableId=tableId;
		this.match=match;
		this.actions=actions;
		this.instructions=instructions;
		this.priority=priority;
		this.hardTimeout=hardTimeout;
		this.idleTimeout=idleTimeout;
	}
	public DatapathId getSwitchId() {
		return switchId;
	}
	public void setSwitchId(DatapathId switchId) {
		this.switchId = switchId;
	}
	public TableId getTableId() {
		return tableId;
	}
	public void setTableId(TableId tableId) {
		this.tableId = tableId;
	}
	public Match getMatch() {
		return match;
	}
	public void setMatch(Match match) {
		this.match = match;
	}
	public List<OFAction> getActions() {
		return actions;
	}
	public void setActions(List<OFAction> actions) {
		this.actions = actions;
	}
	public List<OFInstruction> getInstructions() {
		return instructions;
	}
	public void setInstructions(List<OFInstruction> instructions) {
		this.instructions = instructions;
	}
	public int getPriority() {
		return priority;
	}
	public void setPriority(int priority) {
		this.priority = priority;
	}
	public int getHardTimeout() {
		return hardTimeout;
	}
	public void setHardTimeout(int hardTimeout) {
		this.hardTimeout = hardTimeout;
	}
	public int getIdleTimeout() {
		return idleTimeout;
	}
	public void setIdleTimeout(int idleTimeout) {
		this.idleTimeout = idleTimeout;
	}
}

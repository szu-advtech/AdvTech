package net.floodlightcontroller.savi.flow;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.TableId;
public class FlowRemoveAction extends FlowAction{
	DatapathId switchId;
	TableId tableId;
	Match match;
	public FlowRemoveAction(){
		this.type=FlowActionType.REMOVE;
		this.switchId=null;
		this.tableId=null;
		this.match=null;
	}
	public FlowRemoveAction(DatapathId switchId,TableId tableId,Match match){
		this.type=FlowActionType.REMOVE;
		this.switchId=switchId;
		this.tableId=tableId;
		this.match=match;
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
}

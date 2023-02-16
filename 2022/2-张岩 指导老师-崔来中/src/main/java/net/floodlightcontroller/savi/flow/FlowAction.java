package net.floodlightcontroller.savi.flow;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.TableId;
import org.projectfloodlight.openflow.protocol.action.OFAction;
import org.projectfloodlight.openflow.protocol.instruction.OFInstruction;
public abstract class FlowAction {
	static Map<FlowActionType, Class<? extends FlowAction>> actionStore;
	static{
		actionStore = new HashMap<>();
		actionStore.put(FlowActionType.ADD, FlowAddAction.class);
		actionStore.put(FlowActionType.REMOVE, FlowRemoveAction.class);
		actionStore.put(FlowActionType.MOD, FlowModAction.class);
	}
	FlowActionType 	type;
	public FlowActionType getType() {
		return type;
	}
	public void setType(FlowActionType type){
		this.type = type;
	}
	public static class FlowActionFactory {
			return new FloodAction(switchId, inPort, eth);
		public static FlowAddAction getFlowAddAction(DatapathId switchId,TableId tableId,
				Match match,
				List<OFAction> actions,
				List<OFInstruction> instructions,
				int priority){
			return new FlowAddAction(switchId,tableId,match,actions,instructions,priority);
		}
		public static FlowModAction getFlowModAction(DatapathId switchId,TableId tableId,
				Match match,
				List<OFAction> actions,
				List<OFInstruction> instructions,
				int priority,int hardTimeout,int idleTimeout){
			return new FlowModAction(switchId,tableId,match,actions,instructions,priority,hardTimeout,idleTimeout);
		}
		public static FlowRemoveAction getFlowRemoveAction(DatapathId switchId,TableId tableId,Match match){
			return new FlowRemoveAction(switchId,tableId,match);
		}
	}
}

package net.floodlightcontroller.savi.rest;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.action.OFAction;
import org.projectfloodlight.openflow.protocol.instruction.OFInstruction;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.EthType;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.TableId;
import org.restlet.resource.Post;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.devicemanager.SwitchPort;
import net.floodlightcontroller.savi.flow.FlowAction;
import net.floodlightcontroller.savi.flow.FlowAction.FlowActionFactory;
import net.floodlightcontroller.savi.service.SAVIProviderService;
public class FlowResource extends ServerResource {
	protected static Logger log = LoggerFactory.getLogger(SAVIRest.class);
	public static final String FLOW_ADD="add";
	public static final String FLOW_REMOVE="remove";
	public static final String FLOW_MOD="mod";
	public static final String FLOW_TYPE="type";
	public static final String FLOW_ADD_2 = "add2";
	public static final String FLOW_MOD_2 = "mod2";
	public static final String SWITCHID="swid";
	public static final String TABLEID="tid";
	public static final String INPORT="in_port";
	public static final String IPV6_SRC="src_ipv6";
	public static final String ETH_SRC="src_mac";
	public static final String ETH_TYPE="eth_type";
	public static final String PRIORITY="priority";
	public static final String IDLE_TIMEOUT="idle_timeout";
	public static final String HARD_TIMEOUT="hard_timeout";
	public static final int BIND_PRIORITY = 5;
	public static final int BIND_TABLE_ID = 1;
	public static final int OTHER_TABLE_ID = 2;
	public static final String ACTION="action";
	public static final String OUTPUT="output";
	public static final String INSTRUCTION="instruction";
	public static final String RESUBMIT="resubmit";
	SAVIProviderService saviProvider=null;
	@Post
	public String post(String json){
		List<FlowAction> actions=new ArrayList<>();
		saviProvider=(SAVIProviderService)getContext().getAttributes()
		.get(SAVIProviderService.class.getCanonicalName());
		Map<String, String> jsonMap=SaviUtils.jsonToStringMap(json);
		for(Map.Entry<String, String> entry:jsonMap.entrySet()){
			Map<String,String> map=SaviUtils.splitToStringMap(entry.getValue());
			switch(entry.getKey()){
			case FLOW_ADD:
				doFlowAdd(map, actions);
				break;
			case FLOW_REMOVE:
				doFlowRemove(map, actions);
				break;
			case FLOW_MOD:
				doFlowMod(map, actions);
				break;
			case FLOW_ADD_2:
				doFlowAdd2(map, actions);
				break;
			case FLOW_MOD_2:
				doFlowMod2(map, actions);
				break;
			default:
				break;
			}
		}
		saviProvider.pushFlowActions(actions);
		return "{Success}";
	}
	public void doFlowAdd(Map<String, String> map,List<FlowAction> actions){
		if(map==null||map.size()==0) return;
		String swid=map.get(SWITCHID);
		if(swid==null||swid.isEmpty()) return;
		DatapathId dpId=DatapathId.of(1);
		OFPort port=OFPort.of(1);
		TableId tid=TableId.of(1);
		int priority=0;
		Match.Builder mb=OFFactories.getFactory(OFVersion.OF_13).buildMatch();
		List<OFInstruction> instructions=null;
		List<OFAction> ofActions=null;
		try{
			for(String key:map.keySet()){
				if(key.equals(SWITCHID)){
					dpId = DatapathId.of(swid);
				}
				else if(key.equals(TABLEID)){
					tid=TableId.of(Integer.parseInt(map.get(key)));
				}
				else if(key.equals(INPORT)){
					mb.setExact(MatchField.IN_PORT, OFPort.of(Integer.parseInt(map.get(key))));
					port=OFPort.of(Integer.parseInt(map.get(key)));
				}
				else if(key.equals(IPV6_SRC)){
					mb.setExact(MatchField.IPV6_SRC, IPv6Address.of(map.get(key)));
				}
				else if(key.equals(ETH_SRC)){
					mb.setExact(MatchField.ETH_SRC, MacAddress.of(map.get(key)));
				}
				else if(key.equals(PRIORITY)){
					priority=Integer.parseInt(map.get(key));
				}
				else if(key.equals(ACTION)){
					String[] splits=map.get(key).split(":");
					if(splits.length<2||!splits[0].equals(OUTPUT))
						continue;
					ofActions=new ArrayList<>();
					if(splits[1].equals("0")){
						ofActions.add(OFFactories.getFactory(OFVersion.OF_13).actions().output(OFPort.CONTROLLER, Integer.MAX_VALUE));
					}
					else{
						ofActions.add(OFFactories.getFactory(OFVersion.OF_13).actions().output(OFPort.of(Integer.parseInt(splits[1])), Integer.MAX_VALUE));
					}
				}
				else if(key.equals(INSTRUCTION)){
					String[] splits=map.get(key).split(":");
					if(splits.length<2||!splits[0].equals(RESUBMIT))
						continue;
					instructions=new ArrayList<>();
					instructions.add(OFFactories.getFactory(OFVersion.OF_13).instructions().gotoTable(TableId.of(Integer.parseInt(splits[1]))));
				}
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		if(dpId.getLong()!=1&&port.getPortNumber()!=1) {
			saviProvider.getPushFlowToSwitchPorts().add(new SwitchPort(dpId, port));
		}
		try {
			Thread.sleep(1000);
			mb.setExact(MatchField.ETH_TYPE, EthType.IPv6);
			FlowAction action=FlowActionFactory.getFlowAddAction(
					dpId, tid, mb.build(), ofActions, instructions, priority);
			actions.add(action);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	public void doFlowAdd2(Map<String, String> map,List<FlowAction> actions){
		if(map==null||map.size()==0) return;
		String swid=map.get(SWITCHID);
		if(swid==null||swid.isEmpty()) return;
		String inport = map.get(INPORT);
		if(inport == null || inport.isEmpty()) return;
		DatapathId dpId=DatapathId.of(1);
		OFPort port=OFPort.of(1);
		TableId tid = TableId.of(1);
		int priority = BIND_PRIORITY;
		Match.Builder mb = OFFactories.getFactory(OFVersion.OF_13).buildMatch();
		Match.Builder mb2 = OFFactories.getFactory(OFVersion.OF_13).buildMatch();
		List<OFInstruction> instructions = null;
		try{
			for(String key:map.keySet()){
				if(key.equals(SWITCHID)){
					dpId=DatapathId.of(swid);
				}
				else if(key.equals(INPORT)){
					mb.setExact(MatchField.IN_PORT, OFPort.of(Integer.parseInt(map.get(key))));
					mb2.setExact(MatchField.IN_PORT, OFPort.of(Integer.parseInt(map.get(key))));
					port=OFPort.of(Integer.parseInt(map.get(key)));
				}
				else if(key.equals(IPV6_SRC)){
					mb.setExact(MatchField.IPV6_SRC, IPv6Address.of(map.get(key)));
				}
				else if(key.equals(ETH_SRC)){
					mb.setExact(MatchField.ETH_SRC, MacAddress.of(map.get(key)));
				}
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		mb.setExact(MatchField.ETH_TYPE, EthType.IPv6);
		if(dpId.getLong()!=1&&port.getPortNumber()!=1) {
			saviProvider.getPushFlowToSwitchPorts().add(new SwitchPort(dpId, port));
		}
		try {
			Thread.sleep(1000);
			instructions=new ArrayList<>();
			instructions.add(OFFactories.getFactory(OFVersion.OF_13).instructions().gotoTable(TableId.of(2)));
			FlowAction action = FlowActionFactory.getFlowAddAction(
					dpId, tid, mb.build(), null, instructions, priority);
			FlowAction anyAction = FlowActionFactory.getFlowAddAction(
					dpId, tid, mb2.build(), null, null, priority - 1);
			actions.add(action);
			actions.add(anyAction);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	public void doFlowMod2(Map<String, String> map,List<FlowAction> actions){
		if(map==null||map.size()==0) return;
		String swid=map.get(SWITCHID);
		if(swid==null||swid.isEmpty()) return;
		String inport = map.get(INPORT);
		if(inport == null || inport.isEmpty()) return;
		DatapathId dpId=DatapathId.of(1);
		TableId tid = TableId.of(1);
		int priority = BIND_PRIORITY;
		Match.Builder mb = OFFactories.getFactory(OFVersion.OF_13).buildMatch();
		Match.Builder mb2 = OFFactories.getFactory(OFVersion.OF_13).buildMatch();
		List<OFInstruction> instructions = null;
		int idleTimeout = 0;
		int hardTimeout = 0;
		try{
			for(String key:map.keySet()){
				if(key.equals(SWITCHID)){
					dpId=DatapathId.of(swid);
				}
				else if(key.equals(INPORT)){
					mb.setExact(MatchField.IN_PORT, OFPort.of(Integer.parseInt(map.get(key))));
					mb2.setExact(MatchField.IN_PORT, OFPort.of(Integer.parseInt(map.get(key))));
				}
				else if(key.equals(IPV6_SRC)){
					mb.setExact(MatchField.IPV6_SRC, IPv6Address.of(map.get(key)));
				}
				else if(key.equals(ETH_SRC)){
					mb.setExact(MatchField.ETH_SRC, MacAddress.of(map.get(key)));
				}
				else if(key.equals(IDLE_TIMEOUT)){
					idleTimeout=Integer.parseInt(map.get(key));
				}
				else if(key.equals(HARD_TIMEOUT)){
					hardTimeout=Integer.parseInt(map.get(key));
				}
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		mb.setExact(MatchField.ETH_TYPE, EthType.IPv6);
		instructions=new ArrayList<>();
		instructions.add(OFFactories.getFactory(OFVersion.OF_13).instructions().gotoTable(TableId.of(2)));
		FlowAction action = FlowActionFactory.getFlowModAction(
				dpId, tid, mb.build(), null, instructions, priority, hardTimeout, idleTimeout);
		FlowAction anyAction = FlowActionFactory.getFlowModAction(
				dpId, tid, mb2.build(), null, null, priority - 1, hardTimeout, idleTimeout);
		actions.add(action);
		actions.add(anyAction);
	}
	public void doFlowMod(Map<String, String> map,List<FlowAction> actions){
		if(map==null||map.size()==0) return;
		String swid=map.get(SWITCHID);
		if(swid==null||swid.isEmpty()) return;
		DatapathId dpId=DatapathId.of(1);
		TableId tid=TableId.of(1);
		int priority=0;
		Match.Builder mb = OFFactories.getFactory(OFVersion.OF_13).buildMatch();
		List<OFInstruction> instructions=null;
		List<OFAction> ofActions=null;
		int idleTimeout=0;
		int hardTimeout=0;
		try{
			for(String key:map.keySet()){
				if(key.equals(SWITCHID)){
					dpId=DatapathId.of(swid);
				}
				else if(key.equals(TABLEID)){
					tid=TableId.of(Integer.parseInt(map.get(key)));
				}
				else if(key.equals(INPORT)){
					mb.setExact(MatchField.IN_PORT, OFPort.of(Integer.parseInt(map.get(key))));
				}
				else if(key.equals(IPV6_SRC)){
					mb.setExact(MatchField.IPV6_SRC, IPv6Address.of(map.get(key)));
				}
				else if(key.equals(ETH_SRC)){
					mb.setExact(MatchField.ETH_SRC, MacAddress.of(map.get(key)));
				}
				else if(key.equals(PRIORITY)){
					priority=Integer.parseInt(map.get(key));
				}
				else if(key.equals(ACTION)){
					String[] splits=map.get(key).split(":");
					if(splits.length<2||!splits[0].equals(OUTPUT))
						continue;
					ofActions=new ArrayList<>();
					if(splits[1].equals("0")){
						ofActions.add(OFFactories.getFactory(OFVersion.OF_13).actions().output(OFPort.CONTROLLER, Integer.MAX_VALUE));
					}
					else{
						ofActions.add(OFFactories.getFactory(OFVersion.OF_13).actions().output(OFPort.of(Integer.parseInt(splits[1])), Integer.MAX_VALUE));
					}
				}
				else if(key.equals(INSTRUCTION)){
					String[] splits=map.get(key).split(":");
					if(splits.length<2||!splits[0].equals(RESUBMIT))
						continue;
					instructions=new ArrayList<>();
					instructions.add(OFFactories.getFactory(OFVersion.OF_13).instructions().gotoTable(TableId.of(Integer.parseInt(splits[1]))));
				}
				else if(key.equals(IDLE_TIMEOUT)){
					idleTimeout=Integer.parseInt(map.get(key));
				}
				else if(key.equals(HARD_TIMEOUT)){
					hardTimeout=Integer.parseInt(map.get(key));
				}
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		mb.setExact(MatchField.ETH_TYPE, EthType.IPv6);
		FlowAction action=FlowActionFactory.getFlowModAction(
				dpId, tid, mb.build(), ofActions, instructions, priority, hardTimeout, idleTimeout);
		actions.add(action);
	}
	public void doFlowRemove(Map<String, String> map,List<FlowAction> actions){
		if(map==null||map.size()==0) return;
		String swid=map.get(SWITCHID);
		if(swid==null||swid.isEmpty()) return;
		DatapathId datapathId=DatapathId.of(1);
		TableId tid=TableId.of(1);
		Match.Builder mb=OFFactories.getFactory(OFVersion.OF_13).buildMatch();
		OFPort port=OFPort.of(1);
		try{
			for(String key:map.keySet()){
				if(key.equals(SWITCHID)){
					datapathId=DatapathId.of(swid);
				}
				else if(key.equals(TABLEID)){
					tid=TableId.of(Integer.parseInt(map.get(key)));
				}
				else if(key.equals(INPORT)){
					mb.setExact(MatchField.IN_PORT, OFPort.of(Integer.parseInt(map.get(key))));
				}
				else if(key.equals(IPV6_SRC)){
					mb.setExact(MatchField.IPV6_SRC, IPv6Address.of(map.get(key)));
				}
				else if(key.equals(ETH_SRC)){
					mb.setExact(MatchField.ETH_SRC, MacAddress.of(map.get(key)));
				}
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		if(saviProvider.getPushFlowToSwitchPorts().contains(new SwitchPort(datapathId, port))) {
			log.info("交换机端口：  "+datapathId+"--"+port+"  不存在手动下发的验证规则，删除动作无效");
			return ;
		}
		mb.setExact(MatchField.ETH_TYPE, EthType.IPv6);
		FlowAction action=FlowActionFactory.getFlowRemoveAction(
				datapathId, tid,mb.build());
		actions.add(action);
	}
}

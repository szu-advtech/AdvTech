package net.floodlightcontroller.savi.analysis;
import java.util.Map;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.U64;
import net.floodlightcontroller.core.module.IFloodlightService;
public interface IAnalysisService extends IFloodlightService {
	public void changeStatusByRest(int flag);
	public void enableAnalysis(boolean flag);
	public void changePlanByRest(int flag);
	public U64 getInPacketsNum(DatapathId dpid,OFPort p);
	public U64 getOutPacketsNum(DatapathId dpid,OFPort p);
	public PacketOfFlow getPacketOfFlow(DatapathId dpid,OFPort p);
	public Object getAllPacketOfFlow();
	public Object getPortSet();
	public Object showOutFlow();
	public void updateOutFlow();
	public Object showMaxTraffic();
	public void updateMaxTraffic();
	public void setPriorityLevel(int priorityLevel);
	public Object getHostsCredit();
	public void setAutoCheck(boolean setAuto);
	public int calculateRule(DatapathId dpid);
	public Map<DatapathId, Integer> calculateRule();
	public String getFilePath();
}

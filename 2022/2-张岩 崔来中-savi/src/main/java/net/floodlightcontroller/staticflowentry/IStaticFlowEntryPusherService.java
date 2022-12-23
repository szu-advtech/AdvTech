package net.floodlightcontroller.staticflowentry;
import java.util.Map;
import org.projectfloodlight.openflow.protocol.OFFlowMod;
import org.projectfloodlight.openflow.types.DatapathId;
import net.floodlightcontroller.core.module.IFloodlightService;
public interface IStaticFlowEntryPusherService extends IFloodlightService {
    public void addFlow(String name, OFFlowMod fm, DatapathId swDpid);
    public void deleteFlow(String name);
    public void deleteFlowsForSwitch(DatapathId dpid);
    public void deleteAllFlows();
    public Map<String, Map<String, OFFlowMod>> getFlows();
    public Map<String, OFFlowMod> getFlows(DatapathId dpid);
}

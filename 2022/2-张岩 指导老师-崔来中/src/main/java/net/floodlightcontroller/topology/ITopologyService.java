package net.floodlightcontroller.topology;
import java.util.Date;
import java.util.Map;
import java.util.Set;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.OFPort;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.routing.Link;
public interface ITopologyService extends IFloodlightService  {
	public void addListener(ITopologyListener listener);
	public Date getLastUpdateTime();
	public boolean isAttachmentPointPort(DatapathId switchid, OFPort port);
	public boolean isAttachmentPointPort(DatapathId switchid, OFPort port, boolean tunnelEnabled);
   	public boolean isEdge(DatapathId sw, OFPort p);
	public Set<OFPort> getSwitchBroadcastPorts(DatapathId sw);
	public boolean isBroadcastDomainPort(DatapathId sw, OFPort port);
	public boolean isBroadcastDomainPort(DatapathId sw, OFPort port, boolean tunnelEnabled);
	public boolean isAllowed(DatapathId sw, OFPort portId);
	public boolean isAllowed(DatapathId sw, OFPort portId, boolean tunnelEnabled);
	public boolean isConsistent(DatapathId oldSw, OFPort oldPort, 
			DatapathId newSw, OFPort newPort);
	public boolean isConsistent(DatapathId oldSw, OFPort oldPort,
			DatapathId newSw, OFPort newPort, boolean tunnelEnabled);
	public boolean isInSameBroadcastDomain(DatapathId s1, OFPort p1,
			DatapathId s2, OFPort p2);
	public boolean isInSameBroadcastDomain(DatapathId s1, OFPort p1,
			DatapathId s2, OFPort p2,
			boolean tunnelEnabled);
	public Set<OFPort> getBroadcastPorts(DatapathId targetSw, DatapathId src, OFPort srcPort);
	public Set<OFPort> getBroadcastPorts(DatapathId targetSw, DatapathId src, OFPort srcPort, boolean tunnelEnabled);
	public boolean isIncomingBroadcastAllowed(DatapathId sw, OFPort portId);
	public boolean isIncomingBroadcastAllowed(DatapathId sw, OFPort portId, boolean tunnelEnabled);
	public NodePortTuple getAllowedIncomingBroadcastPort(DatapathId src, OFPort srcPort);
	public NodePortTuple getAllowedIncomingBroadcastPort(DatapathId src, OFPort srcPort, boolean tunnelEnabled);
	public Set<NodePortTuple> getBroadcastDomainPorts();
	public Set<NodePortTuple> getTunnelPorts();
	public Set<NodePortTuple> getBlockedPorts();
	public Set<OFPort> getPorts(DatapathId sw);
	public DatapathId getOpenflowDomainId(DatapathId switchId);
	public DatapathId getOpenflowDomainId(DatapathId switchId, boolean tunnelEnabled);
	public boolean inSameOpenflowDomain(DatapathId switch1, DatapathId switch2);
	public boolean inSameOpenflowDomain(DatapathId switch1, DatapathId switch2, boolean tunnelEnabled);
	public Set<DatapathId> getSwitchesInOpenflowDomain(DatapathId switchDPID);
	public Set<DatapathId> getSwitchesInOpenflowDomain(DatapathId switchDPID, boolean tunnelEnabled);
	public Map<DatapathId, Set<Link>> getAllLinks();
	public Set<OFPort> getPortsWithLinks(DatapathId sw);
	public Set<OFPort> getPortsWithLinks(DatapathId sw, boolean tunnelEnabled);
	public NodePortTuple getOutgoingSwitchPort(DatapathId src, OFPort srcPort, DatapathId dst, OFPort dstPort);
	public NodePortTuple getOutgoingSwitchPort(DatapathId src, OFPort srcPort,
			DatapathId dst, OFPort dstPort, boolean tunnelEnabled);
	public NodePortTuple getIncomingSwitchPort(DatapathId src, OFPort srcPort, DatapathId dst, OFPort dstPort);
	public NodePortTuple getIncomingSwitchPort(DatapathId src, OFPort srcPort,
			DatapathId dst, OFPort dstPort, boolean tunnelEnabled);
	public NodePortTuple getAllowedOutgoingBroadcastPort(DatapathId src, OFPort srcPort, DatapathId dst, OFPort dstPort);
	public NodePortTuple getAllowedOutgoingBroadcastPort(DatapathId src, OFPort srcPort, DatapathId dst, OFPort dstPort, boolean tunnelEnabled);
}
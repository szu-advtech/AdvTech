package net.floodlightcontroller.linkdiscovery;
import java.util.Map;
import java.util.Set;
import org.projectfloodlight.openflow.protocol.OFPacketOut;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFPort;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.linkdiscovery.internal.LinkInfo;
import net.floodlightcontroller.routing.Link;
import net.floodlightcontroller.topology.NodePortTuple;
public interface ILinkDiscoveryService extends IFloodlightService {
    public boolean isTunnelPort(DatapathId sw, OFPort port);
    public Map<Link, LinkInfo> getLinks();
    public LinkInfo getLinkInfo(Link link);
    public ILinkDiscovery.LinkType getLinkType(Link lt, LinkInfo info);
    public OFPacketOut generateLLDPMessage(IOFSwitch iofSwitch, OFPort port,
                                           boolean isStandard,
                                           boolean isReverse);
    public Map<DatapathId, Set<Link>> getSwitchLinks();
    public void addListener(ILinkDiscoveryListener listener);
    public Set<NodePortTuple> getSuppressLLDPsInfo();
    public void AddToSuppressLLDPs(DatapathId sw, OFPort port);
    public void RemoveFromSuppressLLDPs(DatapathId sw, OFPort port);
    public Set<OFPort> getQuarantinedPorts(DatapathId sw);
    public boolean isAutoPortFastFeature();
    public void setAutoPortFastFeature(boolean autoPortFastFeature);
    public Map<NodePortTuple, Set<Link>> getPortLinks();
    public void addMACToIgnoreList(MacAddress mac, int ignoreBits);
}

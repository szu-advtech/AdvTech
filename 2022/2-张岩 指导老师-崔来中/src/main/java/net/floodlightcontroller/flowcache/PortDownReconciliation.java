package net.floodlightcontroller.flowcache;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.protocol.OFFlowDelete;
import org.projectfloodlight.openflow.protocol.OFFlowStatsEntry;
import org.projectfloodlight.openflow.protocol.OFType;
import org.projectfloodlight.openflow.protocol.OFFlowStatsReply;
import org.projectfloodlight.openflow.protocol.OFFlowStatsRequest;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.TableId;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.internal.IOFSwitchService;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.flowcache.IFlowReconcileListener;
import net.floodlightcontroller.flowcache.IFlowReconcileService;
import net.floodlightcontroller.flowcache.OFMatchReconcile;
import net.floodlightcontroller.flowcache.PriorityPendingQueue.EventPriority;
import net.floodlightcontroller.linkdiscovery.ILinkDiscovery;
import net.floodlightcontroller.linkdiscovery.ILinkDiscovery.LDUpdate;
import net.floodlightcontroller.linkdiscovery.internal.LinkInfo;
import net.floodlightcontroller.linkdiscovery.ILinkDiscoveryService;
import net.floodlightcontroller.routing.Link;
import net.floodlightcontroller.topology.ITopologyListener;
import net.floodlightcontroller.topology.ITopologyService;
import net.floodlightcontroller.util.OFMatchWithSwDpid;
@Deprecated
public class PortDownReconciliation implements IFloodlightModule,
    ITopologyListener, IFlowReconcileListener {
    protected static Logger log = LoggerFactory.getLogger(PortDownReconciliation.class);
    protected ITopologyService topology;
    protected IOFSwitchService switchService;
    protected IFlowReconcileService frm;
    protected ILinkDiscoveryService lds;
    protected Map<Link, LinkInfo> links;
    protected FloodlightContext cntx;
    protected static boolean waiting = false;
    protected int statsQueryXId;
    protected static List<OFFlowStatsReply> statsReply;
    @Override
    public void topologyChanged(List<LDUpdate> appliedUpdates) {
        for (LDUpdate ldu : appliedUpdates) {
            if (ldu.getOperation()
                   .equals(ILinkDiscovery.UpdateOperation.PORT_DOWN)) {
                IOFSwitch affectedSwitch = switchService.getSwitch(ldu.getSrc());
                OFMatchReconcile ofmr = new OFMatchReconcile();
                OFMatchWithSwDpid ofmatchsw = new OFMatchWithSwDpid(match, affectedSwitch.getId());
                ofmr.rcAction = OFMatchReconcile.ReconcileAction.UPDATE_PATH;
                ofmr.ofmWithSwDpid = ofmatchsw;
                ofmr.outPort = ldu.getSrcPort();
                frm.reconcileFlow(ofmr, EventPriority.HIGH);
            }
        }
    }
    @Override
    public Collection<Class<? extends IFloodlightService>>
            getModuleServices() {
        return null;
    }
    @Override
    public Map<Class<? extends IFloodlightService>, IFloodlightService>
            getServiceImpls() {
        return null;
    }
    @Override
    public Collection<Class<? extends IFloodlightService>>
            getModuleDependencies() {
        Collection<Class<? extends IFloodlightService>> l = new ArrayList<Class<? extends IFloodlightService>>();
        l.add(IFloodlightProviderService.class);
        l.add(ITopologyService.class);
        l.add(IFlowReconcileService.class);
        l.add(ILinkDiscoveryService.class);
        return l;
    }
    @Override
    public
            void
            init(FloodlightModuleContext context)
                                                 throws FloodlightModuleException {
        switchService = context.getServiceImpl(IOFSwitchService.class);
        topology = context.getServiceImpl(ITopologyService.class);
        frm = context.getServiceImpl(IFlowReconcileService.class);
        lds = context.getServiceImpl(ILinkDiscoveryService.class);
        cntx = new FloodlightContext();
    }
    @Override
    public void startUp(FloodlightModuleContext context) {
        topology.addListener(this);
        frm.addFlowReconcileListener(this);
    }
    @Override
    public String getName() {
        return "portdownreconciliation";
    }
    @Override
    public boolean isCallbackOrderingPrereq(OFType type, String name) {
        return false;
    }
    @Override
    public boolean isCallbackOrderingPostreq(OFType type, String name) {
        return true;
    }
    @Override
    public net.floodlightcontroller.core.IListener.Command reconcileFlows(ArrayList<OFMatchReconcile> ofmRcList) {
        if (lds != null) {
            links = new HashMap<Link, LinkInfo>();
            if (lds.getLinks() != null) links.putAll(lds.getLinks());
            for (OFMatchReconcile ofmr : ofmRcList) {
                if (ofmr.rcAction.equals(OFMatchReconcile.ReconcileAction.UPDATE_PATH)) {
                    IOFSwitch sw = switchService.getSwitch(ofmr.ofmWithSwDpid.getDpid());
                    Map<OFPort, List<Match>> invalidBaseIngressAndMatches = new HashMap<OFPort, List<Match>>();
                    List<OFFlowStatsReply> flows = getFlows(sw, ofmr.outPort);
                    for (OFFlowStatsReply flow : flows) {
                    	for (OFFlowStatsEntry entry : flow.getEntries()) {
                    		Match match = entry.getMatch();
                    		if (invalidBaseIngressAndMatches.containsKey(match.get(MatchField.IN_PORT)))
                    			invalidBaseIngressAndMatches.get(match.get(MatchField.IN_PORT))
                    			.add(match);
                    		else {
                    			List<Match> matches = new ArrayList<Match>();
                    			matches.add(match);
                    			invalidBaseIngressAndMatches.put(match.get(MatchField.IN_PORT), matches);
                    		}
                    	}
                    }
                    if (!flows.isEmpty()) {
                        log.debug("Removing flows on switch : " + sw.getId()
                                  + " with outport: " + ofmr.outPort);
                        clearFlowMods(sw, ofmr.outPort);
                    }
                    Map<IOFSwitch, Map<OFPort, List<Match>>> neighborSwitches = new HashMap<IOFSwitch, Map<OFPort, List<Match>>>();
                    for (Link link : links.keySet()) {
                        if (link.getDst() == sw.getId()) {
                            for (Entry<OFPort, List<Match>> invalidBaseIngressAndMatch : invalidBaseIngressAndMatches.entrySet()) {
                                if (link.getDstPort() == invalidBaseIngressAndMatch.getKey()) {
                                    Map<OFPort, List<Match>> invalidNeighborOutportAndMatch = new HashMap<OFPort, List<Match>>();
                                    invalidNeighborOutportAndMatch.put(link.getSrcPort(),
                                                                       invalidBaseIngressAndMatch.getValue());
                                    neighborSwitches.put(switchService.getSwitch(link.getSrc()), invalidNeighborOutportAndMatch);
                                }
                            }
                        }
                    }
                    log.debug("We have " + neighborSwitches.size()
                              + " neighboring switches to deal with!");
                    for (IOFSwitch neighborSwitch : neighborSwitches.keySet()) {
                        log.debug("NeighborSwitch ID : " + neighborSwitch.getId());
                        if (neighborSwitches.get(neighborSwitch) != null)
                             deleteInvalidFlows(neighborSwitch, neighborSwitches.get(neighborSwitch));
                    }
                }
                return Command.CONTINUE;
            }
        } else {
            log.error("Link Discovery Service Is Null");
        }
        return Command.CONTINUE;
    }
    public List<OFFlowStatsReply> getFlows(IOFSwitch sw, OFPort outPort) {
        statsReply = new ArrayList<OFFlowStatsReply>();
        List<OFFlowStatsReply> values = null;
        Future<List<OFFlowStatsReply>> future;
        OFFlowStatsRequest req = sw.getOFFactory().buildFlowStatsRequest()
        		.setMatch(sw.getOFFactory().buildMatch().build())
        		.setOutPort(outPort)
        		.setTableId(TableId.ALL)
        		.build();
        try {
            future = sw.writeStatsRequest(req);
            values = future.get(10, TimeUnit.SECONDS);
            if (values != null) {
                for (OFFlowStatsReply stat : values) {
                    statsReply.add(stat);
                }
            }
        } catch (Exception e) {
            log.error("Failure retrieving statistics from switch " + sw, e);
        }
        return statsReply;
    }
    public void clearFlowMods(IOFSwitch sw, OFPort outPort) {
    	Match match = sw.getOFFactory().buildMatch().build();
    	OFFlowDelete fm = sw.getOFFactory().buildFlowDelete()
    			.setMatch(match)
    			.setOutPort(outPort)
    			.build();
    	try {
    		sw.write(fm);
    	} catch (Exception e) {
    		log.error("Failed to clear flows on switch {} - {}", this, e);
    	}
    }
    public void clearFlowMods(IOFSwitch sw, Match match, OFPort outPort) {
        OFFlowDelete fm = sw.getOFFactory().buildFlowDelete()
        		.setMatch(match)
        		.setOutPort(outPort)
        		.build();
        try {
            sw.write(fm);
        } catch (Exception e) {
            log.error("Failed to clear flows on switch {} - {}", this, e);
        }
    }
    public void deleteInvalidFlows(IOFSwitch sw, Map<OFPort, List<Match>> invalidOutportAndMatch) {
        log.debug("Deleting invalid flows on switch : " + sw.getId());
        Map<OFPort, List<Match>> invalidNeighborIngressAndMatches = new HashMap<OFPort, List<Match>>();
        for (OFPort outPort : invalidOutportAndMatch.keySet()) {
            List<OFFlowStatsReply> flows = getFlows(sw, outPort);
            for (OFFlowStatsReply flow : flows) {
            	for (OFFlowStatsEntry entry : flow.getEntries()) {
            		for (Match match : invalidOutportAndMatch.get(outPort)) {
            			if (entry.getMatch().get(MatchField.ETH_DST).equals(match.get(MatchField.ETH_DST))
            				&& entry.getMatch().get(MatchField.ETH_SRC).equals(match.get(MatchField.ETH_SRC))
            				&& entry.getMatch().get(MatchField.ETH_TYPE).equals(match.get(MatchField.ETH_TYPE))
            				&& entry.getMatch().get(MatchField.VLAN_VID).equals(match.get(MatchField.VLAN_VID))
            				&& entry.getMatch().get(MatchField.IPV4_DST).equals(match.get(MatchField.IPV4_DST))
            				&& entry.getMatch().get(MatchField.IP_PROTO).equals(match.get(MatchField.IP_PROTO))
            				&& entry.getMatch().get(MatchField.IPV4_SRC).equals(match.get(MatchField.IPV4_SRC))
            				&& entry.getMatch().get(MatchField.IP_ECN).equals(match.get(MatchField.IP_ECN))) {
            					if (invalidNeighborIngressAndMatches.containsKey(match.get(MatchField.IN_PORT)))
            						invalidNeighborIngressAndMatches.get(match.get(MatchField.IN_PORT))
            						.add(match);
            					else {
            						List<Match> matches = new ArrayList<Match>();
            						matches.add(match);
            						invalidNeighborIngressAndMatches.put(match.get(MatchField.IN_PORT), matches);
            					}
            					clearFlowMods(sw, entry.getMatch(), outPort);
            				}
            		}
            	}
            }
            Map<IOFSwitch, Map<OFPort, List<Match>>> neighborSwitches = new HashMap<IOFSwitch, Map<OFPort, List<Match>>>();
            for (Link link : links.keySet()) {
                if (link.getDst().equals(sw.getId())) {
                    for (Entry<OFPort, List<Match>> ingressPort : invalidNeighborIngressAndMatches.entrySet()) {
                        if (link.getDstPort().equals(ingressPort.getKey())) {
                            Map<OFPort, List<Match>> invalidNeighborOutportAndMatch = new HashMap<OFPort, List<Match>>();
                            invalidNeighborOutportAndMatch.put(link.getSrcPort(),
                                                               ingressPort.getValue());
                            neighborSwitches.put(switchService.getSwitch(link.getSrc()), invalidNeighborOutportAndMatch);
                        }
                    }
                }
            }
            log.debug("We have " + neighborSwitches.size() + " neighbors to deal with!");
            for (IOFSwitch neighborSwitch : neighborSwitches.keySet()) {
                log.debug("NeighborSwitch ID : " + neighborSwitch.getId());
                deleteInvalidFlows(neighborSwitch, neighborSwitches.get(neighborSwitch));
            }
        }
    }
}

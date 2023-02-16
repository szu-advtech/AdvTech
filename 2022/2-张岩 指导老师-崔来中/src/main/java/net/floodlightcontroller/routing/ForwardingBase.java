package net.floodlightcontroller.routing;
import java.io.IOException;
import java.util.EnumSet;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IOFMessageListener;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.internal.IOFSwitchService;
import net.floodlightcontroller.core.util.AppCookie;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import net.floodlightcontroller.devicemanager.IDeviceService;
import net.floodlightcontroller.devicemanager.SwitchPort;
import net.floodlightcontroller.packet.IPacket;
import net.floodlightcontroller.routing.IRoutingService;
import net.floodlightcontroller.routing.IRoutingDecision;
import net.floodlightcontroller.routing.Route;
import net.floodlightcontroller.topology.ITopologyService;
import net.floodlightcontroller.topology.NodePortTuple;
import net.floodlightcontroller.util.FlowModUtils;
import net.floodlightcontroller.util.MatchUtils;
import net.floodlightcontroller.util.OFDPAUtils;
import net.floodlightcontroller.util.OFMessageDamper;
import net.floodlightcontroller.util.TimedCache;
import org.projectfloodlight.openflow.protocol.OFFlowMod;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.protocol.OFFlowModCommand;
import org.projectfloodlight.openflow.protocol.OFFlowModFlags;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFPacketIn;
import org.projectfloodlight.openflow.protocol.OFPacketOut;
import org.projectfloodlight.openflow.protocol.OFType;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.action.OFAction;
import org.projectfloodlight.openflow.protocol.action.OFActionOutput;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFBufferId;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.TableId;
import org.projectfloodlight.openflow.types.U64;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public abstract class ForwardingBase implements IOFMessageListener {
	protected static Logger log = LoggerFactory.getLogger(ForwardingBase.class);
	protected static boolean FLOWMOD_DEFAULT_SET_SEND_FLOW_REM_FLAG = false;
	protected static boolean FLOWMOD_DEFAULT_MATCH_VLAN = true;
	protected static boolean FLOWMOD_DEFAULT_MATCH_MAC = true;
	protected static boolean FLOWMOD_DEFAULT_MATCH_IP_ADDR = true;
	protected static boolean FLOWMOD_DEFAULT_MATCH_TRANSPORT = true;
	protected static final short FLOWMOD_DEFAULT_IDLE_TIMEOUT_CONSTANT = 5;
	protected static final short FLOWMOD_DEFAULT_HARD_TIMEOUT_CONSTANT = 0;
	protected static boolean FLOOD_ALL_ARP_PACKETS = false;
	protected static TableId tableId = null;
	protected IFloodlightProviderService floodlightProviderService;
	protected IOFSwitchService switchService;
	protected IDeviceService deviceManagerService;
	protected IRoutingService routingEngineService;
	protected ITopologyService topologyService;
	protected IDebugCounterService debugCounterService;
	protected OFMessageDamper messageDamper;
	protected boolean broadcastCacheFeature = true;
	static {
		AppCookie.registerApp(FORWARDING_APP_ID, "Forwarding");
	}
	public static final U64 appCookie = AppCookie.makeCookie(FORWARDING_APP_ID, 0);
	public Comparator<SwitchPort> clusterIdComparator =
			new Comparator<SwitchPort>() {
		@Override
		public int compare(SwitchPort d1, SwitchPort d2) {
			DatapathId d1ClusterId = topologyService.getOpenflowDomainId(d1.getSwitchDPID());
			DatapathId d2ClusterId = topologyService.getOpenflowDomainId(d2.getSwitchDPID());
			return d1ClusterId.compareTo(d2ClusterId);
		}
	};
	protected void init() {
		messageDamper = new OFMessageDamper(OFMESSAGE_DAMPER_CAPACITY,
				EnumSet.of(OFType.FLOW_MOD),
				OFMESSAGE_DAMPER_TIMEOUT);
	}
	protected void startUp() {
		floodlightProviderService.addOFMessageListener(OFType.PACKET_IN, this);
	}
	@Override
	public String getName() {
		return "forwarding";
	}
	public abstract Command processPacketInMessage(IOFSwitch sw, OFPacketIn pi, 
			IRoutingDecision decision, FloodlightContext cntx);
	@Override
	public Command receive(IOFSwitch sw, OFMessage msg, FloodlightContext cntx) {
		switch (msg.getType()) {
		case PACKET_IN:
			IRoutingDecision decision = null;
			if (cntx != null) {
				decision = RoutingDecision.rtStore.get(cntx, IRoutingDecision.CONTEXT_DECISION);
			}
			return this.processPacketInMessage(sw, (OFPacketIn) msg, decision, cntx);
		default:
			break;
		}
		return Command.CONTINUE;
	}
	public boolean pushRoute(Route route, Match match, OFPacketIn pi,
			DatapathId pinSwitch, U64 cookie, FloodlightContext cntx,
			boolean requestFlowRemovedNotification, OFFlowModCommand flowModCommand) {
		boolean packetOutSent = false;
		List<NodePortTuple> switchPortList = route.getPath();
		for (int indx = switchPortList.size() - 1; indx > 0; indx -= 2) {
			DatapathId switchDPID = switchPortList.get(indx).getNodeId();
			IOFSwitch sw = switchService.getSwitch(switchDPID);
			if (sw == null) {
				if (log.isWarnEnabled()) {
					log.warn("Unable to push route, switch at DPID {} " + "not available", switchDPID);
				}
				return packetOutSent;
			}
			OFFlowMod.Builder fmb;
			switch (flowModCommand) {
			case ADD:
				fmb = sw.getOFFactory().buildFlowAdd();
				break;
			case DELETE:
				fmb = sw.getOFFactory().buildFlowDelete();
				break;
			case DELETE_STRICT:
				fmb = sw.getOFFactory().buildFlowDeleteStrict();
				break;
			case MODIFY:
				fmb = sw.getOFFactory().buildFlowModify();
				break;
			default:
				log.error("Could not decode OFFlowModCommand. Using MODIFY_STRICT. (Should another be used as the default?)");        
			case MODIFY_STRICT:
				fmb = sw.getOFFactory().buildFlowModifyStrict();
				break;			
			}
			OFActionOutput.Builder aob = sw.getOFFactory().actions().buildOutput();
			List<OFAction> actions = new ArrayList<OFAction>();	
 			Match.Builder mb = MatchUtils.convertToVersion(match, sw.getOFFactory().getVersion());
			OFPort outPort = switchPortList.get(indx).getPortId();
			OFPort inPort = switchPortList.get(indx - 1).getPortId();
			mb.setExact(MatchField.IN_PORT, inPort);
			aob.setPort(outPort);
			aob.setMaxLen(Integer.MAX_VALUE);
			actions.add(aob.build());
			if (FLOWMOD_DEFAULT_SET_SEND_FLOW_REM_FLAG || requestFlowRemovedNotification) {
				Set<OFFlowModFlags> flags = new HashSet<>();
				flags.add(OFFlowModFlags.SEND_FLOW_REM);
				fmb.setFlags(flags);
			}
			fmb.setMatch(mb.build())
			.setIdleTimeout(FLOWMOD_DEFAULT_IDLE_TIMEOUT)
			.setHardTimeout(FLOWMOD_DEFAULT_HARD_TIMEOUT)
			.setBufferId(OFBufferId.NO_BUFFER)
			.setCookie(cookie)
			.setOutPort(outPort)
			.setPriority(FLOWMOD_DEFAULT_PRIORITY);
			if(tableId != null)
			{
				fmb.setTableId(tableId);
			}
			FlowModUtils.setActions(fmb, actions, sw);
			try {
				if (log.isTraceEnabled()) {
					log.trace("Pushing Route flowmod routeIndx={} " +
							"sw={} inPort={} outPort={}",
							new Object[] {indx,
							sw,
							fmb.getMatch().get(MatchField.IN_PORT),
							outPort });
				}
				if (OFDPAUtils.isOFDPASwitch(sw)) {
					OFDPAUtils.addLearningSwitchFlow(sw, cookie, 
							FLOWMOD_DEFAULT_PRIORITY, 
							FLOWMOD_DEFAULT_HARD_TIMEOUT,
							FLOWMOD_DEFAULT_IDLE_TIMEOUT,
							fmb.getMatch(), 
							outPort);
				} else {
					messageDamper.write(sw, fmb.build());
				}
				if (sw.getId().equals(pinSwitch) &&
						!fmb.getCommand().equals(OFFlowModCommand.DELETE) &&
						!fmb.getCommand().equals(OFFlowModCommand.DELETE_STRICT)) {
					pushPacket(sw, pi, outPort, true, cntx);
					packetOutSent = true;
				}
			} catch (IOException e) {
				log.error("Failure writing flow mod", e);
			}
		}
		return packetOutSent;
	}
	protected void pushPacket(IOFSwitch sw, OFPacketIn pi, OFPort outport, boolean useBufferedPacket, FloodlightContext cntx) {
		if (pi == null) {
			return;
		}
		if ((pi.getVersion().compareTo(OFVersion.OF_12) < 0 ? pi.getInPort() : pi.getMatch().get(MatchField.IN_PORT)).equals(outport)) {
			if (log.isDebugEnabled()) {
				log.debug("Attempting to do packet-out to the same " +
						"interface as packet-in. Dropping packet. " +
						" SrcSwitch={}, pi={}",
						new Object[]{sw, pi});
				return;
			}
		}
		if (log.isTraceEnabled()) {
			log.trace("PacketOut srcSwitch={} pi={}",
					new Object[] {sw, pi});
		}
		OFPacketOut.Builder pob = sw.getOFFactory().buildPacketOut();
		List<OFAction> actions = new ArrayList<OFAction>();
		actions.add(sw.getOFFactory().actions().output(outport, Integer.MAX_VALUE));
		pob.setActions(actions);
		if (useBufferedPacket) {
		} else {
			pob.setBufferId(OFBufferId.NO_BUFFER);
		}
		if (pob.getBufferId().equals(OFBufferId.NO_BUFFER)) {
			byte[] packetData = pi.getData();
			pob.setData(packetData);
		}
		pob.setInPort((pi.getVersion().compareTo(OFVersion.OF_12) < 0 ? pi.getInPort() : pi.getMatch().get(MatchField.IN_PORT)));
		try {
			messageDamper.write(sw, pob.build());
		} catch (IOException e) {
			log.error("Failure writing packet out", e);
		}
	}
	public void packetOutMultiPort(byte[] packetData, IOFSwitch sw, 
			OFPort inPort, Set<OFPort> outPorts, FloodlightContext cntx) {
		List<OFAction> actions = new ArrayList<OFAction>();
		Iterator<OFPort> j = outPorts.iterator();
		while (j.hasNext()) {
			actions.add(sw.getOFFactory().actions().output(j.next(), 0));
		}
		OFPacketOut.Builder pob = sw.getOFFactory().buildPacketOut();
		pob.setActions(actions);
		pob.setBufferId(OFBufferId.NO_BUFFER);
		pob.setInPort(inPort);
		pob.setData(packetData);
		try {
			if (log.isTraceEnabled()) {
				log.trace("write broadcast packet on switch-id={} " +
						"interfaces={} packet-out={}",
						new Object[] {sw.getId(), outPorts, pob.build()});
			}
			messageDamper.write(sw, pob.build());
		} catch (IOException e) {
			log.error("Failure writing packet out", e);
		}
	}
	public void packetOutMultiPort(OFPacketIn pi, IOFSwitch sw,
			OFPort inPort, Set<OFPort> outPorts, FloodlightContext cntx) {
		packetOutMultiPort(pi.getData(), sw, inPort, outPorts, cntx);
	}
	public void packetOutMultiPort(IPacket packet, IOFSwitch sw,
			OFPort inPort, Set<OFPort> outPorts, FloodlightContext cntx) {
		packetOutMultiPort(packet.serialize(), sw, inPort, outPorts, cntx);
	}
	public static boolean blockHost(IOFSwitchService switchService,
			SwitchPort sw_tup, MacAddress host_mac, short hardTimeout, U64 cookie) {
		if (sw_tup == null) {
			return false;
		}
		IOFSwitch sw = switchService.getSwitch(sw_tup.getSwitchDPID());
		if (sw == null) {
			return false;
		}
		OFPort inputPort = sw_tup.getPort();
		log.debug("blockHost sw={} port={} mac={}",
				new Object[] { sw, sw_tup.getPort(), host_mac.getLong() });
		OFFlowMod.Builder fmb = sw.getOFFactory().buildFlowAdd();
		Match.Builder mb = sw.getOFFactory().buildMatch();
		mb.setExact(MatchField.IN_PORT, inputPort);
		if (host_mac.getLong() != -1L) {
			mb.setExact(MatchField.ETH_SRC, host_mac);
		}
		fmb.setCookie(cookie)
		.setHardTimeout(hardTimeout)
		.setIdleTimeout(FLOWMOD_DEFAULT_IDLE_TIMEOUT)
		.setPriority(FLOWMOD_DEFAULT_PRIORITY)
		.setBufferId(OFBufferId.NO_BUFFER)
		.setMatch(mb.build());
		FlowModUtils.setActions(fmb, actions, sw);
		log.debug("write drop flow-mod sw={} match={} flow-mod={}",
					new Object[] { sw, mb.build(), fmb.build() });
		sw.write(fmb.build());
		return true;
	}
	@Override
	public boolean isCallbackOrderingPrereq(OFType type, String name) {
		return (type.equals(OFType.PACKET_IN) && (name.equals("topology") || name.equals("devicemanager")));
	}
	@Override
	public boolean isCallbackOrderingPostreq(OFType type, String name) {
		return false;
	}
}
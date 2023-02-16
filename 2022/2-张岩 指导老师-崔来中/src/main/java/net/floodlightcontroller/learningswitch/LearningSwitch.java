package net.floodlightcontroller.learningswitch;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.IControllerCompletionListener;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IOFMessageListener;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.core.types.MacVlanPair;
import net.floodlightcontroller.debugcounter.IDebugCounter;
import net.floodlightcontroller.debugcounter.IDebugCounterService;
import net.floodlightcontroller.debugcounter.IDebugCounterService.MetaData;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.restserver.IRestApiService;
import net.floodlightcontroller.util.OFMessageUtils;
import org.projectfloodlight.openflow.protocol.OFFlowMod;
import org.projectfloodlight.openflow.protocol.OFFlowRemoved;
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
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFBufferId;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.OFVlanVidMatch;
import org.projectfloodlight.openflow.types.U64;
import org.projectfloodlight.openflow.types.VlanVid;
import org.projectfloodlight.openflow.util.LRULinkedHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class LearningSwitch
implements IFloodlightModule, ILearningSwitchService, IOFMessageListener, IControllerCompletionListener {
	protected static Logger log = LoggerFactory.getLogger(LearningSwitch.class);
	protected IFloodlightProviderService floodlightProviderService;
	protected IRestApiService restApiService;
	protected IDebugCounterService debugCounterService;
	private IDebugCounter counterFlowMod;
	private IDebugCounter counterPacketOut;
	protected Map<IOFSwitch, Map<MacVlanPair, OFPort>> macVlanToSwitchPortMap;
	public static final int LEARNING_SWITCH_APP_ID = 1;
	public static final int APP_ID_BITS = 12;
	public static final int APP_ID_SHIFT = (64 - APP_ID_BITS);
	public static final long LEARNING_SWITCH_COOKIE = (long) (LEARNING_SWITCH_APP_ID & ((1 << APP_ID_BITS) - 1)) << APP_ID_SHIFT;
	protected static short FLOWMOD_PRIORITY = 100;
	protected static final int MAX_MACS_PER_SWITCH  = 1000;
	protected static final boolean LEARNING_SWITCH_REVERSE_FLOW = true;
	protected final boolean flushAtCompletion = false;
	public void setFloodlightProvider(IFloodlightProviderService floodlightProviderService) {
		this.floodlightProviderService = floodlightProviderService;
	}
	@Override
	public String getName() {
		return "learningswitch";
	}
	protected void addToPortMap(IOFSwitch sw, MacAddress mac, VlanVid vlan, OFPort portVal) {
		Map<MacVlanPair, OFPort> swMap = macVlanToSwitchPortMap.get(sw);
		if (vlan == VlanVid.FULL_MASK || vlan == null) {
			vlan = VlanVid.ofVlan(0);
		}
		if (swMap == null) {
			swMap = Collections.synchronizedMap(new LRULinkedHashMap<MacVlanPair, OFPort>(MAX_MACS_PER_SWITCH));
			macVlanToSwitchPortMap.put(sw, swMap);
		}
		swMap.put(new MacVlanPair(mac, vlan), portVal);
	}
	protected void removeFromPortMap(IOFSwitch sw, MacAddress mac, VlanVid vlan) {
		if (vlan == VlanVid.FULL_MASK) {
			vlan = VlanVid.ofVlan(0);
		}
		Map<MacVlanPair, OFPort> swMap = macVlanToSwitchPortMap.get(sw);
		if (swMap != null) {
			swMap.remove(new MacVlanPair(mac, vlan));
		}
	}
	public OFPort getFromPortMap(IOFSwitch sw, MacAddress mac, VlanVid vlan) {
		if (vlan == VlanVid.FULL_MASK || vlan == null) {
			vlan = VlanVid.ofVlan(0);
		}
		Map<MacVlanPair, OFPort> swMap = macVlanToSwitchPortMap.get(sw);
		if (swMap != null) {
			return swMap.get(new MacVlanPair(mac, vlan));
		}
		return null;
	}
	public void clearLearnedTable() {
		macVlanToSwitchPortMap.clear();
	}
	public void clearLearnedTable(IOFSwitch sw) {
		Map<MacVlanPair, OFPort> swMap = macVlanToSwitchPortMap.get(sw);
		if (swMap != null) {
			swMap.clear();
		}
	}
	@Override
	public synchronized Map<IOFSwitch, Map<MacVlanPair, OFPort>> getTable() {
		return macVlanToSwitchPortMap;
	}
	private void writeFlowMod(IOFSwitch sw, OFFlowModCommand command, OFBufferId bufferId,
			Match match, OFPort outPort) {
		OFFlowMod.Builder fmb;
		if (command == OFFlowModCommand.DELETE) {
			fmb = sw.getOFFactory().buildFlowDelete();
		} else {
			fmb = sw.getOFFactory().buildFlowAdd();
		}
		fmb.setMatch(match);
		fmb.setCookie((U64.of(LearningSwitch.LEARNING_SWITCH_COOKIE)));
		fmb.setIdleTimeout(LearningSwitch.FLOWMOD_DEFAULT_IDLE_TIMEOUT);
		fmb.setHardTimeout(LearningSwitch.FLOWMOD_DEFAULT_HARD_TIMEOUT);
		fmb.setPriority(LearningSwitch.FLOWMOD_PRIORITY);
		fmb.setBufferId(bufferId);
		fmb.setOutPort((command == OFFlowModCommand.DELETE) ? OFPort.ANY : outPort);
		Set<OFFlowModFlags> sfmf = new HashSet<OFFlowModFlags>();
		if (command != OFFlowModCommand.DELETE) {
			sfmf.add(OFFlowModFlags.SEND_FLOW_REM);
		}
		fmb.setFlags(sfmf);
		List<OFAction> al = new ArrayList<OFAction>();
		al.add(sw.getOFFactory().actions().buildOutput().setPort(outPort).setMaxLen(0xffFFffFF).build());
		fmb.setActions(al);
		if (log.isTraceEnabled()) {
			log.trace("{} {} flow mod {}",
					new Object[]{ sw, (command == OFFlowModCommand.DELETE) ? "deleting" : "adding", fmb.build() });
		}
		counterFlowMod.increment();
		sw.write(fmb.build());
	}
	private void pushPacket(IOFSwitch sw, Match match, OFPacketIn pi, OFPort outport) {
		if (pi == null) {
			return;
		}
		OFPort inPort = (pi.getVersion().compareTo(OFVersion.OF_12) < 0 ? pi.getInPort() : pi.getMatch().get(MatchField.IN_PORT));
		if (inPort.equals(outport)) {
			if (log.isDebugEnabled()) {
				log.debug("Attempting to do packet-out to the same " +
						"interface as packet-in. Dropping packet. " +
						" SrcSwitch={}, match = {}, pi={}",
						new Object[]{sw, match, pi});
				return;
			}
		}
		if (log.isTraceEnabled()) {
			log.trace("PacketOut srcSwitch={} match={} pi={}",
					new Object[] {sw, match, pi});
		}
		OFPacketOut.Builder pob = sw.getOFFactory().buildPacketOut();
		List<OFAction> actions = new ArrayList<OFAction>();
		actions.add(sw.getOFFactory().actions().buildOutput().setPort(outport).setMaxLen(0xffFFffFF).build());
		pob.setActions(actions);
		if (sw.getBuffers() == 0) {
			pi = pi.createBuilder().setBufferId(OFBufferId.NO_BUFFER).build();
			pob.setBufferId(OFBufferId.NO_BUFFER);
		} else {
			pob.setBufferId(pi.getBufferId());
		}
		pob.setInPort(inPort);
		if (pi.getBufferId() == OFBufferId.NO_BUFFER) {
			byte[] packetData = pi.getData();
			pob.setData(packetData);
		}
		counterPacketOut.increment();
		sw.write(pob.build());
	}
	private void writePacketOutForPacketIn(IOFSwitch sw, OFPacketIn packetInMessage, OFPort egressPort) {
		OFMessageUtils.writePacketOutForPacketIn(sw, packetInMessage, egressPort);
		counterPacketOut.increment();
	}
	protected Match createMatchFromPacket(IOFSwitch sw, OFPort inPort, FloodlightContext cntx) {
		Ethernet eth = IFloodlightProviderService.bcStore.get(cntx, IFloodlightProviderService.CONTEXT_PI_PAYLOAD);
		VlanVid vlan = VlanVid.ofVlan(eth.getVlanID());
		MacAddress srcMac = eth.getSourceMACAddress();
		MacAddress dstMac = eth.getDestinationMACAddress();
		Match.Builder mb = sw.getOFFactory().buildMatch();
		mb.setExact(MatchField.IN_PORT, inPort)
		.setExact(MatchField.ETH_SRC, srcMac)
		.setExact(MatchField.ETH_DST, dstMac);
		if (!vlan.equals(VlanVid.ZERO)) {
			mb.setExact(MatchField.VLAN_VID, OFVlanVidMatch.ofVlanVid(vlan));
		}
		return mb.build();
	}
	private Command processPacketInMessage(IOFSwitch sw, OFPacketIn pi, FloodlightContext cntx) {
		OFPort inPort = (pi.getVersion().compareTo(OFVersion.OF_12) < 0 ? pi.getInPort() : pi.getMatch().get(MatchField.IN_PORT));
		Match m = createMatchFromPacket(sw, inPort, cntx);
		MacAddress sourceMac = m.get(MatchField.ETH_SRC);
		MacAddress destMac = m.get(MatchField.ETH_DST);
		VlanVid vlan = m.get(MatchField.VLAN_VID) == null ? VlanVid.ZERO : m.get(MatchField.VLAN_VID).getVlanVid();
		if (sourceMac == null) {
			sourceMac = MacAddress.NONE;
		}
		if (destMac == null) {
			destMac = MacAddress.NONE;
		}
		if (vlan == null) {
			vlan = VlanVid.ZERO;
		}
		if ((destMac.getLong() & 0xfffffffffff0L) == 0x0180c2000000L) {
			if (log.isTraceEnabled()) {
				log.trace("ignoring packet addressed to 802.1D/Q reserved addr: switch {} vlan {} dest MAC {}",
						new Object[]{ sw, vlan, destMac.toString() });
			}
			return Command.STOP;
		}
		if ((sourceMac.getLong() & 0x010000000000L) == 0) {
			this.addToPortMap(sw, sourceMac, vlan, inPort);
		}
		OFPort outPort = getFromPortMap(sw, destMac, vlan);
		if (outPort == null) {
			this.writePacketOutForPacketIn(sw, pi, OFPort.FLOOD);
		} else if (outPort.equals(inPort)) {
			log.trace("ignoring packet that arrived on same port as learned destination:"
					+ " switch {} vlan {} dest MAC {} port {}",
					new Object[]{ sw, vlan, destMac.toString(), outPort.getPortNumber() });
		} else {
			this.pushPacket(sw, m, pi, outPort);
			this.writeFlowMod(sw, OFFlowModCommand.ADD, OFBufferId.NO_BUFFER, m, outPort);
			if (LEARNING_SWITCH_REVERSE_FLOW) {
				Match.Builder mb = m.createBuilder();
				mb.setExact(MatchField.ETH_SRC, m.get(MatchField.ETH_DST))                         
				.setExact(MatchField.ETH_DST, m.get(MatchField.ETH_SRC))     
				.setExact(MatchField.IN_PORT, outPort);
				if (m.get(MatchField.VLAN_VID) != null) {
					mb.setExact(MatchField.VLAN_VID, m.get(MatchField.VLAN_VID));
				}
				this.writeFlowMod(sw, OFFlowModCommand.ADD, OFBufferId.NO_BUFFER, mb.build(), inPort);
			}
		}
		return Command.CONTINUE;
	}
	private Command processFlowRemovedMessage(IOFSwitch sw, OFFlowRemoved flowRemovedMessage) {
		if (!flowRemovedMessage.getCookie().equals(U64.of(LearningSwitch.LEARNING_SWITCH_COOKIE))) {
			return Command.CONTINUE;
		}
		if (log.isTraceEnabled()) {
			log.trace("{} flow entry removed {}", sw, flowRemovedMessage);
		}
		Match match = flowRemovedMessage.getMatch();
		this.removeFromPortMap(sw, match.get(MatchField.ETH_SRC), 
				match.get(MatchField.VLAN_VID) == null 
				? VlanVid.ZERO 
				: match.get(MatchField.VLAN_VID).getVlanVid());
		Match.Builder mb = sw.getOFFactory().buildMatch();
		mb.setExact(MatchField.ETH_SRC, match.get(MatchField.ETH_DST))                         
		.setExact(MatchField.ETH_DST, match.get(MatchField.ETH_SRC));
		if (match.get(MatchField.VLAN_VID) != null) {
			mb.setExact(MatchField.VLAN_VID, match.get(MatchField.VLAN_VID));                    
		}
		this.writeFlowMod(sw, OFFlowModCommand.DELETE, OFBufferId.NO_BUFFER, mb.build(), match.get(MatchField.IN_PORT));
		return Command.CONTINUE;
	}
	@Override
	public Command receive(IOFSwitch sw, OFMessage msg, FloodlightContext cntx) {
		switch (msg.getType()) {
		case PACKET_IN:
			return this.processPacketInMessage(sw, (OFPacketIn) msg, cntx);
		case FLOW_REMOVED:
			return this.processFlowRemovedMessage(sw, (OFFlowRemoved) msg);
		case ERROR:
			log.info("received an error {} from switch {}", msg, sw);
			return Command.CONTINUE;
		default:
			log.error("received an unexpected message {} from switch {}", msg, sw);
			return Command.CONTINUE;
		}
	}
	@Override
	public boolean isCallbackOrderingPrereq(OFType type, String name) {
		return false;
	}
	@Override
	public boolean isCallbackOrderingPostreq(OFType type, String name) {
		return false;
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleServices() {
		Collection<Class<? extends IFloodlightService>> l =
				new ArrayList<Class<? extends IFloodlightService>>();
		l.add(ILearningSwitchService.class);
		return l;
	}
	@Override
	public Map<Class<? extends IFloodlightService>, IFloodlightService> getServiceImpls() {
		Map<Class<? extends IFloodlightService>,  IFloodlightService> m = 
				new HashMap<Class<? extends IFloodlightService>, IFloodlightService>();
		m.put(ILearningSwitchService.class, this);
		return m;
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleDependencies() {
		Collection<Class<? extends IFloodlightService>> l =
				new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IFloodlightProviderService.class);
		l.add(IDebugCounterService.class);
		l.add(IRestApiService.class);
		return l;
	}
	@Override
	public void init(FloodlightModuleContext context) throws FloodlightModuleException {
		macVlanToSwitchPortMap = new ConcurrentHashMap<IOFSwitch, Map<MacVlanPair, OFPort>>();
		floodlightProviderService = context.getServiceImpl(IFloodlightProviderService.class);
		debugCounterService = context.getServiceImpl(IDebugCounterService.class);
		restApiService = context.getServiceImpl(IRestApiService.class);
	}
	@Override
	public void startUp(FloodlightModuleContext context) {
		floodlightProviderService.addCompletionListener(this);
		floodlightProviderService.addOFMessageListener(OFType.PACKET_IN, this);
		floodlightProviderService.addOFMessageListener(OFType.FLOW_REMOVED, this);
		floodlightProviderService.addOFMessageListener(OFType.ERROR, this);
		restApiService.addRestletRoutable(new LearningSwitchWebRoutable());
		Map<String, String> configOptions = context.getConfigParams(this);
		try {
			String idleTimeout = configOptions.get("idletimeout");
			if (idleTimeout != null) {
				FLOWMOD_DEFAULT_IDLE_TIMEOUT = Short.parseShort(idleTimeout);
			}
		} catch (NumberFormatException e) {
			log.warn("Error parsing flow idle timeout, " +
					"using default of {} seconds", FLOWMOD_DEFAULT_IDLE_TIMEOUT);
		}
		try {
			String hardTimeout = configOptions.get("hardtimeout");
			if (hardTimeout != null) {
				FLOWMOD_DEFAULT_HARD_TIMEOUT = Short.parseShort(hardTimeout);
			}
		} catch (NumberFormatException e) {
			log.warn("Error parsing flow hard timeout, " +
					"using default of {} seconds", FLOWMOD_DEFAULT_HARD_TIMEOUT);
		}
		try {
			String priority = configOptions.get("priority");
			if (priority != null) {
				FLOWMOD_PRIORITY = Short.parseShort(priority);
			}
		} catch (NumberFormatException e) {
			log.warn("Error parsing flow priority, " +
					"using default of {}",
					FLOWMOD_PRIORITY);
		}
		log.debug("FlowMod idle timeout set to {} seconds", FLOWMOD_DEFAULT_IDLE_TIMEOUT);
		log.debug("FlowMod hard timeout set to {} seconds", FLOWMOD_DEFAULT_HARD_TIMEOUT);
		log.debug("FlowMod priority set to {}", FLOWMOD_PRIORITY);
		debugCounterService.registerModule(this.getName());
		counterFlowMod = debugCounterService.registerCounter(this.getName(), "flow-mods-written", "Flow mods written to switches by LearningSwitch", MetaData.WARN);
		counterPacketOut = debugCounterService.registerCounter(this.getName(), "packet-outs-written", "Packet outs written to switches by LearningSwitch", MetaData.WARN);
	}
	@Override
	public void onMessageConsumed(IOFSwitch sw, OFMessage msg, FloodlightContext cntx) {
		if (this.flushAtCompletion) {
			log.debug("Learning switch: ended processing packet {}",msg.toString());
		}
	}
}

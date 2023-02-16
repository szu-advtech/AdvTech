package net.floodlightcontroller.firewall;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFPacketIn;
import org.projectfloodlight.openflow.protocol.OFType;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.EthType;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv4AddressWithMask;
import org.projectfloodlight.openflow.types.IpProtocol;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.TransportPort;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.IOFMessageListener;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.devicemanager.IDeviceService;
import java.util.ArrayList;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.packet.IPv4;
import net.floodlightcontroller.packet.TCP;
import net.floodlightcontroller.packet.UDP;
import net.floodlightcontroller.restserver.IRestApiService;
import net.floodlightcontroller.routing.IRoutingDecision;
import net.floodlightcontroller.routing.RoutingDecision;
import net.floodlightcontroller.storage.IResultSet;
import net.floodlightcontroller.storage.IStorageSourceService;
import net.floodlightcontroller.storage.StorageException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class Firewall implements IFirewallService, IOFMessageListener,
IFloodlightModule {
	protected IFloodlightProviderService floodlightProvider;
	protected IStorageSourceService storageSource;
	protected IRestApiService restApi;
	protected static Logger logger;
	protected boolean enabled;
	protected IPv4Address subnet_mask = IPv4Address.of("255.255.255.0");
	public static final String TABLE_NAME = "controller_firewallrules";
	public static final String COLUMN_RULEID = "ruleid";
	public static final String COLUMN_DPID = "dpid";
	public static final String COLUMN_IN_PORT = "in_port";
	public static final String COLUMN_DL_SRC = "dl_src";
	public static final String COLUMN_DL_DST = "dl_dst";
	public static final String COLUMN_DL_TYPE = "dl_type";
	public static final String COLUMN_NW_SRC_PREFIX = "nw_src_prefix";
	public static final String COLUMN_NW_SRC_MASKBITS = "nw_src_maskbits";
	public static final String COLUMN_NW_DST_PREFIX = "nw_dst_prefix";
	public static final String COLUMN_NW_DST_MASKBITS = "nw_dst_maskbits";
	public static final String COLUMN_NW_PROTO = "nw_proto";
	public static final String COLUMN_TP_SRC = "tp_src";
	public static final String COLUMN_TP_DST = "tp_dst";
	public static final String COLUMN_WILDCARD_DPID = "wildcard_dpid";
	public static final String COLUMN_WILDCARD_IN_PORT = "any_in_port";
	public static final String COLUMN_WILDCARD_DL_SRC = "any_dl_src";
	public static final String COLUMN_WILDCARD_DL_DST = "any_dl_dst";
	public static final String COLUMN_WILDCARD_DL_TYPE = "any_dl_type";
	public static final String COLUMN_WILDCARD_NW_SRC = "any_nw_src";
	public static final String COLUMN_WILDCARD_NW_DST = "any_nw_dst";
	public static final String COLUMN_WILDCARD_NW_PROTO = "any_nw_proto";
	public static final String COLUMN_WILDCARD_TP_SRC = "any_tp_src";
	public static final String COLUMN_WILDCARD_TP_DST = "any_tp_dst";
	public static final String COLUMN_PRIORITY = "priority";
	public static final String COLUMN_ACTION = "action";
	public static String ColumnNames[] = { COLUMN_RULEID, COLUMN_DPID,
		COLUMN_IN_PORT, COLUMN_DL_SRC, COLUMN_DL_DST, COLUMN_DL_TYPE,
		COLUMN_NW_SRC_PREFIX, COLUMN_NW_SRC_MASKBITS, COLUMN_NW_DST_PREFIX,
		COLUMN_NW_DST_MASKBITS, COLUMN_NW_PROTO, COLUMN_TP_SRC,
		COLUMN_TP_DST, COLUMN_WILDCARD_DPID, COLUMN_WILDCARD_IN_PORT,
		COLUMN_WILDCARD_DL_SRC, COLUMN_WILDCARD_DL_DST,
		COLUMN_WILDCARD_DL_TYPE, COLUMN_WILDCARD_NW_SRC,
		COLUMN_WILDCARD_NW_DST, COLUMN_WILDCARD_NW_PROTO, COLUMN_PRIORITY,
		COLUMN_ACTION };
	@Override
	public String getName() {
		return "firewall";
	}
	@Override
	public boolean isCallbackOrderingPrereq(OFType type, String name) {
		return false;
	}
	@Override
	public boolean isCallbackOrderingPostreq(OFType type, String name) {
		return (type.equals(OFType.PACKET_IN) && name.equals("forwarding"));
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleServices() {
		Collection<Class<? extends IFloodlightService>> l = new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IFirewallService.class);
		return l;
	}
	@Override
	public Map<Class<? extends IFloodlightService>, IFloodlightService> getServiceImpls() {
		Map<Class<? extends IFloodlightService>, IFloodlightService> m = new HashMap<Class<? extends IFloodlightService>, IFloodlightService>();
		m.put(IFirewallService.class, this);
		return m;
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleDependencies() {
		Collection<Class<? extends IFloodlightService>> l = new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IFloodlightProviderService.class);
		l.add(IStorageSourceService.class);
		l.add(IRestApiService.class);
		return l;
	}
	protected ArrayList<FirewallRule> readRulesFromStorage() {
		ArrayList<FirewallRule> l = new ArrayList<FirewallRule>();
		try {
			Map<String, Object> row;
			IResultSet resultSet = storageSource.executeQuery(TABLE_NAME, ColumnNames, null, null);
			for (Iterator<IResultSet> it = resultSet.iterator(); it.hasNext();) {
				row = it.next().getRow();
				FirewallRule r = new FirewallRule();
				if (!row.containsKey(COLUMN_RULEID) || !row.containsKey(COLUMN_DPID)) {
					logger.error( "skipping entry with missing required 'ruleid' or 'switchid' entry: {}", row);
					return l;
				}
				try {
					r.ruleid = Integer
							.parseInt((String) row.get(COLUMN_RULEID));
					r.dpid = DatapathId.of((String) row.get(COLUMN_DPID));
					for (String key : row.keySet()) {
						if (row.get(key) == null) {
							continue;
						}
						if (key.equals(COLUMN_RULEID) || key.equals(COLUMN_DPID) || key.equals("id")) {
						} else if (key.equals(COLUMN_IN_PORT)) {
							r.in_port = OFPort.of(Integer.parseInt((String) row.get(COLUMN_IN_PORT)));
						} else if (key.equals(COLUMN_DL_SRC)) {
							r.dl_src = MacAddress.of(Long.parseLong((String) row.get(COLUMN_DL_SRC)));
						}  else if (key.equals(COLUMN_DL_DST)) {
							r.dl_dst = MacAddress.of(Long.parseLong((String) row.get(COLUMN_DL_DST)));
						} else if (key.equals(COLUMN_DL_TYPE)) {
							r.dl_type = EthType.of(Integer.parseInt((String) row.get(COLUMN_DL_TYPE)));
						} else if (key.equals(COLUMN_NW_SRC_PREFIX)) {
							r.nw_src_prefix_and_mask = IPv4AddressWithMask.of(IPv4Address.of(Integer.parseInt((String) row.get(COLUMN_NW_SRC_PREFIX))), r.nw_src_prefix_and_mask.getMask());
						} else if (key.equals(COLUMN_NW_SRC_MASKBITS)) {
							r.nw_src_prefix_and_mask = IPv4AddressWithMask.of(r.nw_src_prefix_and_mask.getValue(), IPv4Address.of(Integer.parseInt((String) row.get(COLUMN_NW_SRC_MASKBITS))));
						} else if (key.equals(COLUMN_NW_DST_PREFIX)) {
							r.nw_dst_prefix_and_mask = IPv4AddressWithMask.of(IPv4Address.of(Integer.parseInt((String) row.get(COLUMN_NW_DST_PREFIX))), r.nw_dst_prefix_and_mask.getMask());
						} else if (key.equals(COLUMN_NW_DST_MASKBITS)) {
							r.nw_dst_prefix_and_mask = IPv4AddressWithMask.of(r.nw_dst_prefix_and_mask.getValue(), IPv4Address.of(Integer.parseInt((String) row.get(COLUMN_NW_DST_MASKBITS))));
						} else if (key.equals(COLUMN_NW_PROTO)) {
							r.nw_proto = IpProtocol.of(Short.parseShort((String) row.get(COLUMN_NW_PROTO)));
						} else if (key.equals(COLUMN_TP_SRC)) {
							r.tp_src = TransportPort.of(Integer.parseInt((String) row.get(COLUMN_TP_SRC)));
						} else if (key.equals(COLUMN_TP_DST)) {
							r.tp_dst = TransportPort.of(Integer.parseInt((String) row.get(COLUMN_TP_DST)));
						} else if (key.equals(COLUMN_WILDCARD_DPID)) {
							r.any_dpid = Boolean.parseBoolean((String) row.get(COLUMN_WILDCARD_DPID));
						} else if (key.equals(COLUMN_WILDCARD_IN_PORT)) {
							r.any_in_port = Boolean.parseBoolean((String) row.get(COLUMN_WILDCARD_IN_PORT));
						} else if (key.equals(COLUMN_WILDCARD_DL_SRC)) {
							r.any_dl_src = Boolean.parseBoolean((String) row.get(COLUMN_WILDCARD_DL_SRC));
						} else if (key.equals(COLUMN_WILDCARD_DL_DST)) {
							r.any_dl_dst = Boolean.parseBoolean((String) row.get(COLUMN_WILDCARD_DL_DST));
						} else if (key.equals(COLUMN_WILDCARD_DL_TYPE)) {
							r.any_dl_type = Boolean.parseBoolean((String) row.get(COLUMN_WILDCARD_DL_TYPE));
						} else if (key.equals(COLUMN_WILDCARD_NW_SRC)) {
							r.any_nw_src = Boolean.parseBoolean((String) row.get(COLUMN_WILDCARD_NW_SRC));
						} else if (key.equals(COLUMN_WILDCARD_NW_DST)) {
							r.any_nw_dst = Boolean.parseBoolean((String) row.get(COLUMN_WILDCARD_NW_DST));
						} else if (key.equals(COLUMN_WILDCARD_NW_PROTO)) {
							r.any_nw_proto = Boolean.parseBoolean((String) row.get(COLUMN_WILDCARD_NW_PROTO));
						} else if (key.equals(COLUMN_PRIORITY)) {
							r.priority = Integer.parseInt((String) row.get(COLUMN_PRIORITY));
						} else if (key.equals(COLUMN_ACTION)) {
							int tmp = Integer.parseInt((String) row.get(COLUMN_ACTION));
							if (tmp == FirewallRule.FirewallAction.DROP.ordinal()) {
								r.action = FirewallRule.FirewallAction.DROP;
							} else if (tmp == FirewallRule.FirewallAction.ALLOW.ordinal()) {
								r.action = FirewallRule.FirewallAction.ALLOW;
							} else {
								r.action = null;
								logger.error("action not recognized");
							}
						}
					}
				} catch (ClassCastException e) {
					logger.error("skipping rule {} with bad data : " + e.getMessage(), r.ruleid);
				}
				if (r.action != null) {
					l.add(r);
				}
			}
		} catch (StorageException e) {
			logger.error("failed to access storage: {}", e.getMessage());
		}
		Collections.sort(l);
		return l;
	}
	@Override
	public void init(FloodlightModuleContext context) throws FloodlightModuleException {
		floodlightProvider = context.getServiceImpl(IFloodlightProviderService.class);
		storageSource = context.getServiceImpl(IStorageSourceService.class);
		restApi = context.getServiceImpl(IRestApiService.class);
		rules = new ArrayList<FirewallRule>();
		logger = LoggerFactory.getLogger(Firewall.class);
		enabled = false;
	}
	@Override
	public void startUp(FloodlightModuleContext context) {
		restApi.addRestletRoutable(new FirewallWebRoutable());
		floodlightProvider.addOFMessageListener(OFType.PACKET_IN, this);
		storageSource.createTable(TABLE_NAME, null);
		storageSource.setTablePrimaryKeyName(TABLE_NAME, COLUMN_RULEID);
		this.rules = readRulesFromStorage();
	}
	@Override
	public Command receive(IOFSwitch sw, OFMessage msg, FloodlightContext cntx) {
		if (!this.enabled) {
			return Command.CONTINUE;
		}
		switch (msg.getType()) {
		case PACKET_IN:
			IRoutingDecision decision = null;
			if (cntx != null) {
				decision = IRoutingDecision.rtStore.get(cntx, IRoutingDecision.CONTEXT_DECISION);
				return this.processPacketInMessage(sw, (OFPacketIn) msg, decision, cntx);
			}
			break;
		default:
			break;
		}
		return Command.CONTINUE;
	}
	@Override
	public void enableFirewall(boolean enabled) {
		logger.info("Setting firewall to {}", enabled);
		this.enabled = enabled;
	}
	@Override
	public List<FirewallRule> getRules() {
		return this.rules;
	}
	@Override
	public List<Map<String, Object>> getStorageRules() {
		ArrayList<Map<String, Object>> l = new ArrayList<Map<String, Object>>();
		try {
			IResultSet resultSet = storageSource.executeQuery(TABLE_NAME, ColumnNames, null, null);
			for (Iterator<IResultSet> it = resultSet.iterator(); it.hasNext();) {
				l.add(it.next().getRow());
			}
		} catch (StorageException e) {
			logger.error("failed to access storage: {}", e.getMessage());
		}
		return l;
	}
	@Override
	public String getSubnetMask() {
		return this.subnet_mask.toString();
	}
	@Override
	public void setSubnetMask(String newMask) {
		if (newMask.trim().isEmpty())
			return;
		this.subnet_mask = IPv4Address.of(newMask.trim());
	}
	@Override
	public synchronized void addRule(FirewallRule rule) {
		rule.ruleid = rule.genID();
		int i = 0;
		for (i = 0; i < this.rules.size(); i++) {
			if (this.rules.get(i).priority >= rule.priority)
				break;
		}
		if (i <= this.rules.size()) {
			this.rules.add(i, rule);
		} else {
			this.rules.add(rule);
		}
		Map<String, Object> entry = new HashMap<String, Object>();
		entry.put(COLUMN_RULEID, Integer.toString(rule.ruleid));
		entry.put(COLUMN_DPID, Long.toString(rule.dpid.getLong()));
		entry.put(COLUMN_IN_PORT, Integer.toString(rule.in_port.getPortNumber()));
		entry.put(COLUMN_DL_SRC, Long.toString(rule.dl_src.getLong()));
		entry.put(COLUMN_DL_DST, Long.toString(rule.dl_dst.getLong()));
		entry.put(COLUMN_DL_TYPE, Integer.toString(rule.dl_type.getValue()));
		entry.put(COLUMN_NW_SRC_PREFIX, Integer.toString(rule.nw_src_prefix_and_mask.getValue().getInt()));
		entry.put(COLUMN_NW_SRC_MASKBITS, Integer.toString(rule.nw_src_prefix_and_mask.getMask().getInt()));
		entry.put(COLUMN_NW_DST_PREFIX, Integer.toString(rule.nw_dst_prefix_and_mask.getValue().getInt()));
		entry.put(COLUMN_NW_DST_MASKBITS, Integer.toString(rule.nw_dst_prefix_and_mask.getMask().getInt()));
		entry.put(COLUMN_NW_PROTO, Short.toString(rule.nw_proto.getIpProtocolNumber()));
		entry.put(COLUMN_TP_SRC, Integer.toString(rule.tp_src.getPort()));
		entry.put(COLUMN_TP_DST, Integer.toString(rule.tp_dst.getPort()));
		entry.put(COLUMN_WILDCARD_DPID, Boolean.toString(rule.any_dpid));
		entry.put(COLUMN_WILDCARD_IN_PORT, Boolean.toString(rule.any_in_port));
		entry.put(COLUMN_WILDCARD_DL_SRC, Boolean.toString(rule.any_dl_src));
		entry.put(COLUMN_WILDCARD_DL_DST, Boolean.toString(rule.any_dl_dst));
		entry.put(COLUMN_WILDCARD_DL_TYPE, Boolean.toString(rule.any_dl_type));
		entry.put(COLUMN_WILDCARD_NW_SRC, Boolean.toString(rule.any_nw_src));
		entry.put(COLUMN_WILDCARD_NW_DST, Boolean.toString(rule.any_nw_dst));
		entry.put(COLUMN_WILDCARD_NW_PROTO, Boolean.toString(rule.any_nw_proto));
		entry.put(COLUMN_WILDCARD_TP_SRC, Boolean.toString(rule.any_tp_src));
		entry.put(COLUMN_WILDCARD_TP_DST, Boolean.toString(rule.any_tp_dst));
		entry.put(COLUMN_PRIORITY, Integer.toString(rule.priority));
		entry.put(COLUMN_ACTION, Integer.toString(rule.action.ordinal()));
		storageSource.insertRow(TABLE_NAME, entry);
	}
	@Override
	public synchronized void deleteRule(int ruleid) {
		Iterator<FirewallRule> iter = this.rules.iterator();
		while (iter.hasNext()) {
			FirewallRule r = iter.next();
			if (r.ruleid == ruleid) {
				iter.remove();
				break;
			}
		}
		storageSource.deleteRow(TABLE_NAME, Integer.toString(ruleid));
	}
	protected RuleMatchPair matchWithRule(IOFSwitch sw, OFPacketIn pi, FloodlightContext cntx) {
		FirewallRule matched_rule = null;
		Ethernet eth = IFloodlightProviderService.bcStore.get(cntx, IFloodlightProviderService.CONTEXT_PI_PAYLOAD);
		AllowDropPair adp = new AllowDropPair(sw.getOFFactory());
		synchronized (rules) {
			Iterator<FirewallRule> iter = this.rules.iterator();
			FirewallRule rule = null;
			while (iter.hasNext()) {
				rule = iter.next();
				if (rule.matchesThisPacket(sw.getId(), (pi.getVersion().compareTo(OFVersion.OF_12) < 0 ? pi.getInPort() : pi.getMatch().get(MatchField.IN_PORT)), eth, adp) == true) {
					matched_rule = rule;
					break;
				}
			}
		}
		RuleMatchPair rmp = new RuleMatchPair();
		rmp.rule = matched_rule;
		if (matched_rule == null) {
			Match.Builder mb = OFFactories.getFactory(pi.getVersion()).buildMatch();
			mb.setExact(MatchField.IN_PORT, (pi.getVersion().compareTo(OFVersion.OF_12) < 0 ? pi.getInPort() : pi.getMatch().get(MatchField.IN_PORT)))
			.setExact(MatchField.ETH_SRC, eth.getSourceMACAddress())
			.setExact(MatchField.ETH_DST, eth.getDestinationMACAddress())
			.setExact(MatchField.ETH_TYPE, eth.getEtherType());
			if (mb.get(MatchField.ETH_TYPE).equals(EthType.IPv4)) {
				IPv4 ipv4 = (IPv4) eth.getPayload();
				mb.setExact(MatchField.IPV4_SRC, ipv4.getSourceAddress())
				.setExact(MatchField.IPV4_DST, ipv4.getDestinationAddress())
				.setExact(MatchField.IP_PROTO, ipv4.getProtocol());
				if (mb.get(MatchField.IP_PROTO).equals(IpProtocol.TCP)) {
					TCP tcp = (TCP) ipv4.getPayload();
					mb.setExact(MatchField.TCP_SRC, tcp.getSourcePort())
					.setExact(MatchField.TCP_DST, tcp.getDestinationPort());
				} else if (mb.get(MatchField.IP_PROTO).equals(IpProtocol.UDP)) {
					UDP udp = (UDP) ipv4.getPayload();
					mb.setExact(MatchField.UDP_SRC, udp.getSourcePort())
					.setExact(MatchField.UDP_DST, udp.getDestinationPort());
				} else {
				}
			}
			rmp.match = mb.build();
		} else if (matched_rule.action == FirewallRule.FirewallAction.DROP) {
			rmp.match = adp.drop.build();
		} else {
			rmp.match = adp.allow.build();
		}
		return rmp;
	}
	protected boolean isIPBroadcast(IPv4Address ip) {
		IPv4Address inv_subnet_mask = subnet_mask.not();
		return ip.and(inv_subnet_mask).equals(inv_subnet_mask);
	}
	public Command processPacketInMessage(IOFSwitch sw, OFPacketIn pi, IRoutingDecision decision, FloodlightContext cntx) {
		Ethernet eth = IFloodlightProviderService.bcStore.get(cntx, IFloodlightProviderService.CONTEXT_PI_PAYLOAD);
		OFPort inPort = (pi.getVersion().compareTo(OFVersion.OF_12) < 0 ? pi.getInPort() : pi.getMatch().get(MatchField.IN_PORT));
		if (eth.isBroadcast() == true) {
			boolean allowBroadcast = true;
			if ((eth.getPayload() instanceof IPv4) && !isIPBroadcast(((IPv4) eth.getPayload()).getDestinationAddress())) {
				allowBroadcast = false;
			}
			if (allowBroadcast == true) {
				if (logger.isTraceEnabled()) {
					logger.trace("Allowing broadcast traffic for PacketIn={}", pi);
				}
				decision = new RoutingDecision(sw.getId(), inPort, 
						IDeviceService.fcStore.get(cntx, IDeviceService.CONTEXT_SRC_DEVICE),
						IRoutingDecision.RoutingAction.MULTICAST);
				decision.addToContext(cntx);
			} else {
				if (logger.isTraceEnabled()) {
					logger.trace("Blocking malformed broadcast traffic for PacketIn={}", pi);
				}
				decision = new RoutingDecision(sw.getId(), inPort,
						IDeviceService.fcStore.get(cntx, IDeviceService.CONTEXT_SRC_DEVICE),
						IRoutingDecision.RoutingAction.DROP);
				decision.addToContext(cntx);
			}
			return Command.CONTINUE;
		}
		if (decision == null) {
			RuleMatchPair rmp = this.matchWithRule(sw, pi, cntx);
			FirewallRule rule = rmp.rule;
			if (rule == null || rule.action == FirewallRule.FirewallAction.DROP) {
				decision = new RoutingDecision(sw.getId(), inPort, 
						IDeviceService.fcStore.get(cntx, IDeviceService.CONTEXT_SRC_DEVICE), 
						IRoutingDecision.RoutingAction.DROP);
				decision.setMatch(rmp.match);
				decision.addToContext(cntx);
				if (logger.isTraceEnabled()) {
					if (rule == null) {
						logger.trace("No firewall rule found for PacketIn={}, blocking flow", pi);
					} else if (rule.action == FirewallRule.FirewallAction.DROP) {
						logger.trace("Deny rule={} match for PacketIn={}", rule, pi);
					}
				}
			} else {
				decision = new RoutingDecision(sw.getId(), inPort, 
						IDeviceService.fcStore.get(cntx, IDeviceService.CONTEXT_SRC_DEVICE),
						IRoutingDecision.RoutingAction.FORWARD_OR_FLOOD);
				decision.setMatch(rmp.match);
				decision.addToContext(cntx);
				if (logger.isTraceEnabled()) {
					logger.trace("Allow rule={} match for PacketIn={}", rule, pi);
				}
			}
		}
		return Command.CONTINUE;
	}
	@Override
	public boolean isEnabled() {
		return enabled;
	}
}

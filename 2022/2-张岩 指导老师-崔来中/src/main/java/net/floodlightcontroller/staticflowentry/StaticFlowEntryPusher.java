package net.floodlightcontroller.staticflowentry;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import net.floodlightcontroller.core.FloodlightContext;
import net.floodlightcontroller.core.HAListenerTypeMarker;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.IHAListener;
import net.floodlightcontroller.core.IOFMessageListener;
import net.floodlightcontroller.core.IOFSwitch;
import net.floodlightcontroller.core.IOFSwitchListener;
import net.floodlightcontroller.core.PortChangeType;
import net.floodlightcontroller.core.internal.IOFSwitchService;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.core.util.AppCookie;
import net.floodlightcontroller.restserver.IRestApiService;
import net.floodlightcontroller.staticflowentry.web.StaticFlowEntryWebRoutable;
import net.floodlightcontroller.storage.IResultSet;
import net.floodlightcontroller.storage.IStorageSourceListener;
import net.floodlightcontroller.storage.IStorageSourceService;
import net.floodlightcontroller.storage.StorageException;
import net.floodlightcontroller.util.ActionUtils;
import net.floodlightcontroller.util.FlowModUtils;
import net.floodlightcontroller.util.InstructionUtils;
import net.floodlightcontroller.util.MatchUtils;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFFlowAdd;
import org.projectfloodlight.openflow.protocol.OFFlowDeleteStrict;
import org.projectfloodlight.openflow.protocol.OFFlowMod;
import org.projectfloodlight.openflow.protocol.OFFlowRemoved;
import org.projectfloodlight.openflow.protocol.OFFlowRemovedReason;
import org.projectfloodlight.openflow.protocol.OFPortDesc;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFType;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.ver10.OFFlowRemovedReasonSerializerVer10;
import org.projectfloodlight.openflow.protocol.ver11.OFFlowRemovedReasonSerializerVer11;
import org.projectfloodlight.openflow.protocol.ver12.OFFlowRemovedReasonSerializerVer12;
import org.projectfloodlight.openflow.protocol.ver13.OFFlowRemovedReasonSerializerVer13;
import org.projectfloodlight.openflow.protocol.ver14.OFFlowRemovedReasonSerializerVer14;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.TableId;
import org.projectfloodlight.openflow.types.U16;
import org.projectfloodlight.openflow.types.U64;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class StaticFlowEntryPusher
implements IOFSwitchListener, IFloodlightModule, IStaticFlowEntryPusherService, IStorageSourceListener, IOFMessageListener {
	protected static Logger log = LoggerFactory.getLogger(StaticFlowEntryPusher.class);
	public static final String StaticFlowName = "staticflowentry";
	public static final int STATIC_FLOW_APP_ID = 10;
	static {
		AppCookie.registerApp(STATIC_FLOW_APP_ID, StaticFlowName);
	}
	public static final String TABLE_NAME = "controller_staticflowtableentry";
	public static final String COLUMN_NAME = "name";
	public static final String COLUMN_SWITCH = "switch";
	public static final String COLUMN_TABLE_ID = "table";
	public static final String COLUMN_ACTIVE = "active";
	public static final String COLUMN_IDLE_TIMEOUT = "idle_timeout";
	public static final String COLUMN_HARD_TIMEOUT = "hard_timeout";
	public static final String COLUMN_PRIORITY = "priority";
	public static final String COLUMN_COOKIE = "cookie";
	public static final String COLUMN_IN_PORT = MatchUtils.STR_IN_PORT;
	public static final String COLUMN_DL_SRC = MatchUtils.STR_DL_SRC;
	public static final String COLUMN_DL_DST = MatchUtils.STR_DL_DST;
	public static final String COLUMN_DL_VLAN = MatchUtils.STR_DL_VLAN;
	public static final String COLUMN_DL_VLAN_PCP = MatchUtils.STR_DL_VLAN_PCP;
	public static final String COLUMN_DL_TYPE = MatchUtils.STR_DL_TYPE;
	public static final String COLUMN_NW_TOS = MatchUtils.STR_NW_TOS;
	public static final String COLUMN_NW_ECN = MatchUtils.STR_NW_ECN;
	public static final String COLUMN_NW_DSCP = MatchUtils.STR_NW_DSCP;
	public static final String COLUMN_NW_PROTO = MatchUtils.STR_NW_PROTO;
	public static final String COLUMN_NW_DST = MatchUtils.STR_NW_DST;
	public static final String COLUMN_SCTP_SRC = MatchUtils.STR_SCTP_SRC;
	public static final String COLUMN_SCTP_DST = MatchUtils.STR_SCTP_DST;
	public static final String COLUMN_UDP_SRC = MatchUtils.STR_UDP_SRC;
	public static final String COLUMN_UDP_DST = MatchUtils.STR_UDP_DST;
	public static final String COLUMN_TCP_SRC = MatchUtils.STR_TCP_SRC;
	public static final String COLUMN_TCP_DST = MatchUtils.STR_TCP_DST;
	public static final String COLUMN_TP_DST = MatchUtils.STR_TP_DST;
	public static final String COLUMN_ICMP_TYPE = MatchUtils.STR_ICMP_TYPE;
	public static final String COLUMN_ICMP_CODE = MatchUtils.STR_ICMP_CODE;
	public static final String COLUMN_ARP_OPCODE = MatchUtils.STR_ARP_OPCODE;
	public static final String COLUMN_ARP_SHA = MatchUtils.STR_ARP_SHA;
	public static final String COLUMN_ARP_DHA = MatchUtils.STR_ARP_DHA;
	public static final String COLUMN_ARP_SPA = MatchUtils.STR_ARP_SPA;
	public static final String COLUMN_ARP_DPA = MatchUtils.STR_ARP_DPA;
	public static final String COLUMN_NW6_SRC = MatchUtils.STR_IPV6_SRC;
	public static final String COLUMN_NW6_DST = MatchUtils.STR_IPV6_DST;
	public static final String COLUMN_IPV6_FLOW_LABEL = MatchUtils.STR_IPV6_FLOW_LABEL;
	public static final String COLUMN_ICMP6_TYPE = MatchUtils.STR_ICMPV6_TYPE;
	public static final String COLUMN_ICMP6_CODE = MatchUtils.STR_ICMPV6_CODE;
	public static final String COLUMN_ND_SLL = MatchUtils.STR_IPV6_ND_SSL;
	public static final String COLUMN_ND_TLL = MatchUtils.STR_IPV6_ND_TTL;
	public static final String COLUMN_ND_TARGET = MatchUtils.STR_IPV6_ND_TARGET;	
	public static final String COLUMN_MPLS_LABEL = MatchUtils.STR_MPLS_LABEL;
	public static final String COLUMN_MPLS_TC = MatchUtils.STR_MPLS_TC;
	public static final String COLUMN_MPLS_BOS = MatchUtils.STR_MPLS_BOS;
	public static final String COLUMN_METADATA = MatchUtils.STR_METADATA;
	public static final String COLUMN_TUNNEL_ID = MatchUtils.STR_TUNNEL_ID;
	public static final String COLUMN_PBB_ISID = MatchUtils.STR_PBB_ISID;
	public static final String COLUMN_ACTIONS = "actions";
	public static final String COLUMN_INSTR_WRITE_METADATA = InstructionUtils.STR_WRITE_METADATA;
	public static final String COLUMN_INSTR_WRITE_ACTIONS = InstructionUtils.STR_WRITE_ACTIONS;
	public static final String COLUMN_INSTR_APPLY_ACTIONS = InstructionUtils.STR_APPLY_ACTIONS;
	public static final String COLUMN_INSTR_CLEAR_ACTIONS = InstructionUtils.STR_CLEAR_ACTIONS;
	public static final String COLUMN_INSTR_GOTO_METER = InstructionUtils.STR_GOTO_METER;
	public static final String COLUMN_INSTR_EXPERIMENTER = InstructionUtils.STR_EXPERIMENTER;
	public static String ColumnNames[] = { COLUMN_NAME, COLUMN_SWITCH,
		COLUMN_PRIORITY, COLUMN_COOKIE, COLUMN_IN_PORT,
		COLUMN_DL_SRC, COLUMN_DL_DST, COLUMN_DL_VLAN, COLUMN_DL_VLAN_PCP,
		COLUMN_DL_TYPE, COLUMN_NW_TOS, COLUMN_NW_PROTO, COLUMN_NW_SRC,
		COLUMN_NW_DST, COLUMN_TP_SRC, COLUMN_TP_DST,
		COLUMN_SCTP_SRC, COLUMN_SCTP_DST, 
		COLUMN_UDP_SRC, COLUMN_UDP_DST, COLUMN_TCP_SRC, COLUMN_TCP_DST,
		COLUMN_ICMP_TYPE, COLUMN_ICMP_CODE, 
		COLUMN_ARP_OPCODE, COLUMN_ARP_SHA, COLUMN_ARP_DHA, 
		COLUMN_ARP_SPA, COLUMN_ARP_DPA,
		COLUMN_NW6_SRC, COLUMN_NW6_DST, COLUMN_ICMP6_TYPE, COLUMN_ICMP6_CODE, 
		COLUMN_IPV6_FLOW_LABEL, COLUMN_ND_SLL, COLUMN_ND_TLL, COLUMN_ND_TARGET,		
		COLUMN_MPLS_LABEL, COLUMN_MPLS_TC, COLUMN_MPLS_BOS, 
		COLUMN_METADATA, COLUMN_TUNNEL_ID, COLUMN_PBB_ISID,
		COLUMN_ACTIONS,
		COLUMN_INSTR_GOTO_TABLE, COLUMN_INSTR_WRITE_METADATA,
		COLUMN_INSTR_WRITE_ACTIONS, COLUMN_INSTR_APPLY_ACTIONS,
		COLUMN_INSTR_CLEAR_ACTIONS, COLUMN_INSTR_GOTO_METER,
		COLUMN_INSTR_EXPERIMENTER
	};
	protected IFloodlightProviderService floodlightProviderService;
	protected IOFSwitchService switchService;
	protected IStorageSourceService storageSourceService;
	protected IRestApiService restApiService;
	private IHAListener haListener;
	protected Map<String, Map<String, OFFlowMod>> entriesFromStorage;
	protected Map<String, String> entry2dpid;
	class FlowModSorter implements Comparator<String> {
		private String dpid;
		public FlowModSorter(String dpid) {
			this.dpid = dpid;
		}
		@Override
		public int compare(String o1, String o2) {
			OFFlowMod f1 = entriesFromStorage.get(dpid).get(o1);
			OFFlowMod f2 = entriesFromStorage.get(dpid).get(o2);
				return o1.compareTo(o2);
			return U16.of(f1.getPriority()).getValue() - U16.of(f2.getPriority()).getValue();
		}
	};
	public int countEntries() {
		int size = 0;
		if (entriesFromStorage == null)
			return 0;
		for (String ofswitch : entriesFromStorage.keySet())
			size += entriesFromStorage.get(ofswitch).size();
		return size;
	}
	public IFloodlightProviderService getFloodlightProvider() {
		return floodlightProviderService;
	}
	public void setFloodlightProvider(IFloodlightProviderService floodlightProviderService) {
		this.floodlightProviderService = floodlightProviderService;
	}
	public void setStorageSource(IStorageSourceService storageSourceService) {
		this.storageSourceService = storageSourceService;
	}
	protected void sendEntriesToSwitch(DatapathId switchId) {
		IOFSwitch sw = switchService.getSwitch(switchId);
		if (sw == null)
			return;
		String stringId = sw.getId().toString();
		if ((entriesFromStorage != null) && (entriesFromStorage.containsKey(stringId))) {
			Map<String, OFFlowMod> entries = entriesFromStorage.get(stringId);
			List<String> sortedList = new ArrayList<String>(entries.keySet());
			Collections.sort( sortedList, new FlowModSorter(stringId));
			for (String entryName : sortedList) {
				OFFlowMod flowMod = entries.get(entryName);
				if (flowMod != null) {
					if (log.isDebugEnabled()) {
						log.debug("Pushing static entry {} for {}", stringId, entryName);
					}
					writeFlowModToSwitch(sw, flowMod);
				}
			}
		}
	}
	protected Map<String, String> computeEntry2DpidMap(
			Map<String, Map<String, OFFlowMod>> map) {
		Map<String, String> ret = new ConcurrentHashMap<String, String>();
		for(String dpid : map.keySet()) {
			for( String entry: map.get(dpid).keySet())
				ret.put(entry, dpid);
		}
		return ret;
	}
	private Map<String, Map<String, OFFlowMod>> readEntriesFromStorage() {
		Map<String, Map<String, OFFlowMod>> entries = new ConcurrentHashMap<String, Map<String, OFFlowMod>>();
		try {
			Map<String, Object> row;
			IResultSet resultSet = storageSourceService.executeQuery(TABLE_NAME, ColumnNames, null, null);
			for (Iterator<IResultSet> it = resultSet.iterator(); it.hasNext();) {
				row = it.next().getRow();
				parseRow(row, entries);
			}
		} catch (StorageException e) {
			log.error("failed to access storage: {}", e.getMessage());
		}
		return entries;
	}
	void parseRow(Map<String, Object> row, Map<String, Map<String, OFFlowMod>> entries) {
		String switchName = null;
		String entryName = null;
		StringBuffer matchString = new StringBuffer();
		OFFlowMod.Builder fmb = null; 
		if (!row.containsKey(COLUMN_SWITCH) || !row.containsKey(COLUMN_NAME)) {
			log.debug("skipping entry with missing required 'switch' or 'name' entry: {}", row);
			return;
		}
		try {
			switchName = (String) row.get(COLUMN_SWITCH);
			entryName = (String) row.get(COLUMN_NAME);
			if (!entries.containsKey(switchName)) {
				entries.put(switchName, new HashMap<String, OFFlowMod>());
			}
			try {
				fmb = OFFactories.getFactory(switchService.getSwitch(DatapathId.of(switchName)).getOFFactory().getVersion()).buildFlowModify();
			} catch (NullPointerException e) {
				storageSourceService.deleteRowAsync(TABLE_NAME, entryName);
				log.error("Deleting entry {}. Switch {} was not connected to the controller, and we need to know the OF protocol version to compose the flow mod.", entryName, switchName);
				return;
			}
			StaticFlowEntries.initDefaultFlowMod(fmb, entryName);
			for (String key : row.keySet()) {
				if (row.get(key) == null) {
					continue;
				}
				if (key.equals(COLUMN_SWITCH) || key.equals(COLUMN_NAME) || key.equals("id")) {
				}
				if (key.equals(COLUMN_ACTIVE)) {
					if  (!Boolean.valueOf((String) row.get(COLUMN_ACTIVE))) {
						log.debug("skipping inactive entry {} for switch {}", entryName, switchName);
						return;
					}
				} else if (key.equals(COLUMN_HARD_TIMEOUT)) {
					fmb.setHardTimeout(Integer.valueOf((String) row.get(COLUMN_HARD_TIMEOUT)));
				} else if (key.equals(COLUMN_IDLE_TIMEOUT)) {
					fmb.setIdleTimeout(Integer.valueOf((String) row.get(COLUMN_IDLE_TIMEOUT)));
				} else if (key.equals(COLUMN_TABLE_ID)) {
					if (fmb.getVersion().compareTo(OFVersion.OF_10) > 0) {
					} else {
						log.error("Table not supported in OpenFlow 1.0");
					}
				} else if (key.equals(COLUMN_ACTIONS)) {
					ActionUtils.fromString(fmb, (String) row.get(COLUMN_ACTIONS), log);
				} else if (key.equals(COLUMN_COOKIE)) {
					fmb.setCookie(StaticFlowEntries.computeEntryCookie(Integer.valueOf((String) row.get(COLUMN_COOKIE)), entryName));
				} else if (key.equals(COLUMN_PRIORITY)) {
					fmb.setPriority(U16.t(Integer.valueOf((String) row.get(COLUMN_PRIORITY))));
				} else if (key.equals(COLUMN_INSTR_APPLY_ACTIONS)) {
					InstructionUtils.applyActionsFromString(fmb, (String) row.get(COLUMN_INSTR_APPLY_ACTIONS), log);
				} else if (key.equals(COLUMN_INSTR_CLEAR_ACTIONS)) {
					InstructionUtils.clearActionsFromString(fmb, (String) row.get(COLUMN_INSTR_CLEAR_ACTIONS), log);
				} else if (key.equals(COLUMN_INSTR_EXPERIMENTER)) {
					InstructionUtils.experimenterFromString(fmb, (String) row.get(COLUMN_INSTR_EXPERIMENTER), log);
				} else if (key.equals(COLUMN_INSTR_GOTO_METER)) {
					InstructionUtils.meterFromString(fmb, (String) row.get(COLUMN_INSTR_GOTO_METER), log);
				} else if (key.equals(COLUMN_INSTR_GOTO_TABLE)) {
					InstructionUtils.gotoTableFromString(fmb, (String) row.get(COLUMN_INSTR_GOTO_TABLE), log);
				} else if (key.equals(COLUMN_INSTR_WRITE_ACTIONS)) {
					InstructionUtils.writeActionsFromString(fmb, (String) row.get(COLUMN_INSTR_WRITE_ACTIONS), log);
				} else if (key.equals(COLUMN_INSTR_WRITE_METADATA)) {
					InstructionUtils.writeMetadataFromString(fmb, (String) row.get(COLUMN_INSTR_WRITE_METADATA), log);
					if (matchString.length() > 0) {
						matchString.append(",");
					}
					matchString.append(key + "=" + row.get(key).toString());
				}
			}
		} catch (ClassCastException e) {
			if (entryName != null && switchName != null) {
				log.warn("Skipping entry {} on switch {} with bad data : " + e.getMessage(), entryName, switchName);
			} else {
				log.warn("Skipping entry with bad data: {} :: {} ", e.getMessage(), e.getStackTrace());
			}
		}
		String match = matchString.toString();
		try {
			fmb.setMatch(MatchUtils.fromString(match, fmb.getVersion()));
		} catch (IllegalArgumentException e) {
			log.error(e.toString());
			log.error("Ignoring flow entry {} on switch {} with illegal OFMatch() key: " + match, entryName, switchName);
			return;
		} catch (Exception e) {
			log.error("OF version incompatible for the match: " + match);
			e.printStackTrace();
			return;
		}
	}
	@Override
	public void switchAdded(DatapathId switchId) {
		log.debug("Switch {} connected; processing its static entries",
				switchId.toString());
		sendEntriesToSwitch(switchId);
	}
	@Override
	public void switchRemoved(DatapathId switchId) {
	}
	@Override
	public void switchActivated(DatapathId switchId) {
	}
	@Override
	public void switchChanged(DatapathId switchId) {
	}
	@Override
	public void switchPortChanged(DatapathId switchId,
			OFPortDesc port,
			PortChangeType type) {
	}
	@Override
	public void rowsModified(String tableName, Set<Object> rowKeys) {
		log.debug("Modifying Table {}", tableName);
		HashMap<String, Map<String, OFFlowMod>> entriesToAdd =
				new HashMap<String, Map<String, OFFlowMod>>();
		for (Object key: rowKeys) {
			IResultSet resultSet = storageSourceService.getRow(tableName, key);
			Iterator<IResultSet> it = resultSet.iterator();
			while (it.hasNext()) {
				Map<String, Object> row = it.next().getRow();
				parseRow(row, entriesToAdd);
			}
		}
		for (String dpid : entriesToAdd.keySet()) {
			if (!entriesFromStorage.containsKey(dpid))
				entriesFromStorage.put(dpid, new HashMap<String, OFFlowMod>());
			List<OFMessage> outQueue = new ArrayList<OFMessage>();
			for (String entry : entriesToAdd.get(dpid).keySet()) {
				OFFlowMod newFlowMod = entriesToAdd.get(dpid).get(entry);
				OFFlowMod oldFlowMod = null;
				String dpidOldFlowMod = entry2dpid.get(entry);
				if (dpidOldFlowMod != null) {
					oldFlowMod = entriesFromStorage.get(dpidOldFlowMod).remove(entry);
				}
				if (oldFlowMod != null && newFlowMod != null) { 
					if (oldFlowMod.getMatch().equals(newFlowMod.getMatch())
							&& oldFlowMod.getCookie().equals(newFlowMod.getCookie())
							&& oldFlowMod.getPriority() == newFlowMod.getPriority()
							&& dpidOldFlowMod.equalsIgnoreCase(dpid)) {
						log.debug("ModifyStrict SFP Flow");
						entriesFromStorage.get(dpid).put(entry, newFlowMod);
						entry2dpid.put(entry, dpid);
						newFlowMod = FlowModUtils.toFlowModifyStrict(newFlowMod);
						outQueue.add(newFlowMod);
					} else {
						log.debug("DeleteStrict and Add SFP Flow");
						oldFlowMod = FlowModUtils.toFlowDeleteStrict(oldFlowMod);
						OFFlowAdd addTmp = FlowModUtils.toFlowAdd(newFlowMod);
						if (dpidOldFlowMod.equals(dpid)) {
							outQueue.add(oldFlowMod);
							outQueue.add(addTmp); 
						} else {
							writeOFMessageToSwitch(DatapathId.of(dpidOldFlowMod), oldFlowMod);
							writeOFMessageToSwitch(DatapathId.of(dpid), FlowModUtils.toFlowAdd(newFlowMod)); 
						}
						entriesFromStorage.get(dpid).put(entry, addTmp);
						entry2dpid.put(entry, dpid);			
					}
				} else if (newFlowMod != null && oldFlowMod == null) {
					log.debug("Add SFP Flow");
					OFFlowAdd addTmp = FlowModUtils.toFlowAdd(newFlowMod);
					entriesFromStorage.get(dpid).put(entry, addTmp);
					entry2dpid.put(entry, dpid);
					outQueue.add(addTmp);
				} else if (newFlowMod == null) { 
					entriesFromStorage.get(dpid).remove(entry);
					entry2dpid.remove(entry);
				}
			}
			writeOFMessagesToSwitch(DatapathId.of(dpid), outQueue);
		}
	}
	@Override
	public void rowsDeleted(String tableName, Set<Object> rowKeys) {
		if (log.isDebugEnabled()) {
			log.debug("Deleting from table {}", tableName);
		}
		for(Object obj : rowKeys) {
			if (!(obj instanceof String)) {
				log.debug("Tried to delete non-string key {}; ignoring", obj);
				continue;
			}
			deleteStaticFlowEntry((String) obj);
		}
	}
	private void deleteStaticFlowEntry(String entryName) {
		String dpid = entry2dpid.remove(entryName);
		if (dpid == null) {
			return;
		}
		if (log.isDebugEnabled()) {
			log.debug("Sending delete flow mod for flow {} for switch {}", entryName, dpid);
		}
		if (switchService.getSwitch(DatapathId.of(dpid)) != null) {
			OFFlowDeleteStrict flowMod = FlowModUtils.toFlowDeleteStrict(entriesFromStorage.get(dpid).get(entryName));
			if (entriesFromStorage.containsKey(dpid) && entriesFromStorage.get(dpid).containsKey(entryName)) {
				entriesFromStorage.get(dpid).remove(entryName);
			} else {
				log.debug("Tried to delete non-existent entry {} for switch {}", entryName, dpid);
				return;
			}
			writeFlowModToSwitch(DatapathId.of(dpid), flowMod);
		} else {
			log.debug("Not sending flow delete for disconnected switch.");
		}
		return;
	}
	private void writeOFMessagesToSwitch(DatapathId dpid, List<OFMessage> messages) {
		IOFSwitch ofswitch = switchService.getSwitch(dpid);
			if (log.isDebugEnabled()) {
				log.debug("Sending {} new entries to {}", messages.size(), dpid);
			}
			ofswitch.write(messages);
		}
	}
	private void writeOFMessageToSwitch(DatapathId dpid, OFMessage message) {
		IOFSwitch ofswitch = switchService.getSwitch(dpid);
			if (log.isDebugEnabled()) {
				log.debug("Sending 1 new entries to {}", dpid.toString());
			}
			ofswitch.write(message);
		}
	}
	private void writeFlowModToSwitch(DatapathId dpid, OFFlowMod flowMod) {
		IOFSwitch ofSwitch = switchService.getSwitch(dpid);
		if (ofSwitch == null) {
			if (log.isDebugEnabled()) {
				log.debug("Not deleting key {} :: switch {} not connected", dpid.toString());
			}
			return;
		}
		writeFlowModToSwitch(ofSwitch, flowMod);
	}
	private void writeFlowModToSwitch(IOFSwitch sw, OFFlowMod flowMod) {
		sw.write(flowMod);
	}
	@Override
	public String getName() {
		return StaticFlowName;
	}
	public Command handleFlowRemoved(IOFSwitch sw, OFFlowRemoved msg, FloodlightContext cntx) {
		U64 cookie = msg.getCookie();
		if (AppCookie.extractApp(cookie) == STATIC_FLOW_APP_ID) {
			OFFlowRemovedReason reason = null;
			switch (msg.getVersion()) {
			case OF_10:
				reason = OFFlowRemovedReasonSerializerVer10.ofWireValue((byte) msg.getReason());
				break;
			case OF_11:
				reason = OFFlowRemovedReasonSerializerVer11.ofWireValue((byte) msg.getReason());
				break;
			case OF_12:
				reason = OFFlowRemovedReasonSerializerVer12.ofWireValue((byte) msg.getReason());
				break;
			case OF_13:
				reason = OFFlowRemovedReasonSerializerVer13.ofWireValue((byte) msg.getReason());
				break;
			case OF_14:
				reason = OFFlowRemovedReasonSerializerVer14.ofWireValue((byte) msg.getReason());
				break;
			default:
				log.debug("OpenFlow version {} unsupported for OFFlowRemovedReasonSerializerVerXX", msg.getVersion());
				break;
			}
			if (reason != null) {
				if (OFFlowRemovedReason.DELETE == reason) {
					log.error("Got a FlowRemove message for a infinite " + 
							"timeout flow: {} from switch {}", msg, sw);
				} else if (OFFlowRemovedReason.HARD_TIMEOUT == reason || OFFlowRemovedReason.IDLE_TIMEOUT == reason) {
					log.debug("Received an IDLE or HARD timeout for an SFP flow. Removing it from the SFP DB.");
					String flowToRemove = null;
					Map<String, OFFlowMod> flowsByName = getFlows(sw.getId());
					for (Map.Entry<String, OFFlowMod> entry : flowsByName.entrySet()) {
						if (msg.getCookie().equals(entry.getValue().getCookie()) &&
								(msg.getVersion().compareTo(OFVersion.OF_12) < 0 ? true : msg.getHardTimeout() == entry.getValue().getHardTimeout()) &&
								msg.getIdleTimeout() == entry.getValue().getIdleTimeout() &&
								msg.getMatch().equals(entry.getValue().getMatch()) &&
								msg.getPriority() == entry.getValue().getPriority() &&
								(msg.getVersion().compareTo(OFVersion.OF_10) == 0 ? true : msg.getTableId().equals(entry.getValue().getTableId()))
								) {
							flowToRemove = entry.getKey();
							break;
						}
					}
					log.debug("Flow to Remove: {}", flowToRemove);
					if (flowToRemove != null) {
						deleteFlow(flowToRemove);
					}
				}
				return Command.STOP;
			}
		}
		return Command.CONTINUE;
	}
	@Override
	public Command receive(IOFSwitch sw, OFMessage msg, FloodlightContext cntx) {
		switch (msg.getType()) {
		case FLOW_REMOVED:
			return handleFlowRemoved(sw, (OFFlowRemoved) msg, cntx);
		default:
			return Command.CONTINUE;
		}
	}
	@Override
	public boolean isCallbackOrderingPrereq(OFType type, String name) {
	}
	@Override
	public boolean isCallbackOrderingPostreq(OFType type, String name) {
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleServices() {
		Collection<Class<? extends IFloodlightService>> l =
				new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IStaticFlowEntryPusherService.class);
		return l;
	}
	@Override
	public Map<Class<? extends IFloodlightService>, IFloodlightService> getServiceImpls() {
		Map<Class<? extends IFloodlightService>,
		IFloodlightService> m =
		new HashMap<Class<? extends IFloodlightService>,
		IFloodlightService>();
		m.put(IStaticFlowEntryPusherService.class, this);
		return m;
	}
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleDependencies() {
		Collection<Class<? extends IFloodlightService>> l =
				new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IFloodlightProviderService.class);
		l.add(IOFSwitchService.class);
		l.add(IStorageSourceService.class);
		l.add(IRestApiService.class);
		return l;
	}
	@Override
	public void init(FloodlightModuleContext context) throws FloodlightModuleException {
		floodlightProviderService = context.getServiceImpl(IFloodlightProviderService.class);
		switchService = context.getServiceImpl(IOFSwitchService.class);
		storageSourceService = context.getServiceImpl(IStorageSourceService.class);
		restApiService = context.getServiceImpl(IRestApiService.class);
		haListener = new HAListenerDelegate();
	} 
	@Override
	public void startUp(FloodlightModuleContext context) {
		floodlightProviderService.addOFMessageListener(OFType.FLOW_REMOVED, this);
		switchService.addOFSwitchListener(this);
		floodlightProviderService.addHAListener(this.haListener);
		storageSourceService.createTable(TABLE_NAME, null);
		storageSourceService.setTablePrimaryKeyName(TABLE_NAME, COLUMN_NAME);
		storageSourceService.addListener(TABLE_NAME, this);
		entriesFromStorage = readEntriesFromStorage();
		entry2dpid = computeEntry2DpidMap(entriesFromStorage);
		restApiService.addRestletRoutable(new StaticFlowEntryWebRoutable());
	}
	@Override
	public void addFlow(String name, OFFlowMod fm, DatapathId swDpid) {
		try {
			Map<String, Object> fmMap = StaticFlowEntries.flowModToStorageEntry(fm, swDpid.toString(), name);
			storageSourceService.insertRowAsync(TABLE_NAME, fmMap);
		} catch (Exception e) {
			log.error("Error! Check the fields specified for the flow.Make sure IPv4 fields are not mixed with IPv6 fields or all "
					+ "mandatory fields are specified. ");
		}
	}
	@Override
	public void deleteFlow(String name) {
		storageSourceService.deleteRowAsync(TABLE_NAME, name);
	}
	@Override
	public void deleteAllFlows() {
		for (String entry : entry2dpid.keySet()) {
			deleteFlow(entry);
		}
        FIXME: Since the OF spec 1.0 is not clear on how
        to match on cookies. Once all switches come to a
        common implementation we can possibly re-enable this
        fix.
        Set<String> swSet = new HashSet<String>();
        for (String dpid : entry2dpid.values()) {
            if (!swSet.contains(dpid)) {
                swSet.add(dpid);
                sendDeleteByCookie(HexString.toLong(dpid));
            }
        }
        entry2dpid.clear();
        for (Map<String, OFFlowMod> eMap : entriesFromStorage.values()) {
            eMap.clear();
        }
        storageSource.deleteMatchingRowsAsync(TABLE_NAME, null);
	}
	@Override
	public void deleteFlowsForSwitch(DatapathId dpid) {
		String sDpid = dpid.toString();
		for (Entry<String, String> e : entry2dpid.entrySet()) {
			if (e.getValue().equals(sDpid))
				deleteFlow(e.getKey());
		}
        FIXME: Since the OF spec 1.0 is not clear on how
        to match on cookies. Once all switches come to a
        common implementation we can possibly re-enable this
        fix.
        String sDpid = HexString.toHexString(dpid);
        Map<String, OFFlowMod> sMap = entriesFromStorage.get(sDpid);
        if (sMap != null) {
            for (String entryName : sMap.keySet()) {
                entry2dpid.remove(entryName);
                deleteFlow(entryName);
            }
            sMap.clear();
        } else {
            log.warn("Map of storage entries for switch {} was null", sDpid);
        }
	}
    FIXME: Since the OF spec 1.0 is not clear on how
    to match on cookies. Once all switches come to a
    common implementation we can possibly re-enable this
    fix.
    private void sendDeleteByCookie(long dpid) {
        if (log.isDebugEnabled())
            log.debug("Deleting all static flows on switch {}", HexString.toHexString(dpid));
        IOFSwitch sw = floodlightProvider.getSwitch(dpid);
        if (sw == null) {
            log.warn("Tried to delete static flows for non-existant switch {}",
                    HexString.toHexString(dpid));
            return;
        }
        OFFlowMod fm = (OFFlowMod) floodlightProvider.getOFMessageFactory().
                getMessage(OFType.FLOW_MOD);
        OFMatch ofm = new OFMatch();
        fm.setMatch(ofm);
        fm.setCookie(AppCookie.makeCookie(StaticFlowEntryPusher.STATIC_FLOW_APP_ID, 0));
        fm.setCommand(OFFlowMod.OFPFC_DELETE);
        fm.setOutPort(OFPort.OFPP_NONE);
        try {
            sw.write(fm, null);
            sw.flush();
        } catch (IOException e1) {
            log.error("Error deleting all flows for switch {}:\n {}",
                    HexString.toHexString(dpid), e1.getMessage());
            return;
        }
    }
	@Override
	public Map<String, Map<String, OFFlowMod>> getFlows() {
		return entriesFromStorage;
	}
	@Override
	public Map<String, OFFlowMod> getFlows(DatapathId dpid) {
		return entriesFromStorage.get(dpid.toString());
	}
	private class HAListenerDelegate implements IHAListener {
		@Override
		public void transitionToActive() {
			log.debug("Re-reading static flows from storage due " +
					"to HA change from STANDBY->ACTIVE");
			entriesFromStorage = readEntriesFromStorage();
			entry2dpid = computeEntry2DpidMap(entriesFromStorage);
		}
		@Override
		public void controllerNodeIPsChanged(
				Map<String, String> curControllerNodeIPs,
				Map<String, String> addedControllerNodeIPs,
				Map<String, String> removedControllerNodeIPs) {
		}
		@Override
		public String getName() {
			return StaticFlowEntryPusher.this.getName();
		}
		@Override
		public boolean isCallbackOrderingPrereq(HAListenerTypeMarker type,
				String name) {
			return false;
		}
		@Override
		public boolean isCallbackOrderingPostreq(HAListenerTypeMarker type,
				String name) {
			return false;
		}
		@Override
		public void transitionToStandby() {	
			log.debug("Controller is now in STANDBY role. Clearing static flow entries from store.");
			deleteAllFlows();
		}
	}
}

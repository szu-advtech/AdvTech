package net.floodlightcontroller.util;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import javax.annotation.Nonnull;
import net.floodlightcontroller.core.IOFSwitch;
import org.projectfloodlight.openflow.protocol.OFBucket;
import org.projectfloodlight.openflow.protocol.OFFlowAdd;
import org.projectfloodlight.openflow.protocol.OFGroupAdd;
import org.projectfloodlight.openflow.protocol.OFGroupType;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.action.OFAction;
import org.projectfloodlight.openflow.protocol.instruction.OFInstruction;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.protocol.match.MatchFields;
import org.projectfloodlight.openflow.types.OFBufferId;
import org.projectfloodlight.openflow.types.OFGroup;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.OFVlanVidMatch;
import org.projectfloodlight.openflow.types.TableId;
import org.projectfloodlight.openflow.types.U16;
import org.projectfloodlight.openflow.types.U32;
import org.projectfloodlight.openflow.types.U64;
import org.projectfloodlight.openflow.types.VlanVid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class OFDPAUtils {
	private OFDPAUtils() {}
	private static final Logger log = LoggerFactory.getLogger(OFDPAUtils.class);
	public static final int PRIORITY = 1000;
	public static final int DLF_PRIORITY = 0;
	public static final int HARD_TIMEOUT = 0;
	public static final int IDLE_TIMEOUT = 0;
	private static class OFDPAGroupType {
	}
	private static class L2OverlaySubType {
		private static final int L2_OVERLAY_FLOOD_OVER_UNICAST_TUNNELS = 0;
		private static final int L2_OVERLAY_FLOOD_OVER_MULTICAST_TUNNELS = 1;
		private static final int L2_OVERLAY_MULTICAST_OVER_UNICAST_TUNNELS = 2;
		private static final int L2_OVERLAY_MULTICAST_OVER_MULTICAST_TUNNELS = 3;
	}
	private static class MPLSSubType {
		private static final int MPLS_INTERFACE = 0;
		private static final int MPLS_L2_VPN_LABEL = 1;
		private static final int MPLS_L3_VPN_LABEL = 2;
		private static final int MPLS_TUNNEL_LABEL_1 = 3;
		private static final int MPLS_TUNNEL_LABEL_2 = 4;
		private static final int MPLS_SWAP_LABEL = 5;
		private static final int MPLS_FAST_FAILOVER = 6;
		private static final int MPLS_ECMP = 8;
		private static final int MPLS_L2_TAG = 10;
	}
	public static class Tables {
		public static final TableId INGRESS_PORT = TableId.of(0);
		public static final TableId VLAN = TableId.of(10);
		public static final TableId TERMINATION_MAC = TableId.of(20);
		public static final TableId UNICAST_ROUTING = TableId.of(30);
		public static final TableId MULITCAST_ROUTING = TableId.of(40);
		public static final TableId BRIDGING = TableId.of(50);
		public static final TableId POLICY_ACL = TableId.of(60);
	}
	private static final List<MatchFields> ALLOWED_MATCHES = 
			Collections.unmodifiableList(
					Arrays.asList(
							MatchFields.IN_PORT,
							MatchFields.ETH_SRC, 
							MatchFields.ETH_DST,
							MatchFields.ETH_TYPE,
							MatchFields.VLAN_VID,
							MatchFields.VLAN_PCP,
							MatchFields.TUNNEL_ID,			
							MatchFields.IP_PROTO,
							MatchFields.IPV4_SRC,
							MatchFields.IPV4_DST,
							MatchFields.IP_DSCP,
							MatchFields.IP_ECN,			
							MatchFields.ARP_SPA,
							MatchFields.ICMPV4_CODE,
							MatchFields.ICMPV4_TYPE,
							MatchFields.IPV6_SRC,
							MatchFields.IPV6_DST,
							MatchFields.IPV6_FLABEL,
							MatchFields.ICMPV6_CODE,
							MatchFields.ICMPV6_TYPE,
							MatchFields.TCP_SRC,
							MatchFields.TCP_DST,
							MatchFields.UDP_SRC,
							MatchFields.UDP_DST,
							MatchFields.SCTP_SRC,
							MatchFields.SCTP_DST
							)
					);
	public static boolean isOFDPASwitch(IOFSwitch s) {
		if (s.getSwitchDescription().getSoftwareDescription().toLowerCase().contains("of-dpa")) {
			return true;
		} else {
			return false;
		}
	}
	public static List<MatchFields> getSupportedMatchFields() {
		return ALLOWED_MATCHES;
	}
	public static List<MatchFields> checkMatchFields(Match m) {
		List<MatchFields> unsupported = null;
		Iterator<MatchField<?>> mfi = m.getMatchFields().iterator();
		while (mfi.hasNext()) {
			MatchField<?> mf = mfi.next();
			if (!getSupportedMatchFields().contains(mf.id)) {
				if (unsupported == null) {
					unsupported = new ArrayList<MatchFields>();
				}
				unsupported.add(mf.id);
			}
		}
		return unsupported;
	}
	public static boolean addLearningSwitchPrereqs(@Nonnull IOFSwitch sw, VlanVid vlan, @Nonnull List<OFPortModeTuple> ports) {
		return addLearningSwitchPrereqGroups(sw, vlan, ports) && addLearningSwitchPrereqFlows(sw, vlan, ports);
	}
	private static boolean addLearningSwitchPrereqGroups(@Nonnull IOFSwitch sw, VlanVid vlan, @Nonnull List<OFPortModeTuple> ports) {
		if (sw == null) {
			throw new NullPointerException("Switch cannot be null.");
		}
		if (vlan == null) {
		} else if (vlan.equals(VlanVid.ofVlan(1))) {
			throw new IllegalArgumentException("VLAN cannot be 1. VLAN 1 is an reserved VLAN for internal use inside the OFDPA switch.");
		}
		if (ports == null) {
			throw new NullPointerException("List of ports cannot be null. Must specify at least 2 valid switch ports.");
		} else if (ports.size() < 2) {
			throw new IllegalArgumentException("List of ports must contain at least 2 valid switch ports.");
		} else {
			for (OFPortModeTuple p : ports) {
				if (sw.getOFFactory().getVersion().equals(OFVersion.OF_10) && (sw.getPort(p.getPort()) == null || p.getPort().getShortPortNumber() > 0xFF00)) {
					throw new IllegalArgumentException("Port " + p.getPort().getPortNumber() + " is not a valid port on switch " + sw.getId().toString());
				} else if (!sw.getOFFactory().getVersion().equals(OFVersion.OF_10) && (sw.getPort(p.getPort()) == null || U32.of(p.getPort().getPortNumber()).compareTo(U32.of(0xffFFff00)) != -1)) {
					throw new IllegalArgumentException("Port " + p.getPort().getPortNumber() + " is not a valid port on switch " + sw.getId().toString());
				}
			}
		}
		for (OFPortModeTuple p : ports) {
			List<OFAction> actions = new ArrayList<OFAction>();
				actions.add(sw.getOFFactory().actions().popVlan());
			}
			actions.add(sw.getOFFactory().actions().output(p.getPort(), 0xffFFffFF));
			OFGroupAdd ga = sw.getOFFactory().buildGroupAdd()
					.setGroup(GroupIds.createL2Interface(p.getPort(), vlan.equals(VlanVid.ZERO) ? VlanVid.ofVlan(1) : vlan))
					.setGroupType(OFGroupType.INDIRECT)
					.setBuckets(Collections.singletonList(
							sw.getOFFactory().buildBucket()
							.setActions(actions)
							.build()))
							.build();
			sw.write(ga);
		}
		List<OFBucket> bucketList = new ArrayList<OFBucket>(ports.size());
		for (OFPortModeTuple p : ports) {
			List<OFAction> actions = new ArrayList<OFAction>();
			actions.add(sw.getOFFactory().actions().group(GroupIds.createL2Interface(p.getPort(), vlan.equals(VlanVid.ZERO) ? VlanVid.ofVlan(1) : vlan)));
			bucketList.add(sw.getOFFactory().buildBucket().setActions(actions).build());
		}
				.setGroup(GroupIds.createL2Flood(U16.of((vlan.equals(VlanVid.ZERO) ? VlanVid.ofVlan(1) : vlan).getVlan()), vlan.equals(VlanVid.ZERO) ? VlanVid.ofVlan(1) : vlan))
				.setGroupType(OFGroupType.ALL)
				.setBuckets(bucketList)
				.build();
		sw.write(ga);
		return true;
	}
	private static boolean addLearningSwitchPrereqFlows(@Nonnull IOFSwitch sw, VlanVid vlan, @Nonnull List<OFPortModeTuple> ports) {
		if (sw == null) {
			throw new NullPointerException("Switch cannot be null.");
		}
		if (vlan == null) {
		} else if (vlan.equals(VlanVid.ofVlan(1))) {
			throw new IllegalArgumentException("VLAN cannot be 1. VLAN 1 is an reserved VLAN for internal use inside the OFDPA switch.");
		}
		if (ports == null) {
			throw new NullPointerException("List of ports cannot be null. Must specify at least 2 valid switch ports.");
		} else if (ports.size() < 2) {
			throw new IllegalArgumentException("List of ports must contain at least 2 valid switch ports.");
		} else {
			for (OFPortModeTuple p : ports) {
				if (sw.getOFFactory().getVersion().equals(OFVersion.OF_10) && (sw.getPort(p.getPort()) == null || p.getPort().getShortPortNumber() > 0xFF00)) {
					throw new IllegalArgumentException("Port " + p.getPort().getPortNumber() + " is not a valid port on switch " + sw.getId().toString());
				} else if (!sw.getOFFactory().getVersion().equals(OFVersion.OF_10) && (sw.getPort(p.getPort()) == null || U32.of(p.getPort().getPortNumber()).compareTo(U32.of(0xffFFff00)) != -1)) {
					throw new IllegalArgumentException("Port " + p.getPort().getPortNumber() + " is not a valid port on switch " + sw.getId().toString());
				}
			}
		}
		ArrayList<OFInstruction> instructions = new ArrayList<OFInstruction>();
		ArrayList<OFAction> applyActions = new ArrayList<OFAction>();
		ArrayList<OFAction> writeActions = new ArrayList<OFAction>();
		Match.Builder mb = sw.getOFFactory().buildMatch();
		OFFlowAdd.Builder fab = sw.getOFFactory().buildFlowAdd();
		fab.setBufferId(OFBufferId.NO_BUFFER)
		.setCookie(APP_COOKIE)
		.setHardTimeout(HARD_TIMEOUT)
		.setIdleTimeout(IDLE_TIMEOUT)
		.setPriority(PRIORITY)
		.setTableId(Tables.VLAN);
		for (OFPortModeTuple p : ports) {
			mb.setExact(MatchField.VLAN_VID, OFVlanVidMatch.ofVlanVid((vlan.equals(VlanVid.ZERO) ? VlanVid.ofVlan(1) : vlan)));
			mb.setExact(MatchField.IN_PORT, p.getPort());
			applyActions.add(sw.getOFFactory().actions().setField(sw.getOFFactory().oxms().vlanVid(OFVlanVidMatch.ofVlanVid((vlan.equals(VlanVid.ZERO) ? VlanVid.ofVlan(1) : vlan)))));
			instructions.add(sw.getOFFactory().instructions().applyActions(applyActions));
			instructions.add(sw.getOFFactory().instructions().gotoTable(Tables.TERMINATION_MAC));
			fab.setInstructions(instructions)
			.setMatch(mb.build())
			.build();
			sw.write(fab.build());
			if (log.isDebugEnabled()) {
				log.debug("Writing tagged prereq flow to VLAN flow table {}", fab.build().toString());
			}
			instructions.clear();
			applyActions.clear();
			mb = sw.getOFFactory().buildMatch();
			if (vlan.equals(VlanVid.ZERO) || p.getMode() == OFPortMode.ACCESS) {
				mb.setExact(MatchField.IN_PORT, p.getPort());
				applyActions.add(sw.getOFFactory().actions().setField(sw.getOFFactory().oxms().vlanVid(OFVlanVidMatch.ofVlanVid((vlan.equals(VlanVid.ZERO) ? VlanVid.ofVlan(1) : vlan)))));
				instructions.add(sw.getOFFactory().instructions().applyActions(applyActions));
				instructions.add(sw.getOFFactory().instructions().gotoTable(Tables.TERMINATION_MAC));
				fab.setInstructions(instructions)
				.setMatch(mb.build())
				.build();
				sw.write(fab.build());
				if (log.isDebugEnabled()) {
					log.debug("Writing untagged prereq flow to VLAN flow table {}", fab.build().toString());
				}
				instructions.clear();
				applyActions.clear();
				mb = sw.getOFFactory().buildMatch();
			}
		}
		writeActions.add(sw.getOFFactory().actions().group(OFDPAUtils.GroupIds.createL2Flood(
		instructions.add(sw.getOFFactory().instructions().writeActions(writeActions));
		instructions.add(sw.getOFFactory().instructions().applyActions(applyActions));
		fab = fab.setMatch(sw.getOFFactory().buildMatch()
				.build())
				.setInstructions(instructions)
				.setTableId(Tables.BRIDGING);
		sw.write(fab.build());
		if (log.isDebugEnabled()) {
			log.debug("Writing DLF flow to bridging table {}", fab.build().toString());
		}
		return true;
	}
	public static boolean addLearningSwitchFlow(IOFSwitch sw, U64 cookie, int priority, int hardTimeout, int idleTimeout, Match match, VlanVid outVlan, OFPort outPort) {
		if (!isOFDPASwitch(sw)) {
			log.error("Switch {} is not an OF-DPA switch. Not inserting flows.", sw.getId().toString());
			return false;
		}
		cookie = (cookie == null ? U64.ZERO : cookie);
		priority = (priority < 1 ? 1 : priority);
		hardTimeout = (hardTimeout < 0 ? 0 : hardTimeout);
		idleTimeout = (idleTimeout < 0 ? 0 : idleTimeout);
		if (match == null || !match.isExact(MatchField.ETH_DST)) {
			log.error("OF-DPA 2.0 requires the destination MAC be matched in order to forward through its pipeline.");
			return false;
		} else if (match == null || !match.isExact(MatchField.VLAN_VID)) {
			log.error("OF-DPA 2.0 requires the VLAN be matched in order to forward through its pipeline.");
			return false;
		} else {
			List<MatchFields> mfs = checkMatchFields(match);
			if (mfs != null) {
				log.error("OF-DPA 2.0 does not support matching on the following fields: {}", mfs.toString());
				return false;
			}
		}
		outVlan = (outVlan == null ? VlanVid.ZERO : outVlan);
		outPort = (outPort == null ? OFPort.ZERO : outPort);
		ArrayList<OFInstruction> instructions = new ArrayList<OFInstruction>();
		ArrayList<OFAction> actions = new ArrayList<OFAction>();
		actions.add(sw.getOFFactory().actions().group(GroupIds.createL2Interface(outPort, (outVlan.equals(VlanVid.ZERO) ? VlanVid.ofVlan(1) : outVlan))));
		instructions.add(sw.getOFFactory().instructions().writeActions(actions));
		OFFlowAdd fa = sw.getOFFactory().buildFlowAdd()
				.setMatch(sw.getOFFactory().buildMatch()
						.setExact(MatchField.VLAN_VID, match.get(MatchField.VLAN_VID))
						.setExact(MatchField.ETH_DST, match.get(MatchField.ETH_DST))
						.build())
						.setPriority(priority)
						.setIdleTimeout(idleTimeout)
						.setHardTimeout(hardTimeout)
						.setBufferId(OFBufferId.NO_BUFFER)
						.setCookie(OFDPAUtils.APP_COOKIE)
						.setTableId(OFDPAUtils.Tables.BRIDGING)
						.setInstructions(instructions)
						.build();
		log.debug("Writing learning switch flow to bridging table: {}", fa);
		sw.write(fa);
		fa = sw.getOFFactory().buildFlowAdd()
				.setBufferId(OFBufferId.NO_BUFFER)
				.setCookie(cookie)
				.setHardTimeout(hardTimeout)
				.setIdleTimeout(idleTimeout)
				.setPriority(priority)
				.setTableId(Tables.POLICY_ACL)
				.setMatch(match)
				.setInstructions(instructions)
				.build();
		log.debug("Writing learning switch flow to policy ACL table: {}", fa);
		return true;
	}
	public static class GroupIds {
		private GroupIds() {}
			return OFGroup.of(0 | p.getShortPortNumber() | (v.getVlan() << 16) | (OFDPAGroupType.L2_INTERFACE << 28));
		}
			return OFGroup.of(0 | (id.getRaw() & 0x0FffFFff) | (OFDPAGroupType.L2_REWRITE << 28));
		}
			return OFGroup.of(0 | (id.getRaw() & 0x0FffFFff) | (OFDPAGroupType.L3_UNICAST << 28));
		}
			return OFGroup.of(0 | id.getRaw() | (v.getVlan() << 16) | (OFDPAGroupType.L2_MULTICAST << 28));
		}
			return OFGroup.of(0 | id.getRaw() | (v.getVlan() << 16) | (OFDPAGroupType.L2_FLOOD << 28));
		}
			return OFGroup.of(0 | (id.getRaw() & 0x0FffFFff) | (OFDPAGroupType.L3_INTERFACE << 28));
		}
			return OFGroup.of(0 | id.getRaw() | (v.getVlan() << 16) | (OFDPAGroupType.L3_MULTICAST << 28));
		}
			return OFGroup.of(0 | (id.getRaw() & 0x0FffFFff) | (OFDPAGroupType.L3_ECMP << 28));
		}
			return OFGroup.of(0 | (index.getRaw() & 0x03ff) 
					| (tunnelId.getRaw() << 12)
					| (L2OverlaySubType.L2_OVERLAY_FLOOD_OVER_UNICAST_TUNNELS << 10)
					| (OFDPAGroupType.L2_DATA_CENTER_OVERLAY << 28));
		}
			return OFGroup.of(0 | (index.getRaw() & 0x03ff) 
					| (tunnelId.getRaw() << 12)
					| (L2OverlaySubType.L2_OVERLAY_FLOOD_OVER_MULTICAST_TUNNELS << 10)
					| (OFDPAGroupType.L2_DATA_CENTER_OVERLAY << 28));
		}
			return OFGroup.of(0 | (index.getRaw() & 0x03ff) 
					| (tunnelId.getRaw() << 12)
					| (L2OverlaySubType.L2_OVERLAY_MULTICAST_OVER_UNICAST_TUNNELS << 10)
					| (OFDPAGroupType.L2_DATA_CENTER_OVERLAY << 28));
		}
			return OFGroup.of(0 | (index.getRaw() & 0x03ff) 
					| (tunnelId.getRaw() << 12)
					| (L2OverlaySubType.L2_OVERLAY_MULTICAST_OVER_MULTICAST_TUNNELS << 10)
					| (OFDPAGroupType.L2_DATA_CENTER_OVERLAY << 28));
		}
			return OFGroup.of(0 | (index.getRaw() & 0x00ffFFff) 
					| (MPLSSubType.MPLS_INTERFACE << 24)
					| (OFDPAGroupType.MPLS_LABEL << 28));
		}
			return OFGroup.of(0 | (index.getRaw() & 0x00ffFFff) 
					| (MPLSSubType.MPLS_L2_VPN_LABEL << 24)
					| (OFDPAGroupType.MPLS_LABEL << 28));
		}
			return OFGroup.of(0 | (index.getRaw() & 0x00ffFFff) 
					| (MPLSSubType.MPLS_L3_VPN_LABEL << 24)
					| (OFDPAGroupType.MPLS_LABEL << 28));
		}
			return OFGroup.of(0 | (index.getRaw() & 0x00ffFFff) 
					| (MPLSSubType.MPLS_TUNNEL_LABEL_1 << 24)
					| (OFDPAGroupType.MPLS_LABEL << 28));
		}
			return OFGroup.of(0 | (index.getRaw() & 0x00ffFFff) 
					| (MPLSSubType.MPLS_TUNNEL_LABEL_2 << 24)
					| (OFDPAGroupType.MPLS_LABEL << 28));
		}
			return OFGroup.of(0 | (index.getRaw() & 0x00ffFFff) 
					| (MPLSSubType.MPLS_SWAP_LABEL << 24)
					| (OFDPAGroupType.MPLS_LABEL << 28));
		}
			return OFGroup.of(0 | (index.getRaw() & 0x00ffFFff) 
					| (MPLSSubType.MPLS_FAST_FAILOVER << 24)
					| (OFDPAGroupType.MPLS_FORWARDING << 28));
		}
			return OFGroup.of(0 | (index.getRaw() & 0x00ffFFff) 
					| (MPLSSubType.MPLS_ECMP << 24)
					| (OFDPAGroupType.MPLS_FORWARDING << 28));
		}
			return OFGroup.of(0 | (index.getRaw() & 0x00ffFFff) 
					| (MPLSSubType.MPLS_L2_TAG << 24)
					| (OFDPAGroupType.MPLS_FORWARDING << 28));
		}
			return OFGroup.of(0 | p.getShortPortNumber() | (OFDPAGroupType.L2_UNFILTERED_INTERFACE << 28));
		}
			return OFGroup.of(0 | p.getShortPortNumber() | (OFDPAGroupType.L2_LOOPBACK << 28));
		}
	}
}
package net.floodlightcontroller.core.web.serializers;
import net.floodlightcontroller.util.MatchUtils;
import org.projectfloodlight.openflow.protocol.ver14.OFOxmClassSerializerVer14;
import org.projectfloodlight.openflow.types.ArpOpcode;
import org.projectfloodlight.openflow.types.EthType;
import org.projectfloodlight.openflow.types.ICMPv4Code;
import org.projectfloodlight.openflow.types.ICMPv4Type;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv6Address;
import org.projectfloodlight.openflow.types.IPv6FlowLabel;
import org.projectfloodlight.openflow.types.IpDscp;
import org.projectfloodlight.openflow.types.IpEcn;
import org.projectfloodlight.openflow.types.IpProtocol;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFBooleanValue;
import org.projectfloodlight.openflow.types.OFMetadata;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.TransportPort;
import org.projectfloodlight.openflow.types.U32;
import org.projectfloodlight.openflow.types.U64;
import org.projectfloodlight.openflow.types.U8;
import org.projectfloodlight.openflow.types.VlanPcp;
import org.projectfloodlight.openflow.types.VlanVid;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
public class OXMSerializer {	
	private static final String STR_MASKED = "_masked";
	private static final String STR_NXM = "nxm_";
	private static final int SHIFT_FIELD = 9;
	private static final int SHIFT_CLASS = 16;
	private static final int SHIFT_HAS_MASK = 8;
	private static final int MASKED = (1 << SHIFT_HAS_MASK);
	private static final int OF_BASIC = (OFOxmClassSerializerVer14.OPENFLOW_BASIC_VAL << SHIFT_CLASS) & 0xFFffFFff;
	private static final int NXM_0 = (OFOxmClassSerializerVer14.NXM_0_VAL << SHIFT_CLASS) & 0xFFffFFff;
	private static final int NXM_1 = (OFOxmClassSerializerVer14.NXM_1_VAL << SHIFT_CLASS) & 0xFFffFFff;
	private static final BiMap<U32, String> map = HashBiMap.create();
	static { 
		map.put(U32.ofRaw(OF_BASIC | (0 << SHIFT_FIELD) | OFPort.ZERO.getLength()), MatchUtils.STR_IN_PORT);
		map.put(U32.ofRaw(OF_BASIC | (1 << SHIFT_FIELD) | OFPort.ZERO.getLength()), MatchUtils.STR_IN_PHYS_PORT);
		map.put(U32.ofRaw(OF_BASIC | (2 << SHIFT_FIELD) | OFMetadata.NONE.getLength()), MatchUtils.STR_METADATA);
		map.put(U32.ofRaw(OF_BASIC | (3 << SHIFT_FIELD) | MacAddress.NONE.getLength()), MatchUtils.STR_DL_DST);
		map.put(U32.ofRaw(OF_BASIC | (4 << SHIFT_FIELD) | MacAddress.NONE.getLength()), MatchUtils.STR_DL_SRC);
		map.put(U32.ofRaw(OF_BASIC | (5 << SHIFT_FIELD) | EthType.NONE.getLength()), MatchUtils.STR_DL_TYPE);
		map.put(U32.ofRaw(OF_BASIC | (6 << SHIFT_FIELD) | VlanVid.ZERO.getLength()), MatchUtils.STR_DL_VLAN);
		map.put(U32.ofRaw(OF_BASIC | (7 << SHIFT_FIELD) | VlanPcp.NONE.getLength()), MatchUtils.STR_DL_VLAN_PCP);
		map.put(U32.ofRaw(OF_BASIC | (8 << SHIFT_FIELD) | IpDscp.NONE.getLength()), MatchUtils.STR_NW_DSCP);
		map.put(U32.ofRaw(OF_BASIC | (9 << SHIFT_FIELD) | IpEcn.NONE.getLength()), MatchUtils.STR_NW_ECN);
		map.put(U32.ofRaw(OF_BASIC | (10 << SHIFT_FIELD) | IpProtocol.NONE.getLength()), MatchUtils.STR_NW_PROTO);
		map.put(U32.ofRaw(OF_BASIC | (11 << SHIFT_FIELD) | IPv4Address.NONE.getLength()), MatchUtils.STR_NW_SRC);
		map.put(U32.ofRaw(OF_BASIC | (12 << SHIFT_FIELD) | IPv4Address.NONE.getLength()), MatchUtils.STR_NW_DST);
		map.put(U32.ofRaw(OF_BASIC | (13 << SHIFT_FIELD) | TransportPort.NONE.getLength()), MatchUtils.STR_TCP_SRC);
		map.put(U32.ofRaw(OF_BASIC | (14 << SHIFT_FIELD) | TransportPort.NONE.getLength()), MatchUtils.STR_TCP_DST);
		map.put(U32.ofRaw(OF_BASIC | (15 << SHIFT_FIELD) | TransportPort.NONE.getLength()), MatchUtils.STR_UDP_SRC);
		map.put(U32.ofRaw(OF_BASIC | (16 << SHIFT_FIELD) | TransportPort.NONE.getLength()), MatchUtils.STR_UDP_DST);
		map.put(U32.ofRaw(OF_BASIC | (17 << SHIFT_FIELD) | TransportPort.NONE.getLength()), MatchUtils.STR_SCTP_SRC);
		map.put(U32.ofRaw(OF_BASIC | (18 << SHIFT_FIELD) | TransportPort.NONE.getLength()), MatchUtils.STR_SCTP_DST);
		map.put(U32.ofRaw(OF_BASIC | (19 << SHIFT_FIELD) | ICMPv4Type.NONE.getLength()), MatchUtils.STR_ICMP_TYPE);
		map.put(U32.ofRaw(OF_BASIC | (20 << SHIFT_FIELD) | ICMPv4Code.NONE.getLength()), MatchUtils.STR_ICMP_CODE);
		map.put(U32.ofRaw(OF_BASIC | (21 << SHIFT_FIELD) | ArpOpcode.NONE.getLength()), MatchUtils.STR_ARP_OPCODE);
		map.put(U32.ofRaw(OF_BASIC | (22 << SHIFT_FIELD) | IPv4Address.NONE.getLength()), MatchUtils.STR_ARP_SPA);
		map.put(U32.ofRaw(OF_BASIC | (23 << SHIFT_FIELD) | IPv4Address.NONE.getLength()), MatchUtils.STR_ARP_DPA);
		map.put(U32.ofRaw(OF_BASIC | (24 << SHIFT_FIELD) | MacAddress.NONE.getLength()), MatchUtils.STR_ARP_SHA);
		map.put(U32.ofRaw(OF_BASIC | (25 << SHIFT_FIELD) | MacAddress.NONE.getLength()), MatchUtils.STR_ARP_DHA);
		map.put(U32.ofRaw(OF_BASIC | (26 << SHIFT_FIELD) | IPv6Address.NONE.getLength()), MatchUtils.STR_IPV6_SRC);
		map.put(U32.ofRaw(OF_BASIC | (27 << SHIFT_FIELD) | IPv6Address.NONE.getLength()), MatchUtils.STR_IPV6_DST);
		map.put(U32.ofRaw(OF_BASIC | (28 << SHIFT_FIELD) | IPv6FlowLabel.NONE.getLength()), MatchUtils.STR_IPV6_FLOW_LABEL);
		map.put(U32.ofRaw(OF_BASIC | (29 << SHIFT_FIELD) | U8.ZERO.getLength()), MatchUtils.STR_ICMPV6_TYPE);
		map.put(U32.ofRaw(OF_BASIC | (30 << SHIFT_FIELD) | U8.ZERO.getLength()), MatchUtils.STR_ICMPV6_CODE);
		map.put(U32.ofRaw(OF_BASIC | (31 << SHIFT_FIELD) | IPv6Address.NONE.getLength()), MatchUtils.STR_IPV6_ND_TARGET);
		map.put(U32.ofRaw(OF_BASIC | (32 << SHIFT_FIELD) | MacAddress.NONE.getLength()), MatchUtils.STR_IPV6_ND_SSL);
		map.put(U32.ofRaw(OF_BASIC | (33 << SHIFT_FIELD) | MacAddress.NONE.getLength()), MatchUtils.STR_IPV6_ND_TTL);
		map.put(U32.ofRaw(OF_BASIC | (34 << SHIFT_FIELD) | U32.ZERO.getLength()), MatchUtils.STR_MPLS_LABEL);
		map.put(U32.ofRaw(OF_BASIC | (35 << SHIFT_FIELD) | U8.ZERO.getLength()), MatchUtils.STR_MPLS_TC);
		map.put(U32.ofRaw(OF_BASIC | (36 << SHIFT_FIELD) | OFBooleanValue.TRUE.getLength()), MatchUtils.STR_MPLS_BOS);
		map.put(U32.ofRaw(OF_BASIC | (38 << SHIFT_FIELD) | U64.ZERO.getLength()), MatchUtils.STR_TUNNEL_ID);
		map.put(U32.ofRaw(OF_BASIC | (41 << SHIFT_FIELD) | 1), "pbb_uca");
		map.put(U32.ofRaw(NXM_0 | (1 << SHIFT_FIELD) | MacAddress.NONE.getLength()), STR_NXM + MatchUtils.STR_DL_DST);
		map.put(U32.ofRaw(NXM_0 | (2 << SHIFT_FIELD) | MacAddress.NONE.getLength()), STR_NXM + MatchUtils.STR_DL_SRC);
		map.put(U32.ofRaw(NXM_0 | (3 << SHIFT_FIELD) | EthType.NONE.getLength()), STR_NXM + MatchUtils.STR_DL_TYPE);
		map.put(U32.ofRaw(NXM_0 | (4 << SHIFT_FIELD) | VlanVid.ZERO.getLength()), STR_NXM + MatchUtils.STR_DL_VLAN);
		map.put(U32.ofRaw(NXM_0 | (5 << SHIFT_FIELD) | 1), STR_NXM + MatchUtils.STR_NW_TOS);
		map.put(U32.ofRaw(NXM_0 | (6 << SHIFT_FIELD) | IpProtocol.NONE.getLength()), STR_NXM + MatchUtils.STR_NW_PROTO);
		map.put(U32.ofRaw(NXM_0 | (7 << SHIFT_FIELD) | IPv4Address.NONE.getLength()), STR_NXM + MatchUtils.STR_NW_SRC);
		map.put(U32.ofRaw(NXM_0 | (8 << SHIFT_FIELD) | IPv4Address.NONE.getLength()), STR_NXM + MatchUtils.STR_NW_DST);
		map.put(U32.ofRaw(NXM_0 | (9 << SHIFT_FIELD) | TransportPort.NONE.getLength()), STR_NXM + MatchUtils.STR_TCP_SRC);
		map.put(U32.ofRaw(NXM_0 | (10 << SHIFT_FIELD) | TransportPort.NONE.getLength()), STR_NXM + MatchUtils.STR_TCP_DST);
		map.put(U32.ofRaw(NXM_0 | (11 << SHIFT_FIELD) | TransportPort.NONE.getLength()), STR_NXM + MatchUtils.STR_UDP_SRC);
		map.put(U32.ofRaw(NXM_0 | (12 << SHIFT_FIELD) | TransportPort.NONE.getLength()), STR_NXM + MatchUtils.STR_UDP_DST);
		map.put(U32.ofRaw(NXM_0 | (13 << SHIFT_FIELD) | U8.ZERO.getLength()), STR_NXM + MatchUtils.STR_ICMP_TYPE);
		map.put(U32.ofRaw(NXM_0 | (14 << SHIFT_FIELD) | U8.ZERO.getLength()), STR_NXM + MatchUtils.STR_ICMP_CODE);
		map.put(U32.ofRaw(NXM_0 | (15 << SHIFT_FIELD) | ArpOpcode.NONE.getLength()), STR_NXM + MatchUtils.STR_ARP_OPCODE);
		map.put(U32.ofRaw(NXM_0 | (16 << SHIFT_FIELD) | IPv4Address.NONE.getLength()), STR_NXM + MatchUtils.STR_ARP_SPA);
		map.put(U32.ofRaw(NXM_0 | (17 << SHIFT_FIELD) | IPv4Address.NONE.getLength()), STR_NXM + MatchUtils.STR_ARP_DPA);
		map.put(U32.ofRaw(NXM_1 | (0 << SHIFT_FIELD) | 4), STR_NXM + "reg_0");
		map.put(U32.ofRaw(NXM_1 | (1 << SHIFT_FIELD) | 4), STR_NXM + "reg_1");
		map.put(U32.ofRaw(NXM_1 | (2 << SHIFT_FIELD) | 4), STR_NXM + "reg_2");
		map.put(U32.ofRaw(NXM_1 | (3 << SHIFT_FIELD) | 4), STR_NXM + "reg_3");
		map.put(U32.ofRaw(NXM_1 | (4 << SHIFT_FIELD) | 4), STR_NXM + "reg_4");
		map.put(U32.ofRaw(NXM_1 | (5 << SHIFT_FIELD) | 4), STR_NXM + "reg_5");
		map.put(U32.ofRaw(NXM_1 | (6 << SHIFT_FIELD) | 4), STR_NXM + "reg_6");
		map.put(U32.ofRaw(NXM_1 | (7 << SHIFT_FIELD) | 4), STR_NXM + "reg_7");
		map.put(U32.ofRaw(NXM_1 | (16 << SHIFT_FIELD) | U64.ZERO.getLength()), STR_NXM + MatchUtils.STR_TUNNEL_ID);
		map.put(U32.ofRaw(NXM_1 | (17 << SHIFT_FIELD) | MacAddress.NONE.getLength()), STR_NXM + MatchUtils.STR_ARP_SHA);
		map.put(U32.ofRaw(NXM_1 | (18 << SHIFT_FIELD) | MacAddress.NONE.getLength()), STR_NXM + MatchUtils.STR_ARP_DHA);
		map.put(U32.ofRaw(NXM_1 | (19 << SHIFT_FIELD) | IPv6Address.NONE.getLength()), STR_NXM + MatchUtils.STR_IPV6_SRC);
		map.put(U32.ofRaw(NXM_1 | (20 << SHIFT_FIELD) | IPv6Address.NONE.getLength()), STR_NXM + MatchUtils.STR_IPV6_DST);
		map.put(U32.ofRaw(NXM_1 | (21 << SHIFT_FIELD) | U8.ZERO.getLength()), STR_NXM + MatchUtils.STR_ICMPV6_TYPE);
		map.put(U32.ofRaw(NXM_1 | (22 << SHIFT_FIELD) | U8.ZERO.getLength()), STR_NXM + MatchUtils.STR_ICMPV6_CODE);
		map.put(U32.ofRaw(NXM_1 | (23 << SHIFT_FIELD) | IPv6Address.NONE.getLength()), STR_NXM + MatchUtils.STR_IPV6_ND_TARGET);
		map.put(U32.ofRaw(NXM_1 | (24 << SHIFT_FIELD) | MacAddress.NONE.getLength()), STR_NXM + MatchUtils.STR_IPV6_ND_SSL);
		map.put(U32.ofRaw(NXM_1 | (25 << SHIFT_FIELD) | MacAddress.NONE.getLength()), STR_NXM + MatchUtils.STR_IPV6_ND_TTL);
		map.put(U32.ofRaw(NXM_1 | (26 << SHIFT_FIELD) | 1), STR_NXM + "ip_frag");
		map.put(U32.ofRaw(NXM_1 | (27 << SHIFT_FIELD) | IPv6FlowLabel.NONE.getLength()), STR_NXM + MatchUtils.STR_IPV6_FLOW_LABEL);	
		map.put(U32.ofRaw(NXM_1 | (28 << SHIFT_FIELD) | IpEcn.NONE.getLength()), STR_NXM + MatchUtils.STR_NW_ECN);
		map.put(U32.ofRaw(NXM_1 | (29 << SHIFT_FIELD) | 1), STR_NXM + "ip_ttl");
		map.put(U32.ofRaw(NXM_1 | (30 << SHIFT_FIELD) | 8), STR_NXM + "flow_cookie");
		map.put(U32.ofRaw(NXM_1 | (31 << SHIFT_FIELD) | IPv4Address.NONE.getLength()), MatchUtils.STR_TUNNEL_IPV4_SRC);
		map.put(U32.ofRaw(NXM_1 | (32 << SHIFT_FIELD) | IPv4Address.NONE.getLength()), MatchUtils.STR_TUNNEL_IPV4_DST);
		map.put(U32.ofRaw(NXM_1 | (33 << SHIFT_FIELD) | 4), STR_NXM + "pkt_mark");
		map.put(U32.ofRaw(NXM_1 | (34 << SHIFT_FIELD) | 2), STR_NXM + "tcp_flags");
		map.put(U32.ofRaw(NXM_1 | (35 << SHIFT_FIELD) | 4), STR_NXM + "internal_dp_hash");
		map.put(U32.ofRaw(NXM_1 | (36 << SHIFT_FIELD) | 4), STR_NXM + "internal_recirc_id");
		map.put(U32.ofRaw(NXM_1 | (37 << SHIFT_FIELD) | 4), STR_NXM + MatchUtils.STR_PBB_ISID);
	}
	public static String oxmIdToString(U32 oxmId) {
		if (map.containsKey(oxmId)) {
			return map.get(oxmId);
		} else {
			return "Unknown OXM ID: " + oxmId.toString();
		}
	}
	public static U32 oxmStringToId(String oxmString) {
		if (map.inverse().containsKey(oxmString)) {
			return map.inverse().get(oxmString);
		} else {
			return null;
		}
	}
}

package net.floodlightcontroller.util;
import java.util.ArrayDeque;
import java.util.Iterator;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFVersion;
import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.types.ArpOpcode;
import org.projectfloodlight.openflow.types.EthType;
import org.projectfloodlight.openflow.types.ICMPv4Code;
import org.projectfloodlight.openflow.types.ICMPv4Type;
import org.projectfloodlight.openflow.types.IPv4Address;
import org.projectfloodlight.openflow.types.IPv4AddressWithMask;
import org.projectfloodlight.openflow.types.IPv6AddressWithMask;
import org.projectfloodlight.openflow.types.IPv6FlowLabel;
import org.projectfloodlight.openflow.types.IpDscp;
import org.projectfloodlight.openflow.types.IpEcn;
import org.projectfloodlight.openflow.types.IpProtocol;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFBooleanValue;
import org.projectfloodlight.openflow.types.OFMetadata;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.OFVlanVidMatch;
import org.projectfloodlight.openflow.types.OFVlanVidMatchWithMask;
import org.projectfloodlight.openflow.types.TransportPort;
import org.projectfloodlight.openflow.types.U16;
import org.projectfloodlight.openflow.types.U32;
import org.projectfloodlight.openflow.types.U64;
import org.projectfloodlight.openflow.types.U8;
import org.projectfloodlight.openflow.types.VlanPcp;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class MatchUtils {
	private static final Logger log = LoggerFactory.getLogger(MatchUtils.class);
	public static final String STR_IN_PORT = "in_port";
	public static final String STR_IN_PHYS_PORT = "in_phys_port";
	public static final String STR_DL_DST = "eth_dst";
	public static final String STR_DL_SRC = "eth_src";
	public static final String STR_DL_TYPE = "eth_type";
	public static final String STR_DL_VLAN = "eth_vlan_vid";
	public static final String STR_DL_VLAN_PCP = "eth_vlan_pcp";
	public static final String STR_NW_DST = "ipv4_dst";
	public static final String STR_NW_SRC = "ipv4_src";
	public static final String STR_IPV6_DST = "ipv6_dst";
	public static final String STR_IPV6_SRC = "ipv6_src";
	public static final String STR_IPV6_FLOW_LABEL = "ipv6_label";
	public static final String STR_IPV6_ND_SSL = "ipv6_nd_ssl";
	public static final String STR_IPV6_ND_TARGET = "ipv6_nd_target";
	public static final String STR_IPV6_ND_TTL = "ipv6_nd_ttl";
	public static final String STR_NW_PROTO = "ip_proto";
	public static final String STR_NW_TOS = "ip_tos";
	public static final String STR_NW_ECN = "ip_ecn";
	public static final String STR_NW_DSCP = "ip_dscp";
	public static final String STR_SCTP_DST = "sctp_dst";
	public static final String STR_SCTP_SRC = "sctp_src";
	public static final String STR_UDP_DST = "udp_dst";
	public static final String STR_UDP_SRC = "udp_src";
	public static final String STR_TCP_DST = "tcp_dst";
	public static final String STR_TCP_SRC = "tcp_src";
	public static final String STR_TP_SRC = "tp_src";
	public static final String STR_ICMP_TYPE = "icmpv4_type";
	public static final String STR_ICMP_CODE = "icmpv4_code";
	public static final String STR_ICMPV6_TYPE = "icmpv6_type";
	public static final String STR_ICMPV6_CODE = "icmpv6_code";
	public static final String STR_ARP_OPCODE = "arp_opcode";
	public static final String STR_ARP_SHA = "arp_sha";
	public static final String STR_ARP_DHA = "arp_tha";
	public static final String STR_ARP_SPA = "arp_spa";
	public static final String STR_ARP_DPA = "arp_tpa";
	public static final String STR_MPLS_LABEL = "mpls_label";
	public static final String STR_MPLS_TC = "mpls_tc";
	public static final String STR_MPLS_BOS = "mpls_bos";
	public static final String STR_METADATA = "metadata";
	public static final String STR_TUNNEL_ID = "tunnel_id";
	public static final String STR_TUNNEL_IPV4_SRC = "tunnel_ipv4_src";
	public static final String STR_TUNNEL_IPV4_DST = "tunnel_ipv4_dst";
	public static final String STR_PBB_ISID = "pbb_isid";	
	public static final String SET_FIELD_DELIM = "->";
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public static Match maskL4AndUp(Match m) {
		Match.Builder mb = m.createBuilder(); 
		while(itr.hasNext()) {
			MatchField mf = itr.next();
			if (mf.equals(MatchField.IN_PORT) || mf.equals(MatchField.ETH_TYPE) || mf.equals(MatchField.ETH_SRC) || mf.equals(MatchField.ETH_DST) ||
					mf.equals(MatchField.IPV4_SRC) || mf.equals(MatchField.IPV4_DST) || mf.equals(MatchField.IP_PROTO)) {
				if (m.isExact(mf)) {
					mb.setExact(mf, m.get(mf));
				} else if (m.isPartiallyMasked(mf)) {
					mb.setExact(mf, m.get(mf));
				} else {
				} 
			}
		}
		return mb.build();
	}
	@SuppressWarnings("unchecked")
	public static Match.Builder convertToVersion(Match parent, OFVersion version) {
		Match.Builder mb = OFFactories.getFactory(version).buildMatch(); 
		while(itr.hasNext()) {
			@SuppressWarnings("rawtypes")
			MatchField mf = itr.next();
			if (parent.isExact(mf)) {
				mb.setExact(mf, parent.get(mf));
			} else if (parent.isPartiallyMasked(mf)) {
				mb.setExact(mf, parent.get(mf));
			} else {
			}
		}
		return mb;
	}
	public static Match.Builder createRetentiveBuilder(Match m) {
		return convertToVersion(m, m.getVersion());
	}
	public static Match.Builder createForgetfulBuilder(Match m) {
		return OFFactories.getFactory(m.getVersion()).buildMatch();
	}
	public static Match createCopy(Match m) {
	}
	@Deprecated
	public static String toString(Match match) {
	        match
	        if ((wildcards & OFPFW_IN_PORT) == 0)
	            str += "," + STR_IN_PORT + "=" + U16.f(this.inputPort);
	        if ((wildcards & OFPFW_DL_DST) == 0)
	            str += "," + STR_DL_DST + "="
	                    + match.);
	        if ((wildcards & OFPFW_DL_SRC) == 0)
	            str += "," + STR_DL_SRC + "="
	                    + HexString.toHexString(this.dataLayerSource);
	        if ((wildcards & OFPFW_DL_TYPE) == 0)
	            str += "," + STR_DL_TYPE + "=0x"
	                    + Integer.toHexString(U16.f(this.dataLayerType));
	        if ((wildcards & OFPFW_DL_VLAN) == 0)
	            str += "," + STR_DL_VLAN + "=0x"
	                    + Integer.toHexString(U16.f(this.dataLayerVirtualLan));
	        if ((wildcards & OFPFW_DL_VLAN_PCP) == 0)
	            str += ","
	                    + STR_DL_VLAN_PCP
	                    + "="
	                    + Integer.toHexString(U8
	                            .f(this.dataLayerVirtualLanPriorityCodePoint));
	        if (getNetworkDestinationMaskLen() > 0)
	            str += ","
	                    + STR_NW_DST
	                    + "="
	                    + cidrToString(networkDestination,
	                            getNetworkDestinationMaskLen());
	        if (getNetworkSourceMaskLen() > 0)
	            str += "," + STR_NW_SRC + "="
	                    + cidrToString(networkSource, getNetworkSourceMaskLen());
	        if ((wildcards & OFPFW_NW_PROTO) == 0)
	            str += "," + STR_NW_PROTO + "=" + U8.f(this.networkProtocol);
	        if ((wildcards & OFPFW_NW_TOS) == 0)
	            str += "," + STR_NW_TOS + "=" + U8.f(this.networkTypeOfService);
	        if ((wildcards & OFPFW_TP_DST) == 0)
	            str += "," + STR_TP_DST + "=" + U16.f(this.transportDestination);
	        if ((wildcards & OFPFW_TP_SRC) == 0)
	            str += "," + STR_TP_SRC + "=" + U16.f(this.transportSource);
	        if ((str.length() > 0) && (str.charAt(0) == ','))
		return "";
	}
	public static Match fromString(String match, OFVersion ofVersion) throws IllegalArgumentException {
		boolean ver10 = false;
		if (match.equals("") || match.equalsIgnoreCase("any") || match.equalsIgnoreCase("all") || match.equals("[]")) {
			match = "Match[]";
		}
		String[] tokens = match.split("[\\[,\\]]");
		int initArg = 0;
		if (tokens[0].equals("Match")) {
			initArg = 1;
		}
		int i;
		String[] tmp;
		ArrayDeque<String[]> llValues = new ArrayDeque<String[]>();
		for (i = initArg; i < tokens.length; i++) {
			tmp = tokens[i].split("=");
			if (tmp.length != 2) {
				throw new IllegalArgumentException("Token " + tokens[i] + " does not have form 'key=value' parsing " + match);
			}
		}	
		Match.Builder mb = OFFactories.getFactory(ofVersion).buildMatch();
		if (ofVersion.equals(OFVersion.OF_10)) {
			ver10 = true;
		}
		while (!llValues.isEmpty()) {
			IpProtocol ipProto = null;
			String[] dataMask = key_value[1].split("/");
			if (dataMask.length > 2) {
				throw new IllegalArgumentException("[Data, Mask] " + dataMask + " does not have form 'data/mask' or 'data'" + key_value[1]);
			} else if (dataMask.length == 1) {
				log.debug("No mask detected in Match string: {}", key_value[1]);
			} else if (dataMask.length == 2) {
				log.debug("Detected mask in Match string: {}", key_value[1]);
			}
			switch (key_value[0]) {
			case STR_IN_PORT:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.IN_PORT, OFPort.ofShort(dataMask[0].contains("0x") ? U16.of(Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16)).getRaw() : U16.of(Integer.valueOf(dataMask[0])).getRaw()));
				} else {
					mb.setMasked(MatchField.IN_PORT, OFPort.ofShort(dataMask[0].contains("0x") ? U16.of(Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16)).getRaw() : U16.of(Integer.valueOf(dataMask[0])).getRaw()), 
					OFPort.ofShort(dataMask[1].contains("0x") ? U16.of(Integer.valueOf(dataMask[1].replaceFirst("0x", ""), 16)).getRaw() : U16.of(Integer.valueOf(dataMask[1])).getRaw()));
				}
				break;
				if (dataMask.length == 1) {
					mb.setExact(MatchField.ETH_DST, MacAddress.of(dataMask[0]));
				} else {
					mb.setMasked(MatchField.ETH_DST, MacAddress.of(dataMask[0]), MacAddress.of(dataMask[1]));
				}
				break;
			case STR_DL_SRC:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.ETH_SRC, MacAddress.of(dataMask[0]));
				} else {
					mb.setMasked(MatchField.ETH_SRC, MacAddress.of(dataMask[0]), MacAddress.of(dataMask[1]));
				}
				break;
			case STR_DL_TYPE:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.ETH_TYPE, EthType.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])));
				} else {
					mb.setMasked(MatchField.ETH_TYPE, EthType.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])), 
					EthType.of(dataMask[1].contains("0x") ? Integer.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[1])));
				}
				break;
			case STR_DL_VLAN:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.VLAN_VID, OFVlanVidMatch.ofRawVid(dataMask[0].contains("0x") ? Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Short.valueOf(dataMask[0])));
				} else {
					mb.setMasked(MatchField.VLAN_VID, OFVlanVidMatchWithMask.of(
						OFVlanVidMatch.ofRawVid(dataMask[0].contains("0x") ? Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Short.valueOf(dataMask[0])), 
						OFVlanVidMatch.ofRawVid(dataMask[1].contains("0x") ? Short.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Short.valueOf(dataMask[1]))));
				}
				break;
			case STR_DL_VLAN_PCP:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.VLAN_PCP, VlanPcp.of(dataMask[0].contains("0x") ? U8.t(Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U8.t(Short.valueOf(dataMask[0]))));
				} else {
					mb.setMasked(MatchField.VLAN_PCP, VlanPcp.of(dataMask[0].contains("0x") ? U8.t(Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U8.t(Short.valueOf(dataMask[0]))), 
					VlanPcp.of(dataMask[1].contains("0x") ? U8.t(Short.valueOf(dataMask[1].replaceFirst("0x", ""), 16)) : U8.t(Short.valueOf(dataMask[1]))));
				}
				break;
				mb.setMasked(MatchField.IPV4_DST, IPv4AddressWithMask.of(key_value[1]));
				break;
			case STR_NW_SRC:
				mb.setMasked(MatchField.IPV4_SRC, IPv4AddressWithMask.of(key_value[1]));
				break;
				if (ver10 == true) {
					throw new IllegalArgumentException("OF Version incompatible");
				}
				mb.setMasked(MatchField.IPV6_DST, IPv6AddressWithMask.of(key_value[1]));
				break;
			case STR_IPV6_SRC:
				if (ver10 == true) {
					throw new IllegalArgumentException("OF Version incompatible");
				}
				mb.setMasked(MatchField.IPV6_SRC, IPv6AddressWithMask.of(key_value[1]));
				break;
			case STR_IPV6_FLOW_LABEL:
				if (ver10 == true) {
					throw new IllegalArgumentException("OF Version incompatible");
				}
				if (dataMask.length == 1) {
					mb.setExact(MatchField.IPV6_FLABEL, IPv6FlowLabel.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])));
				} else {
					mb.setMasked(MatchField.IPV6_FLABEL, IPv6FlowLabel.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])), 
					IPv6FlowLabel.of(dataMask[1].contains("0x") ? Integer.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[1])));
				}
				break;
			case STR_NW_PROTO:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.IP_PROTO, IpProtocol.of(dataMask[0].contains("0x") ? Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Short.valueOf(dataMask[0])));
				} else {
					mb.setMasked(MatchField.IP_PROTO, IpProtocol.of(dataMask[0].contains("0x") ? Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Short.valueOf(dataMask[0])), 
					IpProtocol.of(dataMask[1].contains("0x") ? Short.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Short.valueOf(dataMask[1])));
				}
				break;
			case STR_NW_TOS:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.IP_ECN, IpEcn.of(dataMask[0].contains("0x") ? U8.t(Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U8.t(Short.valueOf(dataMask[0]))));
					mb.setExact(MatchField.IP_DSCP, IpDscp.of(dataMask[0].contains("0x") ? U8.t(Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U8.t(Short.valueOf(dataMask[0]))));
				} else {
					mb.setMasked(MatchField.IP_ECN, IpEcn.of(dataMask[0].contains("0x") ? U8.t(Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U8.t(Short.valueOf(dataMask[0]))), 
							IpEcn.of(dataMask[1].contains("0x") ? U8.t(Short.valueOf(dataMask[1].replaceFirst("0x", ""), 16)) : U8.t(Short.valueOf(dataMask[1]))));
					mb.setMasked(MatchField.IP_DSCP, IpDscp.of(dataMask[0].contains("0x") ? U8.t(Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U8.t(Short.valueOf(dataMask[0]))), 
							IpDscp.of(dataMask[1].contains("0x") ? U8.t(Short.valueOf(dataMask[1].replaceFirst("0x", ""), 16)) : U8.t(Short.valueOf(dataMask[1]))));
				}
				break;
			case STR_NW_ECN:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.IP_ECN, IpEcn.of(dataMask[0].contains("0x") ? U8.t(Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U8.t(Short.valueOf(dataMask[0]))));
				} else {
					mb.setMasked(MatchField.IP_ECN, IpEcn.of(dataMask[0].contains("0x") ? U8.t(Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U8.t(Short.valueOf(dataMask[0]))), 
							IpEcn.of(dataMask[1].contains("0x") ? U8.t(Short.valueOf(dataMask[1].replaceFirst("0x", ""), 16)) : U8.t(Short.valueOf(dataMask[1]))));
				}
				break;
			case STR_NW_DSCP:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.IP_DSCP, IpDscp.of(dataMask[0].contains("0x") ? U8.t(Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U8.t(Short.valueOf(dataMask[0]))));
				} else {
					mb.setMasked(MatchField.IP_DSCP, IpDscp.of(dataMask[0].contains("0x") ? U8.t(Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U8.t(Short.valueOf(dataMask[0]))), 
							IpDscp.of(dataMask[1].contains("0x") ? U8.t(Short.valueOf(dataMask[1].replaceFirst("0x", ""), 16)) : U8.t(Short.valueOf(dataMask[1]))));
				}
				break;
				if (mb.get(MatchField.IP_PROTO) == null) {
				} else {
					if (dataMask.length == 1) {
						mb.setExact(MatchField.SCTP_DST, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])));
					} else {
						mb.setMasked(MatchField.SCTP_DST, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])), 
								TransportPort.of(dataMask[1].contains("0x") ? Integer.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[1])));
					}
				}
				break;
			case STR_SCTP_SRC:
				if (mb.get(MatchField.IP_PROTO) == null) {
				} else {
					if (dataMask.length == 1) {
						mb.setExact(MatchField.SCTP_SRC, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])));
					} else {
						mb.setMasked(MatchField.SCTP_SRC, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])), 
								TransportPort.of(dataMask[1].contains("0x") ? Integer.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[1])));
					}
				}
				break;
			case STR_UDP_DST:
				if (mb.get(MatchField.IP_PROTO) == null) {
				} else {
					if (dataMask.length == 1) {
						mb.setExact(MatchField.UDP_DST, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])));
					} else {
						mb.setMasked(MatchField.UDP_DST, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])), 
								TransportPort.of(dataMask[1].contains("0x") ? Integer.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[1])));
					}
				}
				break;
			case STR_UDP_SRC:
				if (mb.get(MatchField.IP_PROTO) == null) {
				} else {
					if (dataMask.length == 1) {
						mb.setExact(MatchField.UDP_SRC, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])));
					} else {
						mb.setMasked(MatchField.UDP_SRC, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])), 
								TransportPort.of(dataMask[1].contains("0x") ? Integer.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[1])));
					}
				}
				break;
			case STR_TCP_DST:
				if (mb.get(MatchField.IP_PROTO) == null) {
				} else {
					if (dataMask.length == 1) {
						mb.setExact(MatchField.TCP_DST, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])));
					} else {
						mb.setMasked(MatchField.TCP_DST, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])), 
								TransportPort.of(dataMask[1].contains("0x") ? Integer.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[1])));
					}
				}
				break;
			case STR_TCP_SRC:
				if (mb.get(MatchField.IP_PROTO) == null) {
				} else {
					if (dataMask.length == 1) {
						mb.setExact(MatchField.TCP_SRC, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])));
					} else {
						mb.setMasked(MatchField.TCP_SRC, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])), 
								TransportPort.of(dataMask[1].contains("0x") ? Integer.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[1])));
					}
				}
				break;
				if ((ipProto = mb.get(MatchField.IP_PROTO)) == null) {
				} else if (ipProto == IpProtocol.TCP){
					if (dataMask.length == 1) {
						mb.setExact(MatchField.TCP_DST, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])));
					} else {
						mb.setMasked(MatchField.TCP_DST, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])), 
								TransportPort.of(dataMask[1].contains("0x") ? Integer.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[1])));
					}
				} else if (ipProto == IpProtocol.UDP){
					if (dataMask.length == 1) {
						mb.setExact(MatchField.UDP_DST, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])));
					} else {
						mb.setMasked(MatchField.UDP_DST, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])), 
								TransportPort.of(dataMask[1].contains("0x") ? Integer.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[1])));
					}
				} else if (ipProto == IpProtocol.SCTP){
					if (dataMask.length == 1) {
						mb.setExact(MatchField.SCTP_DST, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])));
					} else {
						mb.setMasked(MatchField.SCTP_DST, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])), 
								TransportPort.of(dataMask[1].contains("0x") ? Integer.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[1])));
					}
				}
				break;
			case STR_TP_SRC:
				if ((ipProto = mb.get(MatchField.IP_PROTO)) == null) {
				}  else if (ipProto == IpProtocol.TCP){
					if (dataMask.length == 1) {
						mb.setExact(MatchField.TCP_SRC, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])));
					} else {
						mb.setMasked(MatchField.TCP_SRC, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])), 
								TransportPort.of(dataMask[1].contains("0x") ? Integer.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[1])));
					}
				} else if (ipProto == IpProtocol.UDP){
					if (dataMask.length == 1) {
						mb.setExact(MatchField.UDP_SRC, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])));
					} else {
						mb.setMasked(MatchField.UDP_SRC, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])), 
								TransportPort.of(dataMask[1].contains("0x") ? Integer.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[1])));
					}
				} else if (ipProto == IpProtocol.SCTP){
					if (dataMask.length == 1) {
						mb.setExact(MatchField.SCTP_SRC, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])));
					} else {
						mb.setMasked(MatchField.SCTP_SRC, TransportPort.of(dataMask[0].contains("0x") ? Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[0])), 
								TransportPort.of(dataMask[1].contains("0x") ? Integer.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Integer.valueOf(dataMask[1])));
					}
				}
				break;
			case STR_ICMP_TYPE:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.ICMPV4_TYPE, ICMPv4Type.of(dataMask[0].contains("0x") ? Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Short.valueOf(dataMask[0])));
				} else {
					mb.setMasked(MatchField.ICMPV4_TYPE, ICMPv4Type.of(dataMask[0].contains("0x") ? Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Short.valueOf(dataMask[0])), 
							ICMPv4Type.of(dataMask[1].contains("0x") ? Short.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Short.valueOf(dataMask[1])));
				}
				break;
			case STR_ICMP_CODE:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.ICMPV4_CODE, ICMPv4Code.of(dataMask[0].contains("0x") ? Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Short.valueOf(dataMask[0])));
				} else {
					mb.setMasked(MatchField.ICMPV4_CODE, ICMPv4Code.of(dataMask[0].contains("0x") ? Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16) : Short.valueOf(dataMask[0])), 
							ICMPv4Code.of(dataMask[1].contains("0x") ? Short.valueOf(dataMask[1].replaceFirst("0x", ""), 16) : Short.valueOf(dataMask[1])));
				}
				break;
			case STR_ICMPV6_TYPE:
				if (ver10 == true) {
					throw new IllegalArgumentException("OF Version incompatible");
				}
				if (dataMask.length == 1) {
					mb.setExact(MatchField.ICMPV6_TYPE, dataMask[0].contains("0x") ? U8.of(Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U8.of(Short.valueOf(dataMask[0])));
				} else {
					mb.setMasked(MatchField.ICMPV6_TYPE, dataMask[0].contains("0x") ? U8.of(Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U8.of(Short.valueOf(dataMask[0])), 
							dataMask[1].contains("0x") ? U8.of(Short.valueOf(dataMask[1].replaceFirst("0x", ""), 16)) : U8.of(Short.valueOf(dataMask[1])));
				}
				break;
			case STR_ICMPV6_CODE:
				if (ver10 == true) {
					throw new IllegalArgumentException("OF Version incompatible");
				}
				if (dataMask.length == 1) {
					mb.setExact(MatchField.ICMPV6_CODE, dataMask[0].contains("0x") ? U8.of(Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U8.of(Short.valueOf(dataMask[0])));
				} else {
					mb.setMasked(MatchField.ICMPV6_CODE, dataMask[0].contains("0x") ? U8.of(Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U8.of(Short.valueOf(dataMask[0])), 
							dataMask[1].contains("0x") ? U8.of(Short.valueOf(dataMask[1].replaceFirst("0x", ""), 16)) : U8.of(Short.valueOf(dataMask[1])));
				}
				break;
			case STR_IPV6_ND_SSL:
				if (ver10 == true) {
					throw new IllegalArgumentException("OF Version incompatible");
				}
				if (dataMask.length == 1) {
					mb.setExact(MatchField.IPV6_ND_SLL, MacAddress.of(dataMask[0]));
				} else {
					mb.setMasked(MatchField.IPV6_ND_SLL, MacAddress.of(dataMask[0]), MacAddress.of(dataMask[1]));
				}
				break;
			case STR_IPV6_ND_TTL:
				if (ver10 == true) {
					throw new IllegalArgumentException("OF Version incompatible");
				}
				if (dataMask.length == 1) {
					mb.setExact(MatchField.IPV6_ND_TLL, MacAddress.of(dataMask[0]));
				} else {
					mb.setMasked(MatchField.IPV6_ND_TLL, MacAddress.of(dataMask[0]), MacAddress.of(dataMask[1]));
				}
				break;
			case STR_IPV6_ND_TARGET:
				if (ver10 == true) {
					throw new IllegalArgumentException("OF Version incompatible");
				}
				mb.setMasked(MatchField.IPV6_ND_TARGET, IPv6AddressWithMask.of(key_value[1]));
				break;
			case STR_ARP_OPCODE:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.ARP_OP, dataMask[0].contains("0x") ? ArpOpcode.of(Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : ArpOpcode.of(Integer.valueOf(dataMask[0])));
				} else {
					mb.setMasked(MatchField.ARP_OP, dataMask[0].contains("0x") ? ArpOpcode.of(Integer.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : ArpOpcode.of(Integer.valueOf(dataMask[0])), 
							dataMask[1].contains("0x") ? ArpOpcode.of(Integer.valueOf(dataMask[1].replaceFirst("0x", ""), 16)) : ArpOpcode.of(Integer.valueOf(dataMask[1])));
				}
				break;
			case STR_ARP_SHA:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.ARP_SHA, MacAddress.of(dataMask[0]));
				} else {
					mb.setMasked(MatchField.ARP_SHA, MacAddress.of(dataMask[0]), MacAddress.of(dataMask[1]));
				}
				break;
			case STR_ARP_DHA:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.ARP_THA, MacAddress.of(dataMask[0]));
				} else {
					mb.setMasked(MatchField.ARP_THA, MacAddress.of(dataMask[0]), MacAddress.of(dataMask[1]));
				}
				break;
			case STR_ARP_SPA:
				mb.setMasked(MatchField.ARP_SPA, IPv4AddressWithMask.of(key_value[1]));
				break;
			case STR_ARP_DPA:
				mb.setMasked(MatchField.ARP_TPA, IPv4AddressWithMask.of(key_value[1]));
				break;
			case STR_MPLS_LABEL:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.MPLS_LABEL, dataMask[0].contains("0x") ? U32.of(Long.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U32.of(Long.valueOf(dataMask[0])));
				} else {
					mb.setMasked(MatchField.MPLS_LABEL, dataMask[0].contains("0x") ? U32.of(Long.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U32.of(Long.valueOf(dataMask[0])), 
							dataMask[1].contains("0x") ? U32.of(Long.valueOf(dataMask[1].replaceFirst("0x", ""), 16)) : U32.of(Long.valueOf(dataMask[1])));
				}
				break;
			case STR_MPLS_TC:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.MPLS_TC, dataMask[0].contains("0x") ? U8.of(Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U8.of(Short.valueOf(dataMask[0])));
				} else {
					mb.setMasked(MatchField.MPLS_TC, dataMask[0].contains("0x") ? U8.of(Short.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U8.of(Short.valueOf(dataMask[0])), 
							dataMask[1].contains("0x") ? U8.of(Short.valueOf(dataMask[1].replaceFirst("0x", ""), 16)) : U8.of(Short.valueOf(dataMask[1])));
				}
				break;
			case STR_MPLS_BOS:
				mb.setExact(MatchField.MPLS_BOS, key_value[1].equalsIgnoreCase("true") ? OFBooleanValue.TRUE : OFBooleanValue.FALSE);
				break;
			case STR_METADATA:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.METADATA, dataMask[0].contains("0x") ? OFMetadata.ofRaw(Long.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : OFMetadata.ofRaw(Long.valueOf(dataMask[0])));
				} else {
					mb.setMasked(MatchField.METADATA, dataMask[0].contains("0x") ? OFMetadata.ofRaw(Long.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : OFMetadata.ofRaw(Long.valueOf(dataMask[0])), 
							dataMask[1].contains("0x") ? OFMetadata.ofRaw(Long.valueOf(dataMask[1].replaceFirst("0x", ""), 16)) : OFMetadata.ofRaw(Long.valueOf(dataMask[1])));
				}
				break;
			case STR_TUNNEL_ID:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.TUNNEL_ID, dataMask[0].contains("0x") ? U64.of(Long.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U64.of(Long.valueOf(dataMask[0])));
				} else {
					mb.setMasked(MatchField.TUNNEL_ID, dataMask[0].contains("0x") ? U64.of(Long.valueOf(dataMask[0].replaceFirst("0x", ""), 16)) : U64.of(Long.valueOf(dataMask[0])), 
							dataMask[1].contains("0x") ? U64.of(Long.valueOf(dataMask[1].replaceFirst("0x", ""), 16)) : U64.of(Long.valueOf(dataMask[1])));
				}
				break;
			case STR_TUNNEL_IPV4_SRC:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.TUNNEL_IPV4_SRC, IPv4Address.of(key_value[1]));
				} else {
					mb.setMasked(MatchField.TUNNEL_IPV4_SRC, IPv4AddressWithMask.of(key_value[1]));
				}
				break;
			case STR_TUNNEL_IPV4_DST:
				if (dataMask.length == 1) {
					mb.setExact(MatchField.TUNNEL_IPV4_DST, IPv4Address.of(key_value[1]));
				} else {
					mb.setMasked(MatchField.TUNNEL_IPV4_DST, IPv4AddressWithMask.of(key_value[1]));
				}
				break;
			case STR_PBB_ISID:
				if (key_value[1].startsWith("0x")) {
					mb.setExact(MatchField., U64.of(Long.parseLong(key_value[1].replaceFirst("0x", ""), 16)));
				} else {
					mb.setExact(MatchField., U64.of(Long.parseLong(key_value[1])));
				break;
			default:
				throw new IllegalArgumentException("unknown token " + key_value + " parsing " + match);
			} 
		}
		return mb.build();
	}
}

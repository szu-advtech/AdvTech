package net.floodlightcontroller.firewall;
import com.fasterxml.jackson.databind.annotation.JsonSerialize;
import org.projectfloodlight.openflow.protocol.match.MatchField;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.EthType;
import org.projectfloodlight.openflow.types.IPv4AddressWithMask;
import org.projectfloodlight.openflow.types.IpProtocol;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.TransportPort;
import net.floodlightcontroller.packet.Ethernet;
import net.floodlightcontroller.packet.IPacket;
import net.floodlightcontroller.packet.IPv4;
import net.floodlightcontroller.packet.TCP;
import net.floodlightcontroller.packet.UDP;
@JsonSerialize(using=FirewallRuleSerializer.class)
public class FirewallRule implements Comparable<FirewallRule> {
    @Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		FirewallRule other = (FirewallRule) obj;
		if (action != other.action)
			return false;
		if (any_dl_dst != other.any_dl_dst)
			return false;
		if (any_dl_src != other.any_dl_src)
			return false;
		if (any_dl_type != other.any_dl_type)
			return false;
		if (any_dpid != other.any_dpid)
			return false;
		if (any_in_port != other.any_in_port)
			return false;
		if (any_nw_dst != other.any_nw_dst)
			return false;
		if (any_nw_proto != other.any_nw_proto)
			return false;
		if (any_nw_src != other.any_nw_src)
			return false;
		if (any_tp_dst != other.any_tp_dst)
			return false;
		if (any_tp_src != other.any_tp_src)
			return false;
		if (dl_dst == null) {
			if (other.dl_dst != null)
				return false;
		} else if (!dl_dst.equals(other.dl_dst))
			return false;
		if (dl_src == null) {
			if (other.dl_src != null)
				return false;
		} else if (!dl_src.equals(other.dl_src))
			return false;
		if (dl_type == null) {
			if (other.dl_type != null)
				return false;
		} else if (!dl_type.equals(other.dl_type))
			return false;
		if (dpid == null) {
			if (other.dpid != null)
				return false;
		} else if (!dpid.equals(other.dpid))
			return false;
		if (in_port == null) {
			if (other.in_port != null)
				return false;
		} else if (!in_port.equals(other.in_port))
			return false;
		if (nw_dst_prefix_and_mask == null) {
			if (other.nw_dst_prefix_and_mask != null)
				return false;
		} else if (!nw_dst_prefix_and_mask.equals(other.nw_dst_prefix_and_mask))
			return false;
		if (nw_proto == null) {
			if (other.nw_proto != null)
				return false;
		} else if (!nw_proto.equals(other.nw_proto))
			return false;
		if (nw_src_prefix_and_mask == null) {
			if (other.nw_src_prefix_and_mask != null)
				return false;
		} else if (!nw_src_prefix_and_mask.equals(other.nw_src_prefix_and_mask))
			return false;
		if (priority != other.priority)
			return false;
		if (ruleid != other.ruleid)
			return false;
		if (tp_dst == null) {
			if (other.tp_dst != null)
				return false;
		} else if (!tp_dst.equals(other.tp_dst))
			return false;
		if (tp_src == null) {
			if (other.tp_src != null)
				return false;
		} else if (!tp_src.equals(other.tp_src))
			return false;
		return true;
	}
	public int ruleid;
    public DatapathId dpid; 
    public OFPort in_port; 
    public MacAddress dl_src; 
    public MacAddress dl_dst; 
    public EthType dl_type; 
    public IPv4AddressWithMask nw_src_prefix_and_mask; 
    public IPv4AddressWithMask nw_dst_prefix_and_mask;
    public IpProtocol nw_proto;
    public TransportPort tp_src;
    public TransportPort tp_dst;
    public boolean any_dpid;
    public boolean any_in_port; 
    public boolean any_dl_src;
    public boolean any_dl_dst;
    public boolean any_dl_type;
    public boolean any_nw_src;
    public boolean any_nw_dst;
    public boolean any_nw_proto;
    public boolean any_tp_src;
    public boolean any_tp_dst;
    public int priority = 0;
    public FirewallAction action;
    public enum FirewallAction {
        DROP, ALLOW
    }
    public FirewallRule() {
        this.dpid = DatapathId.NONE;
        this.in_port = OFPort.ANY; 
        this.dl_src = MacAddress.NONE;
        this.dl_dst = MacAddress.NONE;
        this.dl_type = EthType.NONE;
        this.nw_src_prefix_and_mask = IPv4AddressWithMask.NONE;
        this.nw_dst_prefix_and_mask = IPv4AddressWithMask.NONE;
        this.nw_proto = IpProtocol.NONE;
        this.tp_src = TransportPort.NONE;
        this.tp_dst = TransportPort.NONE;
        this.any_dpid = true; 
        this.any_in_port = true; 
        this.any_dl_src = true; 
        this.any_dl_dst = true; 
        this.any_dl_type = true; 
        this.any_nw_src = true; 
        this.any_nw_dst = true; 
        this.any_nw_proto = true; 
        this.any_tp_src = true; 
        this.any_tp_dst = true;
        this.priority = 0; 
        this.action = FirewallAction.ALLOW; 
        this.ruleid = 0; 
    }
    public int genID() {
        int uid = this.hashCode();
        if (uid < 0) {
            uid = Math.abs(uid);
        }
        return uid;
    }
    @Override
    public int compareTo(FirewallRule rule) {
        return this.priority - rule.priority;
    }
    public boolean isSameAs(FirewallRule r) {
        if (this.action != r.action
                || this.any_dl_type != r.any_dl_type
                || (this.any_dl_type == false && !this.dl_type.equals(r.dl_type))
                || this.any_tp_src != r.any_tp_src
                || (this.any_tp_src == false && !this.tp_src.equals(r.tp_src))
                || this.any_tp_dst != r.any_tp_dst
                || (this.any_tp_dst == false && !this.tp_dst.equals(r.tp_dst))
                || this.any_dpid != r.any_dpid
                || (this.any_dpid == false && !this.dpid.equals(r.dpid))
                || this.any_in_port != r.any_in_port
                || (this.any_in_port == false && !this.in_port.equals(r.in_port))
                || this.any_nw_src != r.any_nw_src
                || (this.any_nw_src == false && !this.nw_src_prefix_and_mask.equals(r.nw_src_prefix_and_mask))
                || this.any_dl_src != r.any_dl_src
                || (this.any_dl_src == false && !this.dl_src.equals(r.dl_src))
                || this.any_nw_proto != r.any_nw_proto
                || (this.any_nw_proto == false && !this.nw_proto.equals(r.nw_proto))
                || this.any_nw_dst != r.any_nw_dst
                || (this.any_nw_dst == false && !this.nw_dst_prefix_and_mask.equals(r.nw_dst_prefix_and_mask))
                || this.any_dl_dst != r.any_dl_dst                
                || (this.any_dl_dst == false && this.dl_dst != r.dl_dst)) {
            return false;
        }
        return true;
    }
    public boolean matchesThisPacket(DatapathId switchDpid, OFPort inPort, Ethernet packet, AllowDropPair adp) {
        IPacket pkt = packet.getPayload();
        IPv4 pkt_ip = null;
        TCP pkt_tcp = null;
        UDP pkt_udp = null;
        TransportPort pkt_tp_src = TransportPort.NONE;
        TransportPort pkt_tp_dst = TransportPort.NONE;
        if (any_dpid == false && !dpid.equals(switchDpid))
            return false;
        if (any_in_port == false && !in_port.equals(inPort))
            return false;
        if (action == FirewallRule.FirewallAction.DROP) {
        	if (!OFPort.ANY.equals(this.in_port)) {
        		adp.drop.setExact(MatchField.IN_PORT, this.in_port);
        	}
        } else {
        	if (!OFPort.ANY.equals(this.in_port)) {
        		adp.allow.setExact(MatchField.IN_PORT, this.in_port);
        	}
        }
        if (any_dl_src == false && !dl_src.equals(packet.getSourceMACAddress()))
            return false;
        if (action == FirewallRule.FirewallAction.DROP) {
        	if (!MacAddress.NONE.equals(this.dl_src)) {
        		adp.drop.setExact(MatchField.ETH_SRC, this.dl_src);
        	}
        } else {
        	if (!MacAddress.NONE.equals(this.dl_src)) {
        		adp.allow.setExact(MatchField.ETH_SRC, this.dl_src);
        	}
        }
        if (any_dl_dst == false && !dl_dst.equals(packet.getDestinationMACAddress()))
            return false;
        if (action == FirewallRule.FirewallAction.DROP) {
        	if (!MacAddress.NONE.equals(this.dl_dst)) {
        		adp.drop.setExact(MatchField.ETH_DST, this.dl_dst);
        	}
        } else {
        	if (!MacAddress.NONE.equals(this.dl_dst)) {
        		adp.allow.setExact(MatchField.ETH_DST, this.dl_dst);
        	}
        }
        if (any_dl_type == false) {
            if (dl_type.equals(EthType.ARP)) {
                    return false;
                else {
                    if (action == FirewallRule.FirewallAction.DROP) {
                    	if (!EthType.NONE.equals(this.dl_type)) {
                    		adp.drop.setExact(MatchField.ETH_TYPE, this.dl_type);
                    	}
                    } else {
                    	if (!EthType.NONE.equals(this.dl_type)) {
                    		adp.allow.setExact(MatchField.ETH_TYPE, this.dl_type);
                    	}
                    }
                }
            } else if (dl_type.equals(EthType.IPv4)) {
                    return false;
                else {
                    if (action == FirewallRule.FirewallAction.DROP) {
                    	if (!IpProtocol.NONE.equals(this.nw_proto)) {
                    		adp.drop.setExact(MatchField.IP_PROTO, this.nw_proto);
                    	}
                    } else {
                    	if (!IpProtocol.NONE.equals(this.nw_proto)) {
                    		adp.allow.setExact(MatchField.IP_PROTO, this.nw_proto);
                    	}
                    }
                    pkt_ip = (IPv4) pkt;
                    if (any_nw_src == false && !nw_src_prefix_and_mask.matches(pkt_ip.getSourceAddress()))
                        return false;
                    if (action == FirewallRule.FirewallAction.DROP) {
                    	if (!IPv4AddressWithMask.NONE.equals(this.nw_src_prefix_and_mask)) {
                    		adp.drop.setMasked(MatchField.IPV4_SRC, nw_src_prefix_and_mask);
                    	}
                    } else {
                    	if (!IPv4AddressWithMask.NONE.equals(this.nw_src_prefix_and_mask)) {
                    		adp.allow.setMasked(MatchField.IPV4_SRC, nw_src_prefix_and_mask);
                    	}
                    }
                    if (any_nw_dst == false && !nw_dst_prefix_and_mask.matches(pkt_ip.getDestinationAddress()))
                        return false;
                    if (action == FirewallRule.FirewallAction.DROP) {
                    	if (!IPv4AddressWithMask.NONE.equals(this.nw_dst_prefix_and_mask)) {
                    		adp.drop.setMasked(MatchField.IPV4_DST, nw_dst_prefix_and_mask);
                    	}
                    } else {
                    	if (!IPv4AddressWithMask.NONE.equals(this.nw_dst_prefix_and_mask)) {
                    		adp.allow.setMasked(MatchField.IPV4_DST, nw_dst_prefix_and_mask);
                    	}
                    }
                    if (any_nw_proto == false) {
                        if (nw_proto.equals(IpProtocol.TCP)) {
                            if (!pkt_ip.getProtocol().equals(IpProtocol.TCP)) {
                                return false;
                            } else {
                                pkt_tcp = (TCP) pkt_ip.getPayload();
                                pkt_tp_src = pkt_tcp.getSourcePort();
                                pkt_tp_dst = pkt_tcp.getDestinationPort();
                            }
                        } else if (nw_proto.equals(IpProtocol.UDP)) {
                            if (!pkt_ip.getProtocol().equals(IpProtocol.UDP)) {
                                return false;
                            } else {
                                pkt_udp = (UDP) pkt_ip.getPayload();
                                pkt_tp_src = pkt_udp.getSourcePort();
                                pkt_tp_dst = pkt_udp.getDestinationPort();
                            }
                        } else if (nw_proto.equals(IpProtocol.ICMP)) {
                            if (!pkt_ip.getProtocol().equals(IpProtocol.ICMP)) {
                                return false;
                            } else {
                            }
                        }
                        if (action == FirewallRule.FirewallAction.DROP) {
                        	if (!IpProtocol.NONE.equals(this.nw_proto)) {
                        		adp.drop.setExact(MatchField.IP_PROTO, this.nw_proto);
                        	}
                        } else {
                        	if (!IpProtocol.NONE.equals(this.nw_proto)) {
                        		adp.allow.setExact(MatchField.IP_PROTO, this.nw_proto);
                        	}
                        }
                        if (pkt_tcp != null || pkt_udp != null) {
                            if (tp_src.getPort() != 0 && tp_src.getPort() != pkt_tp_src.getPort()) {
                                return false;
                            }
                            if (action == FirewallRule.FirewallAction.DROP) {
                                if (pkt_tcp != null) {
                                	if (!TransportPort.NONE.equals(this.tp_src)) {
                                		adp.drop.setExact(MatchField.TCP_SRC, this.tp_src);
                                	}
                                } else {
                                	if (!TransportPort.NONE.equals(this.tp_src)) {
                                		adp.drop.setExact(MatchField.UDP_SRC, this.tp_src);   
                                	}
                                }
                            } else {
                                if (pkt_tcp != null) {
                                	if (!TransportPort.NONE.equals(this.tp_src)) {
                                		adp.allow.setExact(MatchField.TCP_SRC, this.tp_src);
                                	}
                                } else {
                                	if (!TransportPort.NONE.equals(this.tp_src)) {
                                		adp.allow.setExact(MatchField.UDP_SRC, this.tp_src);   
                                	}
                                }
                            }
                            if (tp_dst.getPort() != 0 && tp_dst.getPort() != pkt_tp_dst.getPort()) {
                                return false;
                            }
                            if (action == FirewallRule.FirewallAction.DROP) {
                                if (pkt_tcp != null) {
                                	if (!TransportPort.NONE.equals(this.tp_dst)) {
                                		adp.drop.setExact(MatchField.TCP_DST, this.tp_dst);
                                	}
                                } else {
                                	if (!TransportPort.NONE.equals(this.tp_dst)) {
                                		adp.drop.setExact(MatchField.UDP_DST, this.tp_dst);   
                                	}
                                }
                            } else {
                            	if (pkt_tcp != null) {
                                	if (!TransportPort.NONE.equals(this.tp_dst)) {
                                		adp.allow.setExact(MatchField.TCP_DST, this.tp_dst);
                                	}
                                } else {
                                	if (!TransportPort.NONE.equals(this.tp_dst)) {
                                		adp.allow.setExact(MatchField.UDP_DST, this.tp_dst);   
                                	}
                                }
                            }
                        }
                    }
                }
            } else {
                return false;
            }
        }
        if (action == FirewallRule.FirewallAction.DROP) {
        	if (!EthType.NONE.equals(this.dl_type)) {
        		adp.drop.setExact(MatchField.ETH_TYPE, this.dl_type);
        	}
        } else {
        	if (!EthType.NONE.equals(this.dl_type)) {
        		adp.allow.setExact(MatchField.ETH_TYPE, this.dl_type);
        	}
        }
        return true;
    }
    @Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime
				+ ((nw_dst_prefix_and_mask == null) ? 0
						: nw_dst_prefix_and_mask.hashCode());
				+ ((nw_proto == null) ? 0 : nw_proto.hashCode());
		result = prime
				+ ((nw_src_prefix_and_mask == null) ? 0
						: nw_src_prefix_and_mask.hashCode());
		return result;
	}
}

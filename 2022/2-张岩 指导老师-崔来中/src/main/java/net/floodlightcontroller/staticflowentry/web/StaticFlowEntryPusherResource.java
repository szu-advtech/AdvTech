package net.floodlightcontroller.staticflowentry.web;
import java.io.IOException;
import java.util.Map;
import org.projectfloodlight.openflow.types.DatapathId;
import org.restlet.resource.Delete;
import org.restlet.resource.Post;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.floodlightcontroller.staticflowentry.StaticFlowEntries;
import net.floodlightcontroller.staticflowentry.StaticFlowEntryPusher;
import net.floodlightcontroller.storage.IStorageSourceService;
import net.floodlightcontroller.util.MatchUtils;
public class StaticFlowEntryPusherResource extends ServerResource {
	protected static Logger log = LoggerFactory.getLogger(StaticFlowEntryPusherResource.class);
	private int checkFlow(Map<String, Object> rows) {    
		int state = 0;
		boolean dl_type = false;
		boolean nw_proto = false;
		boolean nw_layer = false;
		boolean icmp6_type = false;
		boolean icmp6_code = false;
		boolean nd_target = false;
		boolean nd_sll = false;
		boolean nd_tll = false; 
		boolean ip6 = false;
		boolean ip4 = false;
		int eth_type = -1;
		int nw_protocol = -1;
		int icmp_type = -1;
		if (rows.containsKey(StaticFlowEntryPusher.COLUMN_DL_TYPE)) {
			if (((String) rows.get(StaticFlowEntryPusher.COLUMN_DL_TYPE)).startsWith("0x")) {
				eth_type = Integer.parseInt(((String) rows.get(StaticFlowEntryPusher.COLUMN_DL_TYPE)).replaceFirst("0x", ""), 16);
				dl_type = true;
			} else {
				eth_type = Integer.parseInt((String) rows.get(StaticFlowEntryPusher.COLUMN_DL_TYPE));
				dl_type = true;
			}
				ip6 = true;
				dl_type = true;
				ip4 = true;
				dl_type = true;
			}	
		}
		if (rows.containsKey(StaticFlowEntryPusher.COLUMN_NW_DST) ||
				rows.containsKey(StaticFlowEntryPusher.COLUMN_NW_SRC)) {
			nw_layer = true;
			ip4 = true;
		}
		if (rows.containsKey(StaticFlowEntryPusher.COLUMN_ICMP_CODE) ||
				rows.containsKey(StaticFlowEntryPusher.COLUMN_ICMP_TYPE) ||
				rows.containsKey(StaticFlowEntryPusher.COLUMN_ARP_DHA) ||
				rows.containsKey(StaticFlowEntryPusher.COLUMN_ARP_SHA) ||
				rows.containsKey(StaticFlowEntryPusher.COLUMN_ARP_SPA) ||
				rows.containsKey(StaticFlowEntryPusher.COLUMN_ARP_DPA) ||
				rows.containsKey(StaticFlowEntryPusher.COLUMN_ARP_OPCODE)) {
			ip4 = true;
		}
		if (rows.containsKey(StaticFlowEntryPusher.COLUMN_IPV6_FLOW_LABEL) || 
				rows.containsKey(StaticFlowEntryPusher.COLUMN_NW6_SRC) ||
				rows.containsKey(StaticFlowEntryPusher.COLUMN_NW6_DST)) {
			nw_layer = true;
			ip6 = true;
		}
		if (rows.containsKey(StaticFlowEntryPusher.COLUMN_NW_PROTO)) {
			nw_proto = true;
			if (((String) rows.get(StaticFlowEntryPusher.COLUMN_NW_PROTO)).startsWith("0x")) {
				nw_protocol = Integer.parseInt(((String) rows.get(StaticFlowEntryPusher.COLUMN_NW_PROTO)).replaceFirst("0x", ""), 16);
			} else {
				nw_protocol = Integer.parseInt((String) rows.get(StaticFlowEntryPusher.COLUMN_NW_PROTO));
			}
		}
		if (rows.containsKey(StaticFlowEntryPusher.COLUMN_ICMP6_CODE)) {
			icmp6_code = true;
			ip6 = true;
		}
		if (rows.containsKey(StaticFlowEntryPusher.COLUMN_ICMP6_TYPE)) {
			icmp6_type = true;
			ip6 = true;
			if (((String) rows.get(StaticFlowEntryPusher.COLUMN_ICMP6_TYPE)).startsWith("0x")) {
				icmp_type = Integer.parseInt(((String) rows.get(StaticFlowEntryPusher.COLUMN_ICMP6_TYPE)).replaceFirst("0x", ""), 16);
			} else {
				icmp_type = Integer.parseInt((String) rows.get(StaticFlowEntryPusher.COLUMN_ICMP6_TYPE));
			}
		}
		if (rows.containsKey(StaticFlowEntryPusher.COLUMN_ND_SLL)) {
			nd_sll = true;
			ip6 = true;
		}
		if (rows.containsKey(StaticFlowEntryPusher.COLUMN_ND_TLL)) {
			nd_tll = true;
			ip6 = true;
		}
		if (rows.containsKey(StaticFlowEntryPusher.COLUMN_ND_TARGET)) {
			nd_target = true;
			ip6 = true;
		}    
		if (nw_layer == true || nw_proto == true) {
			if (dl_type == true) {
				if (!(ip4 == true || ip6 == true)) {
					state = 2;    
					return state;
				}
			}
			else {
				state = 1;    
				return state;
			}
		}
		if (icmp6_type == true || icmp6_code == true ) {
			if (nw_proto == true) {
					state = 4;    
					return state;
				}
			}
			else {
				state = 3;    
				return state;
			}
		}
		if (nd_sll == true || nd_tll == true || nd_target == true) {
			if (icmp6_type == true) {
				if (nd_target == true) {
						state = 6;
						return state;
					}
				}
				else if (nd_tll == true) {
					if (!(icmp_type == 136)) {
						state = 6;
						return state;
					}
				}
				else if (nd_sll == true) {
					if (!(icmp_type == 135)) {
						state = 6;
						return state;
					}
				}
			}
			else {
				state = 5;    
				return state;
			}
		}
		int result = checkActions(rows);
		if ((ip4 == true && ip6 == true) || (result == -1) ||
				(result == 1 && ip6 == true) || (result == 2 && ip4 == true)) {
			state = 7;    
			return state;
		}
		if (rows.containsKey(StaticFlowEntryPusher.COLUMN_SWITCH)) {
			try {
				DatapathId.of((String) rows.get(StaticFlowEntryPusher.COLUMN_SWITCH));
			} catch (Exception e) {
				state = 9;
			}
		} else {
			state = 8;
		}
		return state;
	}
	public static int checkActions(Map<String, Object> entry) {
		boolean ip6 = false;
		boolean ip4 = false;
		String actions = null;
		if (entry.containsKey(StaticFlowEntryPusher.COLUMN_ACTIONS) || 
				entry.containsKey(StaticFlowEntryPusher.COLUMN_INSTR_APPLY_ACTIONS) ||
				entry.containsKey(StaticFlowEntryPusher.COLUMN_INSTR_WRITE_ACTIONS)) {
			if (entry.containsKey(StaticFlowEntryPusher.COLUMN_ACTIONS)) {
				actions = (String) entry.get(StaticFlowEntryPusher.COLUMN_ACTIONS);
			}
			else if (entry.containsKey(StaticFlowEntryPusher.COLUMN_INSTR_APPLY_ACTIONS)) {
				actions = (String) entry.get(StaticFlowEntryPusher.COLUMN_INSTR_APPLY_ACTIONS);
			}
			else if (entry.containsKey(StaticFlowEntryPusher.COLUMN_INSTR_WRITE_ACTIONS)) {
				actions = (String) entry.get(StaticFlowEntryPusher.COLUMN_INSTR_WRITE_ACTIONS);
			}
			if (actions.contains(MatchUtils.STR_ICMPV6_CODE) || actions.contains(MatchUtils.STR_ICMPV6_TYPE) ||
					actions.contains(MatchUtils.STR_IPV6_DST) || actions.contains(MatchUtils.STR_IPV6_SRC) || 
					actions.contains(MatchUtils.STR_IPV6_FLOW_LABEL) || actions.contains(MatchUtils.STR_IPV6_ND_SSL) ||
					actions.contains(MatchUtils.STR_IPV6_ND_TARGET) || actions.contains(MatchUtils.STR_IPV6_ND_TTL)) {
				ip6 = true;
			}
			if (actions.contains(MatchUtils.STR_NW_SRC) || actions.contains(MatchUtils.STR_NW_DST) || 
					actions.contains(MatchUtils.STR_ARP_OPCODE) || actions.contains(MatchUtils.STR_ARP_SHA) || 
					actions.contains(MatchUtils.STR_ARP_DHA) || actions.contains(MatchUtils.STR_ARP_SPA) || 
					actions.contains(MatchUtils.STR_ARP_DPA) || actions.contains(MatchUtils.STR_ICMP_CODE) || 
					actions.contains(MatchUtils.STR_ICMP_TYPE)) {
				ip4 = true;
			}
		}
		if (ip6 == false && ip4 == false) {
		} else if (ip6 == false && ip4 == true) {
		} else if (ip6 == true && ip4 == false) {
		} else {
		}
	}
	@Post
	public String store(String fmJson) {
		IStorageSourceService storageSource =
				(IStorageSourceService)getContext().getAttributes().
				get(IStorageSourceService.class.getCanonicalName());
		Map<String, Object> rowValues;
		try {
			rowValues = StaticFlowEntries.jsonToStorageEntry(fmJson);
			String status = null;
			int state = checkFlow(rowValues);
			if (state == 1) {
				status = "Warning! Must specify eth_type of IPv4/IPv6 to " +
						"match on IPv4/IPv6 fields! The flow has been discarded.";
				log.error(status);
			} else if (state == 2) {
				status = "Warning! eth_type not recognized! The flow has been discarded.";
				log.error(status);
			} else if (state == 3) {
				status = "Warning! Must specify ip_proto to match! The flow has been discarded.";
				log.error(status);
			} else if (state == 4) {
				status = "Warning! ip_proto invalid! The flow has been discarded.";
				log.error(status);
			} else if (state == 5) {
				status = "Warning! Must specify icmp6_type to match! The flow has been discarded.";
				log.error(status);
			} else if (state == 6) {
				status = "Warning! icmp6_type invalid! The flow has been discarded.";
				log.error(status);
			} else if (state == 7) {
				status = "Warning! IPv4 & IPv6 fields cannot be specified in the same flow! The flow has been discarded.";
				log.error(status);
			} else if (state == 8) {
				status = "Warning! Must specify switch DPID in flow. The flow has been discarded.";
				log.error(status);
			} else if (state == 9) {
				status = "Warning! Switch DPID invalid! The flow has been discarded.";
				log.error(status);
			} else if (state == 0) {
				status = "Entry pushed";            
				storageSource.insertRowAsync(StaticFlowEntryPusher.TABLE_NAME, rowValues);
			}
			return ("{\"status\" : \"" + status + "\"}");
		} catch (IOException e) {
			log.error("Error parsing push flow mod request: " + fmJson, e);
			return "{\"status\" : \"Error! Could not parse flow mod, see log for details.\"}";
		}        
	}
	@Delete
	public String del(String fmJson) {
		IStorageSourceService storageSource =
				(IStorageSourceService)getContext().getAttributes().
				get(IStorageSourceService.class.getCanonicalName());
		String fmName = null;
		if (fmJson == null) {
			return "{\"status\" : \"Error! No data posted.\"}";
		}
		try {
			fmName = StaticFlowEntries.getEntryNameFromJson(fmJson);
			if (fmName == null) {
				return "{\"status\" : \"Error deleting entry, no name provided\"}";
			}
		} catch (IOException e) {
			log.error("Error deleting flow mod request: " + fmJson, e);
			return "{\"status\" : \"Error deleting entry, see log for details\"}";
		}
		storageSource.deleteRowAsync(StaticFlowEntryPusher.TABLE_NAME, fmName);
		return "{\"status\" : \"Entry " + fmName + " deleted\"}";
	}
}

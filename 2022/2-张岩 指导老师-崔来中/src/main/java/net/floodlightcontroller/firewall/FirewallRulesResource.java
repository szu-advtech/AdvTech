package net.floodlightcontroller.firewall;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.MappingJsonFactory;
import org.projectfloodlight.openflow.types.DatapathId;
import org.projectfloodlight.openflow.types.EthType;
import org.projectfloodlight.openflow.types.IPv4AddressWithMask;
import org.projectfloodlight.openflow.types.IpProtocol;
import org.projectfloodlight.openflow.types.MacAddress;
import org.projectfloodlight.openflow.types.OFPort;
import org.projectfloodlight.openflow.types.TransportPort;
import org.restlet.resource.Delete;
import org.restlet.resource.Post;
import org.restlet.resource.Get;
import org.restlet.resource.ServerResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
public class FirewallRulesResource extends ServerResource {
	protected static Logger log = LoggerFactory.getLogger(FirewallRulesResource.class);
	@Get("json")
	public List<FirewallRule> retrieve() {
		IFirewallService firewall =
				(IFirewallService)getContext().getAttributes().
				get(IFirewallService.class.getCanonicalName());
		return firewall.getRules();
	}
	@Post
	public String store(String fmJson) {
		IFirewallService firewall =
				(IFirewallService)getContext().getAttributes().
				get(IFirewallService.class.getCanonicalName());
		FirewallRule rule = jsonToFirewallRule(fmJson);
		if (rule == null) {
			return "{\"status\" : \"Error! Could not parse firewall rule, see log for details.\"}";
		}
		String status = null;
		if (checkRuleExists(rule, firewall.getRules())) {
			status = "Error! A similar firewall rule already exists.";
			log.error(status);
			return ("{\"status\" : \"" + status + "\"}");
		} else {
			firewall.addRule(rule);
			status = "Rule added";
			return ("{\"status\" : \"" + status + "\", \"rule-id\" : \""+ Integer.toString(rule.ruleid) + "\"}");
		}
	}
	@Delete
	public String remove(String fmJson) {
		IFirewallService firewall =
				(IFirewallService)getContext().getAttributes().
				get(IFirewallService.class.getCanonicalName());
		FirewallRule rule = jsonToFirewallRule(fmJson);
		if (rule == null) {
			return "{\"status\" : \"Error! Could not parse firewall rule, see log for details.\"}";
		}
		String status = null;
		boolean exists = false;
		Iterator<FirewallRule> iter = firewall.getRules().iterator();
		while (iter.hasNext()) {
			FirewallRule r = iter.next();
			if (r.ruleid == rule.ruleid) {
				exists = true;
				break;
			}
		}
		if (!exists) {
			status = "Error! Can't delete, a rule with this ID doesn't exist.";
			log.error(status);
		} else {
			firewall.deleteRule(rule.ruleid);
			status = "Rule deleted";
		}
		return ("{\"status\" : \"" + status + "\"}");
	}
	public static FirewallRule jsonToFirewallRule(String fmJson) {
		FirewallRule rule = new FirewallRule();
		MappingJsonFactory f = new MappingJsonFactory();
		JsonParser jp;
		try {
			try {
				jp = f.createParser(fmJson);
			} catch (JsonParseException e) {
				throw new IOException(e);
			}
			jp.nextToken();
			if (jp.getCurrentToken() != JsonToken.START_OBJECT) {
				throw new IOException("Expected START_OBJECT");
			}
			while (jp.nextToken() != JsonToken.END_OBJECT) {
				if (jp.getCurrentToken() != JsonToken.FIELD_NAME) {
					throw new IOException("Expected FIELD_NAME");
				}
				String n = jp.getCurrentName();
				jp.nextToken();
				if (jp.getText().equals("")) {
					continue;
				}
				if (n.equalsIgnoreCase("ruleid")) {
					try {
						rule.ruleid = Integer.parseInt(jp.getText());
					} catch (IllegalArgumentException e) {
						log.error("Unable to parse rule ID: {}", jp.getText());
					}
				}
				else if (n.equalsIgnoreCase("switchid")) {
					rule.any_dpid = false;
					try {
						rule.dpid = DatapathId.of(jp.getText());
					} catch (NumberFormatException e) {
						log.error("Unable to parse switch DPID: {}", jp.getText());
					}
				}
				else if (n.equalsIgnoreCase("src-inport")) {
					rule.any_in_port = false;
					try {
						rule.in_port = OFPort.of(Integer.parseInt(jp.getText()));
					} catch (NumberFormatException e) {
						log.error("Unable to parse ingress port: {}", jp.getText());
					}
				}
				else if (n.equalsIgnoreCase("src-mac")) {
					if (!jp.getText().equalsIgnoreCase("ANY")) {
						rule.any_dl_src = false;
						try {
							rule.dl_src = MacAddress.of(jp.getText());
						} catch (IllegalArgumentException e) {
							log.error("Unable to parse source MAC: {}", jp.getText());
						}
					}
				}
				else if (n.equalsIgnoreCase("dst-mac")) {
					if (!jp.getText().equalsIgnoreCase("ANY")) {
						rule.any_dl_dst = false;
						try {
							rule.dl_dst = MacAddress.of(jp.getText());
						} catch (IllegalArgumentException e) {
							log.error("Unable to parse destination MAC: {}", jp.getText());
						}
					}
				}
				else if (n.equalsIgnoreCase("dl-type")) {
					if (jp.getText().equalsIgnoreCase("ARP")) {
						rule.any_dl_type = false;
						rule.dl_type = EthType.ARP;
					} else if (jp.getText().equalsIgnoreCase("IPv4")) {
						rule.any_dl_type = false;
						rule.dl_type = EthType.IPv4;
					}
				}
				else if (n.equalsIgnoreCase("src-ip")) {
					if (!jp.getText().equalsIgnoreCase("ANY")) {
						rule.any_nw_src = false;
						if (rule.dl_type.equals(EthType.NONE)){
							rule.any_dl_type = false;
							rule.dl_type = EthType.IPv4;
						}
						try {
							rule.nw_src_prefix_and_mask = IPv4AddressWithMask.of(jp.getText());
						} catch (IllegalArgumentException e) {
							log.error("Unable to parse source IP: {}", jp.getText());
						}
					}
				}
				else if (n.equalsIgnoreCase("dst-ip")) {
					if (!jp.getText().equalsIgnoreCase("ANY")) {
						rule.any_nw_dst = false;
						if (rule.dl_type.equals(EthType.NONE)){
							rule.any_dl_type = false;
							rule.dl_type = EthType.IPv4;
						}
						try {
							rule.nw_dst_prefix_and_mask = IPv4AddressWithMask.of(jp.getText());
						} catch (IllegalArgumentException e) {
							log.error("Unable to parse destination IP: {}", jp.getText());
						}
					}
				}
				else if (n.equalsIgnoreCase("nw-proto")) {
					if (jp.getText().equalsIgnoreCase("TCP")) {
						rule.any_nw_proto = false;
						rule.nw_proto = IpProtocol.TCP;
						rule.any_dl_type = false;
						rule.dl_type = EthType.IPv4;
					} else if (jp.getText().equalsIgnoreCase("UDP")) {
						rule.any_nw_proto = false;
						rule.nw_proto = IpProtocol.UDP;
						rule.any_dl_type = false;
						rule.dl_type = EthType.IPv4;
					} else if (jp.getText().equalsIgnoreCase("ICMP")) {
						rule.any_nw_proto = false;
						rule.nw_proto = IpProtocol.ICMP;
						rule.any_dl_type = false;
						rule.dl_type = EthType.IPv4;
					}
				}
				else if (n.equalsIgnoreCase("tp-src")) {
					rule.any_tp_src = false;
					try {
						rule.tp_src = TransportPort.of(Integer.parseInt(jp.getText()));
					} catch (IllegalArgumentException e) {
						log.error("Unable to parse source transport port: {}", jp.getText());
					}
				}
				else if (n.equalsIgnoreCase("tp-dst")) {
					rule.any_tp_dst = false;
					try {
						rule.tp_dst = TransportPort.of(Integer.parseInt(jp.getText()));
					} catch (IllegalArgumentException e) {
						log.error("Unable to parse destination transport port: {}", jp.getText());
					}
				}
				else if (n.equalsIgnoreCase("priority")) {
					try {
						rule.priority = Integer.parseInt(jp.getText());
					} catch (IllegalArgumentException e) {
						log.error("Unable to parse priority: {}", jp.getText());
					}
				}
				else if (n.equalsIgnoreCase("action")) {
					if (jp.getText().equalsIgnoreCase("allow") || jp.getText().equalsIgnoreCase("accept")) {
						rule.action = FirewallRule.FirewallAction.ALLOW;
					} else if (jp.getText().equalsIgnoreCase("deny") || jp.getText().equalsIgnoreCase("drop")) {
						rule.action = FirewallRule.FirewallAction.DROP;
					}
				}
			}
		} catch (IOException e) {
			log.error("Unable to parse JSON string: {}", e);
		}
		return rule;
	}
	public static boolean checkRuleExists(FirewallRule rule, List<FirewallRule> rules) {
		Iterator<FirewallRule> iter = rules.iterator();
		while (iter.hasNext()) {
			FirewallRule r = iter.next();
			if (rule.isSameAs(r)) {
				return true;
			}
		}
		return false;
	}
}
